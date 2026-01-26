#!/usr/bin/env python3
# Minimal RL-FedAvg server (bandit-style client selection), no torch, no typing

import socket
import json
import random
import time
import os

PORT = 38001          # 与 client_rl_fedavg.py 默认一致
NUM_ROUNDS = 200
LOCAL_EPOCHS = 5
LR = 0.02

# 日志文件：记录 bandit 决策和全局 loss，用于后续画图
SERVER_LOG = "/tmp/rl_fedavg_bandit_server.csv"


# ------------- JSON over TCP ------------- #

def send_json(sock, obj):
    data = json.dumps(obj) + "\n"
    sock.sendall(data.encode("utf-8"))


def recv_json_line(fobj):
    line = fobj.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


# ------------- Bandit（多臂老虎机）------------- #

def bandit_init(n_actions):
    Q = [0.0] * n_actions   # 行动价值估计
    N = [0] * n_actions     # 每个动作被选择次数
    return Q, N


def bandit_select_action(Q, N, eps=0.1):
    """
    简单 epsilon-greedy：
      - 以 eps 概率探索（随机选动作）
      - 以 1-eps 概率利用（选当前 Q 最大的动作）
    """
    n_actions = len(Q)
    if random.random() < eps:
        return random.randint(0, n_actions - 1)

    best_q = None
    best_idx = 0
    for i in range(n_actions):
        q = Q[i]
        if best_q is None or q > best_q:
            best_q = q
            best_idx = i
    return best_idx


def bandit_update(Q, N, action_idx, reward):
    """
    增量式平均更新：Q[a] <- Q[a] + (1/N[a]) * (reward - Q[a])
    """
    if action_idx < 0 or action_idx >= len(Q):
        return
    N[action_idx] += 1
    n = float(N[action_idx])
    alpha = 1.0 / n
    old_q = Q[action_idx]
    Q[action_idx] = old_q + alpha * (reward - old_q)


# ------------- Helper ------------- #

def init_server_log():
    if not os.path.exists("/tmp"):
        try:
            os.makedirs("/tmp")
        except Exception:
            pass
    if not os.path.exists(SERVER_LOG):
        with open(SERVER_LOG, "w") as f:
            f.write("round,action_idx,use_client1,use_client2,"
                    "global_loss,round_time_ms,Q0,Q1,Q2\n")


def append_server_log(round_idx, action_idx, use_c1, use_c2,
                      global_loss, round_time_ms, Q):
    line = "%d,%d,%d,%d,%.10f,%.3f,%.10f,%.10f,%.10f\n" % (
        int(round_idx),
        int(action_idx),
        int(use_c1),
        int(use_c2),
        float(global_loss),
        float(round_time_ms),
        float(Q[0]),
        float(Q[1]),
        float(Q[2]),
    )
    with open(SERVER_LOG, "a") as f:
        f.write(line)


# ------------- 主流程 ------------- #

def run_server(port=PORT):
    print("[SERVER] Listening on 0.0.0.0:%d ..." % port)

    # 1) 创建监听 socket
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("0.0.0.0", port))
    lsock.listen(5)

    # 2) 等待两个 client 连接并发送 HELLO
    clients = []   # 每个元素: {"id": "client1", "sock": sock, "file": f}
    while len(clients) < 2:
        csock, addr = lsock.accept()
        fobj = csock.makefile("r")
        msg = recv_json_line(fobj)
        if msg is None or msg.get("type") != "HELLO":
            # 不符合协议就关掉
            try:
                fobj.close()
            except Exception:
                pass
            try:
                csock.close()
            except Exception:
                pass
            continue

        cid = msg.get("client_id", "client%d" % (len(clients) + 1))
        info = {"id": cid, "sock": csock, "file": fobj}
        clients.append(info)
        print("[SERVER] accepted %s from %s:%d" %
              (cid, addr[0], addr[1]))

    # 简单根据 client_id 排一下，让 client1 在下标0，client2在下标1
    def client_sort_key(x):
        cid = x["id"]
        if cid == "client1":
            return 0
        if cid == "client2":
            return 1
        return 2

    clients.sort(key=client_sort_key)

    # 3) 初始化模型参数
    w = 0.0
    b = 0.0

    # bandit 行动：0=两端都选, 1=只选 client1, 2=只选 client2
    actions = [
        [0, 1],
        [0],
        [1],
    ]
    Q, N = bandit_init(len(actions))

    init_server_log()

    # 4) 训练循环
    for rnd in range(1, NUM_ROUNDS + 1):
        start_t = time.time()

        # 4.1 选择 bandit 动作（决定这一轮用哪些 client）
        action_idx = bandit_select_action(Q, N, eps=0.1)
        selected_indices = actions[action_idx]

        use_c1 = 1 if 0 in selected_indices else 0
        use_c2 = 1 if 1 in selected_indices else 0

        print("[SERVER] [round %d] action=%d, selected=%s" %
              (rnd, action_idx, str(selected_indices)))

        # 4.2 向被选中的 client 发送 TRAIN
        for i in selected_indices:
            info = clients[i]
            msg = {
                "type": "TRAIN",
                "round": rnd,
                "w": w,
                "b": b,
                "local_epochs": LOCAL_EPOCHS,
                "lr": LR,
            }
            send_json(info["sock"], msg)

        # 4.3 接收 UPDATE 并做 FedAvg
        agg_w = []
        agg_b = []
        agg_loss = []

        for i in selected_indices:
            info = clients[i]
            resp = recv_json_line(info["file"])
            if resp is None or resp.get("type") != "UPDATE":
                # 如果某个 client 异常，中断训练
                print("[SERVER] client %s disconnected during UPDATE." %
                      info["id"])
                break

            new_w = resp.get("w", w)
            new_b = resp.get("b", b)
            local_loss = resp.get("local_loss", 0.0)

            agg_w.append(float(new_w))
            agg_b.append(float(new_b))
            agg_loss.append(float(local_loss))

            print("[SERVER] <- %s: round=%d, local_loss=%.6f" %
                  (info["id"], rnd, local_loss))

        # 如果这一轮没人被选中，就跳过
        if len(agg_w) > 0:
            w = sum(agg_w) / float(len(agg_w))
            b = sum(agg_b) / float(len(agg_b))
            global_loss = sum(agg_loss) / float(len(agg_loss))
        else:
            global_loss = 0.0

        round_time_ms = (time.time() - start_t) * 1000.0

        # 4.4 用 -global_loss 作为 reward 更新 bandit
        # loss 越小 reward 越大
        reward = -float(global_loss)
        bandit_update(Q, N, action_idx, reward)

        print("[SERVER] [round %d] round_time=%.3f ms, global_loss=%.6f | "
              "Q0=%.6f, Q1=%.6f, Q2=%.6f"
              % (rnd, round_time_ms, global_loss, Q[0], Q[1], Q[2]))

        append_server_log(rnd, action_idx, use_c1, use_c2,
                          global_loss, round_time_ms, Q)

    # 5) 发送 STOP 并关闭连接
    stop_msg = {"type": "STOP"}
    for info in clients:
        try:
            send_json(info["sock"], stop_msg)
        except Exception:
            pass

    for info in clients:
        try:
            info["file"].close()
        except Exception:
            pass
        try:
            info["sock"].close()
        except Exception:
            pass

    try:
        lsock.close()
    except Exception:
        pass

    print("[SERVER] training finished. Final global model: w=%.6f, b=%.6f" %
          (w, b))


if __name__ == "__main__":
    run_server(port=PORT)
