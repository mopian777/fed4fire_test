#!/usr/bin/env python3
# RL-FedAvg server (bandit) with unified global metrics logging (loss + acc + reward)

import socket
import json
import random
import time
import os

PORT = 38001
NUM_ROUNDS = 200
LOCAL_EPOCHS = 5
LR = 0.02

METHOD = "rl_fedavg"
LOG_PATH = "/tmp/global_metrics_rl_fedavg.csv"

# reward 权重（你可以后续调参/归一化）
ALPHA_TIME = 0.001   # round_time_ms 的系数（注意量纲）
BETA_LOSS = 1.0
GAMMA_ACC = 1.0


def send_json(sock, obj):
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def recv_json_line(fobj):
    line = fobj.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


def ensure_log_header():
    if not os.path.exists("/tmp"):
        try:
            os.makedirs("/tmp")
        except Exception:
            pass
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            f.write("method,round,wall_time_s,round_time_ms,selected_clients,global_loss,global_acc,reward\n")


def log_global(round_idx, wall_time_s, round_time_ms, selected_clients, global_loss, global_acc, reward):
    with open(LOG_PATH, "a") as f:
        f.write("%s,%d,%.6f,%.3f,%s,%.10f,%.6f,%.10f\n" % (
            METHOD,
            int(round_idx),
            float(wall_time_s),
            float(round_time_ms),
            "\"%s\"" % ("|".join(selected_clients)),
            float(global_loss),
            float(global_acc),
            float(reward),
        ))


def bandit_init(n_actions):
    Q = [0.0] * n_actions
    N = [0] * n_actions
    return Q, N


def bandit_select(Q, eps=0.1):
    if random.random() < eps:
        return random.randint(0, len(Q) - 1)
    best_i = 0
    best_q = None
    for i, q in enumerate(Q):
        if best_q is None or q > best_q:
            best_q = q
            best_i = i
    return best_i


def bandit_update(Q, N, action, reward):
    N[action] += 1
    alpha = 1.0 / float(N[action])
    Q[action] = Q[action] + alpha * (reward - Q[action])


def run_server(port=PORT):
    ensure_log_header()
    t_start = time.time()

    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("0.0.0.0", port))
    lsock.listen(5)

    print("[SERVER] Listening on 0.0.0.0:%d ..." % port)

    clients = []
    while len(clients) < 2:
        csock, addr = lsock.accept()
        fobj = csock.makefile("r")
        msg = recv_json_line(fobj)
        if msg is None or msg.get("type") != "HELLO":
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
        clients.append({"id": cid, "sock": csock, "file": fobj})
        print("[SERVER] accepted %s from %s:%d" % (cid, addr[0], addr[1]))

    def sort_key(x):
        if x["id"] == "client1":
            return 0
        if x["id"] == "client2":
            return 1
        return 2

    clients.sort(key=sort_key)

    # global model
    w, b = 0.0, 0.0

    # actions: 0=both, 1=only c1, 2=only c2
    actions = [[0, 1], [0], [1]]
    Q, N = bandit_init(len(actions))

    for rnd in range(1, NUM_ROUNDS + 1):
        t0 = time.time()

        action = bandit_select(Q, eps=0.1)
        sel_idx = actions[action]
        sel_clients = [clients[i]["id"] for i in sel_idx]

        # send TRAIN
        for i in sel_idx:
            send_json(clients[i]["sock"], {
                "type": "TRAIN",
                "round": rnd,
                "w": w,
                "b": b,
                "local_epochs": LOCAL_EPOCHS,
                "lr": LR,
            })

        # recv UPDATE + aggregate
        agg_w, agg_b = [], []
        agg_loss, agg_acc, agg_n = [], [], []

        for i in sel_idx:
            resp = recv_json_line(clients[i]["file"])
            if resp is None or resp.get("type") != "UPDATE":
                print("[SERVER] client %s disconnected." % clients[i]["id"])
                return

            agg_w.append(float(resp.get("w", w)))
            agg_b.append(float(resp.get("b", b)))
            agg_loss.append(float(resp.get("local_loss", 0.0)))
            agg_acc.append(float(resp.get("local_acc", 0.0)))
            agg_n.append(int(resp.get("n_samples", 1)))

        # FedAvg（简单平均；也可按 n_samples 加权）
        if len(agg_w) > 0:
            w = sum(agg_w) / float(len(agg_w))
            b = sum(agg_b) / float(len(agg_b))

            # loss/acc 用 n_samples 加权更合理
            tot = float(sum(agg_n))
            global_loss = sum([agg_loss[k] * agg_n[k] for k in range(len(agg_n))]) / tot
            global_acc = sum([agg_acc[k] * agg_n[k] for k in range(len(agg_n))]) / tot
        else:
            global_loss, global_acc = 0.0, 0.0

        round_time_ms = (time.time() - t0) * 1000.0
        wall_time_s = time.time() - t_start

        # reward（示例：越快越好、loss 越低越好、acc 越高越好）
        reward = -ALPHA_TIME * round_time_ms - BETA_LOSS * global_loss + GAMMA_ACC * global_acc

        # bandit 更新
        bandit_update(Q, N, action, reward)

        log_global(rnd, wall_time_s, round_time_ms, sel_clients, global_loss, global_acc, reward)

        print("[SERVER] round=%d sel=%s loss=%.6f acc=%.4f time=%.1fms reward=%.6f" %
              (rnd, str(sel_clients), global_loss, global_acc, round_time_ms, reward))

    # stop
    for c in clients:
        try:
            send_json(c["sock"], {"type": "STOP"})
        except Exception:
            pass
        try:
            c["file"].close()
        except Exception:
            pass
        try:
            c["sock"].close()
        except Exception:
            pass

    try:
        lsock.close()
    except Exception:
        pass

    print("[SERVER] done. log at %s" % LOG_PATH)


if __name__ == "__main__":
    run_server(port=PORT)
