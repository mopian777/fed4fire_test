# -*- coding: utf-8 -*-
"""
RL-Bandit 调度的 FedAvg 服务器（2 个客户端版本）
- 模型: 简单线性回归 y = w * x + b
- 动作空间:
    a0: 使用 client1 + client2
    a1: 只使用 client1
    a2: 只使用 client2
- 强化学习: epsilon-greedy bandit，在每轮用负的 global_loss 作为 reward
"""

from __future__ import print_function

import socket
import json
import time
import math
import random
import threading

HOST = "0.0.0.0"
PORT = 38001          # 你现在使用的端口
MAX_ROUNDS = 200      # 训练轮数
EXPECTED_CLIENTS = 2  # 这里假定就 2 个 client: client1, client2

# ------------ 工具函数 ------------

def send_json(sock, obj):
    data = json.dumps(obj).encode("utf-8") + b"\n"
    sock.sendall(data)


def recv_json_line(fobj):
    """从 socket.makefile() 读取一行 JSON"""
    line = fobj.readline()
    if not line:
        return None
    return json.loads(line.decode("utf-8").strip())


# ------------ Bandit 实现（带 NaN/inf 保护） ------------

class EpsGreedyBandit(object):
    def __init__(self, num_actions, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.q_values = [0.0] * num_actions
        self.counts = [0] * num_actions

    def select_action(self):
        """带兜底的 epsilon-greedy:
        - 只在 Q 是有限数的 action 上做选择
        - 如果全是 NaN/inf，则退化为在所有 action 上随机
        """
        finite_actions = []
        for a, q in enumerate(self.q_values):
            if q is not None and math.isfinite(q):
                finite_actions.append(a)

        if not finite_actions:
            # 全是 NaN / inf，退化为全部动作都可选
            finite_actions = list(range(self.num_actions))

        if random.random() < self.epsilon:
            return random.choice(finite_actions)

        best_a = finite_actions[0]
        best_q = self.q_values[best_a]
        for a in finite_actions[1:]:
            if self.q_values[a] > best_q:
                best_q = self.q_values[a]
                best_a = a
        return best_a

    def update(self, action, reward):
        """增量式更新 Q，忽略 NaN/inf 的 reward"""
        if reward is None or (not math.isfinite(reward)):
            return
        self.counts[action] += 1
        n = self.counts[action]
        old_q = self.q_values[action]
        self.q_values[action] = old_q + (reward - old_q) / float(n)


# ------------ 服务器主逻辑 ------------

def run_server(host=HOST, port=PORT):
    random.seed(2025)

    # 初始化全局模型
    global_w = 0.0
    global_b = 0.0

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 允许端口复用，方便重启
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(5)

    print("[SERVER] Listening on {}:{} ...".format(host, port))

    clients = {}      # client_id -> (sock, fobj)
    client_order = [] # 记录 client_id 的顺序，方便 action 映射

    # -------- 等待客户端连接并握手 --------
    while len(clients) < EXPECTED_CLIENTS:
        conn, addr = server_sock.accept()
        fobj = conn.makefile("rwb", buffering=0)
        hello = recv_json_line(fobj)
        if not hello or hello.get("type") != "hello":
            conn.close()
            continue
        cid = hello.get("client_id", "unknown")
        print("[SERVER] Connection from {} as {}".format(addr, cid))

        clients[cid] = (conn, fobj)
        client_order.append(cid)

    if len(client_order) != EXPECTED_CLIENTS:
        print("[SERVER] Warning: expected {} clients, got {}".format(
            EXPECTED_CLIENTS, len(client_order)
        ))

    # 定义动作 -> 参与的 client_id
    # 这里假定 client_order[0] = "client1", client_order[1] = "client2"
    actions = [
        [client_order[0], client_order[1]],  # a0: 两个都用
        [client_order[0]],                  # a1: 只 client1
        [client_order[1]],                  # a2: 只 client2
    ]
    num_actions = len(actions)
    bandit = EpsGreedyBandit(num_actions=num_actions, epsilon=0.1)

    print("[SERVER] Registered clients:", client_order)
    print("[SERVER] Actions mapping:", actions)

    try:
        for round_idx in range(1, MAX_ROUNDS + 1):
            # ---- 用 bandit 选择本轮动作 ----
            action_idx = bandit.select_action()
            selected_clients = actions[action_idx]

            print("[SERVER][round {}] action={}, selected={}".format(
                round_idx, action_idx, selected_clients
            ))

            # ---- 下发训练指令 ----
            send_ts = {}
            local_epochs = 5
            lr = 0.02  # 稍微小一点，稳定一些

            for cid in selected_clients:
                sock, _ = clients[cid]
                msg = {
                    "type": "train",
                    "round": round_idx,
                    "w": global_w,
                    "b": global_b,
                    "local_epochs": local_epochs,
                    "lr": lr,
                }
                send_json(sock, msg)
                send_ts[cid] = time.time()

            # ---- 接收各 client 的更新 ----
            round_results = []
            for cid in selected_clients:
                _, fobj = clients[cid]
                resp = recv_json_line(fobj)
                if resp is None:
                    print("[SERVER] WARNING: {} closed connection".format(cid))
                    continue

                now = time.time()
                delay_ms = (now - send_ts[cid]) * 1000.0

                new_w = float(resp.get("new_w", global_w))
                new_b = float(resp.get("new_b", global_b))
                local_loss = resp.get("local_loss", None)

                print("[SERVER] <- {}: delay={:.3f} ms, local_loss={}".format(
                    resp.get("client_id", cid),
                    delay_ms,
                    local_loss,
                ))

                round_results.append({
                    "client_id": cid,
                    "new_w": new_w,
                    "new_b": new_b,
                    "local_loss": local_loss,
                    "delay_ms": delay_ms,
                })

            if not round_results:
                print("[SERVER] No results in round {}, stopping".format(round_idx))
                break

            # ---- FedAvg 聚合（简单平均） ----
            avg_w = 0.0
            avg_b = 0.0
            for r in round_results:
                avg_w += r["new_w"]
                avg_b += r["new_b"]
            avg_w /= float(len(round_results))
            avg_b /= float(len(round_results))

            global_w = avg_w
            global_b = avg_b

            # ---- 计算 global_loss，并对 inf/NaN 做保护 ----
            finite_losses = []
            for r in round_results:
                loss = r["local_loss"]
                if loss is None:
                    continue
                try:
                    loss_f = float(loss)
                except Exception:
                    continue
                if math.isfinite(loss_f):
                    finite_losses.append(loss_f)

            if finite_losses:
                global_loss = sum(finite_losses) / float(len(finite_losses))
            else:
                # 全是 inf/NaN，用一个很大的常数表示很差
                global_loss = 1e3

            # 用负的 global_loss 作为 reward（loss 越小 reward 越大）
            reward = -global_loss

            round_time_ms = max(r["delay_ms"] for r in round_results)

            print("[SERVER][round {}] round_time={:.3f} ms, global_loss={}, reward={}".format(
                round_idx,
                round_time_ms,
                global_loss,
                reward,
            ))

            # ---- 更新 bandit Q 值 ----
            bandit.update(action_idx, reward)

            # 打印一下 Q 状态
            parts = []
            for a in range(num_actions):
                parts.append(
                    "a{}: Q={}, n={}".format(a, bandit.q_values[a], bandit.counts[a])
                )
            print("[SERVER] Bandit Q-values:", " | ".join(parts))

        print("[SERVER] Training finished. Final global model: w={}, b={}".format(
            global_w, global_b
        ))

        # 告诉各个 client 停止
        stop_msg = {"type": "stop"}
        for cid, (sock, _) in clients.items():
            try:
                send_json(sock, stop_msg)
            except Exception:
                pass

    finally:
        for cid, (sock, fobj) in clients.items():
            try:
                fobj.close()
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
        server_sock.close()
        print("[SERVER] Closed all connections.")


if __name__ == "__main__":
    run_server(port=PORT)
