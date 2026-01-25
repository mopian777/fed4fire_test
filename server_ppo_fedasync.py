#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PPO + FedAsync demo (server side, no torch, pure numpy).

Protocol (per persistent TCP connection):

Client:
    send  {"type": "hello", "client_id": "client1"}

Server for each training step:
    -> send {"type": "train", "global_step": int,
             "w": float, "b": float,
             "local_epochs": int, "lr": float}

Client:
    -> send {"type": "update", "client_id": str,
             "local_step": int, "global_step": int,
             "w": float, "b": float,
             "local_loss": float}

At the end:
    server sends {"type": "stop"} to all clients.
"""

from __future__ import print_function

import json
import math
import socket
import time
import csv
import random

import numpy as np


HOST = "0.0.0.0"
PORT = 5000          # 可以按需改，比如 38001，一定要和 client 一致
BACKLOG = 8

MAX_GLOBAL_STEPS = 200   # 总共异步更新次数
LOCAL_EPOCHS = 5
LOCAL_LR = 0.02
ASYNC_ALPHA = 0.5        # 异步 soft-update 系数

GAMMA = 0.99             # PPO 折扣
CLIP_EPS = 0.2
PPO_LR = 1e-3
PPO_UPDATES_PER_BATCH = 4
PPO_BATCH_SIZE = 64


# ---------- 简单 socket / json 工具 ----------

def send_json(sock, obj):
    data = json.dumps(obj) + "\n"
    sock.sendall(data.encode("utf-8"))


def recv_json_line(fobj):
    line = fobj.readline()
    if not line:
        raise IOError("socket closed")
    return json.loads(line.strip())


# ---------- 极简 PPO Agent（numpy 实现） ----------

class PPOAgent(object):
    """
    非常小的两层 MLP，用于 discrete action PPO：
    state_dim -> hidden_dim(16) -> 2 actions (for 2 clients)
    另外一条 value head 输出 V(s)。
    """

    def __init__(self, state_dim, n_actions,
                 hidden_dim=16,
                 lr=PPO_LR,
                 gamma=GAMMA,
                 clip_eps=CLIP_EPS):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps

        # 参数初始化（小随机数）
        rng = np.random.RandomState(42)
        self.W1 = 0.1 * rng.randn(state_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)

        self.W2_pi = 0.1 * rng.randn(hidden_dim, n_actions)
        self.b2_pi = np.zeros(n_actions)

        self.W2_v = 0.1 * rng.randn(hidden_dim, 1)
        self.b2_v = np.zeros(1)

        # 经验缓存
        self.reset_buffer()

    def reset_buffer(self):
        self.buf_states = []
        self.buf_actions = []
        self.buf_rewards = []
        self.buf_values = []
        self.buf_logprobs = []

    # 前向：给 state，算 policy probs 和 V(s)
    def _forward(self, state_vector):
        x = np.asarray(state_vector, dtype=np.float32).reshape(1, -1)  # (1, D)
        h_raw = np.dot(x, self.W1) + self.b1  # (1, H)
        h = np.tanh(h_raw)                    # (1, H)

        logits = np.dot(h, self.W2_pi) + self.b2_pi  # (1, A)
        # softmax
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (1, A)

        v = np.dot(h, self.W2_v) + self.b2_v  # (1, 1)

        return probs.ravel(), float(v.ravel()[0]), h.ravel(), h_raw.ravel()

    def select_action(self, state_vector):
        probs, v, _, _ = self._forward(state_vector)
        # 采样一个 action
        a = np.random.choice(self.n_actions, p=probs)
        logp = math.log(max(probs[a], 1e-8))
        # 缓存
        self.buf_states.append(np.asarray(state_vector, dtype=np.float32))
        self.buf_actions.append(a)
        self.buf_values.append(v)
        self.buf_logprobs.append(logp)
        return a, v, logp

    def store_reward(self, r):
        self.buf_rewards.append(float(r))

    def _compute_returns_advantages(self):
        # 简单 discounted return + advantage = return - value
        rewards = np.asarray(self.buf_rewards, dtype=np.float32)
        values = np.asarray(self.buf_values, dtype=np.float32)
        n = len(rewards)

        returns = np.zeros_like(rewards)
        running = 0.0
        for t in range(n - 1, -1, -1):
            running = rewards[t] + self.gamma * running
            returns[t] = running

        advantages = returns - values
        # 标准化 advantage 稳定训练
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        return returns, advantages

    def update(self, epochs=PPO_UPDATES_PER_BATCH, batch_size=PPO_BATCH_SIZE):
        if len(self.buf_rewards) == 0:
            return

        states = np.asarray(self.buf_states, dtype=np.float32)    # (N, D)
        actions = np.asarray(self.buf_actions, dtype=np.int64)    # (N,)
        old_logprobs = np.asarray(self.buf_logprobs, dtype=np.float32)  # (N,)
        returns, advantages = self._compute_returns_advantages()
        N = states.shape[0]

        for _ in range(epochs):
            # 简单 full-batch（N 很小），也可以 shuffle 后做 mini-batch
            idxs = np.arange(N)

            # 梯度累积
            gW1 = np.zeros_like(self.W1)
            gb1 = np.zeros_like(self.b1)
            gW2_pi = np.zeros_like(self.W2_pi)
            gb2_pi = np.zeros_like(self.b2_pi)
            gW2_v = np.zeros_like(self.W2_v)
            gb2_v = np.zeros_like(self.b2_v)

            for i in idxs:
                s = states[i]
                a = int(actions[i])
                old_logp = old_logprobs[i]
                ret = returns[i]
                adv = advantages[i]

                # forward
                x = s.reshape(1, -1)                       # (1, D)
                h_raw = np.dot(x, self.W1) + self.b1       # (1, H)
                h = np.tanh(h_raw)                         # (1, H)

                logits = np.dot(h, self.W2_pi) + self.b2_pi  # (1, A)
                logits = logits - np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (1, A)
                probs_r = probs.ravel()
                logp = math.log(max(probs_r[a], 1e-8))

                v = float(np.dot(h, self.W2_v) + self.b2_v)   # scalar

                # ratio + clipped surrogate
                ratio = math.exp(logp - old_logp)
                clipped_ratio = max(min(ratio, 1.0 + self.clip_eps), 1.0 - self.clip_eps)

                term1 = ratio * adv
                term2 = clipped_ratio * adv

                # 这里我们最小化 -min(term1, term2)
                if term1 < term2:
                    # 有梯度的分支
                    policy_loss_grad_coeff = -adv * ratio
                    # dL/dlogp = policy_loss_grad_coeff
                else:
                    # 被 clip，视作常数，不反向
                    policy_loss_grad_coeff = 0.0

                # value loss: 0.5 * (v - ret)^2
                dv = (v - ret)  # dL/dv
                # 总的对 logit head 和 value head 的梯度叠加在 dL/dh 上

                # ----- policy head: log-softmax 的梯度 -----
                # logits grad: d logpi / d logits = one_hot(a) - probs
                one_hot = np.zeros_like(probs_r)
                one_hot[a] = 1.0
                dlogp_dlogits = one_hot - probs_r  # (A,)

                # dL/dlogits = dL/dlogp * dlogp/dlogits
                dL_dlogits = policy_loss_grad_coeff * dlogp_dlogits  # (A,)

                # 对 W2_pi, b2_pi 的梯度
                # logits = h (1,H) * W2_pi (H,A)
                gW2_pi += np.outer(h.ravel(), dL_dlogits)
                gb2_pi += dL_dlogits

                # 对 h 的梯度来自 policy head
                dL_dh_from_policy = np.dot(self.W2_pi, dL_dlogits)  # (H,)

                # ----- value head -----
                # v = h W2_v + b2_v
                gW2_v += np.outer(h.ravel(), np.array([dv]))
                gb2_v += np.array([dv])

                dL_dh_from_value = (self.W2_v * dv).ravel()  # (H,)

                # 合并两个 head 对 h 的梯度
                dL_dh = dL_dh_from_policy + dL_dh_from_value

                # ----- 反向到 W1, b1 -----
                # h = tanh(h_raw)
                dh_raw = (1.0 - np.tanh(h_raw).ravel() ** 2) * dL_dh  # (H,)

                # h_raw = x W1 + b1
                gW1 += np.outer(x.ravel(), dh_raw)
                gb1 += dh_raw

            # 梯度平均并更新参数
            scale = 1.0 / float(N)
            self.W1 -= self.lr * gW1 * scale
            self.b1 -= self.lr * gb1 * scale
            self.W2_pi -= self.lr * gW2_pi * scale
            self.b2_pi -= self.lr * gb2_pi * scale
            self.W2_v -= self.lr * gW2_v * scale
            self.b2_v -= self.lr * gb2_v * scale

        # update 后清空 buffer
        self.reset_buffer()


# ---------- FedAsync + PPO Server ----------

def build_state(client_stats, client_ids, global_step):
    """
    client_stats[cid] = dict(last_delay_ms, last_loss, last_step)
    state: [d1,d2, loss1,loss2, staleness1,staleness2]  (for 2 clients)
    """
    state = []
    for cid in client_ids:
        st = client_stats.get(cid, {})
        d = st.get("last_delay_ms", 0.0)
        loss = st.get("last_loss", 0.3)
        step = st.get("last_step", 0)
        staleness = max(global_step - step, 0)

        # 粗糙归一化
        d_norm = d / 50.0
        if d_norm > 1.0:
            d_norm = 1.0
        loss_norm = loss  # 这里本身就在 0.x
        staleness_norm = staleness / float(max(global_step, 1))
        state.extend([d_norm, loss_norm, staleness_norm])
    return np.asarray(state, dtype=np.float32)


def run_server(port=PORT):
    print("[SERVER] Listening on 0.0.0.0:{0} ...".format(port))
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind((HOST, port))
    listen_sock.listen(BACKLOG)

    clients = {}       # cid -> (sock, fobj, stats)
    client_ids = []

    # ---- 接收 2 个 client 连接 ----
    while len(clients) < 2:
        sock, addr = listen_sock.accept()
        fobj = sock.makefile("r")
        hello = recv_json_line(fobj)
        cid = hello.get("client_id", "unknown")
        print("[SERVER] Client {0} connected from {1}".format(cid, addr))
        clients[cid] = {"sock": sock, "fobj": fobj, "stats": {}}
        client_ids.append(cid)

    client_ids.sort()
    print("[SERVER] Registered clients:", client_ids)

    # ---- 初始化全局模型 ----
    w = 0.0
    b = 0.0
    global_step = 0

    # ---- PPO agent ----
    state_dim = 3 * len(client_ids)
    agent = PPOAgent(state_dim=state_dim, n_actions=len(client_ids))

    # ---- CSV 记录 ----
    csv_path = "/tmp/ppo_fedasync_curve.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow([
        "step", "chosen_client",
        "global_w", "global_b",
        "global_loss", "delay_ms",
        "reward"
    ])

    # ---- 主循环 ----
    try:
        while global_step < MAX_GLOBAL_STEPS:
            global_step += 1

            # 构造 state，选一个 client
            state = build_state(
                dict((cid, clients[cid]["stats"]) for cid in client_ids),
                client_ids,
                global_step
            )
            action_idx, v_pred, logp = agent.select_action(state)
            chosen_cid = client_ids[action_idx]
            c = clients[chosen_cid]
            sock = c["sock"]
            fobj = c["fobj"]

            # 发训练指令
            cmd = {
                "type": "train",
                "global_step": global_step,
                "w": w,
                "b": b,
                "local_epochs": LOCAL_EPOCHS,
                "lr": LOCAL_LR,
            }
            send_json(sock, cmd)
            t0 = time.time()

            # 等这个 client 回 update
            resp = recv_json_line(fobj)
            if resp.get("type") != "update":
                print("[SERVER] unexpected msg from {0}: {1}".format(chosen_cid, resp))
                break

            t1 = time.time()
            delay_ms = (t1 - t0) * 1000.0

            new_w = float(resp["w"])
            new_b = float(resp["b"])
            local_loss = float(resp["local_loss"])
            local_step = int(resp.get("local_step", global_step))

            # 异步 soft-update
            w = (1.0 - ASYNC_ALPHA) * w + ASYNC_ALPHA * new_w
            b = (1.0 - ASYNC_ALPHA) * b + ASYNC_ALPHA * new_b

            # 这里直接用 local_loss 近似 global_loss
            global_loss = local_loss

            # 更新该 client 的 stats
            st = c["stats"]
            st["last_delay_ms"] = delay_ms
            st["last_loss"] = local_loss
            st["last_step"] = local_step

            # reward：同时惩罚 loss 和 delay
            delay_norm = delay_ms / 50.0
            if delay_norm > 1.0:
                delay_norm = 1.0
            lam = 0.5
            reward = - (global_loss + lam * delay_norm)

            agent.store_reward(reward)

            csv_writer.writerow([
                global_step, chosen_cid,
                w, b,
                global_loss, delay_ms,
                reward
            ])

            if global_step % 10 == 0:
                print("[SERVER] step={0}, chosen={1}, loss={2:.6f}, delay={3:.3f} ms, w={4:.4f}, b={5:.4f}".format(
                    global_step, chosen_cid, global_loss, delay_ms, w, b
                ))

            # 定期更新 PPO
            if global_step % 50 == 0:
                print("[SERVER] PPO update at step", global_step)
                agent.update()

        print("[SERVER] Training finished. Final global model: w={0}, b={1}".format(w, b))

        # 发 stop 给所有 clients
        stop_msg = {"type": "stop"}
        for cid in client_ids:
            send_json(clients[cid]["sock"], stop_msg)

    finally:
        try:
            csv_f.close()
            print("[SERVER] CSV saved to", csv_path)
        except Exception:
            pass

        for cid in client_ids:
            try:
                clients[cid]["sock"].close()
            except Exception:
                pass

        listen_sock.close()
        print("[SERVER] Closed all connections.")


if __name__ == "__main__":
    run_server(port=PORT)
