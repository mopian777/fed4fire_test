#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
jFed-style RL-FedAvg server with PPO-based client selection.

- Listens on 0.0.0.0:5000 and accepts exactly 2 persistent client connections.
- Each client must first send: {"type": "hello", "client_id": "client1" | "client2"}
- In each round, PPO chooses which subset of clients to train:
    action 0 -> {client1, client2}
    action 1 -> {client1}
    action 2 -> {client2}
- For selected clients, server sends a "train" command with current global weights;
  clients train locally and send back updated weights + local loss.
- Server measures per-client round time as (recv_time - send_time).
- Reward per round: r = - (alpha * round_time + beta * global_loss)
- Every K rounds, PPO updates its policy.

NOTE:
- This is a *framework draft*: you still need to adapt the local model and
  training details to your real application.
- For simplicity, the global model is just y = w x + b (two scalars).
"""

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import random
import math
import torch
import torch.nn as nn
import torch.optim as optim


# --------------------------------------------------------------------
#  Utilities: JSON over TCP (length-prefixed)
# --------------------------------------------------------------------

def send_json(sock: socket.socket, obj: dict):
    data = json.dumps(obj).encode("utf-8")
    # 4-byte big-endian length prefix
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf


def recv_json(sock: socket.socket) -> dict:
    # read 4-byte length prefix
    header = recv_exact(sock, 4)
    (length,) = struct.unpack("!I", header)
    data = recv_exact(sock, length)
    return json.loads(data.decode("utf-8"))


# --------------------------------------------------------------------
#  Simple global "model": y = w x + b
#  (we just keep two scalars and let clients do real training)
# --------------------------------------------------------------------

@dataclass
class GlobalModel:
    w: float = 0.0
    b: float = 0.0

    def to_dict(self):
        return {"w": self.w, "b": self.b}

    @staticmethod
    def from_dict(d: dict) -> "GlobalModel":
        return GlobalModel(w=float(d["w"]), b=float(d["b"]))

    @staticmethod
    def fedavg(models: List["GlobalModel"]) -> "GlobalModel":
        if not models:
            return GlobalModel()
        w = sum(m.w for m in models) / len(models)
        b = sum(m.b for m in models) / len(models)
        return GlobalModel(w=w, b=b)


# --------------------------------------------------------------------
#  PPO agent (简化版)，和前面 demo 基本一致
# --------------------------------------------------------------------

class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value


@dataclass
class PPOMemory:
    states: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        update_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.net = PolicyValueNet(state_dim, n_actions).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.memory = PPOMemory()

    def select_action(self, state_vec) -> Tuple[int, torch.Tensor, torch.Tensor]:
        state = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze(0), value.squeeze(0)

    def store(self, state_vec, action, log_prob, value, reward, done):
        self.memory.states.append(torch.tensor(state_vec, dtype=torch.float32))
        self.memory.actions.append(action)
        self.memory.log_probs.append(log_prob.detach())
        self.memory.rewards.append(reward)
        self.memory.dones.append(done)
        self.memory.values.append(value.detach())

    def _compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0.0
        values = values + [torch.tensor(0.0)]  # V_{T+1}=0
        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * (1.0 - float(dones[t]))
                - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)
        returns = [adv + v for adv, v in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self):
        if not self.memory.states:
            return

        device = self.device
        states = torch.stack(self.memory.states).to(device)
        actions = torch.tensor(self.memory.actions, dtype=torch.long, device=device)
        old_log_probs = torch.stack(self.memory.log_probs).to(device)
        values = [v.to(device) for v in self.memory.values]
        rewards = self.memory.rewards
        dones = self.memory.dones

        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages = torch.stack(advantages).detach()
        returns = torch.stack(returns).detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = states.size(0)
        idxs = list(range(n))

        for _ in range(self.update_epochs):
            random.shuffle(idxs)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]
                if not batch_idx:
                    continue

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                logits, values_pred = self.net(b_states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values_pred.squeeze(-1), b_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.opt.step()

        self.memory.clear()


# --------------------------------------------------------------------
#  Client connection metadata
# --------------------------------------------------------------------

@dataclass
class ClientConn:
    sock: socket.socket
    addr: Tuple[str, int]
    client_id: str
    last_delay_ms: float = 0.7
    last_local_loss: float = 1.0


# --------------------------------------------------------------------
#  Main RL-FedAvg server
# --------------------------------------------------------------------

def run_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    n_rounds: int = 200,
    local_epochs: int = 5,
    lr_local: float = 0.05,
    alpha_delay: float = 1.0,
    beta_loss: float = 0.5,
    ppo_update_interval: int = 20,
):
    print(f"[SERVER] Listening on {host}:{port} ...")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(2)

    # Accept 2 clients
    clients: Dict[str, ClientConn] = {}
    while len(clients) < 2:
        sock, addr = server_sock.accept()
        print(f"[SERVER] Connection from {addr}")
        # wait for hello
        msg = recv_json(sock)
        if msg.get("type") != "hello":
            print("[SERVER] Invalid first message, closing.")
            sock.close()
            continue
        cid = msg.get("client_id")
        if cid in clients:
            print(f"[SERVER] Duplicate client_id {cid}, closing.")
            sock.close()
            continue
        clients[cid] = ClientConn(sock=sock, addr=addr, client_id=cid)
        print(f"[SERVER] Registered client_id={cid} from {addr}")

    if not ("client1" in clients and "client2" in clients):
        print("[SERVER] Expecting client1 and client2; got:", list(clients.keys()))
        return

    c1 = clients["client1"]
    c2 = clients["client2"]

    # Init global model
    global_model = GlobalModel(w=0.0, b=0.0)

    # PPO agent state: [delay_c1, delay_c2, loss_c1, loss_c2, global_loss]
    state_dim = 5
    n_actions = 3  # {1,2}, {1}, {2}
    agent = PPOAgent(state_dim, n_actions, device="cpu")

    last_global_loss = 1.0

    def select_subset(action_idx: int) -> List[ClientConn]:
        if action_idx == 0:
            return [c1, c2]
        elif action_idx == 1:
            return [c1]
        else:
            return [c2]

    for r in range(1, n_rounds + 1):
        # 构造 RL 状态
        state_vec = [
            c1.last_delay_ms,
            c2.last_delay_ms,
            c1.last_local_loss,
            c2.last_local_loss,
            last_global_loss,
        ]
        action_idx, log_prob, value_est = agent.select_action(state_vec)
        selected_clients = select_subset(action_idx)
        selected_ids = [c.client_id for c in selected_clients]

        print(f"[SERVER][round {r}] action={action_idx}, selected={selected_ids}, "
              f"weights=({global_model.w:.4f},{global_model.b:.4f})")

        # 向选中的 client 发送 train 指令
        send_times: Dict[str, float] = {}
        for c in selected_clients:
            cmd = {
                "type": "train",
                "round": r,
                "weights": global_model.to_dict(),
                "local_epochs": local_epochs,
                "lr": lr_local,
            }
            send_json(c.sock, cmd)
            send_times[c.client_id] = time.time()

        # 接收更新
        updated_models: List[GlobalModel] = []
        local_losses: List[float] = []
        delays_ms: List[float] = []

        for c in selected_clients:
            msg = recv_json(c.sock)
            if msg.get("type") != "update" or msg.get("round") != r:
                print(f"[SERVER] Unexpected message from {c.client_id}: {msg}")
                continue

            now = time.time()
            delay_ms = (now - send_times[c.client_id]) * 1000.0
            c.last_delay_ms = delay_ms

            wdict = msg["weights"]
            local_loss = float(msg["local_loss"])
            c.last_local_loss = local_loss

            updated_models.append(GlobalModel.from_dict(wdict))
            local_losses.append(local_loss)
            delays_ms.append(delay_ms)

            print(f"[SERVER] <- {c.client_id}: delay={delay_ms:.3f} ms, "
                  f"local_loss={local_loss:.6f}")

        if updated_models:
            # FedAvg 聚合
            global_model = GlobalModel.fedavg(updated_models)
            round_time_ms = max(delays_ms) if delays_ms else 0.0
            # 用选中 client 的 local_loss 平均近似 global_loss
            last_global_loss = sum(local_losses) / len(local_losses)
        else:
            round_time_ms = 0.0
            last_global_loss = last_global_loss  # unchanged

        # RL 奖励
        reward = - (alpha_delay * round_time_ms + beta_loss * last_global_loss)
        done = False

        agent.store(state_vec, action_idx, log_prob, value_est, reward, done)

        print(f"[SERVER][round {r}] round_time={round_time_ms:.3f} ms, "
              f"global_loss≈{last_global_loss:.6f}, reward={reward:.4f}")

        # 定期更新 PPO
        if r % ppo_update_interval == 0:
            print(f"[SERVER] PPO update at round {r}")
            agent.update()

    # 训练结束，通知客户端停止
    stop_msg = {"type": "stop"}
    for c in clients.values():
        try:
            send_json(c.sock, stop_msg)
        except Exception:
            pass
        c.sock.close()

    server_sock.close()
    print("[SERVER] Training finished, sockets closed.")


if __name__ == "__main__":
    run_server()
