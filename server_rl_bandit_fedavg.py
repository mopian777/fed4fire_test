#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RL-FedAvg server (jFed-style) with simple epsilon-greedy bandit (no torch).

- Listens on 0.0.0.0:5000 and accepts exactly 2 client connections.
- Each client must first send: {"type": "hello", "client_id": "client1" | "client2"}.
- In each round, the bandit agent chooses one of 3 actions:
    0 -> use {client1, client2}
    1 -> use {client1}
    2 -> use {client2}

- For selected clients, server sends:
    {
      "type": "train",
      "round": r,
      "weights": {"w": ..., "b": ...},
      "local_epochs": 5,
      "lr": 0.05
    }
  Clients locally train and reply:
    {
      "type": "update",
      "round": r,
      "client_id": "client1",
      "weights": {"w": ..., "b": ...},
      "local_loss": 0.0087
    }

- Server measures per-client round time = (recv_time - send_time) in ms.
- Reward: r = - (alpha_delay * round_time_ms + beta_loss * approx_global_loss).
- Bandit updates Q[action] by incremental average or Q-learning.

This file does NOT import torch or numpy, only standard library.
"""

import json
import socket
import struct
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


# --------------------------------------------------------------------
#  Utilities: JSON over TCP (length-prefixed)
# --------------------------------------------------------------------

def send_json(sock: socket.socket, obj: dict):
    data = json.dumps(obj).encode("utf-8")
    header = struct.pack("!I", len(data))  # 4-byte big-endian length
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
    header = recv_exact(sock, 4)
    (length,) = struct.unpack("!I", header)
    data = recv_exact(sock, length)
    return json.loads(data.decode("utf-8"))


# --------------------------------------------------------------------
#  Simple Global Model: y = w x + b
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
#  Epsilon-greedy multi-armed bandit (no state, no torch)
# --------------------------------------------------------------------

class BanditAgent:
    def __init__(self, n_actions: int, eps: float = 0.1, alpha: float = 0.1):
        self.n_actions = n_actions
        self.eps = eps       # exploration prob
        self.alpha = alpha   # learning rate for Q update
        self.q_values = [0.0 for _ in range(n_actions)]  # estimated value of each action
        self.counts = [0 for _ in range(n_actions)]      # how many times each action used

    def select_action(self) -> int:
        # epsilon-greedy
        if random.random() < self.eps:
            a = random.randint(0, self.n_actions - 1)
        else:
            # argmax q_values
            max_q = max(self.q_values)
            # there may be multiple actions with same max; choose randomly among them
            candidates = [i for i, q in enumerate(self.q_values) if q == max_q]
            a = random.choice(candidates)
        return a

    def update(self, action: int, reward: float):
        self.counts[action] += 1
        # incremental update: Q <- Q + alpha * (r - Q)
        old_q = self.q_values[action]
        new_q = old_q + self.alpha * (reward - old_q)
        self.q_values[action] = new_q

    def summary(self) -> str:
        parts = []
        for a in range(self.n_actions):
            parts.append(f"a{a}: Q={self.q_values[a]:.4f}, n={self.counts[a]}")
        return " | ".join(parts)


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
#  Main RL-FedAvg server with bandit-based selection
# --------------------------------------------------------------------

def run_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    n_rounds: int = 200,
    local_epochs: int = 5,
    lr_local: float = 0.05,
    alpha_delay: float = 1.0,
    beta_loss: float = 0.5,
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
        try:
            msg = recv_json(sock)
        except Exception as e:
            print(f"[SERVER] Failed to receive hello: {e}")
            sock.close()
            continue

        if msg.get("type") != "hello":
            print("[SERVER] First message not 'hello', closing.")
            sock.close()
            continue

        cid = msg.get("client_id")
        if cid in clients:
            print(f"[SERVER] Duplicate client_id={cid}, closing.")
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

    # Bandit agent with 3 actions: {1,2}, {1}, {2}
    bandit = BanditAgent(n_actions=3, eps=0.1, alpha=0.1)
    last_global_loss = 1.0  # just for logging

    def select_subset(action_idx: int) -> List[ClientConn]:
        if action_idx == 0:
            return [c1, c2]
        elif action_idx == 1:
            return [c1]
        else:
            return [c2]

    for r in range(1, n_rounds + 1):
        action_idx = bandit.select_action()
        selected_clients = select_subset(action_idx)
        selected_ids = [c.client_id for c in selected_clients]

        print(f"\n[SERVER][round {r}] action={action_idx}, selected={selected_ids}, "
              f"global_wb=({global_model.w:.4f},{global_model.b:.4f})")

        # Send train command
        send_times = {}
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

        # Receive updates
        updated_models: List[GlobalModel] = []
        local_losses: List[float] = []
        delays_ms: List[float] = []

        for c in selected_clients:
            try:
                msg = recv_json(c.sock)
            except Exception as e:
                print(f"[SERVER] Error receiving from {c.client_id}: {e}")
                continue

            if msg.get("type") != "update" or msg.get("round") != r:
                print(f"[SERVER] Unexpected msg from {c.client_id}: {msg}")
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

        # Aggregate
        if updated_models:
            global_model = GlobalModel.fedavg(updated_models)
            round_time_ms = max(delays_ms) if delays_ms else 0.0
            approx_global_loss = sum(local_losses) / len(local_losses)
        else:
            round_time_ms = 0.0
            approx_global_loss = last_global_loss

        last_global_loss = approx_global_loss

        # Reward: negative of (delay + beta * loss)
        reward = - (alpha_delay * round_time_ms + beta_loss * approx_global_loss)
        bandit.update(action_idx, reward)

        print(f"[SERVER][round {r}] round_time={round_time_ms:.3f} ms, "
              f"global_loss≈{approx_global_loss:.6f}, reward={reward:.4f}")
        print(f"[SERVER] Bandit Q-values: {bandit.summary()}")

    # Finish: notify clients
    stop_msg = {"type": "stop"}
    for c in clients.values():
        try:
            send_json(c.sock, stop_msg)
        except Exception:
            pass
        c.sock.close()

    server_sock.close()
    print("\n[SERVER] Training finished, sockets closed.")
    print("[SERVER] Final bandit stats:", bandit.summary())


if __name__ == "__main__":
    run_server()
