#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RL-FedAvg client for jFed-style PPO-FedAvg server.

Protocol (must match server_rl_fedavg.py):

1. Connect to server (IP, port=5000).
2. Immediately send a "hello" message:
   {"type": "hello", "client_id": "client1" | "client2"}

3. Then enter main loop:
   - Receive JSON messages with 4-byte length prefix.
   - If msg["type"] == "train":
        {
          "type": "train",
          "round": r,
          "weights": {"w": ..., "b": ...},
          "local_epochs": 5,
          "lr": 0.05
        }
     -> Run local training on this client's data starting from (w,b).
     -> Compute final local_loss.
     -> Send:
        {
          "type": "update",
          "round": r,
          "client_id": "<client_id>",
          "weights": {"w": new_w, "b": new_b},
          "local_loss": local_loss
        }

   - If msg["type"] == "stop":
        break and close socket.

This is a minimal demo: local model is y = w*x + b, trained with full-batch
gradient descent on a synthetic dataset (y = 2x + 3 + noise).
"""

import json
import socket
import struct
import sys
import random
from typing import Tuple, List


HOST = "127.0.0.1"
PORT = 5000


# ---------------- JSON over TCP (length-prefixed) ---------------- #

def send_json(sock: socket.socket, obj: dict):
    data = json.dumps(obj).encode("utf-8")
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while reading")
        buf += chunk
    return buf


def recv_json(sock: socket.socket) -> dict:
    header = recv_exact(sock, 4)
    (length,) = struct.unpack("!I", header)
    data = recv_exact(sock, length)
    return json.loads(data.decode("utf-8"))


# ---------------- Local data & training (no extra libs) ---------------- #

def make_local_dataset(n_samples: int, noise_std: float, seed: int) -> Tuple[List[float], List[float]]:
    """
    Synthetic 1D regression: y = 2x + 3 + noise
    """
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n_samples):
        x = rng.random() * 10.0      # [0,10]
        noise = rng.gauss(0.0, noise_std)
        y = 2.0 * x + 3.0 + noise
        xs.append(x)
        ys.append(y)
    return xs, ys


def mse(xs: List[float], ys: List[float], w: float, b: float) -> float:
    n = float(len(xs))
    if n == 0:
        return 0.0
    loss = 0.0
    for x, y in zip(xs, ys):
        diff = w * x + b - y
        loss += diff * diff
    return loss / n


def local_train(
    xs: List[float],
    ys: List[float],
    w_init: float,
    b_init: float,
    lr: float,
    local_epochs: int,
) -> Tuple[float, float, float]:
    """
    Full-batch gradient descent on local data for local_epochs steps.
    Returns (new_w, new_b, final_local_loss).
    """
    w = float(w_init)
    b = float(b_init)
    n = float(len(xs))
    if n == 0:
        return w, b, 0.0

    for _ in range(local_epochs):
        grad_w = 0.0
        grad_b = 0.0
        loss = 0.0

        for x, y in zip(xs, ys):
            y_hat = w * x + b
            diff = y_hat - y
            loss += diff * diff
            grad_w += 2.0 * diff * x
            grad_b += 2.0 * diff

        loss /= n
        grad_w /= n
        grad_b /= n

        w -= lr * grad_w
        b -= lr * grad_b

    final_loss = mse(xs, ys, w, b)
    return w, b, final_loss


# ---------------- Main client logic ---------------- #

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 {} <client_id> <server_ip> [n_samples] [noise_std]".format(sys.argv[0]))
        print("  client_id: e.g. client1 or client2")
        print("  server_ip: e.g. 10.11.16.xxx")
        sys.exit(1)

    client_id = sys.argv[1]
    server_ip = sys.argv[2]
    n_samples = int(sys.argv[3]) if len(sys.argv) >= 4 else 64
    noise_std = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.5

    # 每个 client 用不同 seed，模拟轻微 non-IID
    if client_id.lower() == "client1":
        seed = 1
    elif client_id.lower() == "client2":
        seed = 2
    else:
        seed = 1234

    xs, ys = make_local_dataset(n_samples, noise_std, seed)

    print("[CLIENT {}] connect to {}:{} ...".format(client_id, server_ip, PORT))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_ip, PORT))

    # 发送 hello
    hello = {"type": "hello", "client_id": client_id}
    send_json(sock, hello)
    print("[CLIENT {}] sent hello".format(client_id))

    try:
        while True:
            msg = recv_json(sock)
            mtype = msg.get("type")

            if mtype == "train":
                round_id = msg.get("round")
                weights = msg.get("weights", {})
                local_epochs = int(msg.get("local_epochs", 5))
                lr = float(msg.get("lr", 0.05))

                w = float(weights.get("w", 0.0))
                b = float(weights.get("b", 0.0))

                print("[CLIENT {}] TRAIN round={}, w={:.4f}, b={:.4f}, "
                      "local_epochs={}, lr={:.4f}".format(
                          client_id, round_id, w, b, local_epochs, lr))

                # 本地训练
                new_w, new_b, local_loss = local_train(xs, ys, w, b, lr, local_epochs)

                print("[CLIENT {}] UPDATE round={} -> new_w={:.4f}, new_b={:.4f}, "
                      "local_loss={:.6f}".format(
                          client_id, round_id, new_w, new_b, local_loss))

                reply = {
                    "type": "update",
                    "round": round_id,
                    "client_id": client_id,
                    "weights": {"w": new_w, "b": new_b},
                    "local_loss": local_loss,
                }
                send_json(sock, reply)

            elif mtype == "stop":
                print("[CLIENT {}] received stop, exiting.".format(client_id))
                break

            else:
                print("[CLIENT {}] unknown message type: {}".format(client_id, mtype))
                # 可以选择 break 或忽略，这里忽略
                continue

    except ConnectionError as e:
        print("[CLIENT {}] connection error: {}".format(client_id, e))
    except KeyboardInterrupt:
        print("[CLIENT {}] interrupted by user".format(client_id))
    finally:
        sock.close()
        print("[CLIENT {}] socket closed.".format(client_id))


if __name__ == "__main__":
    main()
