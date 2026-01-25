#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PPO + FedAsync demo (client side, pure numpy).

Usage:
    python3 client_ppo_fedasync.py client1 auto
    python3 client_ppo_fedasync.py client2 auto

或手动指定 server:
    python3 client_ppo_fedasync.py client1 10.11.16.103
"""

from __future__ import print_function

import json
import math
import socket
import sys
import time
import csv
import subprocess

import numpy as np


PORT = 5000      # 要和 server 保持一致
DATA_SIZE = 200


# ---------- socket/json 工具 ----------

def send_json(sock, obj):
    data = json.dumps(obj) + "\n"
    sock.sendall(data.encode("utf-8"))


def recv_json_line(fobj):
    line = fobj.readline()
    if not line:
        raise IOError("socket closed")
    return json.loads(line.strip())


# ---------- 自动发现 server ----------

def get_my_ip():
    """
    用 shell 命令拿第一个非 127 的 IP，和你之前 demo 类似。
    """
    try:
        out = subprocess.check_output("hostname -I", shell=True)
        text = out.decode("utf-8").strip()
        parts = text.split()
        for p in parts:
            if not p.startswith("127."):
                return p
    except Exception:
        pass
    # 兜底：用 gethostbyname
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        return ip
    except Exception:
        return "127.0.0.1"


def discover_server(port):
    my_ip = get_my_ip()
    splits = my_ip.split(".")
    prefix = ".".join(splits[:3])
    print("[discover] my_ip = {0}, prefix = {1}.0/24".format(my_ip, prefix))

    for pass_idx in range(1, 6):
        print("[discover] pass {0} ...".format(pass_idx))
        for last in range(1, 255):
            host = prefix + "." + str(last)
            try:
                sock = socket.create_connection((host, port), timeout=0.2)
                # 简单发个 hello / 再关掉，让正式连接复用这个 IP
                sock.close()
                print("[discover] found server at {0}".format(host))
                return host
            except Exception:
                continue
    raise RuntimeError("Server not found on {0}.0/24 after 5 passes".format(prefix))


# ---------- 本地数据 + 训练 ----------

def make_local_data(client_id):
    """
    生成简单 y = 2x + 3 + noise，但每个 client 的 x 分布略有不同。
    """
    rng = np.random.RandomState(123 if client_id == "client1" else 456)

    x = rng.uniform(-5.0, 5.0, size=(DATA_SIZE, 1))
    if client_id == "client2":
        x = rng.uniform(0.0, 10.0, size=(DATA_SIZE, 1))

    true_w = 2.0
    true_b = 3.0
    noise = rng.normal(0.0, 0.5, size=(DATA_SIZE, 1))

    y = true_w * x + true_b + noise
    return x.astype(np.float32), y.astype(np.float32)


def local_train_step(w, b, x, y, lr):
    """
    对线性模型 y_hat = w*x + b 做一次 full-batch 梯度下降。
    """
    # x:(N,1), y:(N,1)
    y_pred = w * x + b
    err = y_pred - y
    loss = float(np.mean(err ** 2))

    # dL/dw = mean(2 * err * x)
    dw = float(np.mean(2.0 * err * x))
    db = float(np.mean(2.0 * err))

    w_new = w - lr * dw
    b_new = b - lr * db
    return w_new, b_new, loss


def run_client(client_id, server_host):
    print("[CLIENT {0}] connecting to {1}:{2} ...".format(client_id, server_host, PORT))
    sock = socket.create_connection((server_host, PORT))
    fobj = sock.makefile("r")

    # hello
    hello = {"type": "hello", "client_id": client_id}
    send_json(sock, hello)

    # 本地数据
    x, y = make_local_data(client_id)

    local_step = 0

    # CSV 记录
    csv_path = "/tmp/ppo_async_curve_{0}.csv".format(client_id)
    csv_f = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow([
        "client_id", "local_step", "global_step",
        "w", "b", "local_loss", "round_time_ms"
    ])

    try:
        while True:
            msg = recv_json_line(fobj)
            mtype = msg.get("type")
            if mtype == "stop":
                print("[CLIENT {0}] received stop, exiting.".format(client_id))
                break
            if mtype != "train":
                print("[CLIENT {0}] unknown msg: {1}".format(client_id, msg))
                break

            global_step = int(msg.get("global_step", 0))
            w = float(msg.get("w", 0.0))
            b = float(msg.get("b", 0.0))
            local_epochs = int(msg.get("local_epochs", 1))
            lr = float(msg.get("lr", 0.01))

            t0 = time.time()
            for _ in range(local_epochs):
                w, b, loss = local_train_step(w, b, x, y, lr)
            t1 = time.time()
            round_time_ms = (t1 - t0) * 1000.0

            local_step += 1

            # 回传更新
            update = {
                "type": "update",
                "client_id": client_id,
                "local_step": local_step,
                "global_step": global_step,
                "w": w,
                "b": b,
                "local_loss": loss,
            }
            send_json(sock, update)

            csv_writer.writerow([
                client_id, local_step, global_step,
                w, b, loss, round_time_ms
            ])

            if local_step % 10 == 0:
                print("[CLIENT {0}] step={1}, global_step={2}, loss={3:.6f}".format(
                    client_id, local_step, global_step, loss
                ))

    finally:
        csv_f.close()
        sock.close()
        print("[CLIENT {0}] CSV saved to {1}".format(client_id, csv_path))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 client_ppo_fedasync.py <client_id> <server_ip|auto>")
        sys.exit(1)

    cid = sys.argv[1]
    mode = sys.argv[2]

    if mode == "auto":
        host = discover_server(PORT)
    else:
        host = mode

    run_client(cid, host)
