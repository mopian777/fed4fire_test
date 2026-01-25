# -*- coding: utf-8 -*-
"""
RL-FedAvg 客户端（配合 server_rl_bandit_fedavg.py 使用）
- 本地模型: 线性回归 y = w * x + b
- 本地数据: 在启动时按 client_id 生成不同分布的合成数据
- 发现 server:
    - 模式 "auto": 扫描本地前缀 10.x.x.x/24 找打开指定端口的 server
    - 模式 "<ip>": 直接连接指定 server_ip
"""

from __future__ import print_function

import socket
import json
import time
import random
import math
import sys

try:
    import numpy as np
except ImportError:
    np = None

HOST = None          # 由 discover 决定
PORT = 38001         # 要和 server 保持一致
DISCOVER_RETRY = 5


# ------------ 工具函数 ------------

def send_json(sock, obj):
    data = json.dumps(obj).encode("utf-8") + b"\n"
    sock.sendall(data)


def recv_json_line(fobj):
    line = fobj.readline()
    if not line:
        return None
    return json.loads(line.decode("utf-8").strip())


def get_my_ip():
    """取本机在实验网络中的 IP（粗略版本）"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def discover_server(port, max_passes=5):
    """自动扫描同一 /24 网段上的 server:port"""
    my_ip = get_my_ip()
    parts = my_ip.split(".")
    if len(parts) != 4:
        raise RuntimeError("Cannot parse my_ip={}".format(my_ip))
    prefix = ".".join(parts[:3])  # 例如 10.11.16

    print("[discover] my_ip = {}, prefix = {}.0/24".format(my_ip, prefix))

    for p in range(max_passes):
        print("[discover] pass {} ...".format(p + 1))
        for last in range(1, 255):
            cand = "{}.{}".format(prefix, last)
            if cand == my_ip:
                continue
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.2)
                sock.connect((cand, port))
                # 试发一个 ping
                msg = {"type": "ping"}
                send_json(sock, msg)
                sock.close()
                print("[discover] found server at {}:{}".format(cand, port))
                return cand
            except Exception:
                continue
    raise RuntimeError("Server_not_found on {}.0/24 after {} passes".format(prefix, max_passes))


# ------------ 本地数据与训练 ------------

def make_local_data(client_id, n_samples=100):
    """为不同 client 生成稍微不同的线性数据"""
    if np is None:
        raise RuntimeError("numpy is required on client")

    rng = np.random.RandomState(123 if client_id == "client1" else 456)

    x = rng.uniform(-5.0, 5.0, size=(n_samples,))
    noise = rng.normal(loc=0.0, scale=0.5, size=(n_samples,))

    # 目标模型 y = 2x + 3，两个客户端加一点偏移
    if client_id == "client1":
        y = 2.0 * x + 3.0 + noise
    else:
        y = 2.1 * x + 2.9 + noise

    return x, y


def local_train(client_id, global_w, global_b, x, y, local_epochs, lr):
    """在本地数据上做若干步梯度下降，并返回新参数和 MSE loss。
       内置数值稳定保护：loss 爆掉或参数过大时回退到 global_w/b。
    """
    if np is None:
        raise RuntimeError("numpy is required on client")

    w = float(global_w)
    b = float(global_b)

    n = float(len(x))

    for _ in range(local_epochs):
        preds = w * x + b
        errors = preds - y
        grad_w = float(2.0 * np.dot(errors, x) / n)
        grad_b = float(2.0 * np.sum(errors) / n)

        w = w - lr * grad_w
        b = b - lr * grad_b

    # 计算 MSE
    preds = w * x + b
    errors = preds - y
    loss = float(np.mean(errors ** 2))

    # 数值稳定保护
    unstable = False
    if (not math.isfinite(loss)) or loss > 1e3:
        unstable = True
    if abs(w) > 1e6 or abs(b) > 1e6:
        unstable = True

    if unstable:
        print("[CLIENT {}] WARNING: unstable local training: loss={}, w={}, b={}. "
              "resetting to global model.".format(client_id, loss, w, b))
        loss = 1e3
        w = float(global_w)
        b = float(global_b)

    return w, b, loss


# ------------ 客户端主逻辑 ------------

def run_client(client_id, mode, server_host=None, port=PORT):
    if mode == "auto":
        server_host = discover_server(port)
    else:
        # mode 直接给 ip
        server_host = mode

    print("[CLIENT {}] connecting to {}:{} ...".format(client_id, server_host, port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, port))
    fobj = sock.makefile("rwb", buffering=0)

    # 发送 hello
    hello = {"type": "hello", "client_id": client_id}
    send_json(sock, hello)
    print("[CLIENT {}] sent hello".format(client_id))

    # 准备本地数据
    x, y = make_local_data(client_id)

    try:
        while True:
            msg = recv_json_line(fobj)
            if msg is None:
                print("[CLIENT {}] server closed connection".format(client_id))
                break

            mtype = msg.get("type", "")
            if mtype == "stop":
                print("[CLIENT {}] received stop, exiting.".format(client_id))
                break

            if mtype != "train":
                # 忽略其他类型（如 ping）
                continue

            round_idx = msg.get("round", 0)
            global_w = msg.get("w", 0.0)
            global_b = msg.get("b", 0.0)
            local_epochs = int(msg.get("local_epochs", 5))
            lr = float(msg.get("lr", 0.02))

            print("[CLIENT {}] TRAIN round={}, w={:.4f}, b={:.4f}, local_epochs={}, lr={:.4f}".format(
                client_id, round_idx, global_w, global_b, local_epochs, lr
            ))

            t0 = time.time()
            new_w, new_b, local_loss = local_train(
                client_id, global_w, global_b, x, y, local_epochs, lr
            )
            t1 = time.time()

            print("[CLIENT {}] UPDATE round={} -> new_w={:.4f}, new_b={:.4f}, local_loss={}".format(
                client_id, round_idx, new_w, new_b, local_loss
            ))

            resp = {
                "type": "update",
                "client_id": client_id,
                "round": round_idx,
                "new_w": new_w,
                "new_b": new_b,
                "local_loss": local_loss,
                "train_time_ms": (t1 - t0) * 1000.0,
            }
            send_json(sock, resp)

    finally:
        try:
            fobj.close()
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass
        print("[CLIENT {}] socket closed.".format(client_id))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 client_rl_fedavg.py <client_id> <auto|server_ip>")
        sys.exit(1)

    cid = sys.argv[1]
    mode_arg = sys.argv[2]  # "auto" 或者 具体 server IP

    run_client(cid, mode_arg, port=PORT)
