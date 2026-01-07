#!/usr/bin/env python3
import socket
import json
import sys
import random
import subprocess
import math
import time

PORT = 5000

# 每个客户端要执行多少次“异步通信+本地训练”
ASYNC_UPDATES = 100

# 每次从全局参数出发，本地训练多少步
LOCAL_STEPS = 50
LR = 0.1   # 本地 SGD 学习率


# ----------- 工具函数：发现 server IP（沿用你现在的 auto 模式） -----------

def discover_server(port=PORT,
                    per_ip_timeout=0.05,
                    max_passes=5,
                    sleep_between=3.0):
    """
    在本地 /24 网段扫描哪个 IP 的 port=5000 能连通，
    最多扫描 max_passes 轮。
    """
    out = subprocess.check_output(["hostname", "-I"]).decode().split()
    if not out:
        raise RuntimeError("Cannot get local IP from 'hostname -I'")
    my_ip = out[0]
    prefix, last = my_ip.rsplit(".", 1)
    print("[discover] my_ip = {}, prefix = {}".format(my_ip, prefix))

    for attempt in range(1, max_passes + 1):
        print("[discover] scan pass {}/{} on {}.0/24".format(
            attempt, max_passes, prefix
        ))
        for i in range(1, 255):
            ip = "{}.{}".format(prefix, i)
            if ip == my_ip:
                continue
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(per_ip_timeout)
            try:
                s.connect((ip, port))
                s.close()
                print("[discover] found server at {} (pass {})".format(ip, attempt))
                return ip
            except OSError:
                s.close()
                continue
        print("[discover] no server found in pass {}, sleep {}s".format(
            attempt, sleep_between
        ))
        time.sleep(sleep_between)

    raise RuntimeError(
        "Server not found on {}.0/24 after {} passes".format(prefix, max_passes)
    )


# ----------- 简单线性回归数据 & 训练 -----------

def make_local_data(n=100):
    """y = 2*x + 3 + noise"""
    xs, ys = [], []
    for _ in range(n):
        x = random.random()
        noise = random.gauss(0.0, 0.1)
        y = 2.0 * x + 3.0 + noise
        xs.append(x)
        ys.append(y)
    return xs, ys


def train_linear(xs, ys, w_init, b_init, lr=0.1, steps=50):
    """从给定 (w_init, b_init) 出发训练一维线性模型"""
    w = float(w_init)
    b = float(b_init)
    n = float(len(xs))

    for _ in range(steps):
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

    return w, b, loss


def mse_on_data(xs, ys, w, b):
    n = float(len(xs))
    loss = 0.0
    for x, y in zip(xs, ys):
        diff = w * x + b - y
        loss += diff * diff
    return loss / n


# ----------- 主逻辑：FedAsync 客户端循环 -----------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 {} <client_id> [server_ip_or_auto]".format(sys.argv[0]))
        sys.exit(1)

    client_id = sys.argv[1]

    if len(sys.argv) >= 3:
        server_host = sys.argv[2]
    else:
        server_host = "auto"

    if server_host.lower() == "auto":
        server_host = discover_server()

    print("[{}] using server {}".format(client_id, server_host))

    # 固定本地数据集
    xs, ys = make_local_data(n=100)

    # 初始本地模型（会很快被全局覆盖，不敏感）
    w = random.uniform(-1.0, 1.0)
    b = random.uniform(-1.0, 1.0)
    print("[{}] initial local w = {:.4f}, b = {:.4f}".format(client_id, w, b))

    # 为了制造“异步感”，不同客户端可以设不同 sleep
    if client_id.lower() == "client1":
        extra_sleep = 0.0   # 快客户端
    else:
        extra_sleep = 0.2   # 慢一点的客户端（模拟 straggler）

    for step in range(1, ASYNC_UPDATES + 1):
        # 1) 从当前全局（上一次收到的）出发做本地训练
        w, b, local_loss = train_linear(xs, ys, w, b, lr=LR, steps=LOCAL_STEPS)

        # 2) 把本地模型发给 server，请其异步更新并返回最新全局
        payload = {
            "client_id": client_id,
            "client_step": step,
            "weights": [w, b],
        }

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((server_host, PORT))
            s.sendall(json.dumps(payload).encode("utf-8"))
            resp_raw = s.recv(4096).decode("utf-8")
            resp = json.loads(resp_raw)
        finally:
            s.close()

        global_w, global_b = resp["global_weights"]
        server_step = resp.get("server_step", None)

        global_loss = mse_on_data(xs, ys, global_w, global_b)

        print("[{}] async_step {} -> local_loss={:.6f}, "
              "global_loss={:.6f}, global_w={:.4f}, global_b={:.4f}, server_step={}".format(
                  client_id, step, local_loss, global_loss,
                  global_w, global_b, server_step))

        # 3) 用最新全局覆盖本地模型，进入下一次异步更新
        w, b = global_w, global_b

        # 4) 为了模拟异步客户端速度差异，sleep 一下
        if extra_sleep > 0:
            time.sleep(extra_sleep)

    print("[{}] Finished {} async updates.".format(client_id, ASYNC_UPDATES))


if __name__ == "__main__":
    main()
