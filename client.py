#!/usr/bin/env python3
import socket
import json
import sys
import random
import subprocess
import math

PORT = 5000


def discover_server(port=PORT, timeout=0.2):
    """
    在本地 /24 网段扫描哪个 IP 的 port=5000 能连通，
    第一个连通的就当作 server。
    """
    out = subprocess.check_output(["hostname", "-I"]).decode().split()
    if not out:
        raise RuntimeError("Cannot get local IP from 'hostname -I'")
    my_ip = out[0]
    prefix, last = my_ip.rsplit(".", 1)
    print("[discover] my_ip = {}, prefix = {}".format(my_ip, prefix))

    for i in range(1, 255):
        ip = "{}.{}".format(prefix, i)
        if ip == my_ip:
            continue
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect((ip, port))
            s.close()
            print("[discover] found server at {}".format(ip))
            return ip
        except OSError:
            continue

    raise RuntimeError("Server not found on {}.0/24".format(prefix))


def make_local_data(n=100):
    """
    生成简单线性数据：y = 2*x + 3 + 噪声
    这里 x 均匀采样 [0, 1]
    """
    xs = []
    ys = []
    for _ in range(n):
        x = random.random()
        noise = random.gauss(0.0, 0.1)
        y = 2.0 * x + 3.0 + noise
        xs.append(x)
        ys.append(y)
    return xs, ys


def train_linear(xs, ys, lr=0.1, steps=200):
    """
    训练一维线性模型 y_hat = w*x + b，MSE 损失
    返回训练后的 (w, b) 和最终 loss
    """
    w = random.uniform(-1.0, 1.0)
    b = random.uniform(-1.0, 1.0)

    n = float(len(xs))

    for step in range(steps):
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

    # 1. 本地生成数据 + 训练
    xs, ys = make_local_data(n=100)
    w, b, local_loss = train_linear(xs, ys, lr=0.1, steps=200)

    print("[{}] Local model w = {:.4f}, b = {:.4f}, train_loss = {:.6f}".format(
        client_id, w, b, local_loss
    ))

    # 2. 把本地参数发给 server
    payload = {
        "client_id": client_id,
        "weights": [w, b]
    }

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((server_host, PORT))
        s.sendall(json.dumps(payload).encode("utf-8"))
        resp_raw = s.recv(4096).decode("utf-8")
        resp = json.loads(resp_raw)
    finally:
        s.close()

    avg_w, avg_b = resp["avg_weights"]
    print("[{}] Global model w = {:.4f}, b = {:.4f}".format(
        client_id, avg_w, avg_b
    ))

    # 3. 比较本地模型 vs 全局模型在本地数据上的 MSE
    global_loss = mse_on_data(xs, ys, avg_w, avg_b)
    print("[{}] Local MSE = {:.6f}, Global MSE = {:.6f}".format(
        client_id, local_loss, global_loss
    ))


if __name__ == "__main__":
    main()
