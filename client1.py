#!/usr/bin/env python3
import socket
import json
import sys
import random
import subprocess
import math
import time

PORT = 5000
ROUNDS = 5          # 要与 server 保持一致
LOCAL_STEPS = 50    # 每轮本地训练步数
LR = 0.1            # 学习率


def discover_server(port=PORT,
                    per_ip_timeout=0.05,
                    max_passes=5,
                    sleep_between=3.0):
    """
    在本地 /24 网段扫描哪个 IP 的 port=5000 能连通，
    最多扫描 max_passes 轮：
      - 每轮遍历 10.x.x.1~254
      - 每个 IP 尝试连接 per_ip_timeout 秒
      - 一轮没找到就 sleep_between 秒再扫下一轮
    这样即使 server 比 client 晚启动几十秒，也能被发现。
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
                # 连接失败就继续
                continue
        # 这一轮没找到，等等再来
        print("[discover] no server found in pass {}, sleep {}s".format(
            attempt, sleep_between
        ))
        time.sleep(sleep_between)

    raise RuntimeError(
        "Server not found on {}.0/24 after {} passes".format(prefix, max_passes)
    )



def make_local_data(n=100):
    """
    生成简单线性数据：y = 2*x + 3 + 噪声
    x ~ U(0, 1)
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


def train_linear(xs, ys, w_init, b_init, lr=0.1, steps=50):
    """
    从给定 (w_init, b_init) 出发训练一维线性模型 y_hat = w*x + b
    返回 (w, b, loss)
    """
    w = float(w_init)
    b = float(b_init)
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

    # 固定本地数据集，跨轮共享
    xs, ys = make_local_data(n=100)

    # 初始模型参数（各 client 可不同）
    w = random.uniform(-1.0, 1.0)
    b = random.uniform(-1.0, 1.0)

    print("[{}] Initial w = {:.4f}, b = {:.4f}".format(client_id, w, b))

    for r in range(1, ROUNDS + 1):
        # 1) 从当前全局参数 (w, b) 出发做本地训练
        w, b, local_loss = train_linear(xs, ys, w, b, lr=LR, steps=LOCAL_STEPS)

        print("[{}] Round {} local w = {:.4f}, b = {:.4f}, train_loss = {:.6f}".format(
            client_id, r, w, b, local_loss
        ))

        # 2) 把本轮本地参数发给 server
        payload = {
            "client_id": client_id,
            "round": r,
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

        avg_w, avg_b = resp["avg_weights"]
        server_round = resp.get("round", None)

        print("[{}] Round {} global w = {:.4f}, b = {:.4f} (server round = {})".format(
            client_id, r, avg_w, avg_b, server_round
        ))

        # 3) 比较本地模型 vs 全局模型在本地数据上的 MSE
        global_loss = mse_on_data(xs, ys, avg_w, avg_b)
        print("[{}] Round {} Local MSE = {:.6f}, Global MSE = {:.6f}".format(
            client_id, r, local_loss, global_loss
        ))

        # 4) 把当前模型更新为全局模型，进入下一轮
        w, b = avg_w, avg_b

    print("[{}] Training finished after {} rounds.".format(client_id, ROUNDS))


if __name__ == "__main__":
    main()
