#!/usr/bin/env python3
# Minimal RL-FedAvg client with network measurement (ping-based delay logging)

import sys
import socket
import json
import random
import time
import os
import subprocess

try:
    import numpy as np
except ImportError:
    np = None

# ----------------------------
# Config
# ----------------------------

PORT = 38001            # 与 server_rl_bandit_fedavg.py 保持一致（可在命令行覆盖）
BACKLOG = 5

TARGET_W = 2.0
TARGET_B = 3.0

NET_LOG_DIR = "/tmp"
NET_PING_COUNT = 50     # 每轮 ping 包数，和前面实验保持一致
NET_MODE_NAME = "rl_fedavg"


# ----------------------------
# 工具函数：JSON over TCP
# ----------------------------

def send_json(sock, obj):
    data = json.dumps(obj) + "\n"
    sock.sendall(data.encode("utf-8"))


def recv_json_line(fobj):
    line = fobj.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


# ----------------------------
# 工具函数：发现 server（自动扫描 /24）
# ----------------------------

def discover_server(port, max_passes=5):
    """
    和你之前 baseline/FedAvg/FedAsync 用的一样的 auto discover 逻辑简化版：
    - 先通过一个 UDP 探测拿到本机 IP；
    - 扫描 /24 段，尝试 TCP 连接指定端口；
    - 第一台能连通的就认为是 server。
    返回: (server_host, prefix)
    """
    # 1) 猜本机 IP
    my_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        my_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    parts = my_ip.split(".")
    if len(parts) != 4:
        prefix = "10.11.16.0"
    else:
        prefix = ".".join(parts[0:3]) + ".0"

    print("[discover] my_ip = %s, prefix = %s/24" % (my_ip, prefix))

    # 2) 扫描 /24
    base = ".".join(prefix.split(".")[0:3])
    for p in range(max_passes):
        print("[discover] pass %d ..." % (p + 1))
        for i in range(1, 255):
            host = "%s.%d" % (base, i)
            if host == my_ip:
                continue
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.2)
                sock.connect((host, port))
                sock.close()
                print("[discover] found server at %s" % host)
                return host, prefix
            except Exception:
                continue

    raise RuntimeError("Server not found on %s/24 after %d passes" %
                       (prefix, max_passes))


# ----------------------------
# 网络测量：初始化 / ping / 记录 CSV
# ----------------------------

def init_net_logger(client_id):
    """
    确保 /tmp/rl_fedavg_curve_clientX.csv 存在并写好表头。
    """
    if not os.path.exists(NET_LOG_DIR):
        try:
            os.makedirs(NET_LOG_DIR)
        except Exception:
            pass

    path = os.path.join(NET_LOG_DIR,
                        "rl_fedavg_curve_%s.csv" % str(client_id))

    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("mode,client_id,server_host,round,"
                    "sent,received,loss,"
                    "rtt_min_ms,rtt_avg_ms,rtt_max_ms,delay_avg_ms\n")
    return path


def measure_delay(server_host, count):
    """
    对 server_host 执行 ping，解析出 rtt min/avg/max，并返回：
    sent, received, loss(0~1), rtt_min, rtt_avg, rtt_max, delay_avg
    delay_avg = rtt_avg / 2  作为单向 delay 的近似。
    """
    cmd = ["ping", "-c", str(count), server_host]
    try:
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out, err = p.communicate(timeout=15)
        text = out.decode("utf-8", "ignore")
    except Exception:
        # ping 失败时，返回全 0 + loss=1
        return 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0

    sent = 0
    received = 0
    loss = 1.0
    rtt_min = 0.0
    rtt_avg = 0.0
    rtt_max = 0.0

    for line in text.splitlines():
        line = line.strip()

        # "50 packets transmitted, 50 received, 0% packet loss, time 49048ms"
        if "packets transmitted" in line:
            parts = line.split(",")
            try:
                sent = int(parts[0].split()[0])
                received = int(parts[1].split()[0])
                loss_str = parts[2].strip().split()[0].replace("%", "")
                loss = float(loss_str) / 100.0
            except Exception:
                pass

        # "rtt min/avg/max/mdev = 0.305/0.309/0.322/0.004 ms"
        if "min/avg/max" in line and "=" in line:
            try:
                stats = line.split("=")[1].strip().split()[0].split("/")
                rtt_min = float(stats[0])
                rtt_avg = float(stats[1])
                rtt_max = float(stats[2])
            except Exception:
                pass

    delay_avg = rtt_avg / 2.0
    return sent, received, loss, rtt_min, rtt_avg, rtt_max, delay_avg


def log_delay(csv_path, client_id, server_host, round_idx,
              sent, received, loss,
              rtt_min, rtt_avg, rtt_max, delay_avg):
    """
    追加一行到 rl_fedavg_curve_clientX.csv
    """
    line = "{mode},{cid},{host},{rnd},{sent},{recv},{loss:.6f}," \
           "{rtt_min:.6f},{rtt_avg:.6f},{rtt_max:.6f},{delay:.6f}\n".format(
               mode=NET_MODE_NAME,
               cid=str(client_id),
               host=server_host,
               rnd=int(round_idx),
               sent=int(sent),
               recv=int(received),
               loss=float(loss),
               rtt_min=float(rtt_min),
               rtt_avg=float(rtt_avg),
               rtt_max=float(rtt_max),
               delay=float(delay_avg))
    with open(csv_path, "a") as f:
        f.write(line)


# ----------------------------
# 本地数据 & 训练
# ----------------------------

def make_local_data(client_id, n_samples=200):
    if np is None:
        raise RuntimeError("numpy is required on client")
    rng = np.random.RandomState(1234 if str(client_id) == "client1" else 5678)
    x = rng.uniform(-2.0, 2.0, size=(n_samples,))
    noise = rng.normal(0.0, 0.1, size=(n_samples,))
    y = TARGET_W * x + TARGET_B + noise
    return x, y


def mse_loss(w, b, x, y):
    pred = w * x + b
    diff = pred - y
    return float(np.mean(diff * diff))


def train_one_round(x, y, w, b, local_epochs, lr):
    """
    最简单的一阶 SGD：对 y = w*x + b 做几轮本地更新。
    """
    w_local = float(w)
    b_local = float(b)

    for _ in range(local_epochs):
        pred = w_local * x + b_local
        diff = pred - y
        grad_w = float(2.0 * np.mean(diff * x))
        grad_b = float(2.0 * np.mean(diff))
        w_local -= lr * grad_w
        b_local -= lr * grad_b

    loss = mse_loss(w_local, b_local, x, y)
    return w_local, b_local, loss


# ----------------------------
# 主逻辑：客户端
# ----------------------------

def run_client(client_id, mode_arg, port=PORT):
    """
    client_id: "client1" / "client2"
    mode_arg:  "auto" 或直接指定 server IP
    """

    # 1) 发现 server
    if mode_arg == "auto":
        server_host, prefix = discover_server(port)
    else:
        server_host = mode_arg

    print("[CLIENT %s] connect to %s:%d ..." %
          (client_id, server_host, port))

    # 2) 建立 TCP 连接
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, port))
    f = sock.makefile("r")

    # 3) 发送 HELLO
    hello = {
        "type": "HELLO",
        "client_id": str(client_id)
    }
    send_json(sock, hello)
    print("[CLIENT %s] sent HELLO" % client_id)

    # 4) 准备本地数据
    x, y = make_local_data(client_id)

    # 5) 初始化网络测量 CSV
    net_csv_path = init_net_logger(client_id)

    # 可以用 round_idx 直接作为 CSV 里的 round
    # （也可以额外维护 net_round 计数器，看你习惯）
    while True:
        msg = recv_json_line(f)
        if msg is None:
            print("[CLIENT %s] socket closed by server." % client_id)
            break

        mtype = msg.get("type", "")
        if mtype == "TRAIN":
            round_idx = msg.get("round", 0)
            w = msg.get("w", 0.0)
            b = msg.get("b", 0.0)
            local_epochs = msg.get("local_epochs", 5)
            lr = msg.get("lr", 0.05)

            print("[CLIENT %s] TRAIN round=%d, w=%.4f, b=%.4f, "
                  "local_epochs=%d, lr=%.4f"
                  % (client_id, round_idx, w, b, local_epochs, lr))

            new_w, new_b, local_loss = train_one_round(
                x, y, w, b, local_epochs, lr
            )

            print("[CLIENT %s] UPDATE round=%d -> new_w=%.4f, new_b=%.4f, "
                  "local_loss=%.6f"
                  % (client_id, round_idx, new_w, new_b, local_loss))

            resp = {
                "type": "UPDATE",
                "round": round_idx,
                "w": new_w,
                "b": new_b,
                "local_loss": local_loss
            }
            send_json(sock, resp)

            # 6) 网络测量：每完成一轮训练后，ping 一次 server，并记录 CSV
            sent, received, loss, rtt_min, rtt_avg, rtt_max, delay_avg = \
                measure_delay(server_host, NET_PING_COUNT)

            log_delay(net_csv_path,
                      client_id=client_id,
                      server_host=server_host,
                      round_idx=round_idx,
                      sent=sent,
                      received=received,
                      loss=loss,
                      rtt_min=rtt_min,
                      rtt_avg=rtt_avg,
                      rtt_max=rtt_max,
                      delay_avg=delay_avg)

            print("[NET %s] round=%d, sent=%d, recv=%d, loss=%.3f, "
                  "delay_avg=%.3f ms"
                  % (client_id, round_idx, sent, received, loss, delay_avg))

        elif mtype == "STOP":
            print("[CLIENT %s] received STOP. exiting." % client_id)
            break
        else:
            # 未知消息，直接忽略
            print("[CLIENT %s] unknown message type: %s" %
                  (client_id, mtype))

    try:
        f.close()
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass


# ----------------------------
# 入口
# ----------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 client_rl_fedavg.py <client_id> <server_ip|auto> [port]")
        sys.exit(1)

    cid = sys.argv[1]
    mode = sys.argv[2]
    if len(sys.argv) >= 4:
        p = int(sys.argv[3])
    else:
        p = PORT

    run_client(cid, mode_arg=mode, port=p)
