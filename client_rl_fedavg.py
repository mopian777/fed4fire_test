#!/usr/bin/env python3
# RL-FedAvg client with delay logging + local loss/acc reporting (no torch)

import sys
import socket
import json
import time
import os
import subprocess

try:
    import numpy as np
except ImportError:
    np = None

PORT = 38001
TARGET_W = 2.0
TARGET_B = 3.0

NET_LOG_DIR = "/tmp"
NET_PING_COUNT = 50
NET_MODE_NAME = "rl_fedavg"

ACC_EPS = 0.5  # 回归“accuracy”阈值，可调


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


def discover_server(port, max_passes=5):
    my_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        my_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    parts = my_ip.split(".")
    if len(parts) == 4:
        prefix = ".".join(parts[0:3]) + ".0"
    else:
        prefix = "10.11.16.0"

    print("[discover] my_ip=%s prefix=%s/24" % (my_ip, prefix))
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
                return host
            except Exception:
                continue

    raise RuntimeError("Server not found on %s/24 after %d passes" % (prefix, max_passes))


def init_net_logger(client_id):
    if not os.path.exists(NET_LOG_DIR):
        try:
            os.makedirs(NET_LOG_DIR)
        except Exception:
            pass

    path = os.path.join(NET_LOG_DIR, "rl_fedavg_curve_%s.csv" % str(client_id))
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("mode,client_id,server_host,round,"
                    "sent,received,loss,"
                    "rtt_min_ms,rtt_avg_ms,rtt_max_ms,delay_avg_ms\n")
    return path


def measure_delay(server_host, count):
    cmd = ["ping", "-c", str(count), "-i", "0.2", "-q", server_host]
    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=20
        )
    except subprocess.CalledProcessError as e:
        out = e.output
    except Exception:
        return 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0

    sent = received = None
    loss = None
    rtt_min = rtt_avg = rtt_max = None

    for line in out.splitlines():
        line = line.strip()
        if "packets transmitted" in line:
            parts = line.split(",")
            try:
                sent = int(parts[0].split()[0])
                received = int(parts[1].split()[0])
                loss_str = parts[2].strip().split()[0].strip("%")
                loss = float(loss_str) / 100.0
            except Exception:
                pass
        if "rtt min/avg/max" in line and "=" in line:
            try:
                stats = line.split("=")[1].strip().split()[0].split("/")
                rtt_min = float(stats[0])
                rtt_avg = float(stats[1])
                rtt_max = float(stats[2])
            except Exception:
                pass

    if sent is None or received is None or loss is None or rtt_avg is None:
        return 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0

    delay_avg = rtt_avg / 2.0
    return sent, received, loss, rtt_min, rtt_avg, rtt_max, delay_avg


def log_delay(csv_path, client_id, server_host, round_idx,
              sent, received, loss, rtt_min, rtt_avg, rtt_max, delay_avg):
    line = "%s,%s,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (
        NET_MODE_NAME, str(client_id), server_host, int(round_idx),
        int(sent), int(received), float(loss),
        float(rtt_min), float(rtt_avg), float(rtt_max), float(delay_avg)
    )
    with open(csv_path, "a") as f:
        f.write(line)


def make_local_data(client_id, n_samples=200):
    if np is None:
        raise RuntimeError("numpy is required on client")
    rng = np.random.RandomState(1234 if str(client_id) == "client1" else 5678)
    x = rng.uniform(-2.0, 2.0, size=(n_samples,))
    noise = rng.normal(0.0, 0.1, size=(n_samples,))
    y = TARGET_W * x + TARGET_B + noise
    return x, y


def eval_metrics(w, b, x, y, eps=ACC_EPS):
    pred = w * x + b
    diff = pred - y
    mse = float(np.mean(diff * diff))
    acc = float(np.mean(np.abs(diff) <= eps))
    return mse, acc


def train_one_round(x, y, w, b, local_epochs, lr):
    w_local = float(w)
    b_local = float(b)

    for _ in range(local_epochs):
        pred = w_local * x + b_local
        diff = pred - y
        grad_w = float(2.0 * np.mean(diff * x))
        grad_b = float(2.0 * np.mean(diff))
        w_local -= lr * grad_w
        b_local -= lr * grad_b

    mse, acc = eval_metrics(w_local, b_local, x, y)
    return w_local, b_local, mse, acc


def run_client(client_id, mode_arg, port=PORT):
    server_host = discover_server(port) if mode_arg == "auto" else mode_arg
    print("[CLIENT %s] connect to %s:%d ..." % (client_id, server_host, port))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, port))
    f = sock.makefile("r")

    send_json(sock, {"type": "HELLO", "client_id": str(client_id)})
    print("[CLIENT %s] sent HELLO" % client_id)

    x, y = make_local_data(client_id)
    net_csv_path = init_net_logger(client_id)

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

            t0 = time.time()
            new_w, new_b, local_loss, local_acc = train_one_round(x, y, w, b, local_epochs, lr)
            train_time_ms = (time.time() - t0) * 1000.0

            send_json(sock, {
                "type": "UPDATE",
                "round": round_idx,
                "w": new_w,
                "b": new_b,
                "local_loss": float(local_loss),
                "local_acc": float(local_acc),
                "n_samples": int(len(x)),
                "train_time_ms": float(train_time_ms),
            })

            sent, received, loss, rtt_min, rtt_avg, rtt_max, delay_avg = \
                measure_delay(server_host, NET_PING_COUNT)

            log_delay(net_csv_path, client_id, server_host, round_idx,
                      sent, received, loss, rtt_min, rtt_avg, rtt_max, delay_avg)

            print("[CLIENT %s] round=%d loss=%.6f acc=%.4f train=%.1fms | delay=%.4fms" %
                  (client_id, round_idx, local_loss, local_acc, train_time_ms, delay_avg))

        elif mtype == "STOP":
            print("[CLIENT %s] received STOP. exiting." % client_id)
            break

    try:
        f.close()
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 client_rl_fedavg.py <client_id> <server_ip|auto> [port]")
        sys.exit(1)

    cid = sys.argv[1]
    mode = sys.argv[2]
    p = int(sys.argv[3]) if len(sys.argv) >= 4 else PORT
    run_client(cid, mode_arg=mode, port=p)
