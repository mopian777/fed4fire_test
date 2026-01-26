#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal FedAvg client (Python 3.4 compatible)
- Persistent TCP connection
- HELLO then wait for TRAIN
- Local training: toy linear regression on synthetic data
- Sends UPDATE with (w,b,local_loss)
- Optional server auto-discovery within /24
"""

from __future__ import print_function

import os
import sys
import time
import json
import socket
import random
import subprocess

PORT = 5000

TARGET_W = 2.0
TARGET_B = 3.0

# --------------------------
# JSON line helpers
# --------------------------

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

# --------------------------
# Auto-discovery
# --------------------------

def get_my_ip():
    # More robust than hostname -I for some nodes
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        pass
    try:
        out = subprocess.check_output(["hostname", "-I"]).decode("utf-8").split()
        if out:
            return out[0]
    except Exception:
        pass
    return "127.0.0.1"

def discover_server(port, per_ip_timeout=0.08, max_passes=5, sleep_between=1.0):
    my_ip = get_my_ip()
    parts = my_ip.split(".")
    if len(parts) != 4:
        raise RuntimeError("Cannot infer /24 prefix from my_ip={0}".format(my_ip))
    prefix = ".".join(parts[0:3])

    print("[discover] my_ip = {0}, scanning {1}.0/24 ...".format(my_ip, prefix))

    for p in range(1, max_passes + 1):
        print("[discover] pass {0}/{1}".format(p, max_passes))
        for i in range(1, 255):
            ip = "{0}.{1}".format(prefix, i)
            if ip == my_ip:
                continue
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(per_ip_timeout)
            try:
                s.connect((ip, port))
                s.close()
                print("[discover] found server at {0}".format(ip))
                return ip
            except Exception:
                try:
                    s.close()
                except Exception:
                    pass
                continue
        time.sleep(sleep_between)

    raise RuntimeError("Server not found on {0}.0/24 after {1} passes".format(prefix, max_passes))

# --------------------------
# Toy data + training
# --------------------------

def make_local_data(client_id, n=100):
    # Deterministic per-client
    seed = 1234 if str(client_id) == "client1" else 5678
    rnd = random.Random(seed)

    xs = []
    ys = []
    for _ in range(n):
        x = rnd.uniform(-2.0, 2.0)
        noise = rnd.gauss(0.0, 0.1)
        y = TARGET_W * x + TARGET_B + noise
        xs.append(x)
        ys.append(y)
    return xs, ys

def mse(xs, ys, w, b):
    n = float(len(xs))
    s = 0.0
    for x, y in zip(xs, ys):
        d = (w * x + b) - y
        s += d * d
    return s / n

def train_one_round(xs, ys, w_init, b_init, local_epochs, lr):
    w = float(w_init)
    b = float(b_init)

    for _ in range(int(local_epochs)):
        # full batch GD
        gw = 0.0
        gb = 0.0
        n = float(len(xs))
        for x, y in zip(xs, ys):
            diff = (w * x + b) - y
            gw += 2.0 * diff * x
            gb += 2.0 * diff
        gw /= n
        gb /= n
        w -= lr * gw
        b -= lr * gb

    loss = mse(xs, ys, w, b)
    return w, b, loss

# --------------------------
# Main
# --------------------------

def run_client(client_id, server_arg, port):
    if server_arg == "auto":
        server_host = discover_server(port)
    else:
        server_host = server_arg

    print("[CLIENT {0}] connect to {1}:{2}".format(client_id, server_host, port))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, port))
    fobj = sock.makefile("r")

    send_json(sock, {"type": "HELLO", "client_id": str(client_id)})
    print("[CLIENT {0}] sent HELLO".format(client_id))

    xs, ys = make_local_data(client_id)

    # initialize with zeros to match server
    while True:
        msg = recv_json_line(fobj)
        if msg is None:
            print("[CLIENT {0}] socket closed by server.".format(client_id))
            break

        mtype = msg.get("type", "")
        if mtype == "TRAIN":
            rnd = int(msg.get("round", 0))
            w = float(msg.get("w", 0.0))
            b = float(msg.get("b", 0.0))
            local_epochs = int(msg.get("local_epochs", 5))
            lr = float(msg.get("lr", 0.02))

            new_w, new_b, local_loss = train_one_round(xs, ys, w, b, local_epochs, lr)

            print("[CLIENT {0}] round={1} local_loss={2:.6f} -> send UPDATE".format(
                client_id, rnd, local_loss
            ))

            send_json(sock, {
                "type": "UPDATE",
                "round": rnd,
                "w": new_w,
                "b": new_b,
                "local_loss": local_loss
            })

        elif mtype == "STOP":
            print("[CLIENT {0}] got STOP, exit.".format(client_id))
            break
        else:
            print("[CLIENT {0}] unknown msg: {1}".format(client_id, msg))

    try:
        fobj.close()
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass

if __name__ == "__main__":
    # Usage: python3 client_fedavg_py34.py <client_id> <server_ip|auto> [port]
    if len(sys.argv) < 3:
        print("Usage: python3 client_fedavg_py34.py <client_id> <server_ip|auto> [port]")
        sys.exit(1)

    cid = sys.argv[1]
    srv = sys.argv[2]
    p = PORT
    if len(sys.argv) >= 4:
        p = int(sys.argv[3])

    run_client(cid, srv, p)
