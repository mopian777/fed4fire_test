#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import socket
import json
import sys
import random
import subprocess
import time
import os

PORT = 5000
UPDATES = 200         # 每个 client 发送多少次更新
LOCAL_STEPS = 50
LR = 0.1
EPS = 0.5
METHOD = "fedasync"

# 建议写到 HOME，避免 /tmp permission denied
PING_LOG_DIR = os.path.join(os.environ.get("HOME", "/tmp"), "fed4fire_logs")
PING_ENABLED = False


def send_json(sock, obj):
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def recv_json_line(f):
    line = f.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


def discover_server(port=PORT, per_ip_timeout=0.05, max_passes=5, sleep_between=2.0):
    out = subprocess.check_output(["hostname", "-I"]).decode("utf-8").split()
    if not out:
        raise RuntimeError("Cannot get IP from hostname -I")
    my_ip = out[0]
    prefix, _ = my_ip.rsplit(".", 1)

    for attempt in range(1, max_passes + 1):
        for i in range(1, 255):
            ip = "{0}.{1}".format(prefix, i)
            if ip == my_ip:
                continue
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(per_ip_timeout)
            try:
                s.connect((ip, port))
                try:
                    s.close()
                except Exception:
                    pass
                return ip
            except OSError:
                try:
                    s.close()
                except Exception:
                    pass
                continue
        time.sleep(sleep_between)

    raise RuntimeError("Server not found")


def make_local_data(n=100, seed=1234):
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        x = rng.random()
        y = 2.0 * x + 3.0 + rng.gauss(0.0, 0.1)
        xs.append(x)
        ys.append(y)
    return xs, ys


def eval_metrics(xs, ys, w, b, eps=EPS):
    n = float(len(xs))
    mse = 0.0
    ok = 0
    for x, y in zip(xs, ys):
        diff = (w * x + b) - y
        mse += diff * diff
        if abs(diff) <= eps:
            ok += 1
    return mse / n, ok / float(len(xs))


def train_steps(xs, ys, w, b, lr=LR, steps=LOCAL_STEPS):
    n = float(len(xs))
    w = float(w)
    b = float(b)
    for _ in range(int(steps)):
        gw = 0.0
        gb = 0.0
        for x, y in zip(xs, ys):
            diff = (w * x + b) - y
            gw += 2.0 * diff * x
            gb += 2.0 * diff
        gw /= n
        gb /= n
        w -= lr * gw
        b -= lr * gb
    return w, b


def init_ping_log(cid):
    if not os.path.exists(PING_LOG_DIR):
        try:
            os.makedirs(PING_LOG_DIR)
        except Exception:
            pass

    path = os.path.join(PING_LOG_DIR, "{0}_curve_{1}.csv".format(METHOD, cid))
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("mode,client_id,server_host,round,sent,received,loss,"
                    "rtt_min_ms,rtt_avg_ms,rtt_max_ms,delay_avg_ms\n")
    return path


def ping_measure(host, count=PING_COUNT):
    cmd = ["ping", "-c", str(count), "-i", "0.2", "-q", host]
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
    rmin = ravg = rmax = None

    for line in out.splitlines():
        line = line.strip()
        if "packets transmitted" in line:
            parts = line.split(",")
            try:
                sent = int(parts[0].split()[0])
                received = int(parts[1].split()[0])
                loss = float(parts[2].strip().split()[0].strip("%")) / 100.0
            except Exception:
                pass

        if "rtt min/avg/max" in line and "=" in line:
            try:
                stats = line.split("=")[1].strip().split()[0].split("/")
                rmin = float(stats[0])
                ravg = float(stats[1])
                rmax = float(stats[2])
            except Exception:
                pass

    if sent is None or received is None or loss is None or ravg is None:
        return 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0

    return sent, received, loss, rmin, ravg, rmax, ravg / 2.0


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 client_fedasync_py34.py <client_id> [server_ip|auto] [port]")
        sys.exit(1)

    cid = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) >= 3 else "auto"
    port = int(sys.argv[3]) if len(sys.argv) >= 4 else PORT

    if host.lower() == "auto":
        host = discover_server(port=port)

    xs, ys = make_local_data(n=100, seed=(1234 if cid == "client1" else 5678))
    ping_path = init_ping_log(cid)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    f = s.makefile("r")

    send_json(s, {"type": "HELLO", "client_id": cid})

    init = recv_json_line(f)
    if (not init) or init.get("type") != "INIT":
        raise RuntimeError("No INIT from server")

    w = float(init.get("w", 0.0))
    b = float(init.get("b", 0.0))
    version = int(init.get("version", 0))

    for k in range(1, UPDATES + 1):
        base_version = version

        t0 = time.time()
        w2, b2 = train_steps(xs, ys, w, b)
        train_ms = (time.time() - t0) * 1000.0

        send_json(s, {
            "type": "UPDATE",
            "k": k,
            "base_version": base_version,
            "w": w2,
            "b": b2,
            "train_time_ms": train_ms
        })

        g = recv_json_line(f)
        if (not g) or g.get("type") != "GLOBAL":
            print("[{0} {1}] server closed.".format(METHOD, cid))
            break

        w = float(g.get("w", w))
        b = float(g.get("b", b))
        version = int(g.get("version", version))

        gl, ga = eval_metrics(xs, ys, w, b)
        send_json(s, {"type": "EVAL", "global_loss": gl, "global_acc": ga, "n_samples": len(xs)})

        if PING_ENABLED:
            sent, rec, pl, rmin, ravg, rmax, delay = ping_measure(host)
            with open(ping_path, "a") as fp:
                fp.write("{0},{1},{2},{3},{4},{5},{6:.6f},{7:.6f},{8:.6f},{9:.6f},{10:.6f}\n".format(
                    METHOD, cid, host, k, sent, rec, pl, rmin, ravg, rmax, delay
                ))

    # wait STOP (optional)
    while True:
        msg = recv_json_line(f)
        if not msg:
            break
        if msg.get("type") == "STOP":
            break

    try:
        f.close()
    except Exception:
        pass
    try:
        s.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
