#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal FedAsync server (Python 3.4 compatible)
- JSON line over TCP
- Accept N clients (default 2)
- Process incoming UPDATE asynchronously (polling)
- Staleness-aware alpha = BASE_ALPHA / (1 + staleness)
- Log to $HOME/fed4fire_logs/global_metrics_fedasync.csv
"""

from __future__ import print_function

import socket
import json
import time
import os
import math

PORT = 5000
TOTAL_UPDATES = 400        # 总共处理多少次 UPDATE（2 clients * 200 = 400）
EXPECTED_CLIENTS = 2
METHOD = "fedasync"

# 建议写到 HOME，避免 /tmp permission denied
LOG_DIR = os.path.join(os.environ.get("HOME", "/tmp"), "fed4fire_logs")
LOG_PATH = os.path.join(LOG_DIR, "global_metrics_fedasync.csv")

# 异步混合：alpha(staleness)=BASE_ALPHA / (1+staleness)
BASE_ALPHA = 0.6

ALPHA_TIME = 0.001
BETA_LOSS = 1.0
GAMMA_ACC = 1.0


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


def ensure_log():
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
        except Exception:
            pass

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            f.write("method,step,wall_time_s,step_time_ms,active_clients,"
                    "global_loss,global_acc,reward,staleness\n")


def log_row(step, wall_s, step_ms, active, loss, acc, reward, staleness):
    with open(LOG_PATH, "a") as f:
        f.write("{0},{1},{2:.6f},{3:.3f},\"{4}\",{5:.10f},{6:.6f},{7:.10f},{8}\n".format(
            METHOD, int(step), float(wall_s), float(step_ms), str(active),
            float(loss), float(acc), float(reward), int(staleness)
        ))


def main():
    ensure_log()
    t0 = time.time()

    # global params + version
    w, b = 0.0, 0.0
    version = 0

    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("0.0.0.0", PORT))
    ls.listen(10)
    print("[{0}] listening on 0.0.0.0:{1}".format(METHOD, PORT))

    clients = []  # list of dict: id, sock, file
    while len(clients) < EXPECTED_CLIENTS:
        cs, addr = ls.accept()
        f = cs.makefile("r")
        hello = None
        try:
            hello = recv_json_line(f)
        except Exception:
            hello = None

        if (not hello) or hello.get("type") != "HELLO":
            try:
                f.close()
            except Exception:
                pass
            try:
                cs.close()
            except Exception:
                pass
            continue

        cid = hello.get("client_id", None)
        if cid is None:
            cid = "client{0}".format(len(clients) + 1)

        clients.append({"id": str(cid), "sock": cs, "file": f})
        send_json(cs, {"type": "INIT", "w": w, "b": b, "version": version})
        print("[{0}] accepted {1} from {2}:{3}".format(METHOD, cid, addr[0], addr[1]))

    last_update_t = time.time()
    step = 0

    # 主循环：谁先发 UPDATE 就先处理
    while step < TOTAL_UPDATES:
        handled = False

        for c in clients:
            # 非阻塞轮询
            try:
                c["sock"].settimeout(0.05)
            except Exception:
                pass

            try:
                msg = recv_json_line(c["file"])
            except Exception:
                msg = None

            if not msg:
                continue

            mtype = msg.get("type", "")

            # 如果碰到 EVAL（因为 client 可能晚到），这里直接丢弃/忽略即可
            #（我们只在处理 UPDATE 后“就地”读一次 EVAL）
            if mtype != "UPDATE":
                continue

            # --- handle UPDATE ---
            step += 1
            handled = True

            cid = c["id"]
            base_ver = int(msg.get("base_version", 0))
            staleness = version - base_ver
            if staleness < 0:
                staleness = 0

            cw = float(msg.get("w", w))
            cb = float(msg.get("b", b))

            alpha = BASE_ALPHA / (1.0 + float(staleness))
            w = (1.0 - alpha) * w + alpha * cw
            b = (1.0 - alpha) * b + alpha * cb
            version += 1

            # 回传最新全局模型
            send_json(c["sock"], {"type": "GLOBAL", "step": step, "w": w, "b": b, "version": version})

            # 读一次 EVAL（给一个较短超时，避免阻塞）
            global_loss = float("nan")
            global_acc = float("nan")
            try:
                c["sock"].settimeout(0.2)
                evalmsg = recv_json_line(c["file"])
            except Exception:
                evalmsg = None

            if evalmsg and evalmsg.get("type") == "EVAL":
                try:
                    global_loss = float(evalmsg.get("global_loss", 0.0))
                except Exception:
                    global_loss = float("nan")
                try:
                    global_acc = float(evalmsg.get("global_acc", 0.0))
                except Exception:
                    global_acc = float("nan")

            now = time.time()
            step_time_ms = (now - last_update_t) * 1000.0
            last_update_t = now
            wall_s = now - t0

            # reward（示例）
            if (isinstance(global_loss, float) and math.isnan(global_loss)) or \
               (isinstance(global_acc, float) and math.isnan(global_acc)):
                reward = -ALPHA_TIME * step_time_ms
            else:
                reward = -ALPHA_TIME * step_time_ms - BETA_LOSS * global_loss + GAMMA_ACC * global_acc

            log_row(step, wall_s, step_time_ms, cid, global_loss, global_acc, reward, staleness)

            print("[{0}] step={1} from={2} ver={3} stale={4} alpha={5:.3f} "
                  "loss={6:.6f} acc={7:.4f} step_time={8:.1f}ms".format(
                      METHOD, step, cid, version, staleness, alpha,
                      (global_loss if not (isinstance(global_loss, float) and math.isnan(global_loss)) else -1.0),
                      (global_acc if not (isinstance(global_acc, float) and math.isnan(global_acc)) else -1.0),
                      step_time_ms
                  ))

            break  # 处理一个就回到 while，保持“异步逐条处理”的节奏

        if not handled:
            time.sleep(0.01)

    # stop all
    for c in clients:
        try:
            send_json(c["sock"], {"type": "STOP"})
        except Exception:
            pass
        try:
            c["file"].close()
        except Exception:
            pass
        try:
            c["sock"].close()
        except Exception:
            pass

    try:
        ls.close()
    except Exception:
        pass

    print("[{0}] done. log: {1}".format(METHOD, LOG_PATH))


if __name__ == "__main__":
    main()
