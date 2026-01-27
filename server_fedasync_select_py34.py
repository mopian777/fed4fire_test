#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FedAsync server (Python 3.4 compatible)
- JSON line protocol over TCP
- True async scheduling via select() + per-socket line buffering
- State machine per client: expect UPDATE -> send GLOBAL -> expect EVAL -> log -> back to UPDATE
- Avoid starvation caused by fixed-order polling/readline()
"""

from __future__ import print_function

import os
import time
import json
import math
import socket
import select

HOST = "0.0.0.0"
PORT = 5000

MIN_CLIENTS = 2
TOTAL_UPDATES = 400

METHOD = "fedasync"
LOG_PATH = os.path.join(os.environ.get("HOME", "/tmp"), "global_metrics_fedasync.csv")

BASE_ALPHA = 0.6            # alpha = BASE_ALPHA / (1+staleness)
ALPHA_TIME = 0.001
BETA_LOSS = 1.0
GAMMA_ACC = 1.0

EVAL_TIMEOUT_S = 2.0        # if client never sends EVAL after GLOBAL, log NaN and continue
SELECT_TICK_S = 0.2


def send_json(sock, obj):
    data = json.dumps(obj) + "\n"
    sock.sendall(data.encode("utf-8"))


def ensure_log():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            f.write("method,step,wall_time_s,step_time_ms,active_clients,global_loss,global_acc,reward,staleness\n")


def log_row(step, wall_s, step_ms, active, loss, acc, reward, staleness):
    # keep CSV simple and compatible
    with open(LOG_PATH, "a") as f:
        # active is a single client id string
        f.write("{0},{1},{2:.6f},{3:.3f},\"{4}\",{5:.10f},{6:.6f},{7:.10f},{8}\n".format(
            METHOD, int(step), float(wall_s), float(step_ms), str(active),
            float(loss), float(acc), float(reward), int(staleness)
        ))


def parse_lines_from_buffer(buf):
    # buf: bytes; return (lines_as_str_list, remaining_bytes)
    lines = []
    while True:
        idx = buf.find(b"\n")
        if idx < 0:
            break
        line = buf[:idx]
        buf = buf[idx + 1:]
        line = line.strip()
        if line:
            try:
                lines.append(line.decode("utf-8"))
            except Exception:
                # drop undecodable line
                pass
    return lines, buf


def safe_json_loads(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def main():
    ensure_log()
    t0 = time.time()

    # global params
    w = 0.0
    b = 0.0
    version = 0

    # listening socket
    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind((HOST, PORT))
    ls.listen(50)
    ls.setblocking(0)

    print("[{0}] listening on {1}:{2}".format(METHOD, HOST, PORT))

    # client table: sock -> info
    clients = {}  # sock: dict

    step = 0
    last_update_t = time.time()

    def drop_client(cs, reason):
        info = clients.get(cs)
        cid = info.get("id") if info else "unknown"
        try:
            cs.close()
        except Exception:
            pass
        if cs in clients:
            del clients[cs]
        print("[{0}] drop client {1} ({2})".format(METHOD, cid, reason))

    # main loop
    while step < TOTAL_UPDATES:
        # build read set: listening socket + all client sockets
        rlist = [ls]
        for cs in clients.keys():
            rlist.append(cs)

        # select readable
        try:
            readable, _, _ = select.select(rlist, [], [], SELECT_TICK_S)
        except Exception:
            readable = []

        now = time.time()

        # timeout check: clients waiting for EVAL too long
        for cs, info in list(clients.items()):
            if info.get("state") == "WAIT_EVAL":
                t_sent = info.get("global_sent_time", 0.0)
                if t_sent and (now - t_sent) > EVAL_TIMEOUT_S:
                    # log NaN and continue
                    step_i = info.get("pending_step", None)
                    if step_i is not None:
                        staleness = int(info.get("pending_staleness", 0))
                        step_time_ms = float(info.get("pending_step_time_ms", 0.0))
                        wall_s = now - t0
                        gl = float("nan")
                        ga = float("nan")
                        reward = -ALPHA_TIME * step_time_ms
                        log_row(step_i, wall_s, step_time_ms, info.get("id", "unknown"), gl, ga, reward, staleness)
                        print("[{0}] step={1} from={2} EVAL timeout -> log NaN".format(
                            METHOD, step_i, info.get("id", "unknown")
                        ))
                    info["state"] = "WAIT_UPDATE"
                    info["pending_step"] = None

        for s in readable:
            if s is ls:
                # accept as many as possible
                while True:
                    try:
                        cs, addr = ls.accept()
                    except Exception:
                        break
                    cs.setblocking(0)
                    clients[cs] = {
                        "sock": cs,
                        "addr": addr,
                        "buf": b"",
                        "id": None,
                        "state": "WAIT_HELLO",
                        # pending fields for logging
                        "pending_step": None,
                        "pending_staleness": 0,
                        "pending_step_time_ms": 0.0,
                        "global_sent_time": 0.0
                    }
            else:
                cs = s
                info = clients.get(cs)
                if not info:
                    continue
                try:
                    data = cs.recv(4096)
                except Exception:
                    drop_client(cs, "recv error")
                    continue
                if not data:
                    drop_client(cs, "closed")
                    continue

                info["buf"] += data
                lines, rest = parse_lines_from_buffer(info["buf"])
                info["buf"] = rest

                for line in lines:
                    msg = safe_json_loads(line)
                    if not msg:
                        continue

                    st = info.get("state")

                    # -------- HELLO --------
                    if st == "WAIT_HELLO":
                        if msg.get("type") != "HELLO":
                            drop_client(cs, "no HELLO")
                            break
                        cid = str(msg.get("client_id", "unknown"))
                        info["id"] = cid
                        info["state"] = "WAIT_UPDATE"
                        # reply INIT
                        try:
                            send_json(cs, {"type": "INIT", "w": w, "b": b, "version": version})
                        except Exception:
                            drop_client(cs, "send INIT failed")
                            break
                        print("[{0}] accepted {1} from {2}:{3}".format(
                            METHOD, cid, info["addr"][0], info["addr"][1]
                        ))
                        continue

                    # before enough clients, ignore UPDATE/EVAL
                    if len([x for x in clients.values() if x.get("id")]) < MIN_CLIENTS:
                        continue

                    # -------- UPDATE --------
                    if st == "WAIT_UPDATE":
                        if msg.get("type") != "UPDATE":
                            continue

                        # process one UPDATE immediately
                        step += 1

                        cid = info.get("id", "unknown")
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

                        now2 = time.time()
                        step_time_ms = (now2 - last_update_t) * 1000.0
                        last_update_t = now2

                        # send GLOBAL
                        try:
                            send_json(cs, {"type": "GLOBAL", "step": step, "w": w, "b": b, "version": version})
                        except Exception:
                            drop_client(cs, "send GLOBAL failed")
                            break

                        # switch to WAIT_EVAL and store pending info for logging
                        info["state"] = "WAIT_EVAL"
                        info["pending_step"] = step
                        info["pending_staleness"] = staleness
                        info["pending_step_time_ms"] = step_time_ms
                        info["global_sent_time"] = time.time()

                        print("[{0}] step={1} from={2} ver={3} stale={4} alpha={5:.3f}".format(
                            METHOD, step, cid, version, staleness, alpha
                        ))

                        if step >= TOTAL_UPDATES:
                            break
                        continue

                    # -------- EVAL --------
                    if st == "WAIT_EVAL":
                        if msg.get("type") != "EVAL":
                            continue

                        cid = info.get("id", "unknown")
                        gl = float(msg.get("global_loss", 0.0))
                        ga = float(msg.get("global_acc", 0.0))

                        step_i = info.get("pending_step", None)
                        if step_i is None:
                            # stray eval, ignore
                            info["state"] = "WAIT_UPDATE"
                            continue

                        staleness = int(info.get("pending_staleness", 0))
                        step_time_ms = float(info.get("pending_step_time_ms", 0.0))
                        wall_s = time.time() - t0

                        # reward
                        reward = -ALPHA_TIME * step_time_ms - BETA_LOSS * gl + GAMMA_ACC * ga

                        log_row(step_i, wall_s, step_time_ms, cid, gl, ga, reward, staleness)

                        print("[{0}] step={1} from={2} loss={3:.6f} acc={4:.4f} step_time={5:.1f}ms".format(
                            METHOD, step_i, cid, gl, ga, step_time_ms
                        ))

                        # back to update state
                        info["state"] = "WAIT_UPDATE"
                        info["pending_step"] = None
                        continue

        # loop continues until TOTAL_UPDATES

    # send STOP to all
    for cs, info in list(clients.items()):
        try:
            send_json(cs, {"type": "STOP"})
        except Exception:
            pass
        try:
            cs.close()
        except Exception:
            pass
        del clients[cs]

    try:
        ls.close()
    except Exception:
        pass

    print("[{0}] done. log: {1}".format(METHOD, LOG_PATH))


if __name__ == "__main__":
    main()
