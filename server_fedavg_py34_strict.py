#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict synchronous FedAvg server (Python 3.4 compatible)
- JSON over TCP, one line per message
- Persistent client connections
- Strict barrier: requires BOTH client1 & client2 each round (2-of-2)
- CSV overwritten each run: $HOME/global_metrics_fedavg.csv
"""

from __future__ import print_function

import os
import sys
import time
import json
import socket
import threading

HOST = "0.0.0.0"
PORT = 5000

ROUNDS = 200
LOCAL_EPOCHS = 5
LR = 0.02

# Require exactly these clients (edit if your IDs differ)
REQUIRED_CLIENT_IDS = ["client1", "client2"]
RECV_TIMEOUT_SEC = 20.0   # per-round wait for each client UPDATE

CSV_PATH = os.path.join(os.environ.get("HOME", "/tmp"), "global_metrics_fedavg.csv")


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
# Client session
# --------------------------

class ClientSession(object):
    def __init__(self, sock, addr, client_id, fobj):
        self.sock = sock
        self.addr = addr
        self.client_id = client_id
        self.fobj = fobj
        self.lock = threading.Lock()

    def set_timeout(self, sec):
        try:
            self.sock.settimeout(sec)
        except Exception:
            pass

    def send(self, msg):
        with self.lock:
            send_json(self.sock, msg)

    def recv(self):
        return recv_json_line(self.fobj)

    def close(self):
        try:
            self.fobj.close()
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


# --------------------------
# CSV helpers
# --------------------------

def write_csv_header_overwrite(path):
    # overwrite every run to avoid mixing old experiments
    with open(path, "w") as f:
        f.write("method,round,active_clients,global_w,global_b,global_loss,round_time_ms\n")

def append_csv(path, method, rnd, active_clients, w, b, gl_loss, rt_ms):
    with open(path, "a") as f:
        f.write("{0},{1},{2},{3:.10f},{4:.10f},{5:.10f},{6:.3f}\n".format(
            method, int(rnd), json.dumps(active_clients),
            float(w), float(b), float(gl_loss), float(rt_ms)
        ))


# --------------------------
# Server core
# --------------------------

def host_tf(host):
    return host

def run_server(host=HOST, port=PORT):
    write_csv_header_overwrite(CSV_PATH)

    global_w = 0.0
    global_b = 0.0

    clients = {}
    clients_lock = threading.Lock()

    def accept_loop(server_sock):
        while True:
            try:
                csock, addr = server_sock.accept()
            except Exception:
                continue

            # IMPORTANT: create makefile only once, and DO NOT close it after HELLO
            fobj = csock.makefile("r")

            # Expect HELLO
            try:
                hello = recv_json_line(fobj)
            except Exception:
                try:
                    fobj.close()
                except Exception:
                    pass
                try:
                    csock.close()
                except Exception:
                    pass
                continue

            if (not hello) or hello.get("type") != "HELLO":
                try:
                    fobj.close()
                except Exception:
                    pass
                try:
                    csock.close()
                except Exception:
                    pass
                continue

            cid = str(hello.get("client_id", "unknown"))
            sess = ClientSession(csock, addr, cid, fobj)

            with clients_lock:
                # replace if reconnect
                old = clients.get(cid)
                if old:
                    try:
                        old.close()
                    except Exception:
                        pass
                clients[cid] = sess

            print("[SERVER-FEDAVG] HELLO from {0} @ {1}".format(cid, addr))

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host_tf(host), port))
    server_sock.listen(50)
    print("[SERVER-FEDAVG] Listening on {0}:{1} ...".format(host, port))

    t = threading.Thread(target=accept_loop, args=(server_sock,))
    t.daemon = True
    t.start()

    # ----- Barrier: wait required clients -----
    print("[SERVER-FEDAVG] Waiting for required clients: {0}".format(REQUIRED_CLIENT_IDS))
    while True:
        with clients_lock:
            ok = True
            for cid in REQUIRED_CLIENT_IDS:
                if cid not in clients:
                    ok = False
                    break
        if ok:
            break
        time.sleep(0.2)

    print("[SERVER-FEDAVG] All required clients connected. Start training.")

    rnd = 1
    while rnd <= ROUNDS:
        t0 = time.time()

        with clients_lock:
            sessions = []
            missing = []
            for cid in REQUIRED_CLIENT_IDS:
                s = clients.get(cid)
                if s is None:
                    missing.append(cid)
                else:
                    sessions.append(s)

        if missing:
            print("[SERVER-FEDAVG] round={0} missing clients={1}, waiting...".format(rnd, missing))
            time.sleep(0.5)
            continue

        # Broadcast TRAIN (to both)
        train_msg = {
            "type": "TRAIN",
            "round": rnd,
            "w": global_w,
            "b": global_b,
            "local_epochs": LOCAL_EPOCHS,
            "lr": LR
        }

        # send
        for s in sessions:
            try:
                s.send(train_msg)
            except Exception:
                pass

        # collect
        updates = []
        dead = []
        for s in sessions:
            try:
                s.set_timeout(RECV_TIMEOUT_SEC)
                msg = s.recv()
                if (not msg) or msg.get("type") != "UPDATE" or int(msg.get("round", -1)) != int(rnd):
                    dead.append(s.client_id)
                    continue
                w = float(msg.get("w", 0.0))
                b = float(msg.get("b", 0.0))
                local_loss = float(msg.get("local_loss", 0.0))
                updates.append((s.client_id, w, b, local_loss))
            except Exception:
                dead.append(s.client_id)

        # If any client missing -> do NOT advance round; wait reconnect and retry same round
        if dead or (len(updates) != len(REQUIRED_CLIENT_IDS)):
            if dead:
                with clients_lock:
                    for cid in dead:
                        ss = clients.get(cid)
                        if ss:
                            try:
                                ss.close()
                            except Exception:
                                pass
                            try:
                                del clients[cid]
                            except Exception:
                                pass
                print("[SERVER-FEDAVG] round={0} dropped clients: {1}".format(rnd, dead))

            print("[SERVER-FEDAVG] round={0} updates={1}/{2} -> retry same round".format(
                rnd, len(updates), len(REQUIRED_CLIENT_IDS)
            ))
            time.sleep(0.5)
            continue

        # FedAvg aggregate (2-of-2)
        sum_w = 0.0
        sum_b = 0.0
        sum_loss = 0.0
        for (cid, w, b, l) in updates:
            sum_w += w
            sum_b += b
            sum_loss += l

        n = float(len(updates))
        global_w = sum_w / n
        global_b = sum_b / n
        global_loss = sum_loss / n

        rt_ms = (time.time() - t0) * 1000.0
        print("[fedavg] round={0} loss={1:.6f} active=2 time={2:.1f}ms w={3:.4f} b={4:.4f}".format(
            rnd, global_loss, rt_ms, global_w, global_b
        ))
        append_csv(CSV_PATH, "fedavg", rnd, [u[0] for u in updates], global_w, global_b, global_loss, rt_ms)

        rnd += 1  # ONLY advance when both updates collected

    # Stop
    with clients_lock:
        for cid, s in list(clients.items()):
            try:
                s.send({"type": "STOP"})
            except Exception:
                pass
            try:
                s.close()
            except Exception:
                pass
        clients.clear()

    try:
        server_sock.close()
    except Exception:
        pass

    print("[SERVER-FEDAVG] finished. metrics at {0}".format(CSV_PATH))


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        PORT = int(sys.argv[1])
    if len(sys.argv) >= 3:
        ROUNDS = int(sys.argv[2])
    run_server(host=HOST, port=PORT)
