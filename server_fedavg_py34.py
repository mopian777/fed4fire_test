#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal FedAvg server (Python 3.4 compatible)
- JSON over TCP, one line per message
- Persistent client connections (clients HELLO then wait for TRAIN)
- Synchronous rounds: select all connected clients each round
- Aggregation: simple average of (w,b)
- Logs:
  - $HOME/server_fedavg.log (stdout via nohup)
  - $HOME/global_metrics_fedavg.csv
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

# Toy linear regression target (client uses its own data)
# We only aggregate params, so server doesn't need data.

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
# Client session bookkeeping
# --------------------------

class ClientSession(object):
    def __init__(self, sock, addr, client_id):
        self.sock = sock
        self.addr = addr
        self.client_id = client_id
        self.fobj = sock.makefile("r")
        self.lock = threading.Lock()

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
# Server core
# --------------------------

def write_csv_header(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("method,round,active_clients,global_w,global_b,global_loss,round_time_ms\n")

def append_csv(path, method, rnd, active_clients, w, b, gl_loss, rt_ms):
    with open(path, "a") as f:
        f.write("{0},{1},{2},{3:.10f},{4:.10f},{5:.10f},{6:.3f}\n".format(
            method, int(rnd), json.dumps(active_clients), float(w), float(b), float(gl_loss), float(rt_ms)
        ))

def run_server(host=HOST, port=PORT):
    write_csv_header(CSV_PATH)

    # Global model params
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
            csock.settimeout(None)
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
            sess = ClientSession(csock, addr, cid)
            # replace fobj with the one created in ClientSession
            # (close old fobj first)
            try:
                fobj.close()
            except Exception:
                pass

            with clients_lock:
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

    # Wait at least 2 clients (or start with whoever is connected)
    # For demo, we'll wait a bit for clients to connect.
    time.sleep(2.0)

    for rnd in range(1, ROUNDS + 1):
        t0 = time.time()

        with clients_lock:
            active_ids = sorted(clients.keys())
            active_sessions = [clients[cid] for cid in active_ids]

        if not active_sessions:
            print("[SERVER-FEDAVG] round={0} no clients yet, waiting...".format(rnd))
            time.sleep(1.0)
            continue

        # Broadcast TRAIN
        train_msg = {
            "type": "TRAIN",
            "round": rnd,
            "w": global_w,
            "b": global_b,
            "local_epochs": LOCAL_EPOCHS,
            "lr": LR
        }

        for sess in active_sessions:
            try:
                sess.send(train_msg)
            except Exception:
                pass

        # Collect UPDATE from all active sessions
        updates = []
        dead = []
        for sess in active_sessions:
            try:
                msg = sess.recv()
                if (not msg) or msg.get("type") != "UPDATE":
                    dead.append(sess.client_id)
                    continue
                w = float(msg.get("w", 0.0))
                b = float(msg.get("b", 0.0))
                local_loss = float(msg.get("local_loss", 0.0))
                updates.append((sess.client_id, w, b, local_loss))
            except Exception:
                dead.append(sess.client_id)

        # Drop dead clients
        if dead:
            with clients_lock:
                for cid in dead:
                    s = clients.get(cid)
                    if s:
                        try:
                            s.close()
                        except Exception:
                            pass
                        del clients[cid]
            print("[SERVER-FEDAVG] dropped clients: {0}".format(dead))

        if not updates:
            print("[SERVER-FEDAVG] round={0} got 0 updates, skip".format(rnd))
            time.sleep(0.2)
            continue

        # FedAvg aggregation
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
        print("[fedavg] round={0} loss={1:.6f} active={2} time={3:.1f}ms w={4:.4f} b={5:.4f}".format(
            rnd, global_loss, len(updates), rt_ms, global_w, global_b
        ))
        append_csv(CSV_PATH, "fedavg", rnd, [u[0] for u in updates], global_w, global_b, global_loss, rt_ms)

    # Stop
    with clients_lock:
        for cid, sess in clients.items():
            try:
                sess.send({"type": "STOP"})
            except Exception:
                pass
            try:
                sess.close()
            except Exception:
                pass
        clients.clear()

    try:
        server_sock.close()
    except Exception:
        pass

    print("[SERVER-FEDAVG] finished. metrics at {0}".format(CSV_PATH))

def host_tf(host):
    # allow "0.0.0.0" as-is
    return host

if __name__ == "__main__":
    # Optional args: [port] [rounds]
    if len(sys.argv) >= 2:
        PORT = int(sys.argv[1])
    if len(sys.argv) >= 3:
        ROUNDS = int(sys.argv[2])
    run_server(host=HOST, port=PORT)
