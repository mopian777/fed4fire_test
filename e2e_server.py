#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import threading
import json
import time
import os
import argparse

try:
    import queue  # py3
except ImportError:
    import Queue as queue  # py2 fallback

METHOD = "e2e"
DEFAULT_PORT = 6000

def now_ms():
    # monotonic exists in py3.4
    return time.monotonic() * 1000.0

def send_json_line(sock, obj, lock=None):
    data = (json.dumps(obj) + "\n").encode("utf-8")
    if lock is None:
        sock.sendall(data)
    else:
        with lock:
            sock.sendall(data)

def recv_json_line(f):
    line = f.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)

def ensure_dir(p):
    d = os.path.dirname(p)
    if d and (not os.path.exists(d)):
        try:
            os.makedirs(d)
        except:
            pass

def percentile(sorted_vals, p):
    # p in [0,100]
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_vals[0])
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

class E2EServer(object):
    def __init__(self, host, port, queue_size, proc_ms, mode, log_path, workers, drop_on_full):
        self.host = host
        self.port = port
        self.mode = mode
        self.proc_ms = float(proc_ms)
        self.drop_on_full = bool(drop_on_full)
        self.q = queue.Queue(maxsize=int(queue_size))

        self.log_path = log_path
        ensure_dir(self.log_path)
        self._log_lock = threading.Lock()

        self.workers = int(workers)
        self._stop = threading.Event()

        self._client_locks = {}     # sock -> lock
        self._client_locks_lock = threading.Lock()

    def init_log(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("mode,server_host,req_id,client_id,status,arrive_ms,start_ms,done_ms,queue_ms,proc_ms,payload_bytes\n")

    def log_row(self, row):
        with self._log_lock:
            with open(self.log_path, "a") as f:
                f.write(row + "\n")

    def _get_sock_lock(self, sock):
        with self._client_locks_lock:
            lk = self._client_locks.get(sock)
            if lk is None:
                lk = threading.Lock()
                self._client_locks[sock] = lk
            return lk

    def worker_loop(self, wid):
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.2)
            except Exception:
                continue
            if item is None:
                continue

            sock, req = item
            lk = self._get_sock_lock(sock)

            t_start = now_ms()
            queue_ms = max(0.0, t_start - req["arrive_ms"])

            # simulate processing
            # Option 1: sleep (simple & stable)
            if self.proc_ms > 0:
                time.sleep(self.proc_ms / 1000.0)

            t_done = now_ms()
            proc_ms = max(0.0, t_done - t_start)

            resp = {
                "type": "RESP",
                "req_id": req["req_id"],
                "status": "OK",
                "server_start_ms": t_start,
                "server_done_ms": t_done
            }

            try:
                send_json_line(sock, resp, lock=lk)
            except Exception:
                # client likely disconnected
                pass

            # log
            row = "%s,%s,%s,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%d" % (
                self.mode,
                self.host,
                req["req_id"],
                req["client_id"],
                "OK",
                req["arrive_ms"],
                t_start,
                t_done,
                queue_ms,
                proc_ms,
                req.get("payload_bytes", 0)
            )
            self.log_row(row)

            try:
                self.q.task_done()
            except Exception:
                pass

    def client_thread(self, cs, addr):
        f = cs.makefile("r")
        while not self._stop.is_set():
            try:
                msg = recv_json_line(f)
            except Exception:
                msg = None
            if not msg:
                break
            if msg.get("type") != "REQ":
                continue

            t_arrive = now_ms()
            client_id = msg.get("client_id", "unknown")
            req_id = msg.get("req_id", "na")
            payload_bytes = int(msg.get("payload_bytes", 0))

            req = {
                "client_id": client_id,
                "req_id": req_id,
                "arrive_ms": t_arrive,
                "payload_bytes": payload_bytes
            }

            # enqueue or drop
            if self.drop_on_full:
                try:
                    self.q.put_nowait((cs, req))
                except Exception:
                    # drop immediately
                    resp = {"type": "RESP", "req_id": req_id, "status": "DROP"}
                    try:
                        send_json_line(cs, resp, lock=self._get_sock_lock(cs))
                    except Exception:
                        pass

                    row = "%s,%s,%s,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%d" % (
                        self.mode,
                        self.host,
                        req_id,
                        client_id,
                        "DROP",
                        t_arrive,
                        t_arrive,
                        t_arrive,
                        0.0,
                        0.0,
                        payload_bytes
                    )
                    self.log_row(row)
            else:
                # blocking put (can increase queueing -> higher delay)
                try:
                    self.q.put((cs, req))
                except Exception:
                    pass

        try:
            f.close()
        except:
            pass
        try:
            cs.close()
        except:
            pass

    def serve_forever(self):
        self.init_log()

        # start workers
        for i in range(self.workers):
            th = threading.Thread(target=self.worker_loop, args=(i,))
            th.daemon = True
            th.start()

        ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ls.bind((self.host, self.port))
        ls.listen(64)
        print("[E2E_SERVER] listen %s:%d mode=%s proc_ms=%.1f qsize=%d workers=%d drop_on_full=%s log=%s" % (
            self.host, self.port, self.mode, self.proc_ms, self.q.maxsize, self.workers, str(self.drop_on_full), self.log_path
        ))

        try:
            while True:
                cs, addr = ls.accept()
                th = threading.Thread(target=self.client_thread, args=(cs, addr))
                th.daemon = True
                th.start()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            try:
                ls.close()
            except:
                pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--mode", default=METHOD)
    ap.add_argument("--queue_size", type=int, default=200)
    ap.add_argument("--proc_ms", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--drop_on_full", action="store_true")
    ap.add_argument("--log", default=os.path.join(os.environ.get("HOME", "/tmp"), "e2e_server_log.csv"))
    args = ap.parse_args()

    srv = E2EServer(
        host=args.host,
        port=args.port,
        queue_size=args.queue_size,
        proc_ms=args.proc_ms,
        mode=args.mode,
        log_path=args.log,
        workers=args.workers,
        drop_on_full=args.drop_on_full
    )
    srv.serve_forever()

if __name__ == "__main__":
    main()
