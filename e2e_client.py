#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import threading
import json
import time
import os
import argparse
import random
import string

METHOD = "e2e"

def now_ms():
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

class E2EClient(object):
    def __init__(self, host, port, client_id, qps, duration_s, payload_bytes, timeout_ms,
                 window_s, lambda_d, lambda_to, mode, perreq_log, summary_log):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.qps = float(qps)
        self.duration_s = float(duration_s)
        self.payload_bytes = int(payload_bytes)
        self.timeout_ms = float(timeout_ms)
        self.window_s = float(window_s)

        # reward: r = -lambda_d * p95_ms - lambda_to * timeout_rate
        self.lambda_d = float(lambda_d)
        self.lambda_to = float(lambda_to)

        self.mode = mode

        self.perreq_log = perreq_log
        self.summary_log = summary_log
        ensure_dir(self.perreq_log)
        ensure_dir(self.summary_log)

        self.sock = None
        self.f = None
        self.send_lock = threading.Lock()

        self.inflight = {}  # req_id -> (t_send_ms, status)
        self.inflight_lock = threading.Lock()

        self._stop = threading.Event()

        # window stats (client-local)
        self.win_start_ms = None
        self.win_lat = []
        self.win_sent = 0
        self.win_ok = 0
        self.win_timeout = 0
        self.win_drop = 0
        self.round_idx = 0

        # total
        self.total_sent = 0

    def init_logs(self):
        if not os.path.exists(self.perreq_log):
            with open(self.perreq_log, "w") as f:
                f.write("mode,client_id,server_host,req_id,t_send_ms,t_recv_ms,e2e_ms,status,payload_bytes\n")
        if not os.path.exists(self.summary_log):
            with open(self.summary_log, "w") as f:
                f.write("mode,client_id,server_host,round,window_s,sent,ok,timeout,drop,timeout_rate,p50_ms,p95_ms,p99_ms,mean_ms,reward\n")

    def log_perreq(self, row):
        with open(self.perreq_log, "a") as f:
            f.write(row + "\n")

    def log_summary(self, row):
        with open(self.summary_log, "a") as f:
            f.write(row + "\n")

    def receiver_loop(self):
        while not self._stop.is_set():
            try:
                msg = recv_json_line(self.f)
            except Exception:
                msg = None
            if not msg:
                break
            if msg.get("type") != "RESP":
                continue

            req_id = msg.get("req_id", "na")
            status = msg.get("status", "OK")
            t_recv = now_ms()

            with self.inflight_lock:
                item = self.inflight.pop(req_id, None)

            if item is None:
                continue

            t_send = item[0]
            e2e = max(0.0, t_recv - t_send)

            self.win_ok += 1 if status == "OK" else 0
            self.win_drop += 1 if status == "DROP" else 0
            self.win_lat.append(e2e)

            row = "%s,%s,%s,%s,%.3f,%.3f,%.3f,%s,%d" % (
                self.mode, self.client_id, self.host, req_id, t_send, t_recv, e2e, status, self.payload_bytes
            )
            self.log_perreq(row)

    def flush_window_if_needed(self, t_now_ms):
        if self.win_start_ms is None:
            self.win_start_ms = t_now_ms
            return
        if (t_now_ms - self.win_start_ms) < (self.window_s * 1000.0):
            return

        # compute stats
        sent = self.win_sent
        ok = self.win_ok
        timeout = self.win_timeout
        drop = self.win_drop

        timeout_rate = 0.0
        if sent > 0:
            timeout_rate = float(timeout + drop) / float(sent)

        lat_sorted = sorted(self.win_lat)
        p50 = percentile(lat_sorted, 50.0)
        p95 = percentile(lat_sorted, 95.0)
        p99 = percentile(lat_sorted, 99.0)
        mean = float(sum(self.win_lat)) / float(len(self.win_lat)) if self.win_lat else float("nan")

        reward = 0.0
        # if no ok samples, still penalize timeouts
        if p95 != p95:  # NaN check
            reward = - self.lambda_to * timeout_rate
        else:
            reward = - self.lambda_d * p95 - self.lambda_to * timeout_rate

        self.round_idx += 1
        row = "%s,%s,%s,%d,%.3f,%d,%d,%d,%d,%.6f,%.3f,%.3f,%.3f,%.3f,%.6f" % (
            self.mode, self.client_id, self.host, self.round_idx, self.window_s,
            sent, ok, timeout, drop, timeout_rate,
            p50 if p50 == p50 else -1.0,
            p95 if p95 == p95 else -1.0,
            p99 if p99 == p99 else -1.0,
            mean if mean == mean else -1.0,
            reward
        )
        self.log_summary(row)

        # reset window
        self.win_start_ms = t_now_ms
        self.win_lat = []
        self.win_sent = 0
        self.win_ok = 0
        self.win_timeout = 0
        self.win_drop = 0

    def timeout_sweep(self, t_now_ms):
        # mark inflight older than timeout as TIMEOUT
        timeout_cut = t_now_ms - self.timeout_ms
        to_expire = []
        with self.inflight_lock:
            for rid, item in self.inflight.items():
                if item[0] <= timeout_cut:
                    to_expire.append(rid)
            for rid in to_expire:
                t_send = self.inflight[rid][0]
                del self.inflight[rid]

        for rid in to_expire:
            self.win_timeout += 1
            row = "%s,%s,%s,%s,%.3f,%.3f,%.3f,%s,%d" % (
                self.mode, self.client_id, self.host, rid,
                t_send, t_now_ms, max(0.0, t_now_ms - t_send),
                "TIMEOUT", self.payload_bytes
            )
            self.log_perreq(row)

    def gen_payload(self, n):
        # avoid huge json; just a short string
        # if you want real payload, you can enlarge it
        return "x" * max(0, n)

    def run(self):
        self.init_logs()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.f = self.sock.makefile("r")

        th = threading.Thread(target=self.receiver_loop)
        th.daemon = True
        th.start()

        t0 = now_ms()
        self.win_start_ms = t0

        # fixed-rate send loop
        interval = 1000.0 / self.qps if self.qps > 0 else 1000.0
        next_send = t0

        req_seq = 0
        while True:
            t_now = now_ms()
            elapsed_s = (t_now - t0) / 1000.0
            if elapsed_s >= self.duration_s:
                break

            # send at target QPS
            if t_now >= next_send:
                req_seq += 1
                req_id = "%s-%d-%d" % (self.client_id, int(time.time()), req_seq)
                t_send = t_now

                msg = {
                    "type": "REQ",
                    "client_id": self.client_id,
                    "req_id": req_id,
                    "t_send_ms": t_send,
                    "payload_bytes": self.payload_bytes
                }

                with self.inflight_lock:
                    self.inflight[req_id] = (t_send, "INFLIGHT")

                try:
                    send_json_line(self.sock, msg, lock=self.send_lock)
                    self.win_sent += 1
                    self.total_sent += 1
                except Exception:
                    # connection broken
                    break

                next_send += interval

            # timeout and window flush
            self.timeout_sweep(t_now)
            self.flush_window_if_needed(t_now)

            time.sleep(0.001)

        # final flush
        t_end = now_ms()
        self.timeout_sweep(t_end)
        self.flush_window_if_needed(t_end + self.window_s * 1000.0 + 1.0)

        self._stop.set()
        try:
            self.f.close()
        except:
            pass
        try:
            self.sock.close()
        except:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--client_id", required=True)
    ap.add_argument("--qps", type=float, default=20.0)
    ap.add_argument("--duration_s", type=float, default=60.0)
    ap.add_argument("--payload_bytes", type=int, default=256)
    ap.add_argument("--timeout_ms", type=float, default=300.0)
    ap.add_argument("--window_s", type=float, default=1.0)
    ap.add_argument("--lambda_d", type=float, default=0.01)
    ap.add_argument("--lambda_to", type=float, default=10.0)
    ap.add_argument("--mode", default=METHOD)

    home = os.environ.get("HOME", "/tmp")
    ap.add_argument("--perreq_log", default=os.path.join(home, "e2e_perreq_%s.csv" % METHOD))
    ap.add_argument("--summary_log", default=os.path.join(home, "e2e_summary_%s.csv" % METHOD))
    args = ap.parse_args()

    # personalize filenames by client_id
    perreq = args.perreq_log.replace(".csv", "_%s.csv" % args.client_id)
    summ = args.summary_log.replace(".csv", "_%s.csv" % args.client_id)

    cli = E2EClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        qps=args.qps,
        duration_s=args.duration_s,
        payload_bytes=args.payload_bytes,
        timeout_ms=args.timeout_ms,
        window_s=args.window_s,
        lambda_d=args.lambda_d,
        lambda_to=args.lambda_to,
        mode=args.mode,
        perreq_log=perreq,
        summary_log=summ
    )
    cli.run()

if __name__ == "__main__":
    main()
