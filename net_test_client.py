#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess
import csv
import os
import time


def run_ping(server_ip, count):
    """
    调用系统 ping，返回：
      sent, received, loss_percent, rtt_min_ms, rtt_avg_ms, rtt_max_ms
    """
    # 注意：Linux 上用 -c，Windows 上是 -n；jFed 节点是 Linux，用 -c 没问题
    cmd = ["ping", "-c", str(count), server_ip]
    out = subprocess.check_output(cmd).decode("utf-8", errors="ignore").splitlines()

    sent = received = 0
    loss = 0.0
    rtt_min = rtt_avg = rtt_max = 0.0

    for line in out:
        line = line.strip()
        # 例如: "4 packets transmitted, 4 received, 0% packet loss, time 3051ms"
        if "packets transmitted" in line and "packet loss" in line:
            parts = line.split(",")
            try:
                sent = int(parts[0].split()[0])
                received = int(parts[1].split()[0])
                loss_str = parts[2].strip().split()[0]  # "0%"
                loss = float(loss_str.replace("%", ""))
            except Exception:
                pass

        # 例如: "rtt min/avg/max/mdev = 0.631/0.660/0.676/0.018 ms"
        if line.startswith("rtt min/avg/max"):
            _, rhs = line.split("=")
            stats, _unit = rhs.split("ms")
            stats = stats.strip()
            try:
                rtt_min, rtt_avg, rtt_max, _mdev = [float(x) for x in stats.split("/")]
            except Exception:
                pass

    return sent, received, loss, rtt_min, rtt_avg, rtt_max


def main():
    if len(sys.argv) < 5:
        print("Usage: python3 {} <mode> <client_id> <server_ip> <rounds> [ping_count]".format(
            sys.argv[0]
        ))
        print("  mode: baseline | fedavg | fedasync")
        sys.exit(1)

    mode = sys.argv[1]          # baseline / fedavg / fedasync
    client_id = sys.argv[2]     # client1 / client2
    server_ip = sys.argv[3]
    rounds = int(sys.argv[4])
    ping_count = int(sys.argv[5]) if len(sys.argv) >= 6 else 50

    csv_path = "/tmp/{}_curve_{}.csv".format(mode, client_id)
    new_file = not os.path.exists(csv_path)

    f = open(csv_path, "a", newline="")
    writer = csv.writer(f)

    if new_file:
        writer.writerow([
            "mode",
            "client_id",
            "server_host",
            "round",
            "sent",
            "received",
            "loss",
            "rtt_min_ms",
            "rtt_avg_ms",
            "rtt_max_ms",
            "delay_avg_ms",
        ])

    print("[NET] mode={}, client_id={}, server_ip={}, rounds={}, ping_count={}, log={}".format(
        mode, client_id, server_ip, rounds, ping_count, csv_path
    ))

    try:
        for r in range(1, rounds + 1):
            sent, received, loss, rtt_min, rtt_avg, rtt_max = run_ping(server_ip, ping_count)

            # 简单地用一半 RTT 近似单向 delay
            delay_avg = rtt_avg / 2.0 if rtt_avg > 0 else 0.0

            writer.writerow([
                mode,
                client_id,
                server_ip,
                r,
                sent,
                received,
                loss,
                rtt_min,
                rtt_avg,
                rtt_max,
                delay_avg,
            ])
            f.flush()

            print("[NET][{}][round {}] sent={}, recv={}, loss={}%, rtt_avg={:.3f} ms, delay_avg={:.3f} ms".format(
                client_id, r, sent, received, loss, rtt_avg, delay_avg
            ))

            # 如果你想拉长测试时间，可以在每轮之间 sleep 一下
            # time.sleep(0.1)

    finally:
        f.close()
        print("[NET] done, results in {}".format(csv_path))


if __name__ == "__main__":
    main()
