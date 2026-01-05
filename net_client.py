#!/usr/bin/env python3
import socket
import json
import sys
import time
import subprocess
import os
import csv

PORT = 6000   # 必须与 net_server.py 一致

ROUNDS = 10          # 测 10 轮
NUM_PACKETS = 50     # 每轮发多少个探测包（可调）
INTERVAL = 0.1       # 包间隔秒
TIMEOUT = 1.0        # 等 echo 的超时时间秒


def discover_server(port=PORT,
                    per_ip_timeout=0.03,
                    max_passes=5,
                    sleep_between=1.0):
    """
    在本地 /24 网段扫描哪个 IP 的 UDP port=6000 能响应。
    """
    out = subprocess.check_output(["hostname", "-I"]).decode().split()
    if not out:
        raise RuntimeError("Cannot get local IP from 'hostname -I'")
    my_ip = out[0]
    prefix, last = my_ip.rsplit(".", 1)
    print("[discover-net] my_ip = {}, prefix = {}".format(my_ip, prefix))

    for attempt in range(1, max_passes + 1):
        print("[discover-net] scan pass {}/{} on {}.0/24".format(
            attempt, max_passes, prefix
        ))
        for i in range(1, 255):
            ip = "{}.{}".format(prefix, i)
            if ip == my_ip:
                continue
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(per_ip_timeout)
            try:
                probe = {"client_id": "probe", "seq": -1}
                sock.sendto(json.dumps(probe).encode("utf-8"), (ip, port))
                _data, _addr = sock.recvfrom(4096)
                sock.close()
                print("[discover-net] found UDP server at {} (pass {})".format(
                    ip, attempt
                ))
                return ip
            except OSError:
                sock.close()
                continue
        print("[discover-net] no UDP server in pass {}, sleep {}s".format(
            attempt, sleep_between
        ))
        time.sleep(sleep_between)

    raise RuntimeError(
        "UDP server not found on {}.0/24 after {} passes".format(prefix, max_passes)
    )


def measure_once(client_id, server_host,
                 num_packets=NUM_PACKETS,
                 interval=INTERVAL,
                 timeout=TIMEOUT):
    """
    发送 num_packets 个 UDP 探测包，统计 RTT 与丢包。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)

    rtts = []
    sent = 0
    received = 0

    for seq in range(num_packets):
        msg = {"client_id": client_id, "seq": seq}
        payload = json.dumps(msg).encode("utf-8")
        t0 = time.time()
        try:
            sock.sendto(payload, (server_host, PORT))
            sent += 1

            data, addr = sock.recvfrom(4096)
            t1 = time.time()

            try:
                echo = json.loads(data.decode("utf-8"))
            except ValueError:
                continue

            if echo.get("seq") != seq:
                continue

            rtt_ms = (t1 - t0) * 1000.0
            rtts.append(rtt_ms)
            received += 1
        except socket.timeout:
            # 超时视为丢包
            pass

        time.sleep(interval)

    sock.close()

    loss = 0.0
    if sent > 0:
        loss = float(sent - received) / float(sent)

    if rtts:
        rtt_min = min(rtts)
        rtt_max = max(rtts)
        rtt_avg = sum(rtts) / float(len(rtts))
    else:
        rtt_min = rtt_max = rtt_avg = 0.0

    delay_avg = rtt_avg / 2.0   # 近似单向时延

    stats = {
        "sent": sent,
        "received": received,
        "loss": loss,
        "rtt_min": rtt_min,
        "rtt_avg": rtt_avg,
        "rtt_max": rtt_max,
        "delay_avg": delay_avg,
    }

    return stats


def multi_round_measure(client_id, mode, server_host,
                        rounds=ROUNDS,
                        num_packets=NUM_PACKETS,
                        interval=INTERVAL,
                        timeout=TIMEOUT,
                        pause_between=1.0):
    """
    连续测 rounds 轮，每轮调用一次 measure_once，返回包含 round 字段的列表。
    """
    all_stats = []
    for r in range(1, rounds + 1):
        print("=== [{}-NET][{}] Round {} ===".format(client_id, mode, r))
        stats = measure_once(client_id, server_host,
                             num_packets=num_packets,
                             interval=interval,
                             timeout=timeout)
        stats["round"] = r
        all_stats.append(stats)

        print("[{}-NET][{}] Round {}: sent={}, recv={}, loss={:.2%}, "
              "rtt_avg={:.3f} ms, delay_avg={:.3f} ms".format(
                  client_id, mode, r,
                  stats["sent"], stats["received"], stats["loss"],
                  stats["rtt_avg"], stats["delay_avg"]))
        time.sleep(pause_between)
    return all_stats


def write_csv(stats_list, client_id, mode, server_host, csv_path):
    """
    把多轮测量结果追加写入 CSV。
    CSV 列：mode, client_id, server_host, round, sent, received,
           loss, rtt_min, rtt_avg, rtt_max, delay_avg
    """
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        writer = csv.writer(f)
        if not file_exists:
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
        for s in stats_list:
            writer.writerow([
                mode,
                client_id,
                server_host,
                s["round"],
                s["sent"],
                s["received"],
                s["loss"],
                s["rtt_min"],
                s["rtt_avg"],
                s["rtt_max"],
                s["delay_avg"],
            ])


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 {} <client_id> <mode> [server_ip_or_auto]".format(
            sys.argv[0]
        ))
        print("  example mode: baseline, fedavg")
        sys.exit(1)

    client_id = sys.argv[1]
    mode = sys.argv[2]

    if len(sys.argv) >= 4:
        server_host = sys.argv[3]
    else:
        server_host = "auto"

    if server_host.lower() == "auto":
        server_host = discover_server()

    print("[{}-NET][{}] using server {}".format(client_id, mode, server_host))

    stats_list = multi_round_measure(client_id, mode, server_host)

    # CSV 保存路径：/tmp/net_results_<client_id>.csv
    csv_path = "/tmp/net_results_{}.csv".format(client_id)
    write_csv(stats_list, client_id, mode, server_host, csv_path)

    print("[{}-NET][{}] results written to {}".format(
        client_id, mode, csv_path
    ))


if __name__ == "__main__":
    main()
