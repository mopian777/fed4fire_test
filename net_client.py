#!/usr/bin/env python3
import socket
import json
import sys
import time
import subprocess

PORT = 6000  # 必须与 net_server.py 一致


def discover_server(port=PORT,
                    per_ip_timeout=0.03,
                    max_passes=5,
                    sleep_between=1.0):
    """
    与你 FL 用的 discover_server 类似，在 /24 网段里找在哪个 IP 上 UDP 6000 能响应。
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
                # 发一个探测包，看能不能收到 echo
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


def measure(client_id, server_host, num_packets=100, interval=0.1, timeout=1.0):
    """
    发送 num_packets 个 UDP 探测包，统计 RTT 与丢包。
    interval: 两个包之间的发送间隔（秒）
    timeout: 接收 echo 的超时时间（秒）
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
                # 序号不一致当作异常
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

    # 近似单向 delay
    delay_avg = rtt_avg / 2.0

    print("[{}-NET] sent = {}, received = {}, loss = {:.2%}".format(
        client_id, sent, received, loss
    ))
    print("[{}-NET] RTT min/avg/max = {:.3f}/{:.3f}/{:.3f} ms".format(
        client_id, rtt_min, rtt_avg, rtt_max
    ))
    print("[{}-NET] approx one-way delay(avg) = {:.3f} ms".format(
        client_id, delay_avg
    ))

    return {
        "sent": sent,
        "received": received,
        "loss": loss,
        "rtt_min": rtt_min,
        "rtt_avg": rtt_avg,
        "rtt_max": rtt_max,
        "delay_avg": delay_avg,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 {} <client_id> [server_ip_or_auto]".format(sys.argv[0]))
        sys.exit(1)

    client_id = sys.argv[1]

    if len(sys.argv) >= 3:
        server_host = sys.argv[2]
    else:
        server_host = "auto"

    if server_host.lower() == "auto":
        server_host = discover_server()

    print("[{}-NET] using server {}".format(client_id, server_host))
    measure(client_id, server_host, num_packets=100, interval=0.1, timeout=1.0)


if __name__ == "__main__":
    main()
