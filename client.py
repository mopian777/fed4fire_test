#!/usr/bin/env python3
import socket
import json
import sys
import random
import subprocess

PORT = 5000


def discover_server(port=PORT, timeout=0.2):
    """
    在本地子网（同一个 /24 网段）里扫描哪个 IP 的 port=5000 能连通，
    第一个连通的就认为是 server。
    """
    # 取本机 IP（hostname -I 的第一个）
    out = subprocess.check_output(["hostname", "-I"]).decode().split()
    if not out:
        raise RuntimeError("Cannot get local IP from 'hostname -I'")
    my_ip = out[0]
    prefix, last = my_ip.rsplit(".", 1)
    print("[discover] my_ip = {}, prefix = {}".format(my_ip, prefix))

    for i in range(1, 255):
        ip = "{}.{}".format(prefix, i)
        if ip == my_ip:
            continue
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect((ip, port))
            s.close()
            print("[discover] found server at {}".format(ip))
            return ip
        except OSError:
            continue

    raise RuntimeError("Server not found on {}.0/24".format(prefix))


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

    local_value = random.uniform(0.0, 10.0)
    data = {"client_id": client_id, "value": local_value}
    print("[{}] Local value = {}".format(client_id, local_value))

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((server_host, PORT))
        s.sendall(json.dumps(data).encode("utf-8"))
        resp_raw = s.recv(4096).decode("utf-8")
        resp = json.loads(resp_raw)
    finally:
        s.close()

    avg = resp["avg"]
    print("[{}] Received global average = {} (server={})".format(client_id, avg, server_host))


if __name__ == "__main__":
    main()
