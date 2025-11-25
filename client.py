#!/usr/bin/env python3
import socket
import json
import sys
import random

# TODO: 换成你 server 节点的 IP（就是你刚才 ping 的那个）
SERVER_IP = "10.11.16.103"
PORT = 5000

if len(sys.argv) != 2:
    print(f"Usage: python3 {sys.argv[0]} <client_id>")
    sys.exit(1)

client_id = sys.argv[1]

# 模拟“本地训练结果”（这里就随便一个随机数）
local_value = random.uniform(0, 10)

data = {"client_id": client_id, "value": local_value}
print(f"[{client_id}] Local value = {local_value}")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((SERVER_IP, PORT))
    s.sendall(json.dumps(data).encode("utf-8"))

    resp_raw = s.recv(4096).decode("utf-8")
    resp = json.loads(resp_raw)
    avg = resp["avg"]

print(f"[{client_id}] Received global average = {avg}")
