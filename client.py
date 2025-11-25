#!/usr/bin/env python3
import socket
import json
import sys
import random

if len(sys.argv) < 3:
    print("Usage: python3 {} <client_id> <server_ip>".format(sys.argv[0]))
    sys.exit(1)

client_id = sys.argv[1]
SERVER_IP = sys.argv[2]
PORT = 5000

# 模拟本地“训练结果”
local_value = random.uniform(0.0, 10.0)
data = {"client_id": client_id, "value": local_value}
print("[{}] Local value = {}".format(client_id, local_value))

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.connect((SERVER_IP, PORT))
    s.sendall(json.dumps(data).encode("utf-8"))
    resp_raw = s.recv(4096).decode("utf-8")
    resp = json.loads(resp_raw)
finally:
    s.close()

avg = resp["avg"]
print("[{}] Received global average = {}".format(client_id, avg))
