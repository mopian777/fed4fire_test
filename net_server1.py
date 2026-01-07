#!/usr/bin/env python3
import socket
import json

HOST = "0.0.0.0"
PORT = 6000   # 独立于 FL 的端口

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    print("[NET_SERVER] Listening UDP on {}:{} ...".format(HOST, PORT))

    try:
        while True:
            data, addr = sock.recvfrom(4096)
            try:
                msg = json.loads(data.decode("utf-8"))
            except ValueError:
                # 不是 JSON 就忽略
                continue

            seq = msg.get("seq")
            client_id = msg.get("client_id", "unknown")
            # 简单 echo 回去
            sock.sendto(data, addr)
            print("[NET_SERVER] Echo seq {} from {} at {}".format(
                seq, client_id, addr
            ))
    except KeyboardInterrupt:
        print("\n[NET_SERVER] Stopping.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
