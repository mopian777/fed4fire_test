#!/usr/bin/env python3
import socket
import threading
import json

HOST = "0.0.0.0"      # 监听所有网卡
PORT = 5000
NUM_CLIENTS = 2       # 客户端数量：client1 + client2

clients_values = {}
lock = threading.Lock()
all_received = threading.Event()


def handle_client(conn, addr):
    print(f"[+] Connection from {addr}")
    try:
        raw = conn.recv(4096).decode("utf-8").strip()
        data = json.loads(raw)
        cid = data["client_id"]
        value = float(data["value"])

        with lock:
            clients_values[cid] = value
            print(f"  - Received from {cid}: {value}")
            if len(clients_values) == NUM_CLIENTS:
                all_received.set()

        # 等所有客户端都发完
        all_received.wait()

        with lock:
            avg = sum(clients_values.values()) / len(clients_values)

        resp = json.dumps({"avg": avg}).encode("utf-8")
        conn.sendall(resp)
        print(f"  - Sent avg {avg} to {cid}")
    except Exception as e:
        print("Error:", e)
    finally:
        conn.close()


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(NUM_CLIENTS)
    print(f"[SERVER] Listening on {HOST}:{PORT} ...")

    try:
        while True:
            conn, addr = s.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n[SERVER] Stopping.")
    finally:
        s.close()


if __name__ == "__main__":
    main()
