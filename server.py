#!/usr/bin/env python3
import socket
import threading
import json

HOST = "0.0.0.0"
PORT = 5000
NUM_CLIENTS = 2   # client1 + client2

clients_weights = {}
lock = threading.Lock()
all_received = threading.Event()


def handle_client(conn, addr):
    print("[+] Connection from {}".format(addr))
    try:
        raw = conn.recv(4096).decode("utf-8").strip()
        if not raw:
            # 探测 / 空连接（比如 client 在扫描端口时产生），直接忽略
            print("  - Empty payload from {}, ignoring".format(addr))
            return

        try:
            data = json.loads(raw)
        except ValueError as e:
            print("Error parsing JSON from {}: {}".format(addr, e))
            return

        cid = data["client_id"]
        weights = data["weights"]   # 形如 [w, b]
        print("  - Received from {}: {}".format(cid, weights))

        with lock:
            clients_weights[cid] = weights
            if len(clients_weights) == NUM_CLIENTS:
                all_received.set()

        # 等所有客户端都发完
        all_received.wait()

        # FedAvg：对每个参数维度做平均
        with lock:
            num = float(len(clients_weights))
            dim = len(weights)
            avg = [0.0] * dim
            for wlist in clients_weights.values():
                for i in range(dim):
                    avg[i] += float(wlist[i])
            for i in range(dim):
                avg[i] /= num

        resp = json.dumps({"avg_weights": avg}).encode("utf-8")
        conn.sendall(resp)
        print("  - Sent avg {} to {}".format(avg, cid))

    except Exception as e:
        print("Error: {}".format(e))
    finally:
        conn.close()


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(NUM_CLIENTS)
    print("[SERVER] Listening on {}:{} ...".format(HOST, PORT))

    try:
        while True:
            conn, addr = s.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr))
            t.daemon = True
            t.start()
    except KeyboardInterrupt:
        print("\n[SERVER] Stopping.")
    finally:
        s.close()


if __name__ == "__main__":
    main()
