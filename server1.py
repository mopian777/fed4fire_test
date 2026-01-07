#!/usr/bin/env python3
import socket
import json

HOST = "0.0.0.0"
PORT = 5000

NUM_CLIENTS = 2      # client1 + client2
ROUNDS = 5           # 联邦训练总轮数


def run_round(sock, round_idx):
    """
    单轮联邦训练：
    - 收集 NUM_CLIENTS 个客户端上传的 weights
    - 做一次 FedAvg
    - 把全局 weights 发回每个客户端
    """
    round_weights = []
    conns = []
    received = 0

    print("[SERVER] Round {} waiting for {} client updates...".format(
        round_idx, NUM_CLIENTS
    ))

    while received < NUM_CLIENTS:
        conn, addr = sock.accept()
        print("[+] Connection from {} in round {}".format(addr, round_idx))
        raw = conn.recv(4096).decode("utf-8").strip()

        if not raw:
            # 探测 / 空连接（client 扫描端口时产生），忽略
            print("  - Empty payload from {}, ignoring".format(addr))
            conn.close()
            continue

        try:
            data = json.loads(raw)
        except ValueError as e:
            print("  - JSON error from {}: {}, ignoring".format(addr, e))
            conn.close()
            continue

        cid = data.get("client_id", "unknown")
        weights = data["weights"]  # 形如 [w, b]
        client_round = data.get("round", None)

        print("  - Received from {} (client round {}): {}".format(
            cid, client_round, weights
        ))

        round_weights.append(weights)
        conns.append(conn)
        received += 1

    # FedAvg：对每个维度做平均
    dim = len(round_weights[0])
    avg = [0.0] * dim
    num = float(len(round_weights))

    for wlist in round_weights:
        for i in range(dim):
            avg[i] += float(wlist[i])

    for i in range(dim):
        avg[i] /= num

    print("[SERVER] Round {} global weights = {}".format(round_idx, avg))

    resp_obj = {"avg_weights": avg, "round": round_idx}
    resp = json.dumps(resp_obj).encode("utf-8")

    # 把全局参数发回给本轮所有客户端，然后关闭连接
    for conn in conns:
        conn.sendall(resp)
        conn.close()


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(NUM_CLIENTS)
    print("[SERVER] listening on {}:{} ...".format(HOST, PORT))

    try:
        for r in range(1, ROUNDS + 1):
            run_round(s, r)

        print("[SERVER] All {} rounds finished. Stopping.".format(ROUNDS))
    except KeyboardInterrupt:
        print("\n[SERVER] Interrupted, stopping.")
    finally:
        s.close()


if __name__ == "__main__":
    main()
