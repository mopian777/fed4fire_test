#!/usr/bin/env python3
import socket
import json

HOST = "0.0.0.0"
PORT = 5000

# FedAsync 风格的“步长”（每个客户端更新对全局的影响比例）
ASYNC_LR = 0.3   # 0<L<1，越大表示单个客户端更新权重越大


def async_update(global_w, client_w, lr):
    """FedAsync 式异步更新：global = (1-lr)*global + lr*client"""
    if global_w is None:
        # 第一次就用客户端的参数初始化
        return [float(x) for x in client_w]

    new_w = []
    for g, c in zip(global_w, client_w):
        new_w.append((1.0 - lr) * float(g) + lr * float(c))
    return new_w


def main():
    # 初始全局模型参数（线性回归：w, b）
    global_weights = [0.0, 0.0]
    global_step = 0

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(8)   # 允许多个客户端并发连接队列
    print("[SERVER-ASYNC] Listening on {}:{} ...".format(HOST, PORT))

    try:
        while True:
            conn, addr = s.accept()
            try:
                raw = conn.recv(4096).decode("utf-8").strip()
                if not raw:
                    conn.close()
                    continue

                try:
                    data = json.loads(raw)
                except ValueError as e:
                    print("[SERVER-ASYNC] JSON error from {}: {}".format(addr, e))
                    conn.close()
                    continue

                cid = data.get("client_id", "unknown")
                client_round = data.get("client_step", None)
                client_weights = data.get("weights", None)

                if not isinstance(client_weights, list):
                    print("[SERVER-ASYNC] Invalid weights from {}, ignore".format(cid))
                    conn.close()
                    continue

                print("[SERVER-ASYNC] recv from {} (client_step={}): {}".format(
                    cid, client_round, client_weights
                ))

                # 一收到就异步更新全局模型
                global_weights = async_update(global_weights, client_weights, ASYNC_LR)
                global_step += 1

                print("  -> global_step={}, global_weights={}".format(
                    global_step, global_weights
                ))

                # 把最新全局参数发回客户端
                resp_obj = {
                    "global_weights": global_weights,
                    "server_step": global_step,
                }
                resp = json.dumps(resp_obj).encode("utf-8")
                conn.sendall(resp)
            finally:
                conn.close()

    except KeyboardInterrupt:
        print("\n[SERVER-ASYNC] Interrupted, stopping.")
    finally:
        s.close()


if __name__ == "__main__":
    main()
