#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RL-FedAvg client with simple server auto-discovery.

Usage:
  python3 client_rl_fedavg.py client1 10.11.16.103
  python3 client_rl_fedavg.py client2 auto

- If second arg is 'auto', client will:
    1) detect its own IP, e.g. 10.11.16.101
    2) derive prefix 10.11.16
    3) scan 10.11.16.1-254: try TCP connect(port=PORT) with small timeout
    4) first IP that accepts connection is treated as server_ip

- Then it connects again as real FL client, sends JSON 'hello',
  and participates in RL-FedAvg training with server_rl_bandit_fedavg.py.
"""

import json
import socket
import struct
import sys
import random
import time

PORT = 38001  # 和 server 保持一致




# ------------- JSON over TCP (length-prefixed) ---------------- #

def send_json(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while reading")
        buf += chunk
    return buf


def recv_json(sock):
    header = recv_exact(sock, 4)
    length, = struct.unpack("!I", header)
    data = recv_exact(sock, length)
    return json.loads(data.decode("utf-8"))


# ---------------- IP / server discovery ---------------- #

def get_local_ip():
    """
    Try to get local IP address by opening a dummy UDP socket.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def discover_server(port, max_passes=5, timeout=0.15):
    """
    Scan my /24 subnet: 10.11.16.1-254, try TCP connect to given port.
    First IP that accepts connection is treated as server_ip.

    Returns: server_ip (string)
    Raises: RuntimeError if not found after max_passes.
    """
    my_ip = get_local_ip()
    parts = my_ip.split(".")
    if len(parts) != 4:
        raise RuntimeError("Unexpected local IP: {}".format(my_ip))
    prefix = ".".join(parts[:3])

    print("[discover] my_ip = {}, prefix = {}.0/24".format(my_ip, prefix))

    for p in range(max_passes):
        print("[discover] pass {} ...".format(p + 1))
        for host in range(1, 255):
            addr = "%s.%d" % (prefix, host)
            if addr == my_ip:
                continue
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            try:
                # non-blocking connect test
                err = s.connect_ex((addr, port))
                s.close()
                if err == 0:
                    print("[discover] found server at {}".format(addr))
                    return addr
            except Exception:
                s.close()
                continue
    raise RuntimeError("Server not found on {}.0/24 after {} passes".format(prefix, max_passes))


# ---------------- Local data & training (no extra libs) ---------------- #

def make_local_dataset(n_samples, noise_std, seed):
    """
    Synthetic 1D regression: y = 2x + 3 + noise
    Returns (xs, ys) as two Python lists of floats.
    """
    rng = random.Random(seed)
    xs = []
    ys = []
    for _ in range(n_samples):
        x = rng.random() * 10.0      # [0,10]
        noise = rng.gauss(0.0, noise_std)
        y = 2.0 * x + 3.0 + noise
        xs.append(x)
        ys.append(y)
    return xs, ys


def mse(xs, ys, w, b):
    n = float(len(xs))
    if n == 0:
        return 0.0
    loss = 0.0
    for x, y in zip(xs, ys):
        diff = w * x + b - y
        loss += diff * diff
    return loss / n


def local_train(xs, ys, w_init, b_init, lr, local_epochs):
    """
    Full-batch gradient descent on local data for local_epochs steps.
    Returns (new_w, new_b, final_local_loss).
    """
    w = float(w_init)
    b = float(b_init)
    n = float(len(xs))
    if n == 0:
        return w, b, 0.0

    for _ in range(local_epochs):
        grad_w = 0.0
        grad_b = 0.0

        for x, y in zip(xs, ys):
            y_hat = w * x + b
            diff = y_hat - y
            grad_w += 2.0 * diff * x
            grad_b += 2.0 * diff

        grad_w /= n
        grad_b /= n

        w -= lr * grad_w
        b -= lr * grad_b

    final_loss = mse(xs, ys, w, b)
    return w, b, final_loss


# ---------------- Main client logic ---------------- #

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 {} <client_id> <server_ip|auto> [n_samples] [noise_std]"
              .format(sys.argv[0]))
        sys.exit(1)

    client_id = sys.argv[1]
    server_arg = sys.argv[2]

    if len(sys.argv) >= 4:
        n_samples = int(sys.argv[3])
    else:
        n_samples = 64
    if len(sys.argv) >= 5:
        noise_std = float(sys.argv[4])
    else:
        noise_std = 0.5

    # Different seed per client, to simulate slightly different data
    cid_lower = client_id.lower()
    if cid_lower == "client1":
        seed = 1
    elif cid_lower == "client2":
        seed = 2
    else:
        seed = 1234

    xs, ys = make_local_dataset(n_samples, noise_std, seed)

    # auto-discover server if needed
    if server_arg.lower() == "auto":
        try:
            server_ip = discover_server(PORT)
        except RuntimeError as e:
            print("[CLIENT {}] {}".format(client_id, e))
            sys.exit(1)
    else:
        server_ip = server_arg

    print("[CLIENT {}] connect to {}:{} ...".format(client_id, server_ip, PORT))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_ip, PORT))

    # send hello
    hello = {"type": "hello", "client_id": client_id}
    send_json(sock, hello)
    print("[CLIENT {}] sent hello".format(client_id))

    try:
        while True:
            msg = recv_json(sock)
            mtype = msg.get("type")

            if mtype == "train":
                round_id = msg.get("round")
                weights = msg.get("weights", {})
                local_epochs = int(msg.get("local_epochs", 5))
                lr = float(msg.get("lr", 0.05))

                w = float(weights.get("w", 0.0))
                b = float(weights.get("b", 0.0))

                print("[CLIENT {}] TRAIN round={}, w={:.4f}, b={:.4f}, "
                      "local_epochs={}, lr={:.4f}".format(
                          client_id, round_id, w, b, local_epochs, lr))

                # local training
                new_w, new_b, local_loss = local_train(xs, ys, w, b, lr, local_epochs)

                print("[CLIENT {}] UPDATE round={} -> new_w={:.4f}, new_b={:.4f}, "
                      "local_loss={:.6f}".format(
                          client_id, round_id, new_w, new_b, local_loss))

                reply = {
                    "type": "update",
                    "round": round_id,
                    "client_id": client_id,
                    "weights": {"w": new_w, "b": new_b},
                    "local_loss": local_loss,
                }
                send_json(sock, reply)

            elif mtype == "stop":
                print("[CLIENT {}] received stop, exiting.".format(client_id))
                break

            else:
                print("[CLIENT {}] unknown message type: {}".format(client_id, mtype))
                # ignore and continue
                continue

    except ConnectionError as e:
        print("[CLIENT {}] connection error: {}".format(client_id, e))
    except KeyboardInterrupt:
        print("[CLIENT {}] interrupted by user".format(client_id))
    finally:
        sock.close()
        print("[CLIENT {}] socket closed.".format(client_id))


if __name__ == "__main__":
    main()
