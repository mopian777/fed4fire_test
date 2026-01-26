#!/usr/bin/env python3
import socket, json, time, os

PORT = 5000
ROUNDS = 200
EXPECTED_CLIENTS = 2
METHOD = "baseline"
LOG_PATH = "/tmp/global_metrics_baseline.csv"

# reward 只是为了统一字段（baseline 不做 RL）
ALPHA_TIME = 0.001
BETA_LOSS = 1.0
GAMMA_ACC = 1.0

def send_json(sock, obj):
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))

def recv_json_line(f):
    line = f.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)

def ensure_log():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            f.write("method,step,wall_time_s,step_time_ms,active_clients,global_loss,global_acc,reward,staleness\n")

def log_row(step, wall_s, step_ms, active, loss, acc, reward, staleness):
    with open(LOG_PATH, "a") as f:
        f.write(f"{METHOD},{step},{wall_s:.6f},{step_ms:.3f},\"{active}\",{loss:.10f},{acc:.6f},{reward:.10f},{staleness}\n")

def main():
    ensure_log()
    t0 = time.time()

    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("0.0.0.0", PORT))
    ls.listen(10)
    print(f"[{METHOD}] listening on 0.0.0.0:{PORT}")

    clients = []
    while len(clients) < EXPECTED_CLIENTS:
        cs, addr = ls.accept()
        f = cs.makefile("r")
        hello = recv_json_line(f)
        if not hello or hello.get("type") != "HELLO":
            try: f.close()
            except: pass
            try: cs.close()
            except: pass
            continue
        cid = hello.get("client_id", f"client{len(clients)+1}")
        clients.append({"id": cid, "sock": cs, "file": f})
        print(f"[{METHOD}] accepted {cid} from {addr[0]}:{addr[1]}")

    # 对齐：每轮让所有 client 都训练一步并上报
    for r in range(1, ROUNDS + 1):
        step_t0 = time.time()
        for c in clients:
            send_json(c["sock"], {"type": "TRAIN_LOCAL", "round": r})

        losses, accs, ns, active_ids = [], [], [], []
        for c in clients:
            msg = recv_json_line(c["file"])
            if not msg or msg.get("type") != "METRICS":
                print(f"[{METHOD}] {c['id']} disconnected.")
                return
            losses.append(float(msg.get("local_loss", 0.0)))
            accs.append(float(msg.get("local_acc", 0.0)))
            ns.append(int(msg.get("n_samples", 1)))
            active_ids.append(c["id"])

        tot = float(sum(ns))
        global_loss = sum(losses[i] * ns[i] for i in range(len(ns))) / tot
        global_acc  = sum(accs[i]   * ns[i] for i in range(len(ns))) / tot

        step_ms = (time.time() - step_t0) * 1000.0
        wall_s  = time.time() - t0

        reward = -ALPHA_TIME * step_ms - BETA_LOSS * global_loss + GAMMA_ACC * global_acc
        log_row(r, wall_s, step_ms, "|".join(active_ids), global_loss, global_acc, reward, staleness=0)

        print(f"[{METHOD}] round={r} loss={global_loss:.6f} acc={global_acc:.4f} time={step_ms:.1f}ms")

    for c in clients:
        try: send_json(c["sock"], {"type": "STOP"})
        except: pass
        try: c["file"].close()
        except: pass
        try: c["sock"].close()
        except: pass
    try: ls.close()
    except: pass
    print(f"[{METHOD}] done. log: {LOG_PATH}")

if __name__ == "__main__":
    main()
