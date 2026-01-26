#!/usr/bin/env python3
import socket, json, time, os, math

PORT = 5000
TOTAL_UPDATES = 400   # 总共接受多少次 client 更新（两 client 各 200 次 => 400）
METHOD = "fedasync"
LOG_PATH = "/tmp/global_metrics_fedasync.csv"

# 异步混合：alpha(staleness)=base_alpha / (1+staleness)
BASE_ALPHA = 0.6

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

    w, b = 0.0, 0.0
    version = 0

    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("0.0.0.0", PORT))
    ls.listen(10)
    print(f"[{METHOD}] listening on 0.0.0.0:{PORT}")

    clients = []
    # 接两客户端即可（更多也行）
    while len(clients) < 2:
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
        # 发 INIT
        send_json(cs, {"type":"INIT","w":w,"b":b,"version":version})
        print(f"[{METHOD}] accepted {cid} from {addr[0]}:{addr[1]}")

    last_update_t = time.time()
    step = 0

    # 主循环：谁先发 UPDATE 就先处理
    while step < TOTAL_UPDATES:
        # 轮询式读取（最小实现）：逐个尝试读一行，哪个先有就处理哪个
        handled = False
        for c in clients:
            c["sock"].settimeout(0.05)
            try:
                msg = recv_json_line(c["file"])
            except Exception:
                msg = None

            if not msg:
                continue
            if msg.get("type") != "UPDATE":
                continue

            step_t0 = time.time()
            step += 1
            handled = True

            cid = c["id"]
            base_ver = int(msg.get("base_version", 0))
            staleness = max(0, version - base_ver)

            cw = float(msg.get("w", w))
            cb = float(msg.get("b", b))

            # 异步聚合：w <- (1-alpha)*w + alpha*cw
            alpha = BASE_ALPHA / (1.0 + float(staleness))
            w = (1.0 - alpha) * w + alpha * cw
            b = (1.0 - alpha) * b + alpha * cb
            version += 1

            # 回传最新全局模型让 client 评估
            send_json(c["sock"], {"type":"GLOBAL","step":step,"w":w,"b":b,"version":version})

            # 期待 client 回 EVAL（包含 global_loss/global_acc）
            evalmsg = recv_json_line(c["file"])
            if not evalmsg or evalmsg.get("type") != "EVAL":
                # 不阻塞：如果缺 eval，就用占位
                global_loss = float("nan")
                global_acc = float("nan")
            else:
                global_loss = float(evalmsg.get("global_loss", 0.0))
                global_acc = float(evalmsg.get("global_acc", 0.0))

            now = time.time()
            step_time_ms = (now - last_update_t) * 1000.0
            last_update_t = now
            wall_s = now - t0

            # reward（示例）
            if math.isnan(global_loss) or math.isnan(global_acc):
                reward = -ALPHA_TIME * step_time_ms
            else:
                reward = -ALPHA_TIME * step_time_ms - BETA_LOSS * global_loss + GAMMA_ACC * global_acc

            log_row(step, wall_s, step_time_ms, cid, global_loss, global_acc, reward, staleness)

            print(f"[{METHOD}] step={step} from={cid} ver={version} stale={staleness} alpha={alpha:.3f} "
                  f"loss={global_loss:.6f} acc={global_acc:.4f} step_time={step_time_ms:.1f}ms")

            break  # 本轮只处理一个消息，回到 while

        if not handled:
            time.sleep(0.01)

    # stop all
    for c in clients:
        try: send_json(c["sock"], {"type":"STOP"})
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
