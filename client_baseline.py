#!/usr/bin/env python3
import socket, json, sys, random, subprocess, time, os

PORT = 5000
ROUNDS = 200
LOCAL_STEPS = 50
LR = 0.1
EPS = 0.5
METHOD = "baseline"

PING_COUNT = 50
PING_LOG_DIR = "/tmp"
PING_ENABLED = True  # 不需要就改 False

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

def discover_server(port=PORT, per_ip_timeout=0.05, max_passes=5, sleep_between=2.0):
    out = subprocess.check_output(["hostname", "-I"]).decode().split()
    if not out:
        raise RuntimeError("Cannot get IP from hostname -I")
    my_ip = out[0]
    prefix, _ = my_ip.rsplit(".", 1)
    for attempt in range(1, max_passes + 1):
        for i in range(1, 255):
            ip = f"{prefix}.{i}"
            if ip == my_ip:
                continue
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(per_ip_timeout)
            try:
                s.connect((ip, port))
                s.close()
                return ip
            except OSError:
                continue
        time.sleep(sleep_between)
    raise RuntimeError("Server not found")

def make_local_data(n=100, seed=1234):
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        x = rng.random()
        y = 2.0 * x + 3.0 + rng.gauss(0.0, 0.1)
        xs.append(x)
        ys.append(y)
    return xs, ys

def eval_metrics(xs, ys, w, b, eps=EPS):
    n = float(len(xs))
    mse = 0.0
    ok = 0
    for x, y in zip(xs, ys):
        diff = (w * x + b) - y
        mse += diff * diff
        if abs(diff) <= eps:
            ok += 1
    return mse / n, ok / float(len(xs))

def train_steps(xs, ys, w, b, lr=LR, steps=LOCAL_STEPS):
    n = float(len(xs))
    w = float(w); b = float(b)
    for _ in range(steps):
        gw = 0.0; gb = 0.0
        for x, y in zip(xs, ys):
            diff = (w * x + b) - y
            gw += 2.0 * diff * x
            gb += 2.0 * diff
        gw /= n; gb /= n
        w -= lr * gw
        b -= lr * gb
    loss, acc = eval_metrics(xs, ys, w, b)
    return w, b, loss, acc

def init_ping_log(cid):
    if not os.path.exists(PING_LOG_DIR):
        try: os.makedirs(PING_LOG_DIR)
        except: pass
    path = os.path.join(PING_LOG_DIR, f"{METHOD}_curve_{cid}.csv")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("mode,client_id,server_host,round,sent,received,loss,rtt_min_ms,rtt_avg_ms,rtt_max_ms,delay_avg_ms\n")
    return path

def ping_measure(host, count=PING_COUNT):
    cmd = ["ping", "-c", str(count), "-i", "0.2", "-q", host]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True, timeout=20)
    except subprocess.CalledProcessError as e:
        out = e.output
    except Exception:
        return 0,0,1.0,0.0,0.0,0.0,0.0

    sent=received=None; loss=None; rmin=ravg=rmax=None
    for line in out.splitlines():
        line=line.strip()
        if "packets transmitted" in line:
            parts=line.split(",")
            try:
                sent=int(parts[0].split()[0])
                received=int(parts[1].split()[0])
                loss=float(parts[2].strip().split()[0].strip("%"))/100.0
            except: pass
        if "rtt min/avg/max" in line and "=" in line:
            try:
                stats=line.split("=")[1].strip().split()[0].split("/")
                rmin=float(stats[0]); ravg=float(stats[1]); rmax=float(stats[2])
            except: pass
    if sent is None or received is None or loss is None or ravg is None:
        return 0,0,1.0,0.0,0.0,0.0,0.0
    return sent,received,loss,rmin,ravg,rmax,ravg/2.0

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 client_baseline.py <client_id> [server_ip|auto]")
        sys.exit(1)
    cid = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) >= 3 else "auto"
    if host.lower() == "auto":
        host = discover_server()

    xs, ys = make_local_data(n=100, seed=1234 if cid=="client1" else 5678)
    w = random.uniform(-1.0, 1.0)
    b = random.uniform(-1.0, 1.0)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, PORT))
    f = s.makefile("r")
    send_json(s, {"type": "HELLO", "client_id": cid})
    ping_path = init_ping_log(cid)

    while True:
        msg = recv_json_line(f)
        if not msg:
            break
        if msg.get("type") == "TRAIN_LOCAL":
            r = int(msg.get("round", 0))
            w, b, loss, acc = train_steps(xs, ys, w, b)
            send_json(s, {"type":"METRICS","round":r,"local_loss":loss,"local_acc":acc,"n_samples":len(xs)})

            if PING_ENABLED:
                sent,rec,pl,rmin,ravg,rmax,delay = ping_measure(host)
                with open(ping_path,"a") as fp:
                    fp.write(f"{METHOD},{cid},{host},{r},{sent},{rec},{pl:.6f},{rmin:.6f},{ravg:.6f},{rmax:.6f},{delay:.6f}\n")

        elif msg.get("type") == "STOP":
            break

    try: f.close()
    except: pass
    try: s.close()
    except: pass

if __name__ == "__main__":
    main()
