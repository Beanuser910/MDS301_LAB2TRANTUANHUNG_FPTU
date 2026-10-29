"""Parallel chunked downloader for CIFAR-10 (the host throttles per-connection).

Resumes the torchvision download: keeps already-downloaded prefix bytes from
data/cifar-10-python.tar.gz, then fetches the remaining ranges concurrently.
"""
import os, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
OUT = os.path.join(paths.DATA, "cifar-10-python.tar.gz")
TMP = os.path.join(paths.DATA, "cifar_parts")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
os.makedirs(TMP, exist_ok=True)

TOTAL = 170498071
N_WORKERS = 4
CHUNK = 4 * 1024 * 1024  # 4 MB per task

# Existing prefix from the (slow) torchvision download is reused
have = os.path.getsize(OUT) if os.path.exists(OUT) else 0
prefix = open(OUT, "rb").read() if have else b""
print(f"reusing {have/1e6:.1f} MB prefix from existing download", flush=True)

tasks = []
start = have
i = 0
while start < TOTAL:
    end = min(start + CHUNK, TOTAL) - 1
    part = os.path.join(TMP, f"part{i:04d}")
    tasks.append((i, start, end, part))
    start = end + 1
    i += 1

lock = threading.Lock()
done_bytes = [0]
failed = []


def worker(items):
    for idx, s, e, path in items:
        if os.path.exists(path) and os.path.getsize(path) == (e - s + 1):
            with lock:
                done_bytes[0] += e - s + 1
            continue
        for attempt in range(8):
            try:
                req = urllib.request.Request(URL, headers={"Range": f"bytes={s}-{e}"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = r.read()
                if len(data) != (e - s + 1):
                    raise IOError(f"short read {len(data)} != {e-s+1}")
                with open(path, "wb") as f:
                    f.write(data)
                with lock:
                    done_bytes[0] += len(data)
                    pct = (have + done_bytes[0]) / TOTAL * 100
                    print(f"  part {idx} ok | {have+done_bytes[0]/1e6:.1f} MB "
                          f"({pct:.1f}%)", flush=True)
                break
            except Exception as ex:
                print(f"  part {idx} attempt {attempt+1} failed: {ex}", flush=True)
                time.sleep(5 * (attempt + 1))  # backoff: server limits concurrent conns
        else:
            with lock:
                failed.append(idx)


shards = [tasks[k::N_WORKERS] for k in range(N_WORKERS)]
threads = [threading.Thread(target=worker, args=(sh,)) for sh in shards]
t0 = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
dt = time.time() - t0

if failed:
    print(f"FAILED parts: {failed}", flush=True)
    sys.exit(1)

with open(OUT, "wb") as f:
    f.write(prefix)
    for idx, s, e, path in tasks:
        f.write(open(path, "rb").read())
final = os.path.getsize(OUT)
print(f"assembled {final/1e6:.1f} MB (expected {TOTAL/1e6:.1f}) in {dt:.0f}s", flush=True)
assert final == TOTAL, "size mismatch"
print("DOWNLOAD COMPLETE", flush=True)
