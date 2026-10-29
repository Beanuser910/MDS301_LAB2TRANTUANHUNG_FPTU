"""Parse the CNN training history out of a run_part3 log so the completed
CustomCNN run does not have to be repeated when adding ResNet checkpointing.

Reads lines like:
    CNN ep 3/30 loss 1.0510/1.0772 acc 0.6209/0.6276 lr 0.001000
    CNN test acc=0.8206 time=41762.3s
and writes outputs/metrics/part3_cnn_cache.json.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

ASSETS = paths.METRICS

EP_RE = re.compile(
    r"^CNN ep (\d+)/(\d+) loss ([\d.]+)/([\d.]+) acc ([\d.]+)/([\d.]+) lr ([\d.eE+-]+)\s*$")
TEST_RE = re.compile(r"^CNN test acc=([\d.]+) time=([\d.]+)s\s*$")


def parse(log_path):
    hist = {"tl": [], "ta": [], "vl": [], "va": [], "lr": []}
    test_acc = None
    test_time = None
    n_ep = 0
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = EP_RE.match(line)
            if m:
                ep, n_ep = int(m.group(1)), int(m.group(2))
                hist["tl"].append(float(m.group(3)))
                hist["vl"].append(float(m.group(4)))
                hist["ta"].append(float(m.group(5)))
                hist["va"].append(float(m.group(6)))
                hist["lr"].append(float(m.group(7)))
                continue
            m = TEST_RE.match(line)
            if m:
                test_acc = float(m.group(1))
                test_time = float(m.group(2))
    return hist, test_acc, test_time, n_ep


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else sys.argv[0]
    hist, test_acc, test_time, n_ep = parse(log_path)
    if n_ep == 0:
        raise SystemExit("no CNN epoch lines found in " + log_path)
    os.makedirs(ASSETS, exist_ok=True)
    cache = {
        "epochs": n_ep,
        "optimizer": "Adam(lr=0.001)",
        "scheduler": "StepLR(10,0.1)",
        "train_time_s": test_time,
        "test_accuracy": test_acc,
        "final_lr": hist["lr"][-1],
        "hist": hist,
        "weights": "outputs/models/customcnn.pt",
    }
    out = paths.metric("part3_cnn_cache.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"parsed {n_ep} epochs -> {out}")
    print(f"  test_acc={test_acc} train_time_s={test_time} final_lr={hist['lr'][-1]}")
    print(f"  last: tl={hist['tl'][-1]:.4f} vl={hist['vl'][-1]:.4f} "
          f"ta={hist['ta'][-1]:.4f} va={hist['va'][-1]:.4f}")


if __name__ == "__main__":
    main()
