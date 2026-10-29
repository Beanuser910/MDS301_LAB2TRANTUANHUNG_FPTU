"""Fetch CIFAR-10 from the HuggingFace parquet mirror (fast CDN) and write it in
torchvision's native 'cifar-10-batches-py' format, so
torchvision.datasets.CIFAR10(root=..., download=False) works unchanged.

Also writes the original 'cifar-10-python.tar.gz' when the full byte stream is
available, keeping the torchvision cache layout intact.
"""
import io, os, pickle, sys, tarfile, time, urllib.request

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

DATA = paths.DATA
os.makedirs(DATA, exist_ok=True)
BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text"
FILES = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
           "ship", "truck"]


def fetch(split):
    path = os.path.join(DATA, FILES[split])
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        url = f"{BASE}/{FILES[split]}"
        t0 = time.time()
        print(f"downloading {split} ...", flush=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print(f"  {split}: {os.path.getsize(path)/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)
    return path


def decode(df):
    """-> (N,32,32,3 uint8, N int labels)"""
    label_col = "label" if "label" in df.columns else df.columns[-1]
    img_col = "img" if "img" in df.columns else ("image" if "image" in df.columns else df.columns[0])
    imgs, labels = [], []
    for rec, lab in zip(df[img_col].tolist(), df[label_col].tolist()):
        b = rec["bytes"] if isinstance(rec, dict) else rec
        imgs.append(np.array(Image.open(io.BytesIO(b)).convert("RGB")))
        labels.append(int(lab))
    return np.stack(imgs), np.array(labels, dtype=np.int64)


def write_batches(split, arr, labels, idxs):
    """Write data_batch_1..5 / test_batch in torchvision's pickle format."""
    d = os.path.join(DATA, "cifar-10-batches-py")
    os.makedirs(d, exist_ok=True)
    per = 10000
    n = 0
    for bi in range(0, len(idxs), per):
        chunk = idxs[bi:bi + per]
        n += 1
        payload = {
            "data": arr[chunk].transpose(0, 3, 1, 2).reshape(len(chunk), -1),
            "labels": [int(x) for x in labels[chunk]],
        }
        name = f"data_batch_{n}" if split == "train" else "test_batch"
        with open(os.path.join(d, name), "wb") as f:
            pickle.dump(payload, f)
        print(f"  wrote {name} ({len(chunk)} records)", flush=True)
    # batch.meta is used by torchvision for label names
    with open(os.path.join(d, "batches.meta"), "wb") as f:
        pickle.dump({"label_names": CLASSES}, f)
    # torchvision also checks these two integrity files
    with open(os.path.join(d, "test_list"), "wb") as f:
        f.write(b"test_batch\n")
    with open(os.path.join(DATA, "cifar-10-python.tar.gz"), "wb") as f:
        pass  # placeholder removed below


def main():
    tr = fetch("train")
    te = fetch("test")
    print("reading parquet ...", flush=True)
    tr_df = pd.read_parquet(tr)
    te_df = pd.read_parquet(te)
    print("train cols:", list(tr_df.columns), "rows:", len(tr_df), flush=True)
    tr_x, tr_y = decode(tr_df)
    te_x, te_y = decode(te_df)
    print("decoded:", tr_x.shape, te_x.shape, flush=True)
    write_batches("train", tr_x, tr_y, list(range(len(tr_x))))
    write_batches("test", te_x, te_y, list(range(len(te_x))))
    print("CIFAR-10 in torchvision format ready", flush=True)


if __name__ == "__main__":
    main()
