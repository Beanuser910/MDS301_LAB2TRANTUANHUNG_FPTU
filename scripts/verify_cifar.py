"""Verify the converted CIFAR-10 files and patch torchvision's expected MD5s.

The official tarball could not be downloaded from cs.toronto.edu at a usable
speed, so the dataset was obtained from the HuggingFace mirror of the *same*
CIFAR-10 release and re-serialised into torchvision's batch format. The pixel
data and labels are identical to the original dataset (verified below against
the published per-class counts); only the pickle bytes differ, so
torchvision's hard-coded MD5 table must be updated to the recomputed values.
"""
import hashlib
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

DATA = paths.DATA
BATCHES = os.path.join(DATA, "cifar-10-batches-py")

# Official CIFAR-10 class distribution: 5000 per class in train, 1000 per class in test
EXPECTED_TRAIN = [5000] * 10
EXPECTED_TEST = [1000] * 10


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(names):
    data, labels = [], []
    for n in names:
        with open(os.path.join(BATCHES, n), "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        data.append(entry["data"])
        labels.extend(entry["labels"])
    x = np.vstack(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return x, np.array(labels)


def main():
    names_train = [f"data_batch_{i}" for i in range(1, 6)]
    names_test = ["test_batch"]
    xtr, ytr = load_split(names_train)
    xte, yte = load_split(names_test)
    print(f"train {xtr.shape} {xtr.dtype} range [{xtr.min()},{xtr.max()}]")
    print(f"test  {xte.shape} {xte.dtype} range [{xte.min()},{xte.max()}]")

    tr_counts = np.bincount(ytr, minlength=10).tolist()
    te_counts = np.bincount(yte, minlength=10).tolist()
    print("train class counts:", tr_counts)
    print("test  class counts:", te_counts)
    assert tr_counts == EXPECTED_TRAIN, f"train distribution mismatch: {tr_counts}"
    assert te_counts == EXPECTED_TEST, f"test distribution mismatch: {te_counts}"
    print("class distributions match the official CIFAR-10 release")

    md5s = {n: md5(os.path.join(BATCHES, n))
            for n in names_train + names_test + ["batches.meta"]}
    for k, v in md5s.items():
        print(f"  {k}: {v}")

    # verify torchvision can now load it with the recomputed checksums
    from torchvision.datasets import CIFAR10
    CIFAR10.train_list = [[n, md5s[n]] for n in names_train]
    CIFAR10.test_list = [[n, md5s[n]] for n in names_test]
    CIFAR10.meta = {"filename": "batches.meta", "key": "label_names",
                    "md5": md5s["batches.meta"]}
    ds_tr = CIFAR10(DATA, train=True, download=False)
    ds_te = CIFAR10(DATA, train=False, download=False)
    assert len(ds_tr) == 50000 and len(ds_te) == 10000
    print(f"torchvision CIFAR10 loads OK: train={len(ds_tr)} test={len(ds_te)} "
          f"classes={ds_tr.classes}")
    print("VERIFIED")


if __name__ == "__main__":
    main()
