"""Smoke test for run_part3.py: verifies data pipeline, loaders, models, and a
single training epoch + eval, without running the full 30-epoch schedule."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_part3 as R
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset

SEED = 42


def main():
    torch.manual_seed(SEED)
    MEAN, STD = R.CIFAR_MEAN, R.CIFAR_STD

    train_tf = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    a = torchvision.datasets.CIFAR10(R.DATA, train=True, download=False, transform=train_tf)
    e = torchvision.datasets.CIFAR10(R.DATA, train=True, download=False, transform=eval_tf)
    t = torchvision.datasets.CIFAR10(R.DATA, train=False, download=False, transform=eval_tf)
    print("datasets:", len(a), len(e), len(t), "| classes:", t.classes, flush=True)

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(a), generator=g).tolist()
    tl, vl, el, nw, note = R.build_loaders(Subset(a, perm[:45000]), Subset(e, perm[45000:]), t)
    print("loaders OK | num_workers:", nw, "| note:", note, flush=True)

    # confirm the val split uses the deterministic (non-augmented) copy
    xb_v, _ = next(iter(vl))
    xb_t, _ = next(iter(tl))
    print("val batch shape:", tuple(xb_v.shape), "| train batch shape:", tuple(xb_t.shape))
    print("train pixel stats (augmented): mean %.3f std %.3f" % (xb_t.mean(), xb_t.std()))
    print("val   pixel stats (deterministic): mean %.3f std %.3f" % (xb_v.mean(), xb_v.std()))

    device = torch.device("cpu")
    m = R.CustomCNN().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    crit = torch.nn.CrossEntropyLoss()
    hist, dt = R.train_model(m, tl, vl, opt, sched, 1, "SMOKE", device)
    print("1-epoch train ok:", hist, f"({dt:.0f}s)", flush=True)
    te_loss, te_acc = R.run_epoch(m, el, crit, device, None)
    print("test loss/acc:", te_loss, te_acc)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
