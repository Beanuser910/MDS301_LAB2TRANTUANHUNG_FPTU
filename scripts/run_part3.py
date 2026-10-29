"""Part 3 — PyTorch: fundamentals, CustomCNN on CIFAR-10, ResNet-20. Saves metrics + figures.

Windows note: DataLoader with num_workers>0 uses spawn-based multiprocessing,
which re-imports this module in each worker — hence all execution lives in
main() under an `if __name__ == "__main__"` guard.
"""
import json, os, random, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings("ignore")
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

ASSETS = paths.OUTPUTS
DATA = paths.DATA
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


# ================= model definitions (module level: picklable for workers) =================
class CustomCNN(nn.Module):
    """Block1: Conv(3->32)x2+BN/ReLU+MaxPool; Block2: Conv(32->64)x2+BN/ReLU+MaxPool;
    Block3: Conv(64->128)+BN/ReLU+AdaptiveAvgPool(1); Head: Flatten->Dropout(0.5)->
    Linear(128->256)->ReLU->Linear(256->10)."""
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        return self.classifier(self.block3(self.block2(self.block1(x))))


class ResidualBlock(nn.Module):
    """Conv(C->C,3,pad1)->BN->ReLU->Conv(C->C,3,pad1)->BN + skip -> ReLU.
    Identity skip for same-shape; 1x1 projection shortcut when dims differ."""
    def __init__(self, C, stride=1, in_channels=None):
        super().__init__()
        in_channels = C if in_channels is None else in_channels
        self.conv1 = nn.Conv2d(in_channels, C, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(C)
        if stride == 1 and in_channels == C:
            self.shortcut = nn.Sequential()  # identity skip
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, C, 1, stride=stride, bias=False),
                nn.BatchNorm2d(C))

    def forward(self, x):
        return torch.relu(self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class BottleneckBlock(nn.Module):
    """Conv(C->C//4,1)->BN->ReLU->Conv(3,pad1)->BN->ReLU->Conv(C//4->C,1)->BN + skip -> ReLU."""
    def __init__(self, C, stride=1):
        super().__init__()
        W = C // 4
        self.conv1 = nn.Conv2d(C, W, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(W)
        self.conv2 = nn.Conv2d(W, W, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(W)
        self.conv3 = nn.Conv2d(W, C, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(C)
        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(C, C, 1, stride=stride, bias=False), nn.BatchNorm2d(C))

    def forward(self, x):
        o = torch.relu(self.bn1(self.conv1(x)))
        o = torch.relu(self.bn2(self.conv2(o)))
        o = self.bn3(self.conv3(o))
        return torch.relu(o + self.shortcut(x))


class MiniResNet(nn.Module):
    """ResNet-20: init conv + 3 groups x 3 ResidualBlocks (16/32/64 filters,
    /2 downsample via projection shortcuts) + global pool + fc = 1+18+1 layers."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.g1 = nn.Sequential(*[ResidualBlock(16) for _ in range(3)])
        self.g2 = nn.Sequential(*([ResidualBlock(32, stride=2, in_channels=16)] +
                                   [ResidualBlock(32) for _ in range(2)]))
        self.g3 = nn.Sequential(*([ResidualBlock(64, stride=2, in_channels=32)] +
                                   [ResidualBlock(64) for _ in range(2)]))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        return self.fc(self.pool(x).flatten(1))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, tot_correct, tot_n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * len(xb)
            tot_correct += (logits.argmax(1) == yb).sum().item()
            tot_n += len(xb)
    return tot_loss / tot_n, tot_correct / tot_n


def train_model(model, train_loader, val_loader, opt, sched, epochs, tag, device,
                ckpt_path=None, start_epoch=0, hist=None):
    """Train with per-epoch checkpointing so a long run survives sleep/hibernate.

    ckpt_path: if given, saves {model, optimizer, scheduler, hist, epoch} after
    every epoch; on resume, load and continue from the saved epoch.
    """
    crit = nn.CrossEntropyLoss()
    if hist is None:
        hist = {"tl": [], "ta": [], "vl": [], "va": [], "lr": []}
    t0 = time.time()
    for ep in range(start_epoch, epochs):
        tl, ta = run_epoch(model, train_loader, crit, device, opt)
        vl, va = run_epoch(model, val_loader, crit, device, None)
        sched.step()
        lr = opt.param_groups[0]["lr"]
        hist["tl"].append(tl)
        hist["ta"].append(ta)
        hist["vl"].append(vl)
        hist["va"].append(va)
        hist["lr"].append(lr)
        print(f"{tag} ep {ep + 1}/{epochs} loss {tl:.4f}/{vl:.4f} "
              f"acc {ta:.4f}/{va:.4f} lr {lr:.6f}", flush=True)
        if ckpt_path:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "hist": hist, "epoch": ep + 1},
                       ckpt_path)
    return hist, time.time() - t0


def resume_ckpt(ckpt_path, model, opt, sched, device):
    """Restore a saved run. Returns (start_epoch, hist) or (0, None) if absent."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        return 0, None
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    print(f"resumed {os.path.basename(ckpt_path)} at epoch {ck['epoch']}", flush=True)
    return ck["epoch"], ck["hist"]


def build_loaders(train_set, val_set, test_set, batch_size=64):
    """num_workers=2 as required; worker-spawn failures surface at first-batch
    probe time (not construction), so probe before accepting. Documented
    fallback to num_workers=0 if the runtime rejects worker processes."""
    for nw in (2, 0):
        try:
            tl = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw)
            xb, _ = next(iter(tl))  # real probe: spawns workers
            assert tuple(xb.shape) == (batch_size, 3, 32, 32), f"bad batch {tuple(xb.shape)}"
            vl = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=nw)
            el = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=nw)
            note = ""
            if nw != 2:
                note = ("num_workers=2 unsupported in this runtime (worker probe failed); "
                        "documented compatibility fallback to num_workers=0")
                print("DataLoader fallback:", note, flush=True)
            else:
                print("DataLoaders: num_workers=2 (as required)", flush=True)
            return tl, vl, el, nw, note
        except Exception as e:
            print(f"num_workers={nw} probe failed ({type(e).__name__}: {e}); retrying...", flush=True)
    raise RuntimeError("DataLoader failed even with num_workers=0")


# ================= execution =================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    out = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out["device"] = str(device)
    out["torch_version"] = torch.__version__
    out["torchvision_version"] = torchvision.__version__
    out["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        out["gpu_name"] = torch.cuda.get_device_name(0)
    print("device:", device, "| torch", torch.__version__, flush=True)

    # ---- 3.1A: tensor normalization ----
    img = torch.randn(3, 224, 224)  # float32 RGB image
    img_n = (img - img.mean()) / img.std()
    out["3.1A"] = {"shape": list(img_n.shape), "dtype": str(img_n.dtype),
                   "mean": float(img_n.mean()), "std": float(img_n.std())}
    print("3.1A:", out["3.1A"], flush=True)

    # ---- 3.1B: CPU vs GPU matmul (warm-up + CUDA sync) ----
    N = 2048
    a_cpu, b_cpu = torch.randn(N, N), torch.randn(N, N)
    for _ in range(2):
        _ = a_cpu @ b_cpu  # warm-up
    t0 = time.perf_counter()
    c_cpu = a_cpu @ b_cpu
    t_cpu = time.perf_counter() - t0
    entry_b = {"cpu_time_s": round(t_cpu, 4), "shape": [N, N]}
    if torch.cuda.is_available():
        a_gpu, b_gpu = a_cpu.cuda(), b_cpu.cuda()
        for _ in range(3):
            _ = a_gpu @ b_gpu
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c_gpu = a_gpu @ b_gpu
        torch.cuda.synchronize()
        t_gpu = time.perf_counter() - t0
        entry_b.update({"gpu_time_s": round(t_gpu, 4),
                        "speedup": round(t_cpu / t_gpu, 2)})
        print(f"3.1B cpu={t_cpu:.4f}s gpu={t_gpu:.4f}s speedup={t_cpu / t_gpu:.2f}x", flush=True)
    else:
        entry_b.update({"note": "CUDA not available — GPU comparison not executed"})
        print(f"3.1B cpu={t_cpu:.4f}s (no GPU)", flush=True)
    out["3.1B"] = entry_b

    # ---- 3.1C: autograd f(x)=(x^2+3x-2)^3 at x=2; analytic 3(x^2+3x-2)^2(2x+3) = 1344 ----
    x = torch.tensor([2.0], requires_grad=True)
    f = (x**2 + 3*x - 2)**3
    f.backward()
    ana = 3 * (2**2 + 3*2 - 2)**2 * (2*2 + 3)
    out["3.1C"] = {"autograd": float(x.grad.item()), "analytic": int(ana),
                   "match": bool(abs(float(x.grad.item()) - ana) < 1e-4)}
    print("3.1C:", out["3.1C"], flush=True)

    # ---- 3.2.1 data ----
    train_tf = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                          T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_full_aug = torchvision.datasets.CIFAR10(DATA, train=True, download=False, transform=train_tf)
    train_full_eval = torchvision.datasets.CIFAR10(DATA, train=True, download=False, transform=eval_tf)
    test_set = torchvision.datasets.CIFAR10(DATA, train=False, download=False, transform=eval_tf)
    CLASSES = test_set.classes
    # reproducible 45k/5k split; val uses the EVAL-transform copy (no aug leakage)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(train_full_aug), generator=g).tolist()
    train_idx, val_idx = perm[:45000], perm[45000:]
    train_set = Subset(train_full_aug, train_idx)
    val_set = Subset(train_full_eval, val_idx)
    train_loader, val_loader, test_loader, NW, NW_NOTE = build_loaders(train_set, val_set, test_set)
    out["3.2.1"] = {"train_n": len(train_set), "val_n": len(val_set), "test_n": len(test_set),
                    "batch_size": 64, "num_workers": NW, "num_workers_note": NW_NOTE,
                    "classes": CLASSES}
    print("3.2.1:", {k: v for k, v in out["3.2.1"].items() if k != "classes"}, flush=True)

    vis_set = torchvision.datasets.CIFAR10(DATA, train=True, download=False,
                                           transform=T.Compose([T.ToTensor()]))
    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    rng = np.random.RandomState(SEED)
    for ax in axes.flat:
        i = int(rng.randint(0, len(vis_set)))
        im, lb = vis_set[i]
        ax.imshow(np.transpose(im.numpy(), (1, 2, 0)))
        ax.set_title(CLASSES[lb], fontsize=8)
        ax.axis("off")
    fig.suptitle("CIFAR-10 samples")
    fig.tight_layout()
    fig.savefig(paths.figure("cifar10_samples.png"), dpi=120)
    plt.close(fig)

    # ---- 3.2.2 CustomCNN ----
    cnn = CustomCNN().to(device)
    # single-batch smoke test before the long job: shapes, finite loss, backward, optimizer step
    xb0, yb0 = next(iter(train_loader))
    xb0, yb0 = xb0.to(device), yb0.to(device)
    cnn.train()
    opt0 = torch.optim.Adam(cnn.parameters(), lr=0.001)
    opt0.zero_grad()
    loss0 = nn.CrossEntropyLoss()(cnn(xb0), yb0)
    assert torch.isfinite(loss0), "non-finite smoke-test loss"
    loss0.backward()
    opt0.step()
    with torch.no_grad():
        dummy = cnn(torch.zeros(4, 3, 32, 32).to(device))
    assert tuple(dummy.shape) == (4, 10), f"bad dummy output {tuple(dummy.shape)}"
    cnn_params = count_params(cnn)
    out["3.2.2"] = {"params": cnn_params, "dummy_out_shape": list(dummy.shape),
                    "smoke_test_loss": round(float(loss0.item()), 4)}
    print(f"3.2.2 params={cnn_params} dummy_out={tuple(dummy.shape)} "
          f"smoke_loss={float(loss0.item()):.4f}", flush=True)

    # ---- 3.2.3 train CustomCNN: 30 epochs, Adam 1e-3, StepLR(10, 0.1) ----
    cnn_cache = paths.metric("part3_cnn_cache.json")
    cnn_ck = paths.model("ckpt_cnn.pt")
    if os.path.exists(cnn_cache) and os.path.exists(paths.model("customcnn.pt")):
        # Completed CustomCNN run reused from cache (see parse_cnn_log.py):
        # re-training 30 epochs on CPU costs hours and is not idempotent.
        with open(cnn_cache, encoding="utf-8") as f:
            cached = json.load(f)
        cnn = CustomCNN().to(device)
        cnn.load_state_dict(torch.load(paths.model("customcnn.pt"),
                                      map_location=device, weights_only=True))
        hist_cnn = cached["hist"]
        t_cnn = cached["train_time_s"]
        test_acc_cnn = cached["test_accuracy"]
        crit = nn.CrossEntropyLoss()
        test_loss_cnn, test_acc_cnn = run_epoch(cnn, test_loader, crit, device, None)
        print(f"3.2.3 reused cached CustomCNN run ({cached['epochs']} epochs, "
              f"test acc={test_acc_cnn:.4f})", flush=True)
    else:
        opt = torch.optim.Adam(cnn.parameters(), lr=0.001)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
        start_ep, hist_r = resume_ckpt(cnn_ck, cnn, opt, sched, device)
        hist_cnn, t_cnn = train_model(cnn, train_loader, val_loader, opt, sched, 30, "CNN",
                                      device, ckpt_path=cnn_ck, start_epoch=start_ep, hist=hist_r)
        crit = nn.CrossEntropyLoss()
        test_loss_cnn, test_acc_cnn = run_epoch(cnn, test_loader, crit, device, None)
        with open(cnn_cache, "w", encoding="utf-8") as f:
            json.dump({"epochs": 30, "optimizer": "Adam(lr=0.001)",
                       "scheduler": "StepLR(10,0.1)", "train_time_s": round(t_cnn, 1),
                       "test_accuracy": test_acc_cnn, "test_loss": test_loss_cnn,
                       "final_lr": hist_cnn["lr"][-1], "hist": hist_cnn}, f, indent=2)
    out["3.2.3"] = {"epochs": 30, "optimizer": "Adam(lr=0.001)", "scheduler": "StepLR(10,0.1)",
                    "train_time_s": round(t_cnn, 1), "test_accuracy": test_acc_cnn,
                    "test_loss": test_loss_cnn, "final_lr": hist_cnn["lr"][-1],
                    "hist": hist_cnn}
    print(f"CNN test acc={test_acc_cnn:.4f} time={t_cnn:.1f}s", flush=True)
    torch.save(cnn.state_dict(), paths.model("customcnn.pt"))

    ep = np.arange(1, 31)
    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, hist_cnn["tl"], label="train")
    plt.plot(ep, hist_cnn["vl"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CustomCNN — training vs validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.figure("cnn_loss.png"), dpi=120)
    plt.close()
    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, hist_cnn["ta"], label="train")
    plt.plot(ep, hist_cnn["va"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CustomCNN — training vs validation accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.figure("cnn_acc.png"), dpi=120)
    plt.close()

    # ---- 3.3.1 residual blocks: shapes + gradient flow ----
    rb_id, rb_proj = ResidualBlock(16), ResidualBlock(16, stride=2)
    bb = BottleneckBlock(64)
    checks = {"res_identity": list(rb_id(torch.randn(2, 16, 32, 32)).shape),
              "res_projection": list(rb_proj(torch.randn(2, 16, 32, 32)).shape),
              "bottleneck": list(bb(torch.randn(2, 64, 16, 16)).shape)}
    grads = {}
    for name, blk, shape in [("residual", ResidualBlock(16), (2, 16, 8, 8)),
                             ("bottleneck", BottleneckBlock(64), (2, 64, 8, 8))]:
        blk.train()
        blk(torch.randn(*shape)).sum().backward()
        gn = {n: float(p.grad.norm()) for n, p in blk.named_parameters() if p.grad is not None}
        grads[name] = gn
        checks[name + "_all_grads_nonzero"] = bool(all(v > 0 for v in gn.values()))
    out["3.3.1"] = {"shapes": checks, "grad_norms": grads}
    print("3.3.1:", checks, flush=True)

    # ---- 3.3.2 ResNet-20 ----
    resnet = MiniResNet().to(device)
    with torch.no_grad():
        d = resnet(torch.zeros(4, 3, 32, 32).to(device))
    assert tuple(d.shape) == (4, 10), f"bad resnet output {tuple(d.shape)}"
    res_params = count_params(resnet)
    print(f"ResNet-20 params={res_params} dummy_out={tuple(d.shape)}", flush=True)
    opt2 = torch.optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=30)
    res_ck = paths.model("ckpt_resnet.pt")
    start_ep2, hist_r2 = resume_ckpt(res_ck, resnet, opt2, sched2, device)
    hist_res, t_res = train_model(resnet, train_loader, val_loader, opt2, sched2, 30, "RES",
                                  device, ckpt_path=res_ck, start_epoch=start_ep2, hist=hist_r2)
    test_loss_res, test_acc_res = run_epoch(resnet, test_loader, crit, device, None)
    out["3.3.2"] = {"params": res_params, "epochs": 30,
                    "optimizer": "SGD(lr=0.1,momentum=0.9,wd=1e-4)",
                    "scheduler": "CosineAnnealingLR(T_max=30)",
                    "train_time_s": round(t_res, 1), "test_accuracy": test_acc_res,
                    "test_loss": test_loss_res, "hist": hist_res,
                    "comparison": [
                        {"model": "CustomCNN", "params": cnn_params,
                         "test_accuracy": test_acc_cnn, "train_time_s": round(t_cnn, 1)},
                        {"model": "ResNet-20", "params": res_params,
                         "test_accuracy": test_acc_res, "train_time_s": round(t_res, 1)}]}
    print(f"ResNet test acc={test_acc_res:.4f} time={t_res:.1f}s", flush=True)
    torch.save(resnet.state_dict(), paths.model("resnet20.pt"))

    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, hist_res["tl"], label="train")
    plt.plot(ep, hist_res["vl"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet-20 — training vs validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.figure("resnet_loss.png"), dpi=120)
    plt.close()
    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, hist_res["ta"], label="train")
    plt.plot(ep, hist_res["va"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet-20 — training vs validation accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.figure("resnet_acc.png"), dpi=120)
    plt.close()

    # first-conv feature maps for 3 test images
    resnet.eval()
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    with torch.no_grad():
        for r in range(3):
            xb, yb = test_set[r * 1000]
            fmap = torch.relu(resnet.bn1(resnet.conv1(xb.unsqueeze(0).to(device)))).cpu()[0]
            axes[r, 0].set_ylabel(f"{CLASSES[yb]}\n(ch0-7)", fontsize=9)
            for c in range(8):
                ax = axes[r, c]
                ax.imshow(fmap[c].numpy(), cmap="viridis")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"ch{c}", fontsize=8)
    fig.suptitle("ResNet-20 first-conv feature maps (8 of 16 channels, 3 test images)")
    fig.tight_layout()
    fig.savefig(paths.figure("resnet_featuremaps.png"), dpi=120)
    plt.close(fig)

    with open(paths.metric("part3_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("PART3 DONE", flush=True)


if __name__ == "__main__":
    main()
