"""Part 2 — Classification MLP on Breast Cancer Wisconsin. Saves metrics + figures."""
import json, os, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

warnings.filterwarnings("ignore")
RNG = 42
paths.ensure_dirs()
out = {}
ARCH = (64, 32)

# ---------------- 2.1 Data prep ----------------
data = load_breast_cancer()
X, y = data.data, data.target
out["2.1"] = {"shape": list(X.shape), "feature_names": list(data.feature_names),
              "target_names": list(data.target_names),
              "class_counts": {str(k): int((y == k).sum()) for k in (0, 1)},
              "missing_total": int(np.isnan(X).sum())}
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RNG, stratify=y)
scaler = StandardScaler().fit(X_train)  # fit on train only
Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
out["2.1"].update({"train_shape": list(Xtr.shape), "test_shape": list(Xte.shape)})
print("2.1:", out["2.1"], flush=True)

# ---------------- 2.2 MLP classifier ----------------
clf = MLPClassifier(hidden_layer_sizes=ARCH, activation="relu", solver="adam",
                    max_iter=500, random_state=RNG)
t0 = time.time(); clf.fit(Xtr, y_train); t_clf = time.time() - t0
yp = clf.predict(Xte)
m = {"accuracy": float(accuracy_score(y_test, yp)),
     "precision_malignant": float(precision_score(y_test, yp, pos_label=0)),
     "recall_malignant": float(recall_score(y_test, yp, pos_label=0)),
     "f1_weighted": float(f1_score(y_test, yp, average="weighted")),
     "n_iter": int(clf.n_iter_), "final_loss": float(clf.loss_),
     "train_time_s": round(t_clf, 2),
     "report": classification_report(y_test, yp, target_names=list(data.target_names))}
out["2.2"] = m
print(f"2.2 acc={m['accuracy']:.4f} prec_mal={m['precision_malignant']:.4f} "
      f"rec_mal={m['recall_malignant']:.4f} f1w={m['f1_weighted']:.4f} n_iter={m['n_iter']}", flush=True)

plt.figure(figsize=(7, 4.5))
plt.plot(clf.loss_curve_, linewidth=1.5)
plt.xlabel("Iteration"); plt.ylabel("Training loss (cross-entropy)")
plt.title("Part 2.2 — MLPClassifier training loss (64,32 / relu / adam)")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(paths.figure("part2_loss.png"), dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, yp),
                       display_labels=list(data.target_names)).plot(ax=ax, cmap="Blues")
ax.set_title("Part 2.2 — Confusion matrix (test set)")
fig.tight_layout(); fig.savefig(paths.figure("part2_confusion.png"), dpi=120); plt.close(fig)

# ---------------- 2.3 Optimizer comparison ----------------
opt_results = {}
curves = {}
for solver in ["sgd", "adam", "lbfgs"]:
    kw = dict(hidden_layer_sizes=ARCH, activation="relu", solver=solver,
              max_iter=500, random_state=RNG)
    if solver == "sgd":
        kw.update(learning_rate_init=0.01)
    mdl = MLPClassifier(**kw)
    t0 = time.time(); mdl.fit(Xtr, y_train); dt = time.time() - t0
    acc = float(accuracy_score(y_test, mdl.predict(Xte)))
    entry = {"test_accuracy": acc, "train_time_s": round(dt, 2),
             "n_iter": int(mdl.n_iter_), "final_loss": float(mdl.loss_)}
    if hasattr(mdl, "loss_curve_"):
        entry["has_native_curve"] = True
        curves[solver] = list(mdl.loss_curve_)
    else:
        entry["has_native_curve"] = False  # lbfgs: no per-iteration curve in sklearn
    opt_results[solver] = entry
    print(f"2.3 {solver}: acc={acc:.4f} loss={mdl.loss_:.4f} n_iter={mdl.n_iter_} time={dt:.1f}s", flush=True)
out["2.3"] = opt_results

plt.figure(figsize=(8, 5))
for solver in ["sgd", "adam"]:
    plt.plot(curves[solver], label=f"{solver} (native loss_curve_)", linewidth=1.5)
lb = opt_results["lbfgs"]["final_loss"]
plt.axhline(lb, color="green", linestyle="--",
            label=f"lbfgs final loss={lb:.4f} (no native curve — reference line)")
plt.xlabel("Iteration"); plt.ylabel("Training loss")
plt.title("Part 2.3 — Optimizer comparison (sgd/adam curves + lbfgs final loss)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(paths.figure("part2_optimizers.png"), dpi=120); plt.close()

# ---------------- 2.4 Early stopping ----------------
a = MLPClassifier(hidden_layer_sizes=ARCH, activation="relu", solver="adam",
                  early_stopping=False, max_iter=500, random_state=RNG)
t0 = time.time(); a.fit(Xtr, y_train); ta = time.time() - t0
b = MLPClassifier(hidden_layer_sizes=ARCH, activation="relu", solver="adam",
                  early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=10, max_iter=500, random_state=RNG)
t0 = time.time(); b.fit(Xtr, y_train); tb = time.time() - t0
ea = {"test_accuracy": float(accuracy_score(y_test, a.predict(Xte))),
      "n_iter": int(a.n_iter_), "train_time_s": round(ta, 2)}
eb = {"test_accuracy": float(accuracy_score(y_test, b.predict(Xte))),
      "n_iter": int(b.n_iter_), "train_time_s": round(tb, 2),
      "stopped_early": bool(b.n_iter_ < 500)}
out["2.4"] = {"no_early_stopping": ea, "early_stopping": eb,
              "iterations_saved": int(ea["n_iter"] - eb["n_iter"])}
print(f"2.4 A: acc={ea['test_accuracy']:.4f} it={ea['n_iter']} | "
      f"B: acc={eb['test_accuracy']:.4f} it={eb['n_iter']} early={eb['stopped_early']}", flush=True)

plt.figure(figsize=(8, 5))
plt.plot(a.loss_curve_, label=f"(a) no early stopping (n_iter={ea['n_iter']})", linewidth=1.5)
plt.plot(b.loss_curve_, label=f"(b) early stopping (n_iter={eb['n_iter']})", linewidth=1.5)
if eb["stopped_early"]:
    plt.axvline(eb["n_iter"] - 1, color="red", linestyle="--",
                label=f"early stop @ iter {eb['n_iter']}")
plt.xlabel("Iteration"); plt.ylabel("Training loss")
plt.title("Part 2.4 — Early stopping comparison")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(paths.figure("part2_earlystopping.png"), dpi=120); plt.close()

with open(paths.metric("part2_metrics.json"), "w") as f:
    json.dump(out, f, indent=2)
print("PART2 DONE", flush=True)
