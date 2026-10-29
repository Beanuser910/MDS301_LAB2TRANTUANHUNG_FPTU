"""Part 1 — Regression MLP on California Housing. Saves metrics + figures to outputs/."""
import json, os, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

warnings.filterwarnings("ignore")
RNG = 42
paths.ensure_dirs()
ASSETS = paths.OUTPUTS
out = {}

# ---------------- 1.1 Data preparation ----------------
housing = fetch_california_housing()
X, y = housing.data, housing.target
out["1.1"] = {"shape": list(X.shape), "feature_names": list(housing.feature_names),
              "target_mean": float(np.mean(y)), "target_std": float(np.std(y))}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RNG)
scaler = StandardScaler().fit(X_train)  # fit on train only (no leakage)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
out["1.1"].update({"train_shape": list(X_train_s.shape), "test_shape": list(X_test_s.shape)})
print("Features:", housing.feature_names, "| train", X_train_s.shape, "test", X_test_s.shape, flush=True)

# ---------------- 1.2 Baseline MLP ----------------
base = MLPRegressor(hidden_layer_sizes=(64,), activation="relu", solver="adam",
                    max_iter=500, random_state=RNG)
t0 = time.time()
base.fit(X_train_s, y_train)
t_base = time.time() - t0
pred = base.predict(X_test_s)
rmse_base = float(np.sqrt(mean_squared_error(y_test, pred)))
r2_base = float(r2_score(y_test, pred))
out["1.2"] = {"rmse": rmse_base, "r2": r2_base, "n_iter": int(base.n_iter_),
              "final_loss": float(base.loss_), "train_time_s": round(t_base, 1)}
print(f"Baseline: RMSE={rmse_base:.4f} R2={r2_base:.4f} n_iter={base.n_iter_} time={t_base:.1f}s", flush=True)

plt.figure(figsize=(7, 4.5))
plt.plot(base.loss_curve_, linewidth=1.5)
plt.xlabel("Iteration"); plt.ylabel("Training loss"); plt.title("Part 1.2 — Baseline MLPRegressor training loss")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(paths.figure("part1_baseline_loss.png"), dpi=120)
plt.close()

# ---------------- 1.3 Hyperparameter tuning ----------------
pipe = Pipeline([("scaler", StandardScaler()),
                 ("mlp", MLPRegressor(max_iter=500, random_state=RNG))])
param_grid = {"mlp__hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)],
              "mlp__activation": ["relu", "tanh"],
              "mlp__learning_rate_init": [0.01, 0.001],
              "mlp__alpha": [0.0001, 0.001]}
gs = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
t0 = time.time()
gs.fit(X_train, y_train)  # raw X: scaler is inside the pipeline (no leakage across folds)
t_gs = time.time() - t0
best_cv_rmse = float(-gs.best_score_)
best = gs.best_estimator_
pred_b = best.predict(X_test)
rmse_best = float(np.sqrt(mean_squared_error(y_test, pred_b)))
r2_best = float(r2_score(y_test, pred_b))
best_params = {k.replace("mlp__", ""): (list(v) if isinstance(v, tuple) else v)
               for k, v in gs.best_params_.items()}
out["1.3"] = {"best_params": best_params, "best_cv_rmse": best_cv_rmse,
              "test_rmse": rmse_best, "test_r2": r2_best,
              "n_configs": 24, "n_fits": 120, "grid_time_s": round(t_gs, 1)}
print(f"Tuned: params={best_params} CV_RMSE={best_cv_rmse:.4f} test_RMSE={rmse_best:.4f} R2={r2_best:.4f} time={t_gs:.1f}s", flush=True)

# comparison bar chart (genuine values)
plt.figure(figsize=(6, 4))
plt.bar(["Baseline\n(64,)", "Tuned\n(best CV)"], [rmse_base, rmse_best], color=["steelblue", "darkorange"])
plt.ylabel("Test RMSE"); plt.title("Part 1.3 — Baseline vs tuned test RMSE (lower is better)")
for i, v in enumerate([rmse_base, rmse_best]):
    plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")
plt.tight_layout(); plt.savefig(paths.figure("part1_compare.png"), dpi=120); plt.close()

with open(paths.metric("part1_metrics.json"), "w") as f:
    json.dump(out, f, indent=2)
print("PART1 DONE", flush=True)
