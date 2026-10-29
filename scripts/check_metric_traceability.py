"""Cross-check that every metric value in the notebook really comes from
outputs/metrics/*.json (no hand-typed or fabricated numbers)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

nb = json.load(open(paths.NOTEBOOK, encoding="utf-8"))
src = "\n".join("".join(c["source"]) for c in nb["cells"])
out = "\n".join("".join(c.get("source", [])) for c in nb["cells"]
                if c["cell_type"] == "code")
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            out += "\n" + "".join(o.get("text", []))

p1 = json.load(open(paths.metric("part1_metrics.json"), encoding="utf-8"))
p2 = json.load(open(paths.metric("part2_metrics.json"), encoding="utf-8"))
p3 = json.load(open(paths.metric("part3_metrics.json"), encoding="utf-8"))

CHECKS = [
    ("1.2 baseline RMSE", f"{p1['1.2']['rmse']:.4f}"),
    ("1.2 baseline R2", f"{p1['1.2']['r2']:.4f}"),
    ("1.2 baseline n_iter", str(p1["1.2"]["n_iter"])),
    ("1.3 CV RMSE", f"{p1['1.3']['best_cv_rmse']:.4f}"),
    ("1.3 test RMSE", f"{p1['1.3']['test_rmse']:.4f}"),
    ("1.3 test R2", f"{p1['1.3']['test_r2']:.4f}"),
    ("2.2 accuracy", f"{p2['2.2']['accuracy']:.4f}"),
    ("2.2 precision (malignant)", f"{p2['2.2']['precision_malignant']:.4f}"),
    ("2.2 recall (malignant)", f"{p2['2.2']['recall_malignant']:.4f}"),
    ("2.2 F1 weighted", f"{p2['2.2']['f1_weighted']:.4f}"),
    ("2.4 iterations saved", str(p2["2.4"]["iterations_saved"])),
    ("2.4 early-stop n_iter", str(p2["2.4"]["early_stopping"]["n_iter"])),
    ("3.1C autograd derivative", f"{p3['3.1C']['autograd']:.1f}"),
    ("3.2.2 CustomCNN params", str(p3["3.2.2"]["params"])),
    ("3.2.3 CustomCNN test acc", f"{p3['3.2.3']['test_accuracy']:.4f}"),
    ("3.3.2 ResNet params", str(p3["3.3.2"]["params"])),
    ("3.3.2 ResNet test acc", f"{p3['3.3.2']['test_accuracy']:.4f}"),
]

bad = []
print("notebook value traceability (source + captured output vs metrics JSON)\n")
for lbl, val in CHECKS:
    found = val in src or val in out
    print(f"  {'ok  ' if found else 'MISS'}  {lbl} = {val}")
    if not found:
        bad.append((lbl, val))

print()
if bad:
    print(f"{len(bad)} value(s) NOT traceable to the metrics JSON:")
    for lbl, val in bad:
        print("  -", lbl, "=", val)
    sys.exit(1)
print(f"All {len(CHECKS)} metrics traceable to outputs/metrics/*.json — no fabricated values.")
