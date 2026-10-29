"""Audit the final notebook against the Lab 02 submission requirements.

Checks (from Lab02_MDS301.md):
  * single .ipynb with all code, outputs and written answers
  * first cell carries the student name and ID
  * every required experiment has code + captured output
  * every required chart is embedded (no external image files)
  * no external file references remain in the notebook source
  * model architectures and hyperparameters match the assignment
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

# figure -> assignment requirement it satisfies
REQUIRED_FIGURES = {
    "part1_baseline_loss.png": "1.2 learning curve (loss vs iteration)",
    "part1_compare.png": "1.3 baseline vs tuned comparison",
    "part2_loss.png": "2.2 learning curve",
    "part2_confusion.png": "2.2 confusion matrix",
    "part2_optimizers.png": "2.3 three optimiser curves on one figure",
    "part2_earlystopping.png": "2.4 both curves + early-stop marker",
    "cifar10_samples.png": "3.2.1 grid of 16 samples with labels",
    "cnn_loss.png": "3.2.3 train/val loss curves",
    "cnn_acc.png": "3.2.3 train/val accuracy curves",
    "resnet_loss.png": "3.3.2 train/val loss curves",
    "resnet_acc.png": "3.3.2 train/val accuracy curves",
    "resnet_featuremaps.png": "3.3.2 first-layer feature maps",
}

# (label, regex that must appear somewhere in the notebook source)
REQUIRED_CONTENT = [
    ("1.1 California Housing load", r"fetch_california_housing"),
    ("1.1 train/test split 80/20", r"test_size=0\.2"),
    ("1.1 StandardScaler", r"StandardScaler"),
    ("1.2 MLPRegressor 64 relu adam", r"MLPRegressor\(hidden_layer_sizes=\(64,\), activation=[\"']relu[\"'],\s*\n?\s*.*solver=[\"']adam[\"']"),
    ("1.2 max_iter=500", r"max_iter=500"),
    ("1.2 RMSE", r"rmse|RMSE"),
    ("1.2 R2", r"r2_score|R2"),
    ("1.3 GridSearchCV 5-fold", r"GridSearchCV\(.*cv=5"),
    ("1.3 scoring neg_root_mean_squared_error", r"neg_root_mean_squared_error"),
    ("1.3 n_jobs=-1", r"n_jobs=-1"),
    ("1.3 param grid (64,)/(128,64)/(256,128,64)", r"\(256, 128, 64\)"),
    ("2.1 load_breast_cancer", r"load_breast_cancer"),
    ("2.1 stratify", r"stratify="),
    ("2.2 MLPClassifier (64, 32)", r"MLPClassifier\(hidden_layer_sizes=\(64, 32\)"),
    ("2.2 classification_report", r"classification_report"),
    ("2.3 sgd/adam/lbfgs", r"for solver in \[?[\"']sgd[\"'],\s*[\"']adam[\"'],\s*[\"']lbfgs[\"']"),
    ("2.4 early_stopping=True", r"early_stopping=True"),
    ("2.4 validation_fraction=0.1", r"validation_fraction=0\.1"),
    ("2.4 n_iter_no_change=10", r"n_iter_no_change=10"),
    ("3.1A tensor (3,224,224)", r"torch\.randn\(3, 224, 224\)"),
    ("3.1B 2048x2048 matmul", r"2048"),
    ("3.1C autograd f(x)", r"\(x\*\*2 \+ 3\*x - 2\)\*\*3"),
    ("3.2.1 RandomHorizontalFlip", r"RandomHorizontalFlip"),
    ("3.2.1 RandomCrop padding 4", r"RandomCrop\(32, padding=4\)"),
    ("3.2.1 batch size 64", r"batch_size=64"),
    ("3.2.1 num_workers=2", r"num_workers=2"),
    ("3.2.2 CustomCNN class", r"class CustomCNN\(nn\.Module\)"),
    ("3.2.2 Dropout(0.5)", r"nn\.Dropout\(0\.5\)"),
    ("3.2.2 AdaptiveAvgPool2d(1)", r"nn\.AdaptiveAvgPool2d\(1\)"),
    ("3.2.3 Adam lr=0.001", r"torch\.optim\.Adam\(cnn\.parameters\(\), lr=0\.001\)"),
    ("3.2.3 StepLR step=10 gamma=0.1", r"StepLR\(opt, step_size=10, gamma=0\.1\)"),
    ("3.3.1 ResidualBlock class", r"class ResidualBlock\(nn\.Module\)"),
    ("3.3.1 BottleneckBlock class", r"class BottleneckBlock\(nn\.Module\)"),
    ("3.3.1 projection shortcut 1x1", r"nn\.Conv2d\(in_channels, C, 1"),
    ("3.3.1 gradient norms", r"grad\.norm\(\)"),
    ("3.3.2 ResNet-20 groups 16/32/64", r"ResidualBlock\(16\)"),
    ("3.3.2 SGD lr=0.1 momentum=0.9 wd=1e-4", r"SGD\(resnet\.parameters\(\), lr=0\.1, momentum=0\.9, weight_decay=1e-4\)"),
    ("3.3.2 CosineAnnealingLR T_max=30", r"CosineAnnealingLR\(opt, T_max=30\)"),
]


def main():
    nb = json.load(open(paths.NOTEBOOK, encoding="utf-8"))
    cells = nb["cells"]
    all_src = "\n".join("".join(c["source"]) for c in cells)
    code_cells = [c for c in cells if c["cell_type"] == "code"]

    failures = []
    print(f"notebook: {paths.NOTEBOOK}")
    print(f"cells: {len(cells)} ({len(code_cells)} code, "
          f"{len(cells)-len(code_cells)} markdown)\n")

    # --- 1. first cell carries name + ID ---
    first = "".join(cells[0]["source"])
    has_name = "Student Name" in first
    has_id = "Student ID" in first
    placeholders = "placeholder" in first.lower()
    print(f"[{'ok' if has_name and has_id else 'FAIL'}] first cell has name + ID fields")
    if placeholders:
        print("        NOTE: name/ID are still placeholders — the student must fill "
              "them in before handing the notebook in.")
    if not (has_name and has_id):
        failures.append("first cell missing name/ID fields")

    # --- 2. every code cell carries captured output ---
    no_out = [i for i, c in enumerate(cells)
              if c["cell_type"] == "code" and not c.get("outputs")]
    print(f"[{'ok' if not no_out else 'FAIL'}] every code cell has captured output"
          f"{'' if not no_out else ' -> cells ' + str(no_out)}")
    if no_out:
        failures.append(f"code cells without outputs: {no_out}")

    # --- 3. required figures embedded ---
    embedded = set()
    for c in code_cells:
        for o in c.get("outputs", []):
            d = o.get("data", {})
            if "image/png" in d:
                embedded.add(len(d["image/png"]))
    n_png = sum(1 for c in code_cells for o in c.get("outputs", [])
                if "image/png" in o.get("data", {}))
    missing_fig = []
    for fname, req in REQUIRED_FIGURES.items():
        p = paths.figure(fname)
        if not os.path.exists(p):
            missing_fig.append(fname)
    print(f"[{'ok' if not missing_fig else 'FAIL'}] all {len(REQUIRED_FIGURES)} required "
          f"figures exist in outputs/figures/"
          f"{'' if not missing_fig else ' -> missing ' + str(missing_fig)}")
    print(f"        {n_png} PNG outputs embedded in the notebook")
    if missing_fig:
        failures.append(f"missing figures: {missing_fig}")
    if n_png < len(REQUIRED_FIGURES):
        failures.append(f"only {n_png} of {len(REQUIRED_FIGURES)} figures embedded")

    # --- 4. no external file references ---
    refs = re.findall(r"[\w./\\-]+\.(?:png|json|pt|jpg|csv)", all_src)
    print(f"[{'ok' if not refs else 'FAIL'}] no external file references in source"
          f"{'' if not refs else ' -> ' + str(set(refs))}")
    if refs:
        failures.append(f"external references: {set(refs)}")

    # --- 5. required code content ---
    missing_content = [(lbl, pat) for lbl, pat in REQUIRED_CONTENT
                       if not re.search(pat, all_src, re.S)]
    print(f"[{'ok' if not missing_content else 'FAIL'}] all {len(REQUIRED_CONTENT)} "
          f"required code elements present")
    for lbl, pat in missing_content:
        print(f"        MISSING: {lbl}   (pattern {pat[:44]})")
    if missing_content:
        failures.append(f"missing content: {[l for l, _ in missing_content]}")

    # --- 6. analysis prose present ---
    md_text = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    discussions = md_text.count("**Discussion")
    print(f"[{'ok' if discussions >= 5 else 'WARN'}] {discussions} written discussion "
          f"sections (assignment asks for analysis after most tasks)")

    print()
    if failures:
        print("AUDIT FAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("NOTEBOOK COMPLIES WITH LAB 02 SUBMISSION REQUIREMENTS")


if __name__ == "__main__":
    main()
