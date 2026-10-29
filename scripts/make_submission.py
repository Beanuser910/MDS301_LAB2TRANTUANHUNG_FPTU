"""Create submission/ — a copy of the final notebook — after verification.

Runs the submission audit first and refuses to copy if anything fails, so the
folder only ever holds a notebook that has been checked.
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths


def main():
    # 1. audit must pass
    import audit_submission
    try:
        audit_submission.main()
    except SystemExit:
        print("\nAUDIT FAILED — submission copy not created")
        raise

    # 2. copy the notebook
    os.makedirs(paths.SUBMISSION_DIR, exist_ok=True)
    dest = os.path.join(paths.SUBMISSION_DIR, "Lab02_Neural_Networks.ipynb")
    shutil.copy2(paths.NOTEBOOK, dest)

    # 3. prove the copy is independent of external asset files
    nb = json.load(open(dest, encoding="utf-8"))
    cells = nb["cells"]
    n_png = sum(1 for c in cells if c["cell_type"] == "code"
                for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    src = "\n".join("".join(c["source"]) for c in cells)
    import re
    refs = re.findall(r"[\w./\\-]+\.(?:png|json|pt|jpg|csv)", src)
    n_out = sum(1 for c in cells if c["cell_type"] == "code" and c.get("outputs"))

    print()
    print(f"submission copy: {dest}")
    print(f"  size          : {os.path.getsize(dest)/1024:.0f} KB")
    print(f"  cells         : {len(cells)}")
    print(f"  embedded PNGs : {n_png}")
    print(f"  cells w/ out  : {n_out}")
    print(f"  external refs : {len(refs)} {sorted(set(refs))}")
    assert not refs, "submission notebook still references external files"
    assert n_png == 12, f"expected 12 embedded figures, found {n_png}"
    print("  -> self-contained: displays all saved outputs with no external files")


if __name__ == "__main__":
    main()
