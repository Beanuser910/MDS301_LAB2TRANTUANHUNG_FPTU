"""Shared filesystem layout for the MDS301 Lab 02 project.

Every script resolves paths through this module so the tree can be relocated
without editing each file:

    <root>/
      Lab02_Neural_Networks.ipynb   <- final notebook (submission source)
      Lab02_MDS301.md                <- assignment specification
      scripts/                       <- all python helpers
      outputs/
        figures/                     <- generated PNGs
        metrics/                     <- generated JSON metrics
        models/                      <- trained weights (.pt)
      data/                          <- datasets (git-ignored)
"""
import os

# scripts/ lives directly under the project root
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)

OUTPUTS = os.path.join(ROOT, "outputs")
FIGURES = os.path.join(OUTPUTS, "figures")
METRICS = os.path.join(OUTPUTS, "metrics")
MODELS = os.path.join(OUTPUTS, "models")
DATA = os.path.join(ROOT, "data")

NOTEBOOK = os.path.join(ROOT, "Lab02_Neural_Networks.ipynb")
SUBMISSION_DIR = os.path.join(ROOT, "submission")


def ensure_dirs():
    """Create the output tree if it does not exist yet."""
    for d in (OUTPUTS, FIGURES, METRICS, MODELS, DATA):
        os.makedirs(d, exist_ok=True)


def figure(name):
    """Absolute path of a figure under outputs/figures/."""
    return os.path.join(FIGURES, name)


def metric(name):
    """Absolute path of a metrics JSON under outputs/metrics/."""
    return os.path.join(METRICS, name)


def model(name):
    """Absolute path of a model checkpoint under outputs/models/."""
    return os.path.join(MODELS, name)
