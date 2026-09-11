#!/usr/bin/env python3
"""
common.py
=========
Shared paths, constants, and small helpers used by every script in this
metrics suite. Import this from every other module instead of re-declaring
paths/classes, so the whole pipeline stays consistent with the original
training code in train_bccd_ssdlite_detection.py / train_efficientnet_bccd.py.

Directory layout this suite expects (see README.md):

    project_root/
      pbl-4/                              <- unzipped pbl-4.zip ('PBL-4' repo)
        data/BCCD_Dataset-master/BCCD/    <- original BCCD VOC dataset
        output/ssdlite_bccd_best.pth      <- trained SSDLite detector (optional)
        output/efficientnet_bccd_best.pth <- trained EfficientNet-B0 (optional)
        train_bccd_ssdlite_detection.py
        train_efficientnet_bccd.py
      annotations/annotations/            <- unzipped annotations.zip
        images/*.jpeg                     <- 72 biomed-student images (BCCD subset)
        annotations/*.xml                 <- 72 independent VOC re-annotations
      bcc_metrics/                        <- this folder
        scripts/*.py
        output/

Every path below can be overridden with environment variables so the same
code runs unchanged on a laptop, a lab machine, or a Raspberry Pi:

    BCC_PBL4_DIR          -> path to the pbl-4 repo root
    BCC_ANNOT_DIR         -> path to the extracted annotations/annotations dir
    BCC_OUTPUT_DIR        -> where this suite writes results (default ./output)
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (override via environment variables if your layout differs)
# ----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
SUITE_ROOT = THIS_DIR.parent
PROJECT_ROOT = SUITE_ROOT.parent

PBL4_DIR = Path(os.environ.get("BCC_PBL4_DIR", PROJECT_ROOT / "pbl-4"))
ANNOT_DIR = Path(os.environ.get("BCC_ANNOT_DIR", PROJECT_ROOT / "annotations"))
OUTPUT_DIR = Path(os.environ.get("BCC_OUTPUT_DIR", SUITE_ROOT / "output"))

BCCD_VOC_DIR = PBL4_DIR / "data" / "BCCD_Dataset-master" / "BCCD"
BCCD_ANNOTATIONS_DIR = BCCD_VOC_DIR / "Annotations"
BCCD_IMAGES_DIR = BCCD_VOC_DIR / "JPEGImages"

SEVENTYTWO_IMAGES_DIR = ANNOT_DIR / "images"
SEVENTYTWO_ANNOT_DIR = ANNOT_DIR / "annotations"

SSDLITE_CKPT = PBL4_DIR / "output" / "ssdlite_bccd_best.pth"
EFFICIENTNET_CKPT = PBL4_DIR / "output" / "efficientnet_bccd_best.pth"

FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"
CACHE_DIR = OUTPUT_DIR / "cache"

for d in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Class conventions (must match train_bccd_ssdlite_detection.py exactly)
# ----------------------------------------------------------------------------
DET_CLASSES = ["__background__", "RBC", "WBC", "Platelets"]  # index 0 = background
DET_CLASS_TO_IDX = {c: i for i, c in enumerate(DET_CLASSES)}
CELL_TYPES = ["RBC", "WBC", "Platelets"]  # the 3 cell types we report per-type metrics for

RANDOM_SEED = 42

# Confidence threshold used to turn raw detector output into "kept" detections
# for counting-based metrics (CV%, Bland-Altman, ICC, agreement tables).
DEFAULT_SCORE_THR = 0.35

# IoU threshold used to greedily match predicted boxes to ground-truth boxes
# when computing detection-side precision/recall/F1/confusion matrices.
DEFAULT_MATCH_IOU = 0.5


def log(msg: str, tag: str = "INFO"):
    print(f"[{tag}] {msg}", flush=True)


def check_paths(require_pbl4=True, require_annot72=True, require_checkpoints=False):
    """Sanity-check the expected directory layout and print a clear report.
    Returns True if everything required is present, False otherwise (script
    callers should still be able to run in a degraded mode where possible,
    e.g. by retraining, so this does not raise).
    """
    ok = True

    def _check(path: Path, label: str, required: bool):
        nonlocal ok
        exists = path.exists()
        status = "OK" if exists else ("MISSING" if required else "missing (optional)")
        log(f"{label:55s} -> {path}  [{status}]")
        if required and not exists:
            ok = False

    log("Checking directory layout ...")
    _check(BCCD_VOC_DIR, "BCCD VOC dataset dir", require_pbl4)
    _check(BCCD_ANNOTATIONS_DIR, "BCCD Annotations dir", require_pbl4)
    _check(BCCD_IMAGES_DIR, "BCCD JPEGImages dir", require_pbl4)
    _check(SEVENTYTWO_IMAGES_DIR, "72-image biomed images dir", require_annot72)
    _check(SEVENTYTWO_ANNOT_DIR, "72-image biomed annotations dir", require_annot72)
    _check(SSDLITE_CKPT, "SSDLite checkpoint (.pth)", require_checkpoints)
    _check(EFFICIENTNET_CKPT, "EfficientNet-B0 checkpoint (.pth)", require_checkpoints)
    return ok
