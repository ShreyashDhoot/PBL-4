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

def _project_path(value, default):
    """Resolve configured relative paths from the project root, not cwd."""
    path = Path(value) if value is not None else Path(default)
    return path if path.is_absolute() else PROJECT_ROOT / path


PBL4_DIR = _project_path(os.environ.get("BCC_PBL4_DIR"), PROJECT_ROOT / "pbl-4")
ANNOT_DIR = _project_path(os.environ.get("BCC_ANNOT_DIR"), PROJECT_ROOT / "annotations")
OUTPUT_DIR = _project_path(os.environ.get("BCC_OUTPUT_DIR"), SUITE_ROOT / "output")


def _find_bccd_root():
    """Use either the repo-local or project-root-relative dataset layout."""
    candidates = (
        PBL4_DIR / "data" / "BCCD_Dataset-master" / "BCCD",
        PROJECT_ROOT / "data" / "BCCD_Dataset-master" / "BCCD",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

BCCD_VOC_DIR = _find_bccd_root()
BCCD_ANNOTATIONS_DIR = BCCD_VOC_DIR / "Annotations"
BCCD_IMAGES_DIR = BCCD_VOC_DIR / "JPEGImages"

SEVENTYTWO_IMAGES_DIR = ANNOT_DIR / "images"
SEVENTYTWO_ANNOT_DIR = ANNOT_DIR / "annotations"

def _first_existing(*candidates):
    """Returns the first candidate path that exists on disk, else the first
    (preferred) candidate. Used so that renaming a model's default output
    location doesn't break loading of a checkpoint that was already trained
    under the old location."""
    candidates = [Path(c) for c in candidates]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


# SSDLite (baseline) now trains into output/ssdlite/ like every other model
# (unified layout). The bare "output/ssdlite_bccd_best.pth" path is kept as a
# fallback ONLY so checkpoints trained before this change keep loading --
# new training runs should use the new location.
SSDLITE_CKPT = _first_existing(
    PBL4_DIR / "output" / "ssdlite" / "ssdlite_bccd_best.pth",
    PBL4_DIR / "output" / "ssdlite_bccd_best.pth",  # legacy flat location
)
SSDLITE_ONNX = _first_existing(
    PBL4_DIR / "output" / "ssdlite" / "ssdlite_bccd.onnx",
    PBL4_DIR / "output" / "ssdlite_bccd.onnx",  # legacy flat location
)
EFFICIENTNET_CKPT = PBL4_DIR / "output" / "efficientnet_bccd_best.pth"

# ----------------------------------------------------------------------------
# Multi-model registry. Every model trainable via pbl-4/train_bccd_*.py is
# listed here so detection_eval.py / cv_repeatability.py / agreement_stats.py
# / calibration_eval.py / robustness_eval.py / quantization_bench.py /
# edge_performance.py can all loop over "every model that has actually been
# trained" instead of hardcoding SSDLite. A model is "available" once its
# `onnx` file (or, for the two native-pytorch loaders, its `native_ckpt`
# file) exists on disk -- run pbl-4/train_all_models.py first.
#
# kind:
#   "native_ssd"  -> loaded via models_io.SSDLiteDetector-style code
#                     (native .pth checkpoint, torchvision SSD class)
#   "onnx"        -> loaded via models_io.OnnxDetector (onnxruntime), the
#                     generic path used by every non-torchvision model
#   "native_only" -> trained/evaluated only inside its own train_*.py
#                     (e.g. RTMDet-tiny without an mmdeploy ONNX export);
#                     bcc_metrics scripts skip these with a clear log line
#                     rather than failing.
MODEL_REGISTRY = {
    "ssdlite": {
        "display_name": "SSDLite (baseline, MobileNetV3-Large, 320px)",
        "kind": "native_ssd",
        "native_ckpt": SSDLITE_CKPT,
        "onnx": SSDLITE_ONNX,
    },
    "ssdlite_v2": {
        "display_name": "SSDLite v2 (small-object-optimized, 512px, focal loss)",
        "kind": "native_ssd_v2",
        "native_ckpt": PBL4_DIR / "output" / "ssdlite_v2" / "ssdlite_v2_bccd_best.pth",
        "onnx": PBL4_DIR / "output" / "ssdlite_v2" / "ssdlite_v2_bccd.onnx",
    },
    "yolov8n": {
        "display_name": "YOLOv8n-P2 (Ultralytics)",
        "kind": "onnx",
        "onnx": PBL4_DIR / "output" / "yolov8n" / "yolov8n_bccd.onnx",
    },
    "yolo11n": {
        "display_name": "YOLO11n-P2 (Ultralytics)",
        "kind": "onnx",
        "onnx": PBL4_DIR / "output" / "yolo11n" / "yolo11n_bccd.onnx",
    },
    "rtdetr_nano": {
        "display_name": "RT-DETR (smallest available Ultralytics scale)",
        "kind": "onnx",
        "onnx": PBL4_DIR / "output" / "rtdetr_nano" / "rtdetr_nano_bccd.onnx",
    },
    "nanodet_plus": {
        "display_name": "NanoDet-Plus-style (self-contained, GFL head)",
        "kind": "onnx",
        "onnx": PBL4_DIR / "output" / "nanodet_plus" / "nanodet_plus_bccd.onnx",
    },
    "efficientdet_lite0": {
        "display_name": "EfficientDet-Lite0 (effdet, BiFPN)",
        "kind": "onnx",
        "onnx": PBL4_DIR / "output" / "efficientdet_lite0" / "efficientdet_lite0_bccd.onnx",
    },
    "rtmdet_tiny": {
        "display_name": "RTMDet-tiny (mmyolo)",
        "kind": "native_only",
        "onnx": PBL4_DIR / "output" / "rtmdet_tiny" / "rtmdet_tiny_bccd.onnx",  # only if mmdeploy export was run
    },
}


def _describe_why_unavailable(key, spec):
    """Best-effort diagnosis of *why* a model isn't available yet, so a
    trained-but-not-exported model doesn't look identical to a
    never-trained one in the logs. Looks for that model's own
    run_report_<key>.json (written by every train_bccd_*.py at the end of
    training, regardless of whether ONNX export succeeded) to tell the two
    situations apart."""
    onnx_path = Path(spec["onnx"]) if spec.get("onnx") else None
    report_path = (onnx_path.parent / f"run_report_{key}.json") if onnx_path else None

    if report_path is not None and report_path.exists():
        try:
            import json as _json
            report = _json.loads(report_path.read_text())
            ckpt = (report.get("files") or {}).get("checkpoint") or (report.get("files") or {}).get("best_checkpoint")
            if ckpt and not Path(ckpt).exists():
                return (f"was trained (run_report_{key}.json exists) but its checkpoint "
                        f"'{ckpt}' is missing from disk (moved/deleted, or an Ultralytics run "
                        f"that wrote its weights to a different --project/--name than expected). "
                        f"Re-run pbl-4/train_bccd_{key}_detection.py.")
            return (f"was trained (checkpoint found, run_report_{key}.json exists) but its ONNX "
                    f"export did not succeed (onnx export failures are non-fatal by design, so "
                    f"training completes anyway -- check that run's console output / "
                    f"run_report_{key}.json['files']['onnx'] for the export error). This suite "
                    f"evaluates non-torchvision models exclusively through ONNX, so no ONNX file "
                    f"means no graphs/tables for this model until export succeeds. Re-run "
                    f"pbl-4/train_bccd_{key}_detection.py, or fix/retry just the export step.")
        except Exception:
            pass
    return (f"has no trained checkpoint yet -- train it with "
            f"pbl-4/train_bccd_{key}_detection.py (or pbl-4/train_all_models.py for every model).")


def list_available_models():
    """Returns [(model_key, display_name), ...] for every model in
    MODEL_REGISTRY that actually has a usable checkpoint on disk right now
    (native .pth for the two torchvision-native models, .onnx for every
    other model). Never raises -- an empty list just means 'train something
    first', which every caller logs clearly rather than crashing on."""
    available = []
    for key, spec in MODEL_REGISTRY.items():
        ready = False
        if spec["kind"] in ("native_ssd", "native_ssd_v2"):
            ready = spec["native_ckpt"].exists()
        if not ready and spec.get("onnx") is not None:
            ready = Path(spec["onnx"]).exists()
        if ready:
            available.append((key, spec["display_name"]))
        else:
            log(f"Model '{key}' ({spec['display_name']}) {_describe_why_unavailable(key, spec)}",
                tag="WARN")
    return available

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
