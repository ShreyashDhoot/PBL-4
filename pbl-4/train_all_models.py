#!/usr/bin/env python3
"""
train_all_models.py
====================
Runs every train_bccd_*_detection.py script in this folder, one at a time,
so the full multi-model ablation (baseline SSDLite + small-object-tuned
SSDLite v2 + YOLOv8n-P2 + YOLO11n-P2 + RT-DETR + NanoDet-Plus-style +
EfficientDet-Lite0 + RTMDet-tiny) can be produced with a single command,
matching bcc_metrics/run_all.py's "one entry point, tolerant of individual
step failures" philosophy.

Each model is run in its OWN subprocess (not imported in-process), because
several of these scripts pull in heavy, mutually-incompatible dependency
stacks (ultralytics vs. effdet vs. mmyolo/mmdet/mmengine) that are not
safe to import into the same Python process together. A model whose
optional dependency isn't installed exits cleanly with instructions (see
each script's `_require_*()` guard) and this orchestrator records that as
a skipped, not fatal, step.

Usage:
    cd pbl-4
    python train_all_models.py                          # every model, default hyperparameters
    python train_all_models.py --only ssdlite_v2 yolov8n # a subset
    python train_all_models.py --skip rtmdet_tiny         # everything except the heaviest optional dep
    python train_all_models.py --quick                    # small epoch counts, for a pipeline smoke-test
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

# (model_key, script, extra_args_for_full_run, extra_args_for_quick_run)
MODELS = [
    ("ssdlite", "train_bccd_ssdlite_detection.py", [], ["--epochs", "2"]),
    ("ssdlite_v2", "train_bccd_ssdlite_v2_detection.py", [], ["--epochs", "2"]),
    ("yolov8n", "train_bccd_yolov8n_detection.py", [], ["--epochs", "2"]),
    ("yolo11n", "train_bccd_yolo11n_detection.py", [], ["--epochs", "2"]),
    ("rtdetr_nano", "train_bccd_rtdetr_nano_detection.py", [], ["--epochs", "2"]),
    ("nanodet_plus", "train_bccd_nanodet_plus_detection.py", [], ["--epochs", "2"]),
    ("efficientdet_lite0", "train_bccd_efficientdet_lite0_detection.py", [], ["--epochs", "2"]),
    ("rtmdet_tiny", "train_bccd_rtmdet_tiny_detection.py", [], ["--epochs", "2"]),
]


def main():
    parser = argparse.ArgumentParser(description="Run every BCCD model-training script in this folder.")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these model keys.")
    parser.add_argument("--skip", nargs="*", default=None, help="Skip these model keys.")
    parser.add_argument("--quick", action="store_true", help="Use small epoch counts for a fast pipeline smoke-test.")
    args = parser.parse_args()

    selected = MODELS
    if args.only:
        selected = [m for m in selected if m[0] in args.only]
    if args.skip:
        selected = [m for m in selected if m[0] not in args.skip]

    results = []
    for model_key, script, full_args, quick_args in selected:
        extra = quick_args if args.quick else full_args
        cmd = [sys.executable, str(THIS_DIR / script)] + extra
        print(f"\n{'='*70}\n[TRAIN] {model_key}  ({' '.join(cmd)})\n{'='*70}", flush=True)
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True, cwd=str(THIS_DIR))
            elapsed = time.time() - t0
            print(f"[TRAIN] {model_key} OK ({elapsed/60:.1f} min)")
            results.append({"model": model_key, "status": "ok", "seconds": elapsed, "rerun_cmd": " ".join(cmd)})
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            status = "skipped_missing_dependency" if e.returncode == 1 else "failed"
            print(f"[TRAIN] {model_key} {status.upper()} after {elapsed/60:.1f} min (exit code {e.returncode})")
            results.append({"model": model_key, "status": status, "seconds": elapsed,
                             "exit_code": e.returncode, "rerun_cmd": " ".join(cmd)})
        except Exception as e:
            print(f"[TRAIN] {model_key} FAILED: {e}")
            results.append({"model": model_key, "status": "failed", "error": str(e), "rerun_cmd": " ".join(cmd)})

    out_dir = THIS_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "training_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"models": results}, f, indent=2)

    print(f"\n{'='*70}\nTRAINING SUMMARY\n{'='*70}")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    for r in results:
        line = f"  [{r['status']:28s}] {r['model']}"
        if r["status"] != "ok":
            line += f"   -> re-run with: {r['rerun_cmd']}"
        print(line)
    print(f"\n{n_ok}/{len(results)} models trained successfully. Summary written -> {summary_path}")
    print("Next: run bcc_metrics/run_all.py to evaluate every trained model on the shared test set.")


if __name__ == "__main__":
    main()
