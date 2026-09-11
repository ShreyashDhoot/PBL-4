#!/usr/bin/env python3
"""
run_all.py
==========
Single entry point that runs the ENTIRE metrics suite described in
Comparison_Metrics_Notes.docx, in the correct dependency order, and leaves
every table/figure/report needed to write the paper's Results section under
output/.

Usage
-----
    cd bcc_metrics
    python run_all.py                       # run everything
    python run_all.py --only detection agreement    # run a subset
    python run_all.py --skip robustness quantization  # skip slow steps
    BCC_ON_RASPBERRY_PI=1 python run_all.py  # when physically running on the Pi 4

Environment variables (see scripts/common.py and individual modules):
    BCC_PBL4_DIR              path to the unzipped pbl-4 repo (default: ../pbl-4)
    BCC_ANNOT_DIR              path to unzipped annotations/annotations (default: ../annotations/annotations)
    BCC_OUTPUT_DIR             where results are written (default: ./output)
    BCC_FIELD_IMAGES_DIR       real multi-field capture of ONE slide (Protocol A)
    BCC_SECOND_ANNOTATOR_DIR   a second annotator's relabel of a 72-set subset
    BCC_ON_RASPBERRY_PI        set to 1 when actually running on the target Pi 4
    BCC_POWER_METER_CMD        shell command printing instantaneous watts

Each step is wrapped so that a failure (e.g. missing checkpoint, missing
optional dependency) prints a clear error and lets the remaining steps run,
rather than aborting the whole suite. A final summary lists what
succeeded/failed and exactly which command to re-run for any failed step.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

# (step_name, script_filename, requires_torch, description)
STEPS = [
    ("data_check", "common.py", False,
     "Sanity-check the expected directory layout (BCCD, 72-set, checkpoints)."),
    ("dataset_summary", "dataset_and_model_summary.py", False,
     "Descriptive dataset figures (class distribution, split sizes) + Table I skeleton."),
    ("detection", "detection_eval.py", True,
     "Accuracy/Precision/Recall/F1, confusion matrices, mAP@0.5/0.5:0.95, "
     "BCCD test split vs. 72-image OOD set (generalization gap)."),
    ("cv_repeatability", "cv_repeatability.py", True,
     "CV% Protocol A (field-to-field) and Protocol B (algorithmic reproducibility)."),
    ("agreement", "agreement_stats.py", True,
     "Bland-Altman, ICC(2,1), Pearson/Spearman, Passing-Bablok, Section-5 comparison row, "
     "inter-annotator kappa (if a second annotator's labels are supplied)."),
    ("literature_table", "literature_comparison_table.py", False,
     "Merges the literature analyzer table with our own agreement row (needs 'agreement' to have run)."),
    ("calibration", "calibration_eval.py", True,
     "Expected Calibration Error + reliability diagram."),
    ("robustness", "robustness_eval.py", True,
     "Accuracy vs. corruption-severity curves (blur/obscuration, stain-color shift)."),
    ("quantization", "quantization_bench.py", True,
     "FP32 vs FP16 vs dynamic INT8: accuracy/latency/memory/CV% trade-off table."),
    ("edge_performance", "edge_performance.py", True,
     "Per-image inference latency + memory (+ power, if a meter command is configured)."),
    ("dataset_summary_final", "dataset_and_model_summary.py", False,
     "Re-run once detection/edge/quantization tables exist, to populate a measured Table I."),
]


def run_step(name, script, cwd):
    print(f"\n{'=' * 70}\n[STEP] {name}  ({script})\n{'=' * 70}", flush=True)
    t0 = time.time()
    try:
        subprocess.run([sys.executable, str(SCRIPTS_DIR / script)], check=True, cwd=cwd)
        elapsed = time.time() - t0
        print(f"[STEP] {name} OK ({elapsed:.1f}s)")
        return True, None
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"[STEP] {name} FAILED after {elapsed:.1f}s (exit code {e.returncode})")
        return False, f"python {SCRIPTS_DIR / script}"
    except Exception as e:
        print(f"[STEP] {name} FAILED: {e}")
        return False, f"python {SCRIPTS_DIR / script}"


def main():
    parser = argparse.ArgumentParser(description="Run the full BCC metrics suite.")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these step names.")
    parser.add_argument("--skip", nargs="*", default=None, help="Skip these step names.")
    args = parser.parse_args()

    sys.path.insert(0, str(SCRIPTS_DIR))
    from common import check_paths, log  # noqa: E402

    log("Checking project layout before running anything ...")
    check_paths(require_pbl4=True, require_annot72=True, require_checkpoints=True)

    selected = STEPS
    if args.only:
        selected = [s for s in STEPS if s[0] in args.only]
    if args.skip:
        selected = [s for s in selected if s[0] not in args.skip]

    results = []
    for name, script, _requires_torch, _desc in selected:
        if script == "common.py":
            results.append((name, True, None))
            continue
        ok, rerun_cmd = run_step(name, script, cwd=str(SCRIPTS_DIR))
        results.append((name, ok, rerun_cmd))

    print(f"\n{'=' * 70}\nRUN SUMMARY\n{'=' * 70}")
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, rerun_cmd in results:
        status = "OK" if ok else "FAILED"
        line = f"  [{status:6s}] {name}"
        if not ok and rerun_cmd:
            line += f"   -> re-run with: {rerun_cmd}"
        print(line)
    print(f"\n{n_ok}/{len(results)} steps completed successfully.")
    print(f"Outputs written under: {(SCRIPTS_DIR.parent / 'output').resolve()}")

    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
