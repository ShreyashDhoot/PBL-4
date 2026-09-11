#!/usr/bin/env python3
"""
quantization_bench.py
======================
Notes reference: next-step item #1 ("Quantize the model (2-3 levels),
re-measure accuracy/precision/recall/comparison metric ... A
precision-vs-speed-vs-compression trade-off table is a strong,
self-contained results section on its own" + "run the timing/accuracy
numbers on the actual target hardware").

Three levels, all runnable with stock PyTorch/torchvision (no special
hardware needed to produce the comparison, though absolute latency numbers
should be re-collected on the Raspberry Pi 4 itself for the paper -- see
`--device` and the "ON_PI" flag below):

  fp32          - baseline trained weights
  fp16          - half precision
  dynamic_int8  - torch.quantization.quantize_dynamic (Linear/Conv layers)

At each level we report:
  - accuracy / precision / recall / F1 / mAP@0.5 on the BCCD test split
  - CV% (Protocol B, algorithmic reproducibility) per cell type
  - mean inference latency (ms/image) and its spread
  - model size on disk (MB) as a memory-footprint proxy

Set the environment variable BCC_ON_RASPBERRY_PI=1 when actually running
this on a Pi 4 -- it only changes a label in the output so the paper can
correctly attribute which numbers came from the target hardware vs. a dev
machine, per the notes' explicit caution about this.

Outputs:
  output/tables/quantization_tradeoff.csv
  output/figures/quantization_tradeoff.png
  output/reports/quantization_bench.json
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import CELL_TYPES, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, RANDOM_SEED, log
from voc_data import build_bccd_records, split_bccd_records
from detection_metrics import accumulate_confusion, prf1_from_confusion, compute_map
from models_io import load_ssdlite_detector, perturb_image
from stats_toolkit import coefficient_of_variation

LEVELS = ["fp32", "fp16", "dynamic_int8"]
SCORE_THR = 0.35
N_LATENCY_IMAGES = 20
N_CV_IMAGES = 5
N_CV_REPEATS = 10


def eval_level(mode, test_records):
    on_pi = os.environ.get("BCC_ON_RASPBERRY_PI", "0") == "1"
    device = "cpu"  # dynamic INT8 requires CPU; fp32/fp16 forced to CPU too for a fair apples-to-apples edge comparison
    log(f"--- Quantization level: {mode} (device={device}, on_pi={on_pi}) ---")

    detector = load_ssdlite_detector(device=device, quantize_mode=mode)

    # Accuracy / P / R / F1 / mAP on BCCD test split
    predictions, ground_truths = [], []
    latencies = []
    for i, rec in enumerate(test_records):
        img = Image.open(rec["image_path"]).convert("RGB")
        boxes, labels, scores, ms = detector.timed_predict(img, score_thr=0.0)
        pred_labels_0idx = labels.astype(int) - 1
        gt_labels_0idx = np.asarray(rec["labels"], dtype=int) - 1
        predictions.append((boxes, pred_labels_0idx, scores))
        ground_truths.append((np.asarray(rec["boxes"]), gt_labels_0idx))
        if i < N_LATENCY_IMAGES:
            latencies.append(ms)

    cm = accumulate_confusion(predictions, ground_truths, CELL_TYPES, iou_thr=0.5, score_thr=SCORE_THR)
    prf1 = prf1_from_confusion(cm, CELL_TYPES)
    map_result = compute_map(predictions, ground_truths, CELL_TYPES, iou_thresholds=[0.5])

    # CV% (Protocol B) at this quantization level
    rng = np.random.default_rng(RANDOM_SEED)
    cv_rows = []
    for rec in test_records[:N_CV_IMAGES]:
        img = Image.open(rec["image_path"]).convert("RGB")
        counts_per_type = {c: [] for c in CELL_TYPES}
        for _ in range(N_CV_REPEATS):
            perturbed = perturb_image(img, rng, severity="mild")
            counts = detector.count_cells(perturbed, score_thr=SCORE_THR)
            for c in CELL_TYPES:
                counts_per_type[c].append(counts[c])
        for c in CELL_TYPES:
            cv_rows.append(coefficient_of_variation(np.array(counts_per_type[c], dtype=float)))
    mean_cv = float(np.nanmean(cv_rows)) if cv_rows else np.nan

    result = {
        "quantization_level": mode,
        "on_target_hardware_raspberry_pi4": on_pi,
        "accuracy": prf1["accuracy"],
        "precision_micro": prf1["micro"]["precision"],
        "recall_micro": prf1["micro"]["recall"],
        "f1_micro": prf1["micro"]["f1"],
        "mAP_0.5": map_result["mAP_0.5"],
        "mean_cv_percent_protocol_b": mean_cv,
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "model_size_mb": detector.state_dict_size_mb(),
        "n_parameters": detector.num_parameters(),
    }
    return result


def plot_tradeoff(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].bar(df["quantization_level"], df["accuracy"], color="#3498db")
    axes[0].set_title("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(df["quantization_level"], df["mean_latency_ms"], color="#e67e22")
    axes[1].set_title("Mean latency (ms/image)")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(df["quantization_level"], df["model_size_mb"], color="#2ecc71")
    axes[2].set_title("Model size (MB)")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Precision vs. speed vs. compression trade-off across quantization levels")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")


def main():
    bccd = build_bccd_records()
    _, _, test = split_bccd_records(bccd)
    # Cap evaluation set for speed on CPU-only quantization runs; raise if you have time/GPU.
    test = test[:min(40, len(test))]

    results = []
    for mode in LEVELS:
        try:
            results.append(eval_level(mode, test))
        except Exception as e:
            log(f"Quantization level '{mode}' failed: {e}", tag="WARN")
            results.append({"quantization_level": mode, "error": str(e)})

    df = pd.DataFrame(results)
    df.to_csv(TABLES_DIR / "quantization_tradeoff.csv", index=False)

    ok_df = df[~df.get("error").notna()] if "error" in df.columns else df
    if len(ok_df) > 0:
        plot_tradeoff(ok_df, FIGURES_DIR / "quantization_tradeoff.png")

    with open(REPORTS_DIR / "quantization_bench.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    log("Quantization benchmark complete. See output/tables/quantization_tradeoff.csv")
    log("NOTE: for the paper, re-run with BCC_ON_RASPBERRY_PI=1 physically on the "
        "Raspberry Pi 4 to get target-hardware latency numbers (notes: 'run the "
        "timing/accuracy numbers on the actual Raspberry Pi, not just a dev machine').")


if __name__ == "__main__":
    main()
