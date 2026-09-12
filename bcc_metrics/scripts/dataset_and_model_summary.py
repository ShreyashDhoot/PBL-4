#!/usr/bin/env python3
"""
dataset_and_model_summary.py
==============================
Produces the descriptive, non-inferential figures/tables the paper already
references (Fig. 3 class distribution, dataset split sizes) plus an updated,
measured version of Table I ("Performance Comparison of Models") using real
numbers from this run instead of the placeholder 92/95/90% in the original
draft. Table I now has ONE ROW PER MODEL IN MODEL_REGISTRY (see common.py:
SSDLite, SSDLite v2, YOLOv8n-P2, YOLO11n-P2, RT-DETR, NanoDet-Plus-style,
EfficientDet-Lite0, RTMDet-tiny), built automatically -- filling the "no
YOLO checkpoint ships" gap the notes originally flagged -- plus the
EfficientNet-B0/MobileNetV2 classifier rows kept for backward compatibility.
Any model not yet trained just gets NaN + a "not yet trained" note instead
of an invented number.

Outputs:
  output/figures/class_distribution.png       (Fig. 3 equivalent, BCCD full)
  output/figures/split_sizes.png
  output/tables/table1_model_comparison.csv
  output/reports/dataset_and_model_summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CELL_TYPES, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, log
from voc_data import build_bccd_records, split_bccd_records, class_counts


def plot_class_distribution(records, out_path):
    counts = class_counts(records)
    ordered = {c: counts.get(c, 0) for c in CELL_TYPES}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(ordered.keys(), ordered.values(), color=["#e74c3c", "#3498db", "#2ecc71"])
    ax.set_ylabel("Bounding box count")
    ax.set_title("Distribution of detected blood cell classes (BCCD, full dataset)")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(int(b.get_height())),
                ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")
    return ordered


def plot_split_sizes(train, val, test, out_path):
    sizes = {"train": len(train), "val": len(val), "test": len(test)}
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(sizes.keys(), sizes.values(), color="#9b59b6")
    ax.set_ylabel("Number of images")
    ax.set_title("BCCD dataset split sizes (70/15/15, seed=42)")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(int(b.get_height())),
                ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")
    return sizes


def build_table1_from_measured_results():
    """Reads the CSVs already written by detection_eval.py, edge_performance.py,
    and quantization_bench.py (fp32 row) to assemble an evidence-backed Table I,
    instead of the paper's original hardcoded 92%/95%/90% placeholders -- now
    with ONE ROW PER MODEL IN MODEL_REGISTRY (see common.py), automatically,
    instead of a hardcoded SSDLite-only row + a YOLO "not implemented yet" TODO
    row. Run detection_eval.py / edge_performance.py / quantization_bench.py
    first; any model missing one of those outputs gets NaN in that column
    rather than an invented number.
    """
    from common import MODEL_REGISTRY

    rows = []
    det_path = TABLES_DIR / "map_summary.csv"
    edge_path = TABLES_DIR / "edge_performance.csv"
    quant_path = TABLES_DIR / "quantization_tradeoff.csv"

    det_df = pd.read_csv(det_path) if det_path.exists() else pd.DataFrame()
    edge_df = pd.read_csv(edge_path) if edge_path.exists() else pd.DataFrame()
    quant_df = pd.read_csv(quant_path) if quant_path.exists() else pd.DataFrame()

    for model_key, spec in MODEL_REGISTRY.items():
        acc = mAP50 = latency = mem = np.nan
        if len(det_df):
            bccd_row = det_df[det_df["dataset"] == f"{model_key}__bccd_test"]
            if len(bccd_row):
                acc = float(bccd_row.iloc[0]["accuracy"])
                mAP50 = float(bccd_row.iloc[0]["mAP_0.5"]) if bccd_row.iloc[0]["mAP_0.5"] == bccd_row.iloc[0]["mAP_0.5"] else np.nan
        if len(edge_df) and "model_key" in edge_df.columns:
            edge_row = edge_df[edge_df["model_key"] == model_key]
            if len(edge_row):
                latency = float(edge_row.iloc[0]["mean_latency_ms"])
        if len(quant_df):
            fp32_row = quant_df[(quant_df.get("model") == model_key) & (quant_df["quantization_level"] == "fp32")]
            if len(fp32_row) and "model_size_mb" in fp32_row.columns:
                mem = float(fp32_row.iloc[0]["model_size_mb"])

        checkpoint_exists = (
            spec["native_ckpt"].exists() if spec["kind"] in ("native_ssd", "native_ssd_v2")
            else Path(spec["onnx"]).exists() if spec.get("onnx") else False
        )
        note = "Trained + evaluated" if checkpoint_exists else (
            f"Not yet trained -- run pbl-4/train_bccd_{model_key}_detection.py, then re-run "
            "detection_eval.py/edge_performance.py/quantization_bench.py to populate this row."
        )
        rows.append({
            "Model": spec["display_name"], "model_key": model_key,
            "Accuracy": acc, "mAP@0.5": mAP50, "Latency (ms)": latency, "Memory (MB)": mem,
            "Note": note,
        })

    rows.append({"Model": "EfficientNet-B0 (multi-label presence classifier)", "model_key": "efficientnet",
                 "Accuracy": np.nan, "mAP@0.5": np.nan, "Latency (ms)": np.nan, "Memory (MB)": np.nan,
                 "Note": "Image-level multi-label task, not per-box detection -- see train_efficientnet_bccd.py. "
                          "Fill Accuracy from pbl-4/output/metrics_summary.json if present."})

    eff_acc = np.nan
    metrics_summary_path = TABLES_DIR.parent.parent / "pbl-4" / "output" / "metrics_summary.json"
    if metrics_summary_path.exists():
        with open(metrics_summary_path) as f:
            eff_acc = json.load(f).get("test_accuracy", np.nan)
    if len(edge_df) and "model_key" in edge_df.columns:
        eff_row = edge_df[edge_df["model_key"] == "efficientnet"]
        if len(eff_row):
            rows[-1]["Latency (ms)"] = float(eff_row.iloc[0]["mean_latency_ms"])
    rows[-1]["Accuracy"] = eff_acc

    return pd.DataFrame(rows)


def main():
    bccd = build_bccd_records()
    train, val, test = split_bccd_records(bccd)

    class_dist = plot_class_distribution(bccd, FIGURES_DIR / "class_distribution.png")
    split_sizes = plot_split_sizes(train, val, test, FIGURES_DIR / "split_sizes.png")

    table1 = build_table1_from_measured_results()
    table1.to_csv(TABLES_DIR / "table1_model_comparison.csv", index=False)

    report = {"class_distribution": class_dist, "split_sizes": split_sizes,
               "table1_model_comparison": table1.to_dict(orient="records")}
    with open(REPORTS_DIR / "dataset_and_model_summary.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    log("Dataset/model summary complete. Run detection_eval.py, edge_performance.py, and "
        "quantization_bench.py BEFORE this script for a fully populated Table I.")


if __name__ == "__main__":
    main()
