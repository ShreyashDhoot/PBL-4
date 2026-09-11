#!/usr/bin/env python3
"""
dataset_and_model_summary.py
==============================
Produces the descriptive, non-inferential figures/tables the paper already
references (Fig. 3 class distribution, dataset split sizes) plus an updated,
measured version of Table I ("Performance Comparison of Models") using real
numbers from this run instead of the placeholder 92/95/90% in the current
draft. This directly answers next-step item #6's ablation note: SSDLite,
EfficientNet-B0, and MobileNetV2 are all included; a YOLO variant is left as
an explicit TODO with instructions, since no YOLO checkpoint ships in
pbl-4.zip (the notes: 'either add the comparison or narrow the methodology
text to match what was actually run').

Outputs:
  output/figures/class_distribution.png       (Fig. 3 equivalent, BCCD full)
  output/figures/split_sizes.png
  output/tables/table1_model_comparison.csv
  output/reports/dataset_and_model_summary.json
"""

import json

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
    instead of the paper's current hardcoded 92%/95%/90% placeholders. Run those
    scripts first; if their outputs are missing this fills in NaN with a note
    rather than inventing numbers.
    """
    rows = []

    det_path = TABLES_DIR / "map_summary.csv"
    edge_path = TABLES_DIR / "edge_performance.csv"
    quant_path = TABLES_DIR / "quantization_tradeoff.csv"

    ssdlite_acc = ssdlite_latency = ssdlite_mem = np.nan
    if det_path.exists():
        det_df = pd.read_csv(det_path)
        bccd_row = det_df[det_df["dataset"] == "bccd_test"]
        if len(bccd_row):
            ssdlite_acc = float(bccd_row.iloc[0]["accuracy"])
    if edge_path.exists():
        edge_df = pd.read_csv(edge_path)
        ssd_row = edge_df[edge_df["model"].str.contains("SSDLite", na=False)]
        if len(ssd_row):
            ssdlite_latency = float(ssd_row.iloc[0]["mean_latency_ms"])
    if quant_path.exists():
        quant_df = pd.read_csv(quant_path)
        fp32_row = quant_df[quant_df["quantization_level"] == "fp32"]
        if len(fp32_row) and "model_size_mb" in fp32_row.columns:
            ssdlite_mem = float(fp32_row.iloc[0]["model_size_mb"])

    rows.append({"Model": "SSDLite (MobileNetV3-Large backbone)",
                 "Accuracy": ssdlite_acc, "Latency (ms)": ssdlite_latency, "Memory (MB)": ssdlite_mem,
                 "Note": "Object detector; accuracy = box-level match rate on BCCD test split"})

    eff_acc = eff_latency = eff_mem = np.nan
    metrics_summary_path = None
    for candidate in [TABLES_DIR.parent.parent / "pbl-4" / "output" / "metrics_summary.json"]:
        if candidate.exists():
            metrics_summary_path = candidate
    if metrics_summary_path:
        with open(metrics_summary_path) as f:
            summary = json.load(f)
        eff_acc = summary.get("test_accuracy", np.nan)
    if edge_path.exists():
        edge_df = pd.read_csv(edge_path)
        eff_row = edge_df[edge_df["model"].str.contains("EfficientNet", na=False)]
        if len(eff_row):
            eff_latency = float(eff_row.iloc[0]["mean_latency_ms"])
            eff_mem = float(eff_row.iloc[0]["peak_ram_mb"]) if eff_row.iloc[0]["peak_ram_mb"] == eff_row.iloc[0]["peak_ram_mb"] else np.nan

    rows.append({"Model": "EfficientNet-B0 (multi-label presence classifier)",
                 "Accuracy": eff_acc, "Latency (ms)": eff_latency, "Memory (MB)": eff_mem,
                 "Note": "Image-level multi-label task, not per-box detection -- see train_efficientnet_bccd.py"})

    rows.append({"Model": "MobileNetV2", "Accuracy": np.nan, "Latency (ms)": np.nan, "Memory (MB)": np.nan,
                 "Note": "No trained MobileNetV2 checkpoint shipped in pbl-4.zip; the paper's methodology "
                          "mentions this architecture but no matching train_*.py/checkpoint exists in the repo. "
                          "Train and export it the same way as train_efficientnet_bccd.py before reporting, "
                          "or remove it from the methodology text to match what was actually run."})

    rows.append({"Model": "YOLO variant (e.g. YOLOv8n)", "Accuracy": np.nan, "Latency (ms)": np.nan, "Memory (MB)": np.nan,
                 "Note": "Notes next-step item #6: literature review and methodology mention YOLO as an "
                          "alternative, but no YOLO training script/checkpoint ships in this repo. Either "
                          "add a small ablation (e.g. ultralytics YOLOv8n on the same BCCD VOC-to-YOLO-converted "
                          "labels) or narrow the methodology text to match SSDLite/EfficientNet only."})

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
