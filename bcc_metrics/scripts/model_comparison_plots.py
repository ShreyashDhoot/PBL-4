#!/usr/bin/env python3
"""
model_comparison_plots.py
==========================
Every other step in this suite already writes ONE combined table across all
trained models (table1_model_comparison_detection.csv, map_summary.csv,
calibration_ece_all_models.csv, quantization_tradeoff.csv,
edge_performance.csv, cv_protocol_*_summary.csv) -- but until now nothing
ever turned those combined tables into an actual side-by-side chart, only
per-model figures suffixed `_<model_key>`. This step is that missing piece:
it reads the combined tables (must already exist -- run detection,
calibration, quantization, edge_performance, cv_repeatability first) and
renders one bar chart per metric group, with every trained model side by
side, so "how do the models compare" has an actual picture, not just a CSV.

Deliberately tolerant: if a given combined table doesn't exist yet (that
step wasn't run, or produced no rows) this script logs a WARN and skips
just that chart instead of crashing, so it still produces whatever charts
it can from whatever combined tables are present.

Outputs (all under output/figures/):
  model_comparison_detection.png        (accuracy / F1 / mAP@0.5 / mAP@0.5:0.95)
  model_comparison_calibration.png      (ECE)
  model_comparison_edge_performance.png (mean + p95 latency, peak RAM)
  model_comparison_quantization_fp32.png(model size vs. accuracy, FP32 level)
  model_comparison_cv_repeatability.png (mean CV% Protocol A / B)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import TABLES_DIR, FIGURES_DIR, log


def _grouped_bar(df, label_col, value_cols, value_labels, title, ylabel, out_path, ylim=None):
    """One grouped bar chart: one group of bars per row of df, one bar per
    value column, models on the x-axis."""
    labels = df[label_col].tolist()
    n_groups = len(labels)
    n_bars = len(value_cols)
    x = np.arange(n_groups)
    width = 0.8 / max(n_bars, 1)

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * n_groups), 5.5))
    for i, (col, vlabel) in enumerate(zip(value_cols, value_labels)):
        offset = (i - (n_bars - 1) / 2) * width
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
        ax.bar(x + offset, vals, width, label=vlabel)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")


def plot_detection_comparison():
    path = TABLES_DIR / "table1_model_comparison_detection.csv"
    if not path.exists():
        log(f"{path} not found -- run detection_eval.py first. Skipping detection comparison chart.", tag="WARN")
        return
    df = pd.read_csv(path)
    if df.empty:
        log(f"{path} has no rows -- skipping detection comparison chart.", tag="WARN")
        return
    df = df.sort_values("mAP_0.5", ascending=False)
    _grouped_bar(
        df, "model_display_name",
        ["accuracy", "micro_f1", "mAP_0.5", "mAP_0.5:0.95"],
        ["Accuracy", "F1 (micro)", "mAP@0.5", "mAP@0.5:0.95"],
        "Cross-model comparison — detection accuracy (BCCD test split)",
        "Score", FIGURES_DIR / "model_comparison_detection.png", ylim=(0, 1.05),
    )


def plot_calibration_comparison():
    path = TABLES_DIR / "calibration_ece_all_models.csv"
    if not path.exists():
        log(f"{path} not found -- run calibration_eval.py first. Skipping calibration comparison chart.", tag="WARN")
        return
    df = pd.read_csv(path)
    if df.empty:
        log(f"{path} has no rows -- skipping calibration comparison chart.", tag="WARN")
        return
    df = df.sort_values("ece")
    _grouped_bar(
        df, "model_display_name", ["ece"], ["Expected Calibration Error"],
        "Cross-model comparison — calibration (lower is better)",
        "ECE", FIGURES_DIR / "model_comparison_calibration.png",
    )


def plot_edge_comparison():
    path = TABLES_DIR / "edge_performance.csv"
    if not path.exists():
        log(f"{path} not found -- run edge_performance.py first. Skipping edge-performance comparison chart.", tag="WARN")
        return
    df = pd.read_csv(path)
    if df.empty:
        log(f"{path} has no rows -- skipping edge-performance comparison chart.", tag="WARN")
        return
    label_col = "model" if "model" in df.columns else "model_key"
    df = df.sort_values("mean_latency_ms")

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.8 * len(df)), 5))
    x = np.arange(len(df))
    axes[0].bar(x - 0.2, pd.to_numeric(df["mean_latency_ms"], errors="coerce"), 0.4, label="Mean")
    axes[0].bar(x + 0.2, pd.to_numeric(df["p95_latency_ms"], errors="coerce"), 0.4, label="p95")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df[label_col], rotation=30, ha="right")
    axes[0].set_ylabel("Latency (ms/image)")
    axes[0].set_title("Inference latency")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    if "peak_ram_mb" in df.columns and df["peak_ram_mb"].notna().any():
        axes[1].bar(x, pd.to_numeric(df["peak_ram_mb"], errors="coerce"), color="#9b59b6")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df[label_col], rotation=30, ha="right")
        axes[1].set_ylabel("Peak RAM (MB)")
        axes[1].set_title("Peak memory")
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "peak_ram_mb unavailable\n(install psutil)", ha="center", va="center")

    fig.suptitle("Cross-model comparison — edge performance (CPU)")
    fig.tight_layout()
    out_path = FIGURES_DIR / "model_comparison_edge_performance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")


def plot_quantization_fp32_comparison():
    path = TABLES_DIR / "quantization_tradeoff.csv"
    if not path.exists():
        log(f"{path} not found -- run quantization_bench.py first. Skipping quantization comparison chart.", tag="WARN")
        return
    df = pd.read_csv(path)
    if df.empty or "quantization_level" not in df.columns:
        log(f"{path} has no usable rows -- skipping quantization comparison chart.", tag="WARN")
        return
    fp32 = df[(df["quantization_level"] == "fp32") & (~df.get("model_size_mb").isna())]
    if fp32.empty:
        log("No successful FP32 rows in quantization_tradeoff.csv -- skipping quantization comparison chart.", tag="WARN")
        return
    fp32 = fp32.sort_values("model_size_mb")

    fig, ax1 = plt.subplots(figsize=(max(7, 1.6 * len(fp32)), 5.5))
    x = np.arange(len(fp32))
    label_col = "model_display_name" if "model_display_name" in fp32.columns else "model"
    ax1.bar(x, pd.to_numeric(fp32["model_size_mb"], errors="coerce"), color="#2ecc71", label="Model size (MB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(fp32[label_col], rotation=30, ha="right")
    ax1.set_ylabel("Model size (MB)")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, pd.to_numeric(fp32["accuracy"], errors="coerce"), "o-", color="#e74c3c", label="Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Cross-model comparison — FP32 size vs. accuracy trade-off")
    fig.tight_layout()
    out_path = FIGURES_DIR / "model_comparison_quantization_fp32.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")


def plot_cv_repeatability_comparison():
    path_a = TABLES_DIR / "cv_protocol_a_field_to_field.csv"
    path_b = TABLES_DIR / "cv_protocol_b_algorithmic_summary.csv"
    if not path_a.exists() and not path_b.exists():
        log(f"Neither {path_a} nor {path_b} found -- run cv_repeatability.py first. "
            "Skipping CV%% comparison chart.", tag="WARN")
        return

    rows = []
    if path_a.exists():
        df_a = pd.read_csv(path_a)
        if not df_a.empty and "cv_percent" in df_a.columns:
            agg = df_a.groupby("model", as_index=False)["cv_percent"].mean()
            agg["protocol"] = "A (field-to-field)"
            rows.append(agg)
    if path_b.exists():
        df_b = pd.read_csv(path_b)
        col = "mean_cv_percent" if "mean_cv_percent" in df_b.columns else "cv_percent"
        if not df_b.empty and col in df_b.columns:
            agg = df_b.groupby("model", as_index=False)[col].mean().rename(columns={col: "cv_percent"})
            agg["protocol"] = "B (algorithmic)"
            rows.append(agg)

    if not rows:
        log("No usable CV%% rows found -- skipping CV%% comparison chart.", tag="WARN")
        return

    combined = pd.concat(rows, ignore_index=True)
    pivot = combined.pivot_table(index="model", columns="protocol", values="cv_percent")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(pivot)), 5.5))
    x = np.arange(len(pivot))
    n_bars = len(pivot.columns)
    width = 0.8 / max(n_bars, 1)
    for i, col in enumerate(pivot.columns):
        offset = (i - (n_bars - 1) / 2) * width
        ax.bar(x + offset, pivot[col].to_numpy(), width, label=str(col))
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel("Mean CV%")
    ax.set_title("Cross-model comparison — reproducibility (lower is better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = FIGURES_DIR / "model_comparison_cv_repeatability.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")


def main():
    from common import list_available_models

    models = list_available_models()
    if not models:
        log("No trained models found. Train at least one model first.", tag="WARN")
        return

    log(f"Building cross-model comparison charts for: {', '.join(k for k, _ in models)}")
    plot_detection_comparison()
    plot_calibration_comparison()
    plot_edge_comparison()
    plot_quantization_fp32_comparison()
    plot_cv_repeatability_comparison()
    log("Cross-model comparison charts complete. See output/figures/model_comparison_*.png "
        "(any chart skipped above just means its upstream step/combined table hasn't been run yet).")


if __name__ == "__main__":
    main()
