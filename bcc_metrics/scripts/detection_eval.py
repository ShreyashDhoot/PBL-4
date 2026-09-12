#!/usr/bin/env python3
"""
detection_eval.py
==================
Notes reference: Section 3 ("Held-out generalization check") and Section 7
(Accuracy/Precision/Recall/F1, mAP@0.5, mAP@0.5:0.95, confusion matrix).

Runs the trained SSDLite detector on:
  (a) the BCCD *test* split (same 15% held out by train_bccd_ssdlite_detection.py)
  (b) the 72-image biomed-student set (independent, out-of-distribution)

and reports, for each, per-class and overall:
  - Precision / Recall / F1 (+ confusion matrix)
  - Accuracy with a Wilson 95% CI (small-n proportion, as the notes require)
  - mAP@0.5 and mAP@0.5:0.95

The BCCD-vs-72-set gap this produces is the "accuracy usually drops on our
own captured images" evidence the notes ask for (Section 3, bullet 1).

Outputs:
  output/tables/detection_metrics_bccd_test.csv
  output/tables/detection_metrics_72set.csv
  output/tables/confusion_matrix_bccd_test.csv
  output/tables/confusion_matrix_72set.csv
  output/tables/map_summary.csv
  output/figures/confusion_matrix_bccd_test.png
  output/figures/confusion_matrix_72set.png
  output/figures/generalization_gap.png
  output/reports/detection_eval.json
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import (
    CELL_TYPES,
    DEFAULT_SCORE_THR,
    DEFAULT_MATCH_IOU,
    TABLES_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    log,
)
from voc_data import build_bccd_records, split_bccd_records, build_72_records
from detection_metrics import (
    accumulate_confusion,
    prf1_from_confusion,
    wilson_score_interval,
    compute_map,
)
from models_io import load_detector


def run_predictions(detector, records, score_thr=0.0):
    """Predict on every record; returns predictions[] and ground_truths[]
    lists using 0-indexed labels into CELL_TYPES (i.e. label - 1 vs the
    detector's 1-indexed __background__-first scheme)."""
    predictions, ground_truths = [], []
    for i, rec in enumerate(records, 1):
        img = Image.open(rec["image_path"]).convert("RGB")
        boxes, labels, scores = detector.predict(img, score_thr=score_thr)
        pred_labels_0idx = labels.astype(int) - 1  # drop background offset
        gt_labels_0idx = np.asarray(rec["labels"], dtype=int) - 1 if rec["labels"] and rec["labels"][0] >= 1 else np.asarray(rec["labels"], dtype=int)
        # rec['labels'] from voc_data are already 1-indexed (DET_CLASSES incl. background at 0)
        gt_labels_0idx = np.asarray(rec["labels"], dtype=int) - 1
        predictions.append((boxes, pred_labels_0idx, scores))
        ground_truths.append((np.asarray(rec["boxes"]), gt_labels_0idx))
        if i % 25 == 0:
            log(f"  predicted {i}/{len(records)} images")
    return predictions, ground_truths


def evaluate_split(name, detector, records, score_thr=DEFAULT_SCORE_THR, iou_thr=DEFAULT_MATCH_IOU):
    log(f"Evaluating '{name}' ({len(records)} images) ...")
    predictions, ground_truths = run_predictions(detector, records, score_thr=0.0)  # keep all, threshold in metrics

    cm = accumulate_confusion(predictions, ground_truths, CELL_TYPES, iou_thr=iou_thr, score_thr=score_thr)
    prf1 = prf1_from_confusion(cm, CELL_TYPES)

    acc_point, acc_lo, acc_hi = wilson_score_interval(prf1["n_matched_tp"], prf1["n_total_events"])

    map_result = compute_map(predictions, ground_truths, CELL_TYPES, iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2).tolist())

    per_class_df = pd.DataFrame(prf1["per_class"])
    per_class_df["dataset"] = name

    summary = {
        "dataset": name,
        "n_images": len(records),
        "score_threshold": score_thr,
        "iou_threshold": iou_thr,
        "accuracy": acc_point,
        "accuracy_wilson_ci_lower": acc_lo,
        "accuracy_wilson_ci_upper": acc_hi,
        "micro_precision": prf1["micro"]["precision"],
        "micro_recall": prf1["micro"]["recall"],
        "micro_f1": prf1["micro"]["f1"],
        "macro_precision": prf1["macro"]["precision"],
        "macro_recall": prf1["macro"]["recall"],
        "macro_f1": prf1["macro"]["f1"],
        "mAP_0.5": map_result["mAP_0.5"],
        "mAP_0.5:0.95": map_result["mAP_0.5:0.95"],
    }
    for c in CELL_TYPES:
        summary[f"AP_0.5:0.95_{c}"] = map_result["per_class_mAP_0.5:0.95"][c]

    return per_class_df, cm, summary, map_result


def plot_confusion_matrix(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted")
    ax.set_title(title, fontsize=11)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")


def plot_generalization_gap(summary_bccd, summary_72, out_path):
    metrics = ["accuracy", "micro_precision", "micro_recall", "micro_f1", "mAP_0.5"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "mAP@0.5"]
    bccd_vals = [summary_bccd[m] for m in metrics]
    seventytwo_vals = [summary_72[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, bccd_vals, width, label="BCCD test split (in-distribution)")
    ax.bar(x + width / 2, seventytwo_vals, width, label="72-image biomed set (OOD)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Generalization gap: BCCD test split vs. held-out biomed images")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")


def main():
    from common import list_available_models

    bccd = build_bccd_records()
    _, _, bccd_test = split_bccd_records(bccd)
    seventytwo = build_72_records()

    models = list_available_models()
    if not models:
        log("No trained models found (checked MODEL_REGISTRY in common.py). Train at least one "
            "model first, e.g. `python pbl-4/train_bccd_ssdlite_detection.py`.", tag="WARN")
        return

    all_summaries = []
    full_report = {}
    labels_with_bg = CELL_TYPES + ["background/missed"]

    for model_key, display_name in models:
        log(f"=== Evaluating model: {display_name} ({model_key}) ===")
        try:
            detector = load_detector(model_key)
        except Exception as e:
            log(f"Could not load '{model_key}': {e} -- skipping.", tag="WARN")
            continue

        try:
            per_class_bccd, cm_bccd, summary_bccd, _ = evaluate_split(f"{model_key}__bccd_test", detector, bccd_test)
            per_class_72, cm_72, summary_72, _ = evaluate_split(f"{model_key}__72set_ood", detector, seventytwo)
        except Exception as e:
            log(f"Evaluation failed for '{model_key}': {e} -- skipping.", tag="WARN")
            continue

        plot_confusion_matrix(cm_bccd, labels_with_bg, f"Confusion matrix — BCCD test split — {display_name}",
                               FIGURES_DIR / f"confusion_matrix_bccd_test_{model_key}.png")
        plot_confusion_matrix(cm_72, labels_with_bg, f"Confusion matrix — 72-image biomed set — {display_name}",
                               FIGURES_DIR / f"confusion_matrix_72set_{model_key}.png")
        plot_generalization_gap(summary_bccd, summary_72, FIGURES_DIR / f"generalization_gap_{model_key}.png")

        per_class_bccd.to_csv(TABLES_DIR / f"detection_metrics_bccd_test_{model_key}.csv", index=False)
        per_class_72.to_csv(TABLES_DIR / f"detection_metrics_72set_{model_key}.csv", index=False)
        pd.DataFrame(cm_bccd, index=labels_with_bg, columns=labels_with_bg).to_csv(TABLES_DIR / f"confusion_matrix_bccd_test_{model_key}.csv")
        pd.DataFrame(cm_72, index=labels_with_bg, columns=labels_with_bg).to_csv(TABLES_DIR / f"confusion_matrix_72set_{model_key}.csv")

        summary_bccd["model"] = summary_72["model"] = model_key
        summary_bccd["model_display_name"] = summary_72["model_display_name"] = display_name
        all_summaries.extend([summary_bccd, summary_72])

        full_report[model_key] = {
            "display_name": display_name,
            "bccd_test": summary_bccd,
            "72set_ood": summary_72,
            "generalization_gap": {
                "accuracy_drop": summary_bccd["accuracy"] - summary_72["accuracy"],
                "mAP_0.5_drop": (summary_bccd["mAP_0.5"] or 0) - (summary_72["mAP_0.5"] or 0),
            },
        }
        log(f"{display_name}: BCCD test acc={summary_bccd['accuracy']:.3f} mAP@0.5={summary_bccd['mAP_0.5']:.3f}  |  "
            f"72-set OOD acc={summary_72['accuracy']:.3f} mAP@0.5={summary_72['mAP_0.5']:.3f}")

    if not all_summaries:
        log("No model evaluated successfully.", tag="WARN")
        return

    # Backward-compatible single-model filenames (SSDLite, if present) PLUS
    # the new combined cross-model comparison table every downstream script
    # (dataset_and_model_summary.py, the paper) should actually read from.
    map_df = pd.DataFrame(all_summaries)
    map_df.to_csv(TABLES_DIR / "map_summary.csv", index=False)

    bccd_only = map_df[map_df["dataset"].str.endswith("__bccd_test")].sort_values("mAP_0.5", ascending=False)
    bccd_only.to_csv(TABLES_DIR / "table1_model_comparison_detection.csv", index=False)

    with open(REPORTS_DIR / "detection_eval.json", "w") as f:
        json.dump(full_report, f, indent=2, default=float)

    log("=== Cross-model detection summary (BCCD test split, sorted by mAP@0.5) ===")
    for _, row in bccd_only.iterrows():
        log(f"  {row['model_display_name']:45s} acc={row['accuracy']:.3f}  mAP@0.5={row['mAP_0.5']:.3f}  "
            f"mAP@0.5:0.95={row['mAP_0.5:0.95']:.3f}")
    log("Detection evaluation complete. See output/tables and output/figures "
        "(per-model files suffixed _<model_key>, plus table1_model_comparison_detection.csv).")


if __name__ == "__main__":
    main()
