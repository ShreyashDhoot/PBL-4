#!/usr/bin/env python3
"""
calibration_eval.py
====================
Notes reference: next-step item #3 ("Confidence score + consistency of
confidence ... the formal metric is Expected Calibration Error (ECE), usually
shown alongside a reliability diagram").

For every predicted box (score >= a low inclusion threshold, so we see the
full confidence range) on the BCCD test split, we label it "correct" if it
IoU-matches a ground-truth box of the same class at IoU >= 0.5, and
"incorrect" otherwise (duplicate/unmatched detections). We then bin by
predicted confidence and compare average confidence to empirical accuracy
in each bin -- the standard ECE recipe.

Outputs:
  output/tables/calibration_bins.csv
  output/figures/reliability_diagram.png
  output/reports/calibration_eval.json
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import CELL_TYPES, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, log
from voc_data import build_bccd_records, split_bccd_records
from detection_metrics import greedy_match
from models_io import load_ssdlite_detector
from stats_toolkit import expected_calibration_error

LOW_SCORE_INCLUSION = 0.05  # include low-confidence detections too, so bins near 0 are populated
IOU_THR = 0.5


def collect_confidence_correctness(detector, records):
    all_conf, all_correct = [], []
    for i, rec in enumerate(records, 1):
        img = Image.open(rec["image_path"]).convert("RGB")
        boxes, labels, scores = detector.predict(img, score_thr=LOW_SCORE_INCLUSION)
        pred_labels_0idx = labels.astype(int) - 1
        gt_labels_0idx = np.asarray(rec["labels"], dtype=int) - 1
        gt_boxes = np.asarray(rec["boxes"])

        matches, fp_idx, _ = greedy_match(boxes, pred_labels_0idx, scores, gt_boxes, gt_labels_0idx, iou_thr=IOU_THR)
        matched_pred_idx = {p for p, _ in matches}

        for p_idx in range(len(boxes)):
            all_conf.append(float(scores[p_idx]))
            all_correct.append(1.0 if p_idx in matched_pred_idx else 0.0)

        if i % 25 == 0:
            log(f"  processed {i}/{len(records)} images")

    return np.array(all_conf), np.array(all_correct)


def plot_reliability_diagram(bin_rows, ece, out_path):
    df = pd.DataFrame(bin_rows)
    centers = (df["bin_lower"] + df["bin_upper"]) / 2

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.bar(centers, df["accuracy"], width=0.09, alpha=0.7, edgecolor="black", label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Predicted confidence (binned)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability diagram (ECE = {ece:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")


def main():
    detector = load_ssdlite_detector()
    bccd = build_bccd_records()
    _, _, test = split_bccd_records(bccd)

    log(f"Collecting confidences/correctness on {len(test)} BCCD test images ...")
    conf, correct = collect_confidence_correctness(detector, test)

    ece, bin_rows = expected_calibration_error(conf, correct, n_bins=10)
    pd.DataFrame(bin_rows).to_csv(TABLES_DIR / "calibration_bins.csv", index=False)
    plot_reliability_diagram(bin_rows, ece, FIGURES_DIR / "reliability_diagram.png")

    report = {
        "ece": ece,
        "n_predictions": len(conf),
        "n_bins": 10,
        "score_inclusion_threshold": LOW_SCORE_INCLUSION,
        "iou_match_threshold": IOU_THR,
        "bin_table": bin_rows,
    }
    with open(REPORTS_DIR / "calibration_eval.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    log(f"Expected Calibration Error (ECE) = {ece:.4f} over {len(conf)} detections.")
    log("Calibration evaluation complete. See output/tables and output/figures.")


if __name__ == "__main__":
    main()
