#!/usr/bin/env python3
"""
robustness_eval.py
====================
Notes reference: next-step item #5 ("Obscure/recolor images to imitate dyes
... Frame it as a robustness curve -- accuracy plotted against corruption
severity (mild/moderate/severe blur, stain-color shift, obscuration) --
rather than a single before/after number. This mirrors the standard
ImageNet-C style robustness benchmark").

For each corruption type (blur/obscuration, stain-color shift) and each
severity level (none/mild/moderate/severe), we apply the corruption to every
image in the BCCD test split, run the trained detector, and measure
accuracy / F1 / mAP@0.5 -- producing the robustness curve the notes ask for.

Outputs:
  output/tables/robustness_curve.csv
  output/figures/robustness_curve_blur.png
  output/figures/robustness_curve_stain_shift.png
  output/reports/robustness_eval.json
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import CELL_TYPES, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, RANDOM_SEED, log
from voc_data import build_bccd_records, split_bccd_records
from detection_metrics import accumulate_confusion, prf1_from_confusion, compute_map
from models_io import load_detector, obscure_image, simulate_stain_color_shift

SEVERITIES = ["none", "mild", "moderate", "severe"]
SCORE_THR = 0.35
MAX_TEST_IMAGES = 60  # cap for runtime; raise for a more precise curve


def evaluate_corruption(detector, records, corruption_fn, corruption_name, severity):
    rng = np.random.default_rng(RANDOM_SEED)
    predictions, ground_truths = [], []

    for rec in records:
        img = Image.open(rec["image_path"]).convert("RGB")
        corrupted = corruption_fn(img, rng, severity=severity)
        boxes, labels, scores = detector.predict(corrupted, score_thr=0.0)
        pred_labels_0idx = labels.astype(int) - 1
        gt_labels_0idx = np.asarray(rec["labels"], dtype=int) - 1
        predictions.append((boxes, pred_labels_0idx, scores))
        ground_truths.append((np.asarray(rec["boxes"]), gt_labels_0idx))

    cm = accumulate_confusion(predictions, ground_truths, CELL_TYPES, iou_thr=0.5, score_thr=SCORE_THR)
    prf1 = prf1_from_confusion(cm, CELL_TYPES)
    map_result = compute_map(predictions, ground_truths, CELL_TYPES, iou_thresholds=[0.5])

    return {
        "corruption_type": corruption_name,
        "severity": severity,
        "accuracy": prf1["accuracy"],
        "precision_micro": prf1["micro"]["precision"],
        "recall_micro": prf1["micro"]["recall"],
        "f1_micro": prf1["micro"]["f1"],
        "mAP_0.5": map_result["mAP_0.5"],
    }


def plot_curve(df, corruption_name, out_path):
    sub = df[df["corruption_type"] == corruption_name].set_index("severity").reindex(SEVERITIES)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(SEVERITIES, sub["accuracy"], marker="o", label="Accuracy")
    ax.plot(SEVERITIES, sub["f1_micro"], marker="s", label="F1 (micro)")
    ax.plot(SEVERITIES, sub["mAP_0.5"], marker="^", label="mAP@0.5")
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Robustness curve: {corruption_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")


def main():
    from common import list_available_models

    bccd = build_bccd_records()
    _, _, test = split_bccd_records(bccd)
    test = test[:min(MAX_TEST_IMAGES, len(test))]

    models = list_available_models()
    if not models:
        log("No trained models found. Train at least one model first.", tag="WARN")
        return

    corruptions = {
        "blur_obscuration": obscure_image,
        "stain_color_shift": simulate_stain_color_shift,
    }

    all_rows = []
    full_report = {}
    for model_key, display_name in models:
        log(f"=== Robustness: {display_name} ({model_key}) on {len(test)} images x {len(SEVERITIES)} severities ===")
        try:
            detector = load_detector(model_key)
        except Exception as e:
            log(f"Could not load '{model_key}': {e} -- skipping.", tag="WARN")
            continue

        rows = []
        try:
            for name, fn in corruptions.items():
                for sev in SEVERITIES:
                    log(f"  {name} @ severity={sev}")
                    r = evaluate_corruption(detector, test, fn, name, sev)
                    r["model"] = model_key
                    r["model_display_name"] = display_name
                    rows.append(r)
        except Exception as e:
            log(f"Robustness eval failed for '{model_key}': {e} -- skipping.", tag="WARN")
            continue

        df = pd.DataFrame(rows)
        for name in corruptions:
            plot_curve(df, name, FIGURES_DIR / f"robustness_curve_{name}_{model_key}.png")
        all_rows.extend(rows)
        full_report[model_key] = {"display_name": display_name, "rows": rows}

    if not all_rows:
        log("No model evaluated successfully.", tag="WARN")
        return

    pd.DataFrame(all_rows).to_csv(TABLES_DIR / "robustness_curve.csv", index=False)
    with open(REPORTS_DIR / "robustness_eval.json", "w") as f:
        json.dump(full_report, f, indent=2, default=float)

    log("Robustness evaluation complete for all available models. See output/tables/robustness_curve.csv")


if __name__ == "__main__":
    main()
