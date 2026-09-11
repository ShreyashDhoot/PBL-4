#!/usr/bin/env python3
"""
cv_repeatability.py
====================
Notes reference: Part 1 ("How to compute CV% for our system").

Two protocols, both implemented:

Protocol A — field-to-field CV (directly comparable to the manual/machine
CV% numbers in the teammate's comparison table, since those also reflect
field-to-field / draw-to-draw variation).
  - Requires 10-20 *different* microscope fields from the *same slide*.
  - This script looks for such a set at BCC_FIELD_IMAGES_DIR (a folder of
    images from one slide). If that directory is not supplied/found, it
    builds the closest reproducible proxy from data actually shipped with
    this project: the 72-image biomed set is treated as a stand-in
    "multi-field" sample (documented explicitly in the output so numbers are
    never silently mislabeled as true single-slide CV%). Swap in real
    multi-field captures of one slide before writing the final paper number.

Protocol B — pure algorithmic/model reproducibility under simulated
re-imaging (rotation +/-2-3deg, brightness/contrast jitter, +/-2-3px crop
shift), run 10-20 times per image on a sample of BCCD test images. This is
the "always deterministic on identical input" problem the notes flag, and is
explicitly labelled as a *different, narrower* quantity than Protocol A.

Outputs:
  output/tables/cv_protocol_a_field_to_field.csv
  output/tables/cv_protocol_b_algorithmic.csv
  output/figures/cv_protocol_a_bars.png
  output/figures/cv_protocol_b_bars.png
  output/reports/cv_repeatability.json
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import CELL_TYPES, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, RANDOM_SEED, log
from voc_data import build_bccd_records, split_bccd_records, build_72_records
from models_io import load_ssdlite_detector, perturb_image
from stats_toolkit import coefficient_of_variation, bootstrap_ci

N_REPEATS = 15  # within the notes' recommended 10-20 range
SCORE_THR = 0.35


def protocol_a_field_to_field(detector):
    """Field-to-field CV using whichever multi-field source is available.

    Priority:
      1. BCC_FIELD_IMAGES_DIR env var -> folder of N images, all genuinely
         different fields captured from ONE physical slide. Use this for the
         paper's real number.
      2. Fallback: the 72-image biomed set, clearly labelled as a PROXY
         (different slides/sources, not one slide) so it is never mistaken
         for the real Protocol A number. This still demonstrates the full
         analysis pipeline end-to-end on data shipped with the project.
    """
    field_dir = os.environ.get("BCC_FIELD_IMAGES_DIR")
    is_proxy = False

    if field_dir and Path(field_dir).exists():
        image_paths = sorted(Path(field_dir).glob("*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        source_desc = f"real multi-field capture set: {field_dir}"
    else:
        is_proxy = True
        records = build_72_records()
        image_paths = [Path(r["image_path"]) for r in records[:N_REPEATS]]
        source_desc = (
            "PROXY (no BCC_FIELD_IMAGES_DIR supplied): using the first "
            f"{len(image_paths)} images from the 72-image biomed set as a stand-in "
            "for repeated fields of one slide. These are NOT verified to be the same "
            "physical slide/specimen -- replace with a real 10-20-field capture of one "
            "slide before reporting this as the paper's Protocol A number."
        )

    log(f"Protocol A source: {source_desc}")

    counts_per_type = {c: [] for c in CELL_TYPES}
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        counts = detector.count_cells(img, score_thr=SCORE_THR)
        for c in CELL_TYPES:
            counts_per_type[c].append(counts[c])

    rows = []
    for c in CELL_TYPES:
        vals = np.array(counts_per_type[c], dtype=float)
        cv = coefficient_of_variation(vals)
        ci = bootstrap_ci(vals, coefficient_of_variation, n_boot=2000, seed=RANDOM_SEED) if len(vals) > 2 else {"ci_lower": np.nan, "ci_upper": np.nan}
        rows.append({
            "cell_type": c,
            "n_fields": len(vals),
            "mean_count": vals.mean() if len(vals) else np.nan,
            "sd_count": vals.std(ddof=1) if len(vals) > 1 else np.nan,
            "cv_percent": cv,
            "cv_bootstrap_ci_lower": ci["ci_lower"],
            "cv_bootstrap_ci_upper": ci["ci_upper"],
            "is_proxy_data": is_proxy,
        })

    return pd.DataFrame(rows), source_desc, counts_per_type


def protocol_b_algorithmic(detector, n_images=5, n_repeats=N_REPEATS, severity="mild"):
    """Re-run inference n_repeats times per image with small simulated
    re-imaging perturbations (not identical-tensor reruns, which would
    trivially give CV%=0)."""
    bccd = build_bccd_records()
    _, _, test = split_bccd_records(bccd)
    sample = test[:n_images]

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    per_image_counts = {}

    for rec in sample:
        img = Image.open(rec["image_path"]).convert("RGB")
        counts_per_type = {c: [] for c in CELL_TYPES}
        for _ in range(n_repeats):
            perturbed = perturb_image(img, rng, severity=severity)
            counts = detector.count_cells(perturbed, score_thr=SCORE_THR)
            for c in CELL_TYPES:
                counts_per_type[c].append(counts[c])
        per_image_counts[rec["image_id"]] = counts_per_type

        for c in CELL_TYPES:
            vals = np.array(counts_per_type[c], dtype=float)
            cv = coefficient_of_variation(vals)
            rows.append({
                "image_id": rec["image_id"],
                "cell_type": c,
                "n_repeats": n_repeats,
                "perturbation_severity": severity,
                "mean_count": vals.mean(),
                "sd_count": vals.std(ddof=1) if len(vals) > 1 else np.nan,
                "cv_percent": cv,
            })

    df = pd.DataFrame(rows)
    # Aggregate across images per cell type for a single headline number too.
    agg = df.groupby("cell_type").agg(
        mean_cv_percent=("cv_percent", "mean"),
        median_cv_percent=("cv_percent", "median"),
        n_images=("image_id", "nunique"),
    ).reset_index()
    return df, agg, per_image_counts


def plot_cv_bars(df, value_col, title, out_path, group_col="cell_type"):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    grouped = df.groupby(group_col)[value_col].mean().reindex(CELL_TYPES)
    ax.bar(grouped.index, grouped.values, color=["#e74c3c", "#3498db", "#2ecc71"])
    ax.set_ylabel("CV%")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(grouped.values):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")


def main():
    detector = load_ssdlite_detector()

    log("Running Protocol A (field-to-field CV%) ...")
    df_a, source_desc_a, _ = protocol_a_field_to_field(detector)
    df_a.to_csv(TABLES_DIR / "cv_protocol_a_field_to_field.csv", index=False)
    plot_cv_bars(df_a, "cv_percent", "Protocol A: Field-to-field CV% (specimen-level repeatability)",
                 FIGURES_DIR / "cv_protocol_a_bars.png")

    log("Running Protocol B (algorithmic reproducibility CV%) ...")
    df_b, agg_b, _ = protocol_b_algorithmic(detector)
    df_b.to_csv(TABLES_DIR / "cv_protocol_b_algorithmic.csv", index=False)
    agg_b.to_csv(TABLES_DIR / "cv_protocol_b_algorithmic_summary.csv", index=False)
    plot_cv_bars(agg_b.rename(columns={"mean_cv_percent": "cv_percent"}), "cv_percent",
                 "Protocol B: Algorithmic reproducibility CV% (simulated re-capture)",
                 FIGURES_DIR / "cv_protocol_b_bars.png")

    report = {
        "protocol_a": {
            "label": "Field-to-field CV% (specimen-level repeatability) -- directly comparable to manual/machine CV% in the comparison table",
            "data_source_note": source_desc_a,
            "results": df_a.to_dict(orient="records"),
        },
        "protocol_b": {
            "label": "Algorithmic/model reproducibility under simulated re-imaging -- NOT the same quantity as Protocol A or the machine CV% numbers",
            "aggregate_results": agg_b.to_dict(orient="records"),
        },
    }
    with open(REPORTS_DIR / "cv_repeatability.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    log("CV% repeatability analysis complete. See output/tables and output/figures.")


if __name__ == "__main__":
    main()
