#!/usr/bin/env python3
"""
agreement_stats.py
===================
Notes reference: Part 2 ("method-comparison / agreement statistics", 'no
gold standard') and Part 6 ("Computing the same numbers for our system",
Steps 1-6), which is what actually fills in the "Our system" row of the
Section 5 literature comparison table.

Step 1: Build the paired dataset (image_id, cell_type, system_count,
        manual_count) from the 72 biomed-labelled images -- system counts
        come from running the trained SSDLite detector on the same images
        the biomed student independently labelled.
Step 2: Pearson r / R^2 (+ Spearman as a robustness check, notes suggest
        this for skewed counts like platelets).
Step 3: Passing-Bablok regression (slope/intercept + 95% CI).
Step 4: Bland-Altman bias & 95% limits of agreement (+ bootstrap CIs).
Step 5: Sample size caveat, stated explicitly in the output (n vs the
        40+/CLSI EP09c convention the notes cite).
Step 6: Assembles the "Our system" row in the same r / regression /
        Bland-Altman language as literature Section 5, ready to drop
        straight into the paper's comparison table.

Also computes ICC(2,1) per cell type (Part 2) and, if a second annotator's
relabeling of a subset is available (BCC_SECOND_ANNOTATOR_DIR env var),
inter-annotator Cohen's kappa (Part 3, "flag the single-annotator
limitation, and fix a subset of it").

Outputs:
  output/tables/paired_system_manual_counts.csv
  output/tables/agreement_summary_per_celltype.csv
  output/tables/section5_comparison_row.csv
  output/figures/bland_altman_<CellType>.png
  output/figures/scatter_agreement_<CellType>.png
  output/reports/agreement_stats.json
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import CELL_TYPES, TABLES_DIR, FIGURES_DIR, REPORTS_DIR, RANDOM_SEED, log
from voc_data import build_paired_manual_dataset, build_72_records, per_image_cell_counts
from models_io import load_ssdlite_detector
from stats_toolkit import (
    bland_altman,
    icc_2_1,
    passing_bablok,
    pearson_and_spearman,
    bootstrap_ci_paired,
    cohens_kappa,
)

SCORE_THR = 0.35
CLSI_MIN_N = 40  # CLSI EP09c convention cited in the notes (Step 5)


def build_paired_system_manual(detector):
    """Adds system_<CellType> columns to the manual-count rows from voc_data,
    by running the detector on the same 72 images."""
    rows = build_paired_manual_dataset()
    for row in rows:
        img = Image.open(row["image_path"]).convert("RGB")
        sys_counts = detector.count_cells(img, score_thr=SCORE_THR)
        for c in CELL_TYPES:
            row[f"system_{c}"] = sys_counts[c]
    return pd.DataFrame(rows)


def compute_agreement_per_celltype(df):
    rows = []
    figures = []

    for c in CELL_TYPES:
        sys_vals = df[f"system_{c}"].values.astype(float)
        man_vals = df[f"manual_{c}"].values.astype(float)

        ba = bland_altman(sys_vals, man_vals)
        ba_bias_ci = bootstrap_ci_paired(sys_vals, man_vals,
                                          lambda x, y: (x - y).mean(),
                                          n_boot=2000, seed=RANDOM_SEED)

        def _loa_upper(x, y):
            d = x - y
            return d.mean() + 1.96 * d.std(ddof=1) if len(d) > 1 else np.nan

        def _loa_lower(x, y):
            d = x - y
            return d.mean() - 1.96 * d.std(ddof=1) if len(d) > 1 else np.nan

        loa_u_ci = bootstrap_ci_paired(sys_vals, man_vals, _loa_upper, n_boot=2000, seed=RANDOM_SEED)
        loa_l_ci = bootstrap_ci_paired(sys_vals, man_vals, _loa_lower, n_boot=2000, seed=RANDOM_SEED)

        corr = pearson_and_spearman(sys_vals, man_vals)
        pb = passing_bablok(man_vals, sys_vals)  # x=reference(manual), y=new method(system) per CLSI convention

        ratings = np.stack([sys_vals, man_vals], axis=1)
        icc_result = icc_2_1(ratings)

        rows.append({
            "cell_type": c,
            "n_pairs": len(sys_vals),
            "pearson_r": corr["pearson_r"],
            "r_squared": corr["r_squared"],
            "spearman_rho": corr["spearman_rho"],
            "bland_altman_bias": ba["bias"],
            "bias_bootstrap_ci_lower": ba_bias_ci["ci_lower"],
            "bias_bootstrap_ci_upper": ba_bias_ci["ci_upper"],
            "loa_lower": ba["loa_lower"],
            "loa_lower_ci_lower": loa_l_ci["ci_lower"],
            "loa_lower_ci_upper": loa_l_ci["ci_upper"],
            "loa_upper": ba["loa_upper"],
            "loa_upper_ci_lower": loa_u_ci["ci_lower"],
            "loa_upper_ci_upper": loa_u_ci["ci_upper"],
            "pb_slope": pb["slope"],
            "pb_slope_ci_lower": pb["slope_ci_lower"],
            "pb_slope_ci_upper": pb["slope_ci_upper"],
            "pb_intercept": pb["intercept"],
            "pb_intercept_ci_lower": pb["intercept_ci_lower"],
            "pb_intercept_ci_upper": pb["intercept_ci_upper"],
            "pb_proportional_bias_flag": bool(pb["slope_ci_lower"] > 1 or pb["slope_ci_upper"] < 1),
            "pb_constant_offset_flag": bool(pb["intercept_ci_lower"] > 0 or pb["intercept_ci_upper"] < 0),
            "icc_2_1": icc_result["icc"],
            "icc_ci_lower": icc_result["ci95_lower"],
            "icc_ci_upper": icc_result["ci95_upper"],
            "icc_method": icc_result["method"],
        })

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(ba["means"], ba["diffs"], alpha=0.7, edgecolor="k")
        ax.axhline(ba["bias"], color="black", linestyle="-", label=f"Bias = {ba['bias']:.2f}")
        ax.axhline(ba["loa_upper"], color="red", linestyle="--", label=f"+1.96 SD = {ba['loa_upper']:.2f}")
        ax.axhline(ba["loa_lower"], color="red", linestyle="--", label=f"-1.96 SD = {ba['loa_lower']:.2f}")
        ax.set_xlabel("Mean of system & manual count")
        ax.set_ylabel("System − Manual count")
        ax.set_title(f"Bland-Altman: {c} (n={len(sys_vals)})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = FIGURES_DIR / f"bland_altman_{c}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figures.append(str(p))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(man_vals, sys_vals, alpha=0.7, edgecolor="k")
        lims = [min(man_vals.min(), sys_vals.min()) - 1, max(man_vals.max(), sys_vals.max()) + 1]
        ax.plot(lims, lims, "k--", alpha=0.5, label="y = x (perfect agreement)")
        if not np.isnan(pb["slope"]):
            xs = np.array(lims)
            ax.plot(xs, pb["slope"] * xs + pb["intercept"], color="green", label="Passing-Bablok fit")
        ax.set_xlabel("Manual (biomed student) count")
        ax.set_ylabel("System (SSDLite) count")
        ax.set_title(f"System vs. manual count: {c}  (r={corr['pearson_r']:.3f})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = FIGURES_DIR / f"scatter_agreement_{c}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figures.append(str(p))

    return pd.DataFrame(rows), figures


def build_section5_row(agreement_df, n_pairs):
    """Formats the 'Our system' row in the same style as the literature
    table in the notes' Section 5/6, ready to paste into the paper."""
    parts_r, parts_ba, parts_pb = [], [], []
    for _, row in agreement_df.iterrows():
        parts_r.append(f"r={row['pearson_r']:.3f} ({row['cell_type']})")
        parts_ba.append(f"{row['cell_type']}: bias={row['bland_altman_bias']:.2f}, "
                         f"LoA=[{row['loa_lower']:.2f}, {row['loa_upper']:.2f}]")
        parts_pb.append(f"{row['cell_type']}: slope={row['pb_slope']:.3f} "
                         f"[{row['pb_slope_ci_lower']:.3f}, {row['pb_slope_ci_upper']:.3f}], "
                         f"intercept={row['pb_intercept']:.3f} "
                         f"[{row['pb_intercept_ci_lower']:.3f}, {row['pb_intercept_ci_upper']:.3f}]")

    row_text = {
        "System": "Our system (SSDLite / EfficientNet-B0 on Raspberry Pi)",
        "Compared against": f"Biomed-student manual counts, {n_pairs}-image set",
        "Correlation (r)": "; ".join(parts_r),
        "Bland-Altman bias / LoA": "; ".join(parts_ba),
        "Passing-Bablok slope/intercept": "; ".join(parts_pb),
        "Source": "This study",
        "Sample-size caveat": (
            f"n={n_pairs} paired images, below the CLSI EP09c-style convention of "
            f"~{CLSI_MIN_N}+ paired samples cited in the literature review; report with "
            "confidence intervals rather than as a fully-powered clinical validation."
        ),
    }
    return pd.DataFrame([row_text])


def inter_annotator_kappa_if_available():
    """Optional: if BCC_SECOND_ANNOTATOR_DIR is set and contains VOC XML
    re-labels of a subset of the 72 images by a second annotator, compute
    Cohen's kappa for per-cell-type presence and a Bland-Altman/ICC pass on
    counts too (Part 3: 'get a second person to independently re-label
    15-20 of the 72')."""
    import os
    from pathlib import Path
    second_dir = os.environ.get("BCC_SECOND_ANNOTATOR_DIR")
    if not second_dir or not Path(second_dir).exists():
        log("No BCC_SECOND_ANNOTATOR_DIR set -- skipping inter-annotator kappa "
            "(see notes Part 3: recommend collecting a second annotator's relabel "
            "of 15-20 of the 72 images to report this).")
        return None

    from voc_data import parse_voc_xml
    second_files = sorted(Path(second_dir).glob("*.xml"))
    if not second_files:
        log(f"BCC_SECOND_ANNOTATOR_DIR ({second_dir}) has no .xml files -- skipping.")
        return None

    from common import SEVENTYTWO_IMAGES_DIR
    rows_a, rows_b = [], []
    for f in second_files:
        image_id = f.stem
        orig_xml = list(Path(second_dir).glob(f"{image_id}.xml"))
        rec_b = parse_voc_xml(f, SEVENTYTWO_IMAGES_DIR)

        from common import SEVENTYTWO_ANNOT_DIR
        orig_path = SEVENTYTWO_ANNOT_DIR / f"{image_id}.xml"
        if not orig_path.exists():
            continue
        rec_a = parse_voc_xml(orig_path, SEVENTYTWO_IMAGES_DIR)

        counts_a = per_image_cell_counts(rec_a)
        counts_b = per_image_cell_counts(rec_b)
        for c in CELL_TYPES:
            present_a = 1 if counts_a[c] > 0 else 0
            present_b = 1 if counts_b[c] > 0 else 0
            rows_a.append(present_a)
            rows_b.append(present_b)

    if not rows_a:
        log("No overlapping image_ids between annotators -- skipping kappa.")
        return None

    kappa = cohens_kappa(rows_a, rows_b, labels=[0, 1])
    result = {"n_images_relabelled": len(second_files), "cohens_kappa_cell_presence": kappa}
    log(f"Inter-annotator Cohen's kappa (cell-type presence): {kappa:.3f} over {len(second_files)} images")
    return result


def main():
    detector = load_ssdlite_detector()

    log("Building paired system-vs-manual dataset (72-image biomed set) ...")
    paired_df = build_paired_system_manual(detector)
    paired_df.to_csv(TABLES_DIR / "paired_system_manual_counts.csv", index=False)
    log(f"Paired dataset: {len(paired_df)} images. "
        f"CLSI EP09c-style convention wants ~{CLSI_MIN_N}+ paired samples; "
        f"we have {len(paired_df)} (see notes Part 6, Step 5).")

    agreement_df, figs = compute_agreement_per_celltype(paired_df)
    agreement_df.to_csv(TABLES_DIR / "agreement_summary_per_celltype.csv", index=False)

    section5_row = build_section5_row(agreement_df, len(paired_df))
    section5_row.to_csv(TABLES_DIR / "section5_comparison_row.csv", index=False)

    kappa_result = inter_annotator_kappa_if_available()

    report = {
        "n_paired_images": len(paired_df),
        "clsi_ep09c_min_n_cited_in_literature": CLSI_MIN_N,
        "agreement_per_celltype": agreement_df.to_dict(orient="records"),
        "section5_row": section5_row.to_dict(orient="records")[0],
        "inter_annotator_kappa": kappa_result,
    }
    with open(REPORTS_DIR / "agreement_stats.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    log("=== Agreement / method-comparison summary ===")
    for _, row in agreement_df.iterrows():
        log(f"{row['cell_type']}: r={row['pearson_r']:.3f}  bias={row['bland_altman_bias']:.2f}  "
            f"LoA=[{row['loa_lower']:.2f}, {row['loa_upper']:.2f}]  ICC={row['icc_2_1']:.3f}")
    log("Agreement statistics complete. See output/tables and output/figures.")


if __name__ == "__main__":
    main()
