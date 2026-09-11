# BCC Metrics Suite

Implements every metric and figure requested in `Comparison_Metrics_Notes.docx`
for the "Edge-AI Hematology Screening System" (PBL-4) paper, computed from the
data and trained checkpoints that already exist in `pbl-4.zip` and
`annotations.zip` — no numbers are invented or hardcoded except the literature
table (which is transcribed verbatim from the notes, with sources).

## What this answers, mapped to the notes

| Notes section | Script | Output |
|---|---|---|
| Part 1 — CV% (Protocol A: field-to-field, Protocol B: algorithmic) | `cv_repeatability.py` | `tables/cv_protocol_{a,b}_*.csv`, `figures/cv_protocol_{a,b}_bars.png` |
| Part 2 — Bland-Altman, ICC(2,1) | `agreement_stats.py` | `tables/agreement_summary_per_celltype.csv`, `figures/bland_altman_*.png` |
| Part 3 — 72-image OOD check, inter-annotator kappa | `detection_eval.py`, `agreement_stats.py` | `figures/generalization_gap.png`, kappa in `reports/agreement_stats.json` |
| Part 6 — Pearson/Spearman/Passing-Bablok/Section-5 row | `agreement_stats.py`, `literature_comparison_table.py` | `tables/section5_comparison_row.csv`, `tables/literature_comparison_full.csv` |
| Section 7 checklist — Accuracy/P/R/F1, mAP, confusion matrix | `detection_eval.py` | `tables/detection_metrics_*.csv`, `tables/map_summary.csv`, `figures/confusion_matrix_*.png` |
| Next-step #1 — Quantization trade-off table | `quantization_bench.py` | `tables/quantization_tradeoff.csv`, `figures/quantization_tradeoff.png` |
| Next-step #3 — ECE + reliability diagram | `calibration_eval.py` | `tables/calibration_bins.csv`, `figures/reliability_diagram.png` |
| Next-step #5 — Robustness curve (blur/stain shift) | `robustness_eval.py` | `tables/robustness_curve.csv`, `figures/robustness_curve_*.png` |
| Next-step #6 — mAP, confusion matrix, CI, YOLO ablation note, power/latency | `detection_eval.py`, `edge_performance.py`, `dataset_and_model_summary.py` | see respective tables; YOLO/MobileNetV2 gaps are flagged explicitly, not fabricated |

## Directory layout expected

```
project_root/
  pbl-4/                              <- unzip pbl-4.zip here (rename PBL-4 -> pbl-4, or set BCC_PBL4_DIR)
    data/BCCD_Dataset-master/BCCD/
    output/ssdlite_bccd_best.pth
    output/efficientnet_bccd_best.pth
    train_bccd_ssdlite_detection.py
    train_efficientnet_bccd.py
  annotations/annotations/            <- unzip annotations.zip here (or set BCC_ANNOT_DIR)
    images/*.jpeg
    annotations/*.xml
  bcc_metrics/                        <- this folder
    scripts/*.py
    run_all.py
    output/                           <- created automatically
```

If your extraction paths differ, just set the environment variables instead
of moving files:

```bash
export BCC_PBL4_DIR=/path/to/pbl-4
export BCC_ANNOT_DIR=/path/to/annotations/annotations
export BCC_OUTPUT_DIR=/path/to/bcc_metrics/output
```

## Setup

```bash
cd bcc_metrics
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If the trained checkpoints are NOT already in `pbl-4/output/`, train them
first (from inside the `pbl-4` repo, unchanged from what you already have):

```bash
cd ../pbl-4
python train_bccd_ssdlite_detection.py --epochs 8
python train_efficientnet_bccd.py --epochs 6
```

## Run everything

```bash
cd bcc_metrics
python run_all.py
```

This runs every step in dependency order (dataset summary → detection eval →
CV% → agreement stats → literature table → calibration → robustness →
quantization → edge performance → final Table I) and prints a summary of
which steps succeeded. Each step also runs standalone, e.g.:

```bash
python scripts/detection_eval.py
python scripts/agreement_stats.py
```

Run a subset or skip slow steps:

```bash
python run_all.py --only detection agreement
python run_all.py --skip robustness quantization
```

## Optional inputs that improve the numbers

- **`BCC_FIELD_IMAGES_DIR`** — a folder of 10–20 images that are genuinely
  different microscope fields from ONE physical slide. Without this,
  `cv_repeatability.py` computes Protocol A on the 72-image set as an
  explicitly-labelled **proxy** (different slides/sources) so the number is
  never silently misreported as true single-slide CV%.
- **`BCC_SECOND_ANNOTATOR_DIR`** — a second annotator's independent VOC-XML
  relabeling of 15–20 of the 72 images (same filenames). Enables the
  inter-annotator Cohen's kappa the notes recommend in Part 3.
- **`BCC_ON_RASPBERRY_PI=1`** — set only when physically running on the
  target Raspberry Pi 4; this just labels the latency/quantization numbers
  correctly so dev-machine numbers are never mistaken for target-hardware
  numbers (the notes are explicit that this matters).
- **`BCC_POWER_METER_CMD`** — a shell command that prints instantaneous
  watts to stdout (from a USB power meter or INA219 current-sense HAT on
  the Pi). Without it, `edge_performance.py` reports `power_watts: null`
  rather than fabricating a number.

## Known, explicitly-flagged gaps (not silently papered over)

- **MobileNetV2 and YOLO** appear in the paper's methodology/literature
  review but no trained checkpoint or training script for either ships in
  `pbl-4.zip`. `dataset_and_model_summary.py` writes `NaN` with a note
  explaining this in `table1_model_comparison.csv`, and tells you exactly
  what to add (train and export the same way `train_efficientnet_bccd.py`
  does) or how to narrow the paper's text to match what was actually run.
- **Protocol A CV%** defaults to a proxy (see above) unless you supply real
  same-slide, multi-field images.
- **Inter-annotator kappa** is skipped (not fabricated) unless a second
  annotator's relabeling is supplied.
- **Power draw** is `null` unless a real meter command is wired up.

## Output layout

```
output/
  tables/    *.csv   — every numeric result, ready to paste into paper tables
  figures/   *.png    — every plot referenced above
  reports/   *.json / *.md — machine-readable summaries + a paper-ready
                              markdown version of the literature comparison table
```
