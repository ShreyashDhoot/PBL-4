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

## Multi-model ablation (YOLOv8n-P2, YOLO11n-P2, RT-DETR, NanoDet-Plus-style,
## EfficientDet-Lite0, RTMDet-tiny, SSDLite v2)

`pbl-4/` now ships one `train_bccd_<model>_detection.py` per model, all
trained/validated/tested on the IDENTICAL BCCD 70/15/15 split (seed=42) via
`pbl-4/bccd_data_utils.py`, plus the original baseline SSDLite. Train
everything with:

```
cd pbl-4
python train_all_models.py               # every model
python train_all_models.py --quick        # 2-epoch smoke test of the whole pipeline
python train_all_models.py --only yolov8n nanodet_plus
```

Every model also gets the five "small-object" upgrades applied to the
original SSDLite baseline in `train_bccd_ssdlite_v2_detection.py` (lower
anchor min-scale / an added P2 head, 512px input, many epochs with a
cosine LR schedule, focal-loss classification, and mosaic + platelet
copy-paste augmentation) -- see that script's docstring for the exact
per-model mapping (anchor-based vs. anchor-free models implement the "make
anchors/heads see 15-30px objects" request differently).

Every `train_bccd_*.py` script also exports ONNX (+ a `.meta.json`
sidecar) and writes a standardized `run_report_<model>.{json,md}` under
`pbl-4/output/<model>/` -- hyperparameters, dataset stats, timing, and
metrics in one place, meant to be handed directly to whoever (human or
LLM) writes the paper's Methods/Results section for that model, without
re-deriving anything from raw logs.

Once at least one model is trained, `python run_all.py` (no flags) picks
up every trained model automatically via `MODEL_REGISTRY` in
`scripts/common.py` and runs every evaluation step (detection, CV%,
agreement, calibration, robustness, quantization, edge performance) once
per model, writing per-model files suffixed `_<model_key>` plus a combined
cross-model comparison table per step
(`table1_model_comparison_detection.csv` is the headline one). Pass
`--include-training` to `run_all.py` to train everything first in the same
command.

## Known, explicitly-flagged gaps (not silently papered over)

- **MobileNetV2** appears in the paper's methodology/literature review but
  no trained checkpoint or training script ships in `pbl-4.zip`.
  `dataset_and_model_summary.py` writes `NaN` with a note explaining this
  in `table1_model_comparison.csv` (YOLO no longer has this gap -- see the
  multi-model ablation section above).
- **RTMDet-tiny** trains and evaluates natively inside its own script, but
  needs a separate `mmdeploy` ONNX export to participate in the
  ONNX-based cross-model comparison scripts above; until then it's
  skipped there with a clear log line (see its `run_report`).
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
