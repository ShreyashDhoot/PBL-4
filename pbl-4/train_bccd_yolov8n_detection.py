#!/usr/bin/env python3
"""
train_bccd_yolov8n_detection.py
================================
YOLOv8n on BCCD, via Ultralytics. This is the model the paper's own
literature review flags as the standard small-object recommendation and
the model missing from the current comparison ("directly fills the YOLO
gap your paper already flags" -- see the TODO row Ultralytics adds in
bcc_metrics/scripts/dataset_and_model_summary.py::build_table1_from_measured_results()
before this script existed).

Small-object upgrades applied (mirrors the SSDLite-v2 upgrade list):
  * `yolov8n-p2.yaml` model config: Ultralytics' P2/P3/P4/P5 four-head
    variant of YOLOv8n, which adds a high-resolution P2 (stride-4)
    detection layer specifically for small objects, on top of the
    anchor-free, multi-scale head YOLOv8 already has. Falls back to plain
    `yolov8n.yaml` if the installed Ultralytics version doesn't ship the
    P2 config.
  * 512px input (vs. the 320px SSDLite baseline).
  * >>150 epochs (vs. the SSDLite baseline's 8), AdamW + cosine LR with
    warmup (Ultralytics `cos_lr=True`).
  * `fl_gamma` focal-loss hyperparameter passed through where the
    installed Ultralytics version supports it (older/newer releases
    differ; the shared trainer retries without it if rejected).
  * Ultralytics' own online mosaic/mixup/HSV augmentation (mosaic=1.0)
    PLUS an offline platelet-region copy-paste + mosaic augmentation pass
    baked directly into the YOLO-format training split on disk (see
    ultralytics_common.prepare_yolo_dataset), so the "multiply the
    effective number of platelet positives" request holds regardless of
    Ultralytics version support for its own `copy_paste` (mask-based)
    hyperparameter.

Outputs (under output/yolov8n/):
  history_yolov8n.csv, loss_curves_yolov8n.png, sample_predictions_yolov8n.png,
  yolov8n_bccd.onnx (+ .meta.json), *_ncnn_model/ (best effort),
  run_report_yolov8n.{json,md}

Usage:
    python train_bccd_yolov8n_detection.py --epochs 150 --img_size 512
"""

import argparse
from pathlib import Path

from bccd_data_utils import download_bccd, build_records, split_records, class_counts, box_size_histogram, set_seed
from ultralytics_common import prepare_yolo_dataset, train_ultralytics_model, _require_ultralytics


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8n(-P2) on BCCD.')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--out_dir', default='output/yolov8n')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--lr0', type=float, default=1e-3)
    parser.add_argument('--fl_gamma', type=float, default=1.5, help='focal-loss gamma; ignored if unsupported by installed ultralytics')
    parser.add_argument('--n_mosaic_extra', type=int, default=None)
    parser.add_argument('--n_copy_paste_extra', type=int, default=None)
    parser.add_argument('--pretrained', default='yolov8n.pt', help='COCO-pretrained weights to warm-start from, or "" to train from scratch')
    args = parser.parse_args()

    _require_ultralytics()
    set_seed(42)
    out_dir = Path(args.out_dir)

    voc_dir = download_bccd(Path(args.data_root))
    records = build_records(voc_dir)
    train_recs, val_recs, test_recs = split_records(records)
    print(f'[INFO] Split -> train:{len(train_recs)}  val:{len(val_recs)}  test:{len(test_recs)}')

    data_yaml, n_synth = prepare_yolo_dataset(
        train_recs, val_recs, test_recs, out_dir / 'yolo_dataset',
        n_mosaic_extra=args.n_mosaic_extra, n_copy_paste_extra=args.n_copy_paste_extra,
        img_size=args.img_size,
    )

    from ultralytics import YOLO

    train_ultralytics_model(
        model_key='yolov8n', model_display_name='YOLOv8n-P2 (Ultralytics)',
        description=(
            "YOLOv8n with an added P2 (stride-4) small-object detection head, 512px input, "
            "AdamW + cosine LR over many epochs, and mosaic + offline platelet copy-paste "
            "augmentation. Anchor-free, multi-scale head as per the literature recommendation."
        ),
        ultra_cls=YOLO,
        model_yaml_candidates=['yolov8n-p2.yaml', 'yolov8n.yaml'],
        pretrained_weights=(args.pretrained or None),
        data_yaml=data_yaml, out_dir=out_dir, records_for_pred_grid=test_recs,
        img_size=args.img_size, epochs=args.epochs, batch=args.batch, lr0=args.lr0,
        extra_train_kwargs={'fl_gamma': args.fl_gamma},
        dataset_stats={
            'n_train_real': len(train_recs), 'n_train_synthetic': n_synth, 'n_val': len(val_recs),
            'n_test': len(test_recs), 'class_counts_full_dataset': dict(class_counts(records)),
        },
        hyperparams_extra={'fl_gamma_requested': args.fl_gamma},
    )


if __name__ == '__main__':
    main()
