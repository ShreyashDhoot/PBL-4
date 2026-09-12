#!/usr/bin/env python3
"""
train_bccd_rtdetr_nano_detection.py
====================================
RT-DETR on BCCD, via Ultralytics. RT-DETR is a modern anchor-free,
NMS-free real-time transformer detector; it is included in the comparison
table's "modern anchor-free detectors, strong small-object AP" row.

IMPORTANT Pi 4 caveat (also flagged in the comparison table this script
fills in: "Runs on Pi 4 but check latency budget"): Ultralytics only
ships 'rtdetr-l' and 'rtdetr-x' scale configs -- there is no official
RT-DETR "nano"/"n" scale the way there is for YOLOv8/YOLO11. This script
tries a genuinely small config first (in case a future Ultralytics release
adds one) and falls back to 'rtdetr-l.yaml', the smallest available scale,
while making the caveat explicit in the run report and CLI banner so the
paper doesn't overstate this model's edge-deployment readiness relative to
YOLOv8n-P2/YOLO11n-P2/NanoDet-Plus. bcc_metrics/edge_performance.py's
measured latency number is the actual evidence for whether it fits the Pi
4's latency budget -- read that table before recommending this model for
deployment, not this script's docstring.

512px input, many epochs, AdamW + cosine LR; RT-DETR's Ultralytics
implementation does not use mosaic (transformer detectors are typically
trained without it), so the offline platelet copy-paste pass carries more
of the small-object augmentation weight for this model.

Usage:
    python train_bccd_rtdetr_nano_detection.py --epochs 150 --img_size 512
"""

import argparse
from pathlib import Path

from bccd_data_utils import download_bccd, build_records, split_records, class_counts, set_seed
from ultralytics_common import prepare_yolo_dataset, train_ultralytics_model, _require_ultralytics


def main():
    parser = argparse.ArgumentParser(description='Train RT-DETR (smallest available scale) on BCCD.')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--out_dir', default='output/rtdetr_nano')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=8, help='RT-DETR is heavier than YOLOn; lower default batch')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--lr0', type=float, default=1e-4)
    parser.add_argument('--n_mosaic_extra', type=int, default=None)
    parser.add_argument('--n_copy_paste_extra', type=int, default=None)
    parser.add_argument('--pretrained', default='rtdetr-l.pt')
    args = parser.parse_args()

    _require_ultralytics()
    set_seed(42)
    out_dir = Path(args.out_dir)

    print('[WARN] Ultralytics does not ship an official RT-DETR "nano" scale (only l/x). '
          'This script uses the smallest scale available and documents that explicitly in '
          'run_report_rtdetr_nano.json -- check output/edge_performance.csv before treating this '
          'as Pi-4-latency-equivalent to YOLOv8n-P2/YOLO11n-P2/NanoDet-Plus.')

    voc_dir = download_bccd(Path(args.data_root))
    records = build_records(voc_dir)
    train_recs, val_recs, test_recs = split_records(records)
    print(f'[INFO] Split -> train:{len(train_recs)}  val:{len(val_recs)}  test:{len(test_recs)}')

    data_yaml, n_synth = prepare_yolo_dataset(
        train_recs, val_recs, test_recs, out_dir / 'yolo_dataset',
        n_mosaic_extra=args.n_mosaic_extra, n_copy_paste_extra=args.n_copy_paste_extra,
        img_size=args.img_size,
    )

    from ultralytics import RTDETR

    train_ultralytics_model(
        model_key='rtdetr_nano', model_display_name='RT-DETR (smallest available Ultralytics scale)',
        description=(
            "Ultralytics RT-DETR, NMS-free transformer detector, smallest scale Ultralytics ships "
            "(no true 'nano' config exists upstream -- see docstring/notes for the Pi-4-latency caveat). "
            "512px input, AdamW + cosine LR, offline platelet copy-paste augmentation."
        ),
        ultra_cls=RTDETR,
        model_yaml_candidates=['rtdetr-n.yaml', 'rtdetr-nano.yaml', 'rtdetr-l.yaml'],
        pretrained_weights=(args.pretrained or None),
        data_yaml=data_yaml, out_dir=out_dir, records_for_pred_grid=test_recs,
        img_size=args.img_size, epochs=args.epochs, batch=args.batch, lr0=args.lr0,
        extra_train_kwargs={'mosaic': 0.0},  # RT-DETR: Ultralytics disables mosaic by convention
        dataset_stats={
            'n_train_real': len(train_recs), 'n_train_synthetic': n_synth, 'n_val': len(val_recs),
            'n_test': len(test_recs), 'class_counts_full_dataset': dict(class_counts(records)),
        },
        hyperparams_extra={
            'no_official_nano_scale_caveat': (
                "Ultralytics ships only 'rtdetr-l'/'rtdetr-x'; there is no official RT-DETR-nano. "
                "Latency numbers for this run should NOT be assumed comparable to the other "
                "edge-targeted models without checking bcc_metrics/output/tables/edge_performance.csv."
            ),
        },
    )


if __name__ == '__main__':
    main()
