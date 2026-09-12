#!/usr/bin/env python3
"""
train_bccd_yolo11n_detection.py
================================
YOLO11n on BCCD, via Ultralytics -- the newer sibling of YOLOv8n, same
family of upgrades (anchor-free, multi-scale head; C3k2/C2PSA blocks
generally give a small accuracy/latency edge over YOLOv8n at the 'n'
scale). See train_bccd_yolov8n_detection.py for the detailed rationale;
this script is intentionally near-identical so the two are a clean,
directly comparable pair in the ablation.

Small-object upgrades: `yolo11n-p2.yaml` (P2 head) model config, 512px
input, >>150 epochs with AdamW + cosine LR, focal-loss gamma where
supported, Ultralytics' native augmentation plus the same offline
mosaic + platelet copy-paste pass used for every model in this repo.

Usage:
    python train_bccd_yolo11n_detection.py --epochs 150 --img_size 512
"""

import argparse
from pathlib import Path

from bccd_data_utils import download_bccd, build_records, split_records, class_counts, set_seed
from ultralytics_common import prepare_yolo_dataset, train_ultralytics_model, _require_ultralytics


def main():
    parser = argparse.ArgumentParser(description='Train YOLO11n(-P2) on BCCD.')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--out_dir', default='output/yolo11n')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--lr0', type=float, default=1e-3)
    parser.add_argument('--fl_gamma', type=float, default=1.5)
    parser.add_argument('--n_mosaic_extra', type=int, default=None)
    parser.add_argument('--n_copy_paste_extra', type=int, default=None)
    parser.add_argument('--pretrained', default='yolo11n.pt')
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
        model_key='yolo11n', model_display_name='YOLO11n-P2 (Ultralytics)',
        description=(
            "YOLO11n with an added P2 (stride-4) small-object detection head, 512px input, "
            "AdamW + cosine LR over many epochs, and mosaic + offline platelet copy-paste "
            "augmentation. Directly comparable pair to the YOLOv8n-P2 run in this ablation."
        ),
        ultra_cls=YOLO,
        model_yaml_candidates=['yolo11n-p2.yaml', 'yolo11n.yaml'],
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
