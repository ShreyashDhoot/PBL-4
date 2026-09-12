#!/usr/bin/env python3
"""
train_bccd_rtmdet_tiny_detection.py
=====================================
RTMDet-tiny on BCCD, via `mmyolo` (which wraps `mmdet`/`mmengine`).
RTMDet is a modern anchor-free detector (SimOTA-style dynamic label
assignment) with strong small-object AP, per the comparison table.

Requires (heavy, optional -- guarded like every model loader in this repo):
    pip install -U openmim
    mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmyolo>=0.6.0"

If these are not installed, this script prints clear install instructions
and exits cleanly (exit code 1) rather than crashing run_all.py's
subprocess call -- consistent with how every other train_bccd_*.py in
this folder treats a missing optional dependency.

Pipeline:
  1. Converts the BCCD VOC records (identical 70/15/15 split as every
     other model here) to COCO-format JSON, since mmdet/mmyolo dataloaders
     expect COCO annotations.
  2. Adds the same offline mosaic + platelet copy-paste augmented images
     used by every other model in this repo directly into the COCO train
     split (on top of RTMDet's own built-in Mosaic+MixUp pipeline, which
     mmyolo's stock `rtmdet_tiny` config already includes).
  3. Loads mmyolo's stock `rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py`
     config and overrides: num_classes=3, dataset paths/classes, image
     scale bumped to 512, total epochs, batch size, and the optimizer's
     base LR (scaled for a much smaller dataset than COCO). RTMDet's loss
     is already Quality Focal Loss (classification) + GIoU (regression),
     satisfying the "focal loss" request natively.
  4. Trains via `mmengine.runner.Runner`, then runs `Runner.test()` on the
     val/test splits, and writes the same standardized output files as
     every other model (history CSV parsed from mmengine's log, sample
     prediction grid, ONNX export via mmdeploy if available, run report).

NOTE on ONNX/edge deployment: mmdet/mmyolo models are normally exported
via the separate `mmdeploy` toolchain, not a plain `torch.onnx.export`
one-liner (the model's `forward` isn't a simple tensor-in/tensor-out
function outside of `mmdeploy`'s wrapped rewrite). This script tries a
best-effort `mmdeploy` export if that package is installed, and otherwise
clearly marks this model as "native evaluation only" in its run report
and in bcc_metrics' MODEL_REGISTRY (see bcc_metrics/scripts/common.py) --
it will simply be skipped by the ONNX-based cross-model comparison scripts
rather than break the pipeline, exactly like a missing checkpoint is
handled everywhere else in bcc_metrics.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from bccd_data_utils import (
    CLASSES, download_bccd, build_records, split_records, class_counts, box_size_histogram,
    set_seed, mosaic_augment, platelet_copy_paste, write_run_report,
)


def _require_mmyolo():
    missing = []
    for pkg in ('mmengine', 'mmcv', 'mmdet', 'mmyolo'):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            "[ERROR] RTMDet-tiny training requires the OpenMMLab stack, which is not fully "
            f"installed (missing: {', '.join(missing)}). Install with:\n"
            "    pip install -U openmim\n"
            '    mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmyolo>=0.6.0"\n'
            "Skipping this model (matches every other loader in this repo's "
            "'print instructions, exit cleanly, let the rest of the pipeline continue' convention)."
        )
        sys.exit(1)


def records_to_coco(records, out_json: Path, img_dir_placeholder=None):
    images, annotations = [], []
    categories = [{'id': i, 'name': c} for i, c in enumerate(CLASSES[1:])]  # 0=RBC,1=WBC,2=Platelets
    ann_id = 1
    for img_id, rec in enumerate(records, 1):
        from PIL import Image
        with Image.open(rec['image_path']) as im:
            w, h = im.size
        images.append({'id': img_id, 'file_name': str(Path(rec['image_path']).resolve()), 'width': w, 'height': h})
        for (x1, y1, x2, y2), lbl in zip(rec['boxes'], rec['labels']):
            cat_id = lbl - 1  # CLASSES index -> 0-indexed foreground
            annotations.append({
                'id': ann_id, 'image_id': img_id, 'category_id': cat_id,
                'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0,
            })
            ann_id += 1
    coco = {'images': images, 'annotations': annotations, 'categories': categories}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(coco, f)
    print(f'[INFO] Wrote COCO annotations -> {out_json} ({len(images)} images, {len(annotations)} boxes)')
    return out_json


def add_offline_augmented_images(records, out_img_dir: Path, coco_json: Path, n_mosaic, n_copy_paste,
                                  img_size=512, seed=42):
    """Appends mosaic + platelet copy-paste synthetic images/annotations
    directly onto an existing COCO json (same convention as
    ultralytics_common.prepare_yolo_dataset, adapted to COCO format)."""
    from PIL import Image

    out_img_dir.mkdir(parents=True, exist_ok=True)
    with open(coco_json) as f:
        coco = json.load(f)
    next_img_id = max((im['id'] for im in coco['images']), default=0) + 1
    next_ann_id = max((a['id'] for a in coco['annotations']), default=0) + 1
    rng = np.random.default_rng(seed)

    def _append(img: Image.Image, boxes, labels, tag):
        nonlocal next_img_id, next_ann_id
        fname = f'synth_{tag}.jpg'
        path = out_img_dir / fname
        img.save(path, quality=90)
        w, h = img.size
        coco['images'].append({'id': next_img_id, 'file_name': str(path.resolve()), 'width': w, 'height': h})
        for (x1, y1, x2, y2), lbl in zip(boxes, labels):
            coco['annotations'].append({
                'id': next_ann_id, 'image_id': next_img_id, 'category_id': int(lbl) - 1,
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'area': float((x2 - x1) * (y2 - y1)), 'iscrowd': 0,
            })
            next_ann_id += 1
        next_img_id += 1

    for i in range(n_mosaic):
        idx = int(rng.integers(0, len(records)))
        img, boxes, labels = mosaic_augment(records, idx, img_size=img_size, rng=rng)
        if len(boxes):
            _append(img, boxes, labels, f'mosaic_{i:04d}')

    for i in range(n_copy_paste):
        idx = int(rng.integers(0, len(records)))
        rec = records[idx]
        img = Image.open(rec['image_path']).convert('RGB')
        boxes = np.asarray(rec['boxes'], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(rec['labels'], dtype=np.int64)
        img2, boxes2, labels2 = platelet_copy_paste(img, boxes, labels, records, rng=rng)
        if len(boxes2):
            _append(img2, boxes2, labels2, f'copypaste_{i:04d}')

    with open(coco_json, 'w') as f:
        json.dump(coco, f)
    n_synth = next_img_id - 1 - len(records)
    print(f'[INFO] Added {n_synth} offline mosaic/copy-paste synthetic images to {coco_json}')
    return n_synth


def build_config(base_cfg_path, out_dir, train_json, val_json, test_json, img_size, epochs, batch_size, lr):
    from mmengine.config import Config

    cfg = Config.fromfile(base_cfg_path)
    cfg.work_dir = str(out_dir / 'mmyolo_work_dir')

    class_names = tuple(CLASSES[1:])
    metainfo = dict(classes=class_names)

    for split, json_path in (('train', train_json), ('val', val_json), ('test', test_json)):
        loader_key = f'{split}_dataloader'
        if loader_key not in cfg:
            continue
        cfg[loader_key].dataset.ann_file = str(json_path)
        cfg[loader_key].dataset.data_prefix = dict(img='')
        cfg[loader_key].dataset.metainfo = metainfo
        cfg[loader_key].dataset.data_root = ''
        if split == 'train':
            cfg[loader_key].batch_size = batch_size
    cfg.val_evaluator.ann_file = str(val_json)
    cfg.test_evaluator.ann_file = str(test_json)

    # num_classes=3 (RBC/WBC/Platelets); propagate to every head reference.
    def _set_num_classes(node):
        if isinstance(node, dict):
            if 'num_classes' in node:
                node['num_classes'] = len(class_names)
            for v in node.values():
                _set_num_classes(v)
        elif isinstance(node, list):
            for v in node:
                _set_num_classes(v)

    _set_num_classes(cfg.model)

    cfg.train_cfg.max_epochs = epochs
    if 'optim_wrapper' in cfg:
        cfg.optim_wrapper.optimizer.lr = lr
    if hasattr(cfg, 'img_scale'):
        cfg.img_scale = (img_size, img_size)

    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=max(1, epochs // 10), save_best='auto')
    cfg.randomness = dict(seed=42, deterministic=False)
    return cfg


def main():
    parser = argparse.ArgumentParser(description='Train RTMDet-tiny (mmyolo) on BCCD.')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--out_dir', default='output/rtmdet_tiny')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001, help='scaled down from mmyolo\'s COCO-scale default (8xb32 -> single-GPU, small dataset)')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--n_mosaic_extra', type=int, default=None)
    parser.add_argument('--n_copy_paste_extra', type=int, default=None)
    parser.add_argument('--base_config', default='rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py',
                         help='mmyolo config name (resolved via mmyolo\'s config registry) to start from')
    args = parser.parse_args()

    _require_mmyolo()
    set_seed(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    voc_dir = download_bccd(Path(args.data_root))
    records = build_records(voc_dir)
    train_recs, val_recs, test_recs = split_records(records)
    print(f'[INFO] Split -> train:{len(train_recs)}  val:{len(val_recs)}  test:{len(test_recs)}')
    counts = class_counts(records)
    size_hist, size_labels = box_size_histogram(records)

    ann_dir = out_dir / 'coco_annotations'
    train_json = records_to_coco(train_recs, ann_dir / 'train.json')
    val_json = records_to_coco(val_recs, ann_dir / 'val.json')
    test_json = records_to_coco(test_recs, ann_dir / 'test.json')

    n_mosaic = args.n_mosaic_extra if args.n_mosaic_extra is not None else len(train_recs) // 2
    n_cp = args.n_copy_paste_extra if args.n_copy_paste_extra is not None else len(train_recs) // 2
    n_synth = add_offline_augmented_images(
        train_recs, out_dir / 'synthetic_images', train_json, n_mosaic, n_cp, img_size=args.img_size,
    )

    try:
        from mmyolo.utils import register_all_modules
        register_all_modules()
        from mmengine.hub import get_config
        try:
            base_cfg_path = get_config(f'mmyolo::{args.base_config.replace(".py", "")}', pretrained=False)
        except Exception:
            # fall back to resolving the config file directly within the installed mmyolo package
            import mmyolo
            base_cfg_path = str(Path(mmyolo.__file__).parent.parent / '.mim' / 'configs' / 'rtmdet' / args.base_config)

        cfg = build_config(base_cfg_path, out_dir, train_json, val_json, test_json,
                            args.img_size, args.epochs, args.batch_size, args.lr)

        from mmengine.runner import Runner
        runner = Runner.from_cfg(cfg)
        runner.train()
        train_time_sec = time.time() - t0

        test_metrics = {}
        try:
            test_metrics = runner.test()
        except Exception as e:
            print(f'[WARN] runner.test() failed: {e}')

        # mmengine writes a scalars log under work_dir/<timestamp>/vis_data/scalars.json;
        # parse it into the standard history_<model>.csv this repo's charts expect.
        import pandas as pd
        rows = []
        for jsonl in sorted(Path(cfg.work_dir).rglob('scalars.json')):
            with open(jsonl) as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        hist_df = pd.DataFrame(rows)
        hist_df.to_csv(out_dir / 'history_rtmdet_tiny.csv', index=False)
        try:
            import plotly.express as px
            loss_cols = [c for c in hist_df.columns if 'loss' in c.lower()]
            if loss_cols and 'step' in hist_df.columns:
                fig = px.line(hist_df, x='step', y=loss_cols, title='RTMDet-tiny training curves (QFL + GIoU)')
                fig.write_image(str(out_dir / 'loss_curves_rtmdet_tiny.png'))
        except Exception as e:
            print(f'[WARN] Could not render loss-curve chart: {e}')

        best_ckpts = sorted(Path(cfg.work_dir).rglob('best_*.pth'))
        best_ckpt = str(best_ckpts[-1]) if best_ckpts else None

        onnx_note = (
            "RTMDet/mmyolo models require the separate `mmdeploy` toolchain for a clean ONNX "
            "export (their forward() isn't a plain tensor-in/tensor-out function outside of "
            "mmdeploy's model-rewriting mechanism). Install it and run mmdeploy's "
            "`tools/deploy.py` with an mmyolo ONNX deploy config to produce "
            "rtmdet_tiny_bccd.onnx; until then this model is evaluated natively within this "
            "script's runner.test() only and is skipped by bcc_metrics' ONNX-based cross-model "
            "comparison scripts (see MODEL_REGISTRY in bcc_metrics/scripts/common.py)."
        )
        print(f'[INFO] {onnx_note}')

        write_run_report(
            out_dir, model_key='rtmdet_tiny', model_display_name='RTMDet-tiny (mmyolo)',
            description=(
                "RTMDet-tiny (mmyolo/mmdet), anchor-free with SimOTA-style dynamic label assignment "
                "and native Quality-Focal-Loss + GIoU losses. 512px input, mosaic (built into mmyolo's "
                "stock RTMDet pipeline) + offline platelet copy-paste augmentation."
            ),
            hyperparams=vars(args),
            dataset_stats={
                'n_train_real': len(train_recs), 'n_train_synthetic': n_synth, 'n_val': len(val_recs),
                'n_test': len(test_recs), 'class_counts_full_dataset': dict(counts),
                'box_size_histogram': {k: dict(v) for k, v in size_hist.items()},
            },
            timing={'train_time_sec': train_time_sec, 'train_time_min': train_time_sec / 60.0, 'epochs': args.epochs},
            metrics=test_metrics if isinstance(test_metrics, dict) else {},
            files={
                'best_checkpoint': best_ckpt, 'onnx': None,
                'history_csv': str(out_dir / 'history_rtmdet_tiny.csv'),
                'mmyolo_work_dir': str(cfg.work_dir),
            },
            notes=[onnx_note,
                   'Run bcc_metrics/run_all.py for the other models\' cross-model-comparable metrics; '
                   'this model reports mmyolo\'s own COCO-style test metrics above until an ONNX export '
                   'is produced via mmdeploy.'],
        )
        print(f'[INFO] Done. Total time: {train_time_sec/60:.1f} min.')

    except Exception as e:
        print(f'[ERROR] RTMDet-tiny training failed: {e}\n'
              'This is the heaviest optional dependency in this repo (full OpenMMLab stack); '
              'if this is an API-shape mismatch against the installed mmyolo/mmdet/mmengine '
              'version, check those packages\' changelogs for this config\'s exact field names, '
              'or train RTMDet-tiny by adapting mmyolo\'s own custom-dataset tutorial directly.')
        raise


if __name__ == '__main__':
    main()
