#!/usr/bin/env python3
"""
train_bccd_efficientdet_lite0_detection.py
============================================
EfficientDet-Lite0 on BCCD, via the `effdet` (rwightman) package. BiFPN
multi-scale feature fusion has a strong small-object literature track
record, per the comparison table.

Requires: pip install effdet timm

Small-object upgrades:
  * Anchor generator overridden with a lower minimum scale
    (`--anchor_scale`, default 2.5 vs. EfficientDet's stock 4.0, plus an
    extra `--min_level 2` to keep the P2/stride-4 pyramid level in the
    BiFPN, both of which push the smallest anchor tier down toward
    ~15-30px objects at 512px input -- the anchor-based analog of the
    "add a P2 layer" / "lower min_ratio" requests applied to the other
    models in this repo).
  * 512px input (`--img_size`, Lite0's native default is 320/384).
  * Long cosine-annealed training (`--epochs`, default 150) instead of a
    handful of epochs.
  * `effdet`'s loss is already a Focal Loss (RetinaNet-style) by
    construction, matching the "use focal loss" request out of the box.
  * Mosaic + platelet copy-paste augmentation (bccd_data_utils), applied
    identically to every other model here.

TFLite export: PyTorch has no first-party TFLite exporter. This script
exports ONNX (used by bcc_metrics for the cross-model comparison) and
prints the exact follow-up command to convert that ONNX graph to TFLite
via `onnx2tf` (the standard ONNX->TFLite path), rather than silently
skipping the "Lite"/TFLite half of this model's name.

Outputs (under output/efficientdet_lite0/): history_efficientdet_lite0.csv,
loss_curves_efficientdet_lite0.png, sample_predictions_efficientdet_lite0.png,
efficientdet_lite0_bccd_best.pth, efficientdet_lite0_bccd.onnx (+.meta.json),
run_report_efficientdet_lite0.{json,md}.
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import batched_nms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from bccd_data_utils import (
    CLASSES, download_bccd, build_records, split_records, class_counts, box_size_histogram,
    set_seed, mosaic_augment, platelet_copy_paste, draw_prediction_grid, write_run_report, write_onnx_meta,
)

NUM_FG_CLASSES = 3


def _require_effdet():
    try:
        import effdet  # noqa: F401
        return effdet
    except ImportError as e:
        raise RuntimeError(
            "The `effdet` package is required for this training script but is not installed "
            f"({e}). Install with:\n    pip install effdet timm\n"
        )


def build_model(img_size=512, anchor_scale=2.5, min_level=2, max_level=6, pretrained_backbone=True):
    effdet = _require_effdet()
    from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
    from effdet.efficientdet import HeadNet

    config = get_efficientdet_config('tf_efficientdet_lite0')
    config.num_classes = NUM_FG_CLASSES
    config.image_size = (img_size, img_size)
    # Small-object upgrades: lower anchor_scale + lower min_level pulls the
    # smallest anchor tier down toward ~15-30px objects at `img_size`.
    config.anchor_scale = anchor_scale
    config.min_level = min_level
    config.max_level = max_level

    net = EfficientDet(config, pretrained_backbone=pretrained_backbone)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)

    train_bench = DetBenchTrain(net, config)
    predict_bench = DetBenchPredict(net)
    return train_bench, predict_bench, config


class BCCDDatasetEffDet(Dataset):
    """effdet's DetBenchTrain expects target dict keys 'bbox' (yxyx!) and
    'cls' per image, batched via its own collate convention."""

    def __init__(self, records, img_size=512, train=False, mosaic_prob=0.5, copy_paste_prob=0.5, seed=42):
        self.records = records
        self.img_size = img_size
        self.train = train
        self.mosaic_prob = mosaic_prob if train else 0.0
        self.copy_paste_prob = copy_paste_prob if train else 0.0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.records)

    def _load_plain(self, rec):
        from PIL import Image
        img = Image.open(rec['image_path']).convert('RGB')
        boxes = np.asarray(rec['boxes'], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(rec['labels'], dtype=np.int64)
        sx, sy = self.img_size / img.width, self.img_size / img.height
        img = img.resize((self.img_size, self.img_size))
        if len(boxes):
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
        return img, boxes, labels

    def __getitem__(self, idx):
        rec = self.records[idx]
        if self.train and self.rng.random() < self.mosaic_prob:
            img, boxes, labels = mosaic_augment(self.records, idx, img_size=self.img_size, rng=self.rng)
        else:
            img, boxes, labels = self._load_plain(rec)
        if self.train and self.rng.random() < self.copy_paste_prob:
            img, boxes, labels = platelet_copy_paste(img, boxes, labels, self.records, rng=self.rng)
        if len(boxes):
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, labels = boxes[valid], labels[valid]

        img_t = TF.to_tensor(img)
        fg_labels = labels - 1  # 0-indexed foreground (RBC=0,WBC=1,Platelets=2)
        # effdet convention: bbox in [ymin, xmin, ymax, xmax], cls 1-indexed
        # foreground (0 reserved internally); effdet adds 1 to cls internally
        # via its num_classes bookkeeping, so we pass 1-indexed-foreground here.
        if len(boxes):
            bbox_yxyx = np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)
        else:
            bbox_yxyx = np.zeros((0, 4), dtype=np.float32)
        target = {
            'bbox': torch.as_tensor(bbox_yxyx, dtype=torch.float32),
            'cls': torch.as_tensor(fg_labels + 1, dtype=torch.float32),
            'img_size': torch.as_tensor([self.img_size, self.img_size], dtype=torch.float32),
            'img_scale': torch.as_tensor(1.0, dtype=torch.float32),
        }
        return img_t, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    out_targets = {
        'bbox': [t['bbox'] for t in targets],
        'cls': [t['cls'] for t in targets],
        'img_size': torch.stack([t['img_size'] for t in targets]),
        'img_scale': torch.stack([t['img_scale'] for t in targets]),
    }
    return imgs, out_targets


@torch.no_grad()
def predict_image(predict_bench, pil_img, device, img_size, score_thr=0.05):
    predict_bench.eval()
    w0, h0 = pil_img.size
    resized = pil_img.resize((img_size, img_size))
    x = TF.to_tensor(resized).unsqueeze(0).to(device)
    img_info = {
        'img_size': torch.as_tensor([[img_size, img_size]], dtype=torch.float32, device=device),
        'img_scale': torch.as_tensor([1.0], dtype=torch.float32, device=device),
    }
    out = predict_bench(x, img_info)[0].cpu().numpy()  # [N, 6]: x1,y1,x2,y2,score,cls(1-indexed fg)
    keep = out[:, 4] > score_thr
    out = out[keep]
    if len(out) == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int), np.zeros((0,))
    sx, sy = w0 / img_size, h0 / img_size
    boxes = out[:, :4].copy()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    scores = out[:, 4]
    labels = out[:, 5].astype(int)  # already 1-indexed foreground -> matches CLASSES (bg=0)
    return boxes, labels, scores


def main():
    parser = argparse.ArgumentParser(description='Train EfficientDet-Lite0 (effdet) on BCCD.')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--out_dir', default='output/efficientdet_lite0')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--anchor_scale', type=float, default=2.5, help='EfficientDet stock default is 4.0')
    parser.add_argument('--min_level', type=int, default=2, help='EfficientDet-Lite0 stock default is 3')
    parser.add_argument('--max_level', type=int, default=6)
    parser.add_argument('--mosaic_prob', type=float, default=0.5)
    parser.add_argument('--copy_paste_prob', type=float, default=0.5)
    parser.add_argument('--score_thr', type=float, default=0.35)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    _require_effdet()
    set_seed(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Device: {device}')

    t0 = time.time()
    voc_dir = download_bccd(Path(args.data_root))
    records = build_records(voc_dir)
    train_recs, val_recs, test_recs = split_records(records)
    print(f'[INFO] Split -> train:{len(train_recs)}  val:{len(val_recs)}  test:{len(test_recs)}')

    counts = class_counts(records)
    size_hist, size_labels = box_size_histogram(records)

    train_ds = BCCDDatasetEffDet(train_recs, img_size=args.img_size, train=True,
                                  mosaic_prob=args.mosaic_prob, copy_paste_prob=args.copy_paste_prob)
    val_ds = BCCDDatasetEffDet(val_recs, img_size=args.img_size, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    train_bench, predict_bench, config = build_model(
        img_size=args.img_size, anchor_scale=args.anchor_scale, min_level=args.min_level, max_level=args.max_level,
    )
    train_bench, predict_bench = train_bench.to(device), predict_bench.to(device)

    optimizer = torch.optim.AdamW(train_bench.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_iters = max(1, args.warmup_epochs)

    def lr_lambda(epoch):
        if epoch < warmup_iters:
            return (epoch + 1) / warmup_iters
        progress = (epoch - warmup_iters) / max(1, args.epochs - warmup_iters)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    best_val = float('inf')
    best_path = out_dir / 'efficientdet_lite0_bccd_best.pth'

    print(f'[INFO] Starting EfficientDet-Lite0 training for {args.epochs} epochs '
          f'(img_size={args.img_size}, anchor_scale={args.anchor_scale}, min_level={args.min_level}) ...')
    for epoch in range(1, args.epochs + 1):
        train_bench.train()
        ep_start = time.time()
        train_losses = []
        pbar = tqdm(train_dl, desc=f'Train {epoch}/{args.epochs}', leave=False)
        for images, targets in pbar:
            images = images.to(device)
            targets = {
                'bbox': [b.to(device) for b in targets['bbox']],
                'cls': [c.to(device) for c in targets['cls']],
                'img_size': targets['img_size'].to(device),
                'img_scale': targets['img_scale'].to(device),
            }
            out = train_bench(images, targets)
            loss = out['loss']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_bench.parameters(), max_norm=10.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        train_bench.eval()
        val_losses = []
        with torch.no_grad():
            for images, targets in val_dl:
                images = images.to(device)
                targets = {
                    'bbox': [b.to(device) for b in targets['bbox']],
                    'cls': [c.to(device) for c in targets['cls']],
                    'img_size': targets['img_size'].to(device),
                    'img_scale': targets['img_scale'].to(device),
                }
                out = train_bench(images, targets)
                val_losses.append(float(out['loss'].item()))

        scheduler.step()
        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
        history.append({'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
                         'train_loss_total': train_loss, 'val_loss_total': val_loss})
        print(f'[EPOCH {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} '
              f'lr={optimizer.param_groups[0]["lr"]:.2e} time={time.time()-ep_start:.1f}s')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(train_bench.model.state_dict(), best_path)
            print(f'[INFO] New best model saved -> {best_path} (val_loss={best_val:.4f})')

    train_time_sec = time.time() - t0
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / 'history_efficientdet_lite0.csv', index=False)
    try:
        import plotly.express as px
        fig = px.line(hist_df, x='epoch', y=['train_loss_total', 'val_loss_total'], markers=True,
                       title='EfficientDet-Lite0 Detection Loss Curves (BiFPN, focal loss)')
        fig.write_image(str(out_dir / 'loss_curves_efficientdet_lite0.png'))
    except Exception as e:
        print(f'[WARN] Could not render loss-curve chart: {e}')

    if best_path.exists():
        predict_bench.model.load_state_dict(torch.load(best_path, map_location=device))

    def predict_fn(pil_img):
        return predict_image(predict_bench, pil_img, device, args.img_size, score_thr=0.05)

    draw_prediction_grid(test_recs, predict_fn, out_dir / 'sample_predictions_efficientdet_lite0.png', score_thr=args.score_thr)

    onnx_path = out_dir / 'efficientdet_lite0_bccd.onnx'
    tflite_cmd = None
    try:
        predict_bench.eval()

        class _Wrapper(nn.Module):
            def __init__(self, bench, img_size):
                super().__init__()
                self.bench = bench
                self.img_size = img_size

            def forward(self, x):
                img_info = {
                    'img_size': torch.as_tensor([[self.img_size, self.img_size]] * x.shape[0], dtype=torch.float32, device=x.device),
                    'img_scale': torch.ones(x.shape[0], dtype=torch.float32, device=x.device),
                }
                out = self.bench(x, img_info)[0]  # [N,6]: x1,y1,x2,y2,score,cls
                return out[:, :4], out[:, 4:5], out[:, 5:6]

        wrapper = _Wrapper(predict_bench, args.img_size).to(device)
        dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        torch.onnx.export(wrapper, dummy, str(onnx_path), input_names=['images'],
                           output_names=['boxes', 'scores', 'labels'], opset_version=12)
        write_onnx_meta(
            onnx_path, input_size=(args.img_size, args.img_size), mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],
            letterbox=False, class_map={"0": "RBC", "1": "WBC", "2": "Platelets"},
            extra={'output_layout': 'separate_boxes_scores_labels'},
        )
        tflite_cmd = (
            f"pip install onnx2tf && onnx2tf -i {onnx_path} -o {out_dir / 'efficientdet_lite0_tflite'} "
            f"# converts the ONNX graph to a TFLite model for the 'Lite0'/on-device deployment path"
        )
        print(f'[INFO] To produce an actual .tflite artifact (this script only exports ONNX from '
              f'PyTorch, which has no first-party TFLite exporter), run:\n    {tflite_cmd}')
    except Exception as e:
        warnings.warn(f'ONNX export failed: {e}. Training outputs are still saved.')
        onnx_path = None

    write_run_report(
        out_dir, model_key='efficientdet_lite0', model_display_name='EfficientDet-Lite0 (effdet, BiFPN)',
        description=(
            "EfficientDet-Lite0 (effdet package) with a lowered anchor_scale and min_level pulling "
            "the smallest anchor tier toward ~15-30px platelet-sized objects, 512px input, focal-loss "
            "classification (native to EfficientDet/RetinaNet-style heads), cosine LR, and mosaic + "
            "platelet copy-paste augmentation."
        ),
        hyperparams=vars(args),
        dataset_stats={
            'n_train': len(train_recs), 'n_val': len(val_recs), 'n_test': len(test_recs),
            'class_counts_full_dataset': dict(counts),
            'box_size_histogram': {k: dict(v) for k, v in size_hist.items()},
        },
        timing={'train_time_sec': train_time_sec, 'train_time_min': train_time_sec / 60.0, 'epochs': args.epochs},
        metrics={'best_val_loss': best_val},
        files={
            'checkpoint': str(best_path), 'onnx': str(onnx_path) if onnx_path else None,
            'history_csv': str(out_dir / 'history_efficientdet_lite0.csv'),
            'sample_predictions': str(out_dir / 'sample_predictions_efficientdet_lite0.png'),
        },
        notes=[
            'PyTorch has no first-party TFLite exporter; a true .tflite deployment artifact requires '
            f'an extra ONNX->TFLite conversion step: {tflite_cmd}' if tflite_cmd else
            'ONNX export failed, so no TFLite conversion command is available for this run.',
            'Run bcc_metrics/run_all.py after this for cross-model-comparable mAP/P/R/F1 on the same '
            'BCCD test split and 72-image OOD set used for every other model.',
        ],
    )

    print(f'[INFO] Done. Best val loss: {best_val:.4f}. Total time: {train_time_sec/60:.1f} min.')


if __name__ == '__main__':
    main()
