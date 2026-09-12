#!/usr/bin/env python3
"""
train_bccd_ssdlite_v2_detection.py
===================================
"Small-object-optimized" variant of train_bccd_ssdlite_detection.py, built
to directly address the small-object gap the baseline model has on
Platelets (15-30px objects, the smallest and rarest class in BCCD). This is
NOT a new architecture -- it is the same SSDLite / MobileNetV3-Large
backbone, with five concrete changes, each one requested explicitly:

  1. Anchor generator overridden with a much lower `min_ratio` (default
     0.03 instead of torchvision's stock 0.2), so the smallest anchor tier
     is sized for ~15-30px objects at the model's input resolution instead
     of ~64px+.
  2. Input resolution bumped 320 -> 512 (`--img_size`, default 512):
     torchvision's ssdlite320_mobilenet_v3_large() hardcodes 320x320
     internally and does not expose `size` as an overridable kwarg, so this
     script reconstructs the SSD model from its public building blocks
     (mobilenet_v3_large backbone + SSDLiteFeatureExtractorMobileNet +
     SSDLiteHead + DefaultBoxGenerator + SSD) to allow a custom size.
  3. Default epochs raised from 8 to 60, with an AdamW + cosine-annealing
     LR schedule and a short linear warmup, instead of a flat LR for a
     handful of epochs.
  4. Classification loss replaced with sigmoid focal loss (Lin et al. 2017,
     via torchvision.ops.sigmoid_focal_loss) instead of the stock SSD hard-
     negative-mined cross-entropy, so the many tiny/hard platelet positives
     aren't outweighed by easy background anchors.
  5. Mosaic augmentation + platelet-region copy-paste augmentation
     (see bccd_data_utils.py) applied during training, multiplying the
     effective number of platelet positives seen per epoch.

Produces the same family of outputs as the baseline script (history CSV,
loss-curve/class-count/split-size charts, sample-prediction grid, ONNX
export) plus a standardized run_report_ssdlite_v2.{json,md} under
output/ssdlite_v2/, so this run can be compared apples-to-apples against
every other model in the ablation and against the original baseline.

Usage:
    python train_bccd_ssdlite_v2_detection.py --epochs 60 --img_size 512
"""

import argparse
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from bccd_data_utils import (
    CLASSES, download_bccd, build_records, split_records, class_counts, box_size_histogram,
    set_seed, mosaic_augment, platelet_copy_paste, draw_prediction_grid, write_run_report,
    write_onnx_meta, collect_environment_info,
)

NUM_CLASSES = len(CLASSES)  # incl. background


# ----------------------------------------------------------------------------
# 1 + 2. Custom SSDLite model at arbitrary input size with a lower min_ratio
#         anchor generator. Mirrors torchvision's internal
#         ssdlite320_mobilenet_v3_large() builder, which hardcodes size and
#         min_ratio/max_ratio and does not expose them as kwargs.
# ----------------------------------------------------------------------------
def build_model_v2(num_classes=NUM_CLASSES, img_size=512, min_ratio=0.03, max_ratio=0.8,
                    pretrained_backbone=True, focal_alpha=0.25, focal_gamma=2.0,
                    trainable_backbone_layers=6):
    try:
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        from torchvision.models.detection.ssd import SSD
        from torchvision.models.detection.ssdlite import SSDLiteHead, _mobilenet_extractor
        from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
        from torchvision.models.detection import _utils as det_utils
        from torchvision.ops.misc import Conv2dNormActivation
        from functools import partial
        import torch.nn as nn
    except ImportError as e:
        raise RuntimeError(
            "Could not import the torchvision internal SSD/SSDLite building blocks "
            f"needed to construct a custom-resolution model ({e}). This script relies "
            "on torchvision>=0.13's torchvision.models.detection.ssdlite / ssd / "
            "anchor_utils modules; pin torchvision to a 0.13-0.18-range release if a "
            "newer version has moved these internals."
        )

    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    backbone = mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None,
        reduced_tail=True,
        norm_layer=norm_layer,
    )
    backbone = _mobilenet_extractor(backbone, trainable_backbone_layers, norm_layer)

    size = (img_size, img_size)
    # Small-object upgrade #1: much lower min_ratio than torchvision's stock
    # 0.2 so the smallest anchor tier targets ~15-30px objects at `img_size`.
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=min_ratio, max_ratio=max_ratio)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    if len(out_channels) != len(anchor_generator.aspect_ratios):
        raise RuntimeError(
            f"Backbone produced {len(out_channels)} feature maps but the anchor "
            f"generator expects {len(anchor_generator.aspect_ratios)} -- torchvision's "
            "internal feature-map count for this backbone/size combination may differ "
            "from what this script assumes; check the installed torchvision version."
        )

    head = SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer)

    class FocalSSD(SSD):
        """SSD with sigmoid focal-loss classification instead of stock
        hard-negative-mined cross-entropy (small-object upgrade #4)."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.focal_alpha = focal_alpha
            self.focal_gamma = focal_gamma

        def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
            from torchvision.ops import sigmoid_focal_loss

            bbox_regression = head_outputs["bbox_regression"]
            cls_logits = head_outputs["cls_logits"]
            num_classes = cls_logits.size(-1)

            num_foreground = 0
            bbox_loss = []
            cls_targets = []
            for (targets_per_image, bbox_regression_per_image, cls_logits_per_image,
                 anchors_per_image, matched_idxs_per_image) in zip(
                    targets, bbox_regression, cls_logits, anchors, matched_idxs):
                fg_idx = torch.where(matched_idxs_per_image >= 0)[0]
                fg_matched_idx = matched_idxs_per_image[fg_idx]
                num_foreground += fg_matched_idx.numel()

                matched_gt_boxes = targets_per_image["boxes"][fg_matched_idx]
                bbox_reg_fg = bbox_regression_per_image[fg_idx, :]
                anchors_fg = anchors_per_image[fg_idx, :]
                target_regression = self.box_coder.encode_single(matched_gt_boxes, anchors_fg)
                bbox_loss.append(F.smooth_l1_loss(bbox_reg_fg, target_regression, reduction="sum"))

                gt_classes_target = torch.zeros(
                    (cls_logits_per_image.size(0),), dtype=targets_per_image["labels"].dtype,
                    device=targets_per_image["labels"].device,
                )
                gt_classes_target[fg_idx] = targets_per_image["labels"][fg_matched_idx]
                cls_targets.append(gt_classes_target)

            bbox_loss_t = torch.stack(bbox_loss)
            cls_targets_t = torch.stack(cls_targets).long()  # [B, A]

            one_hot = torch.zeros((*cls_targets_t.shape, num_classes), device=cls_logits.device, dtype=cls_logits.dtype)
            one_hot.scatter_(2, cls_targets_t.unsqueeze(-1), 1.0)
            one_hot[..., 0] = 0.0  # background is the "all-zero" target for sigmoid focal loss

            cls_loss = sigmoid_focal_loss(
                cls_logits, one_hot, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="sum"
            )

            N = max(1, num_foreground)
            return {
                "bbox_regression": bbox_loss_t.sum() / N,
                "classification": cls_loss / N,
            }

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    model = FocalSSD(backbone, anchor_generator, size, num_classes, head=head, **defaults)
    return model


# ----------------------------------------------------------------------------
# 5. Dataset with mosaic + platelet copy-paste
# ----------------------------------------------------------------------------
class BCCDDatasetV2(Dataset):
    def __init__(self, records, img_size=512, train=False, mosaic_prob=0.5, copy_paste_prob=0.5,
                 copy_paste_pool=None, seed=42):
        self.records = records
        self.img_size = img_size
        self.train = train
        self.mosaic_prob = mosaic_prob if train else 0.0
        self.copy_paste_prob = copy_paste_prob if train else 0.0
        self.copy_paste_pool = copy_paste_pool or records
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.records)

    def _load_plain(self, rec):
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
            img, boxes, labels = platelet_copy_paste(img, boxes, labels, self.copy_paste_pool, rng=self.rng)

        if self.train and self.rng.random() < 0.5 and len(boxes):
            img = TF.hflip(img)
            w = img.width
            new_boxes = boxes.copy()
            new_boxes[:, 0] = w - boxes[:, 2]
            new_boxes[:, 2] = w - boxes[:, 0]
            boxes = new_boxes

        # Drop any degenerate boxes produced by augmentation edge cases.
        if len(boxes):
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, labels = boxes[valid], labels[valid]

        img_t = TF.to_tensor(img)
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.as_tensor(
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else np.zeros((0,)),
                dtype=torch.float32,
            ),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),
        }
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ----------------------------------------------------------------------------
# Training / eval loop (same shape as the baseline script for easy diffing)
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    running = []
    pbar = tqdm(loader, desc=f'Train {epoch}/{total_epochs}', unit='batch', leave=False)
    for batch_idx, (images, targets) in enumerate(pbar, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Skip batches with zero total boxes (can happen right after an
        # aggressive copy-paste failure); SSD's loss requires >=1 positive.
        if sum(t['boxes'].shape[0] for t in targets) == 0:
            continue

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        row = {'loss_total': float(loss.item())}
        row.update({k: float(v.item()) for k, v in loss_dict.items()})
        running.append(row)
        pbar.set_postfix(loss=f'{loss.item():.4f}')

        if batch_idx % 20 == 0:
            print(f'  [Train E{epoch}] batch {batch_idx}/{len(loader)} | loss={loss.item():.4f}')

    return pd.DataFrame(running).mean().to_dict() if running else {'loss_total': float('nan')}


def eval_loss(model, loader, device, epoch, total_epochs):
    model.train()  # torchvision detection models only return losses in train() mode
    vals = []
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Val   {epoch}/{total_epochs}', unit='batch', leave=False)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if sum(t['boxes'].shape[0] for t in targets) == 0:
                continue
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            row = {'loss_total': float(loss.item())}
            row.update({k: float(v.item()) for k, v in loss_dict.items()})
            vals.append(row)
            pbar.set_postfix(loss=f'{loss.item():.4f}')
    avg = pd.DataFrame(vals).mean().to_dict() if vals else {'loss_total': float('nan')}
    print(f'  [Val   E{epoch}] avg_loss={avg["loss_total"]:.4f}')
    return avg


def export_onnx(model, out_path, device, img_size):
    class _Wrapper(torch.nn.Module):
        def __init__(self, det):
            super().__init__()
            self.det = det

        def forward(self, x):
            preds = self.det(list(x))
            return preds[0]['boxes'], preds[0]['scores'].unsqueeze(1), preds[0]['labels'].float().unsqueeze(1)

    wrapper = _Wrapper(model.eval()).to(device)
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    torch.onnx.export(
        wrapper, dummy, str(out_path),
        input_names=['images'], output_names=['boxes', 'scores', 'labels'],
        dynamic_axes={'images': {0: 'batch'}}, opset_version=12,
    )
    print(f'[INFO] ONNX model saved -> {out_path}')


def save_charts(hist_df, class_df, split_df, size_hist, size_labels, out_dir):
    fig = px.bar(class_df, x='class', y='count',
                 title='BCCD Box Counts (SSDLite v2 run)<br>'
                       '<span style="font-size:18px;font-weight:normal;">Source: VOC labels | objects per class</span>')
    p = out_dir / 'class_counts_ssdlite_v2.png'
    fig.write_image(str(p))

    fig = px.bar(split_df, x='split', y='images',
                 title='Dataset Split Sizes (SSDLite v2 run)')
    p = out_dir / 'split_sizes_ssdlite_v2.png'
    fig.write_image(str(p))

    rows = []
    for cls, ctr in size_hist.items():
        for bucket in size_labels:
            rows.append({'class': cls, 'size_bucket': bucket, 'count': ctr.get(bucket, 0)})
    size_df = pd.DataFrame(rows)
    fig = px.bar(size_df, x='size_bucket', y='count', color='class', barmode='group',
                 category_orders={'size_bucket': size_labels},
                 title='GT box size distribution (max(w,h) in px) — motivates the small-object upgrades')
    p = out_dir / 'box_size_histogram_ssdlite_v2.png'
    fig.write_image(str(p))

    cols = [c for c in ['train_loss_total', 'val_loss_total'] if c in hist_df.columns]
    if cols:
        fig = px.line(hist_df, x='epoch', y=cols, markers=True,
                       title='SSDLite v2 Detection Loss Curves (focal loss, 512px, cosine LR)')
        p = out_dir / 'loss_curves_ssdlite_v2.png'
        fig.write_image(str(p))
    print('[INFO] Charts saved.')


def main():
    parser = argparse.ArgumentParser(description='Train the small-object-optimized SSDLite v2 on BCCD.')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--out_dir', default='output/ssdlite_v2')
    parser.add_argument('--epochs', type=int, default=60, help='vs. the 8-epoch baseline default')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=512, help='vs. the 320px baseline')
    parser.add_argument('--min_ratio', type=float, default=0.03, help='anchor generator min_ratio (baseline default: 0.2)')
    parser.add_argument('--max_ratio', type=float, default=0.8)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--mosaic_prob', type=float, default=0.5)
    parser.add_argument('--copy_paste_prob', type=float, default=0.5)
    parser.add_argument('--score_thr', type=float, default=0.35)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    set_seed(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Device: {device}')

    t_start = time.time()
    voc_dir = download_bccd(Path(args.data_root))
    records = build_records(voc_dir)
    train_recs, val_recs, test_recs = split_records(records)
    print(f'[INFO] Split -> train:{len(train_recs)}  val:{len(val_recs)}  test:{len(test_recs)}')

    counts = class_counts(records)
    class_df = pd.DataFrame({'class': list(counts.keys()), 'count': list(counts.values())})
    split_df = pd.DataFrame({'split': ['train', 'val', 'test'],
                              'images': [len(train_recs), len(val_recs), len(test_recs)]})
    size_hist, size_labels = box_size_histogram(records)

    train_ds = BCCDDatasetV2(train_recs, img_size=args.img_size, train=True,
                              mosaic_prob=args.mosaic_prob, copy_paste_prob=args.copy_paste_prob,
                              copy_paste_pool=train_recs)
    val_ds = BCCDDatasetV2(val_recs, img_size=args.img_size, train=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate_fn, num_workers=args.num_workers)
    print(f'[INFO] Train batches: {len(train_dl)} | Val batches: {len(val_dl)}')

    model = build_model_v2(
        num_classes=NUM_CLASSES, img_size=args.img_size, min_ratio=args.min_ratio,
        max_ratio=args.max_ratio, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
    ).to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=1e-4)

    warmup_iters = max(1, args.warmup_epochs)

    def lr_lambda(epoch):
        if epoch < warmup_iters:
            return (epoch + 1) / warmup_iters
        progress = (epoch - warmup_iters) / max(1, args.epochs - warmup_iters)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    best_val = float('inf')
    best_path = out_dir / 'ssdlite_v2_bccd_best.pth'

    print(f'[INFO] Starting SSDLite v2 training for {args.epochs} epochs '
          f'(img_size={args.img_size}, min_ratio={args.min_ratio}, focal(alpha={args.focal_alpha},gamma={args.focal_gamma}), '
          f'mosaic_prob={args.mosaic_prob}, copy_paste_prob={args.copy_paste_prob}) ...')
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_stats = train_one_epoch(model, train_dl, optimizer, device, epoch, args.epochs)
        val_stats = eval_loss(model, val_dl, device, epoch, args.epochs)
        scheduler.step()

        row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}
        row.update({f'train_{k}': v for k, v in train_stats.items()})
        row.update({f'val_{k}': v for k, v in val_stats.items()})
        history.append(row)

        val_total = row.get('val_loss_total', float('inf'))
        print(f'[EPOCH {epoch:03d}] train_loss={row.get("train_loss_total", float("nan")):.4f} '
              f'val_loss={val_total:.4f} lr={row["lr"]:.2e} time={time.time() - epoch_start:.1f}s')

        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), best_path)
            print(f'[INFO] New best model saved -> {best_path} (val_loss={best_val:.4f})')

    train_time_sec = time.time() - t_start

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / 'history_ssdlite_v2.csv', index=False)
    save_charts(hist_df, class_df, split_df, size_hist, size_labels, out_dir)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    def predict_fn(pil_img):
        x = TF.to_tensor(pil_img).to(device)
        sx, sy = pil_img.width / args.img_size, pil_img.height / args.img_size
        resized = pil_img.resize((args.img_size, args.img_size))
        x = TF.to_tensor(resized).to(device)
        with torch.no_grad():
            pred = model([x])[0]
        boxes = pred['boxes'].cpu().numpy()
        if len(boxes):
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
        return boxes, pred['labels'].cpu().numpy(), pred['scores'].cpu().numpy()

    draw_prediction_grid(test_recs, predict_fn, out_dir / 'sample_predictions_ssdlite_v2.png', score_thr=args.score_thr)

    onnx_path = out_dir / 'ssdlite_v2_bccd.onnx'
    try:
        export_onnx(model, onnx_path, device, args.img_size)
        write_onnx_meta(
            onnx_path, input_size=(args.img_size, args.img_size), mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],
            letterbox=False, class_map={"0": "RBC", "1": "WBC", "2": "Platelets"},
            extra={'raw_output_labels_are_1indexed_with_background': True, 'score_key': 'scores'},
        )
    except Exception as e:
        warnings.warn(f'ONNX export failed: {e}. Training outputs are still saved.')
        onnx_path = None

    write_run_report(
        out_dir, model_key='ssdlite_v2', model_display_name='SSDLite v2 (small-object-optimized, MobileNetV3-Large)',
        description=(
            "Baseline SSDLite320-MobileNetV3-Large re-tuned for small-object (platelet) recall: "
            "lower-min_ratio anchor generator, 512px input, sigmoid focal-loss classification, "
            "cosine LR schedule with warmup, and mosaic + platelet copy-paste augmentation."
        ),
        hyperparams=vars(args),
        dataset_stats={
            'n_train': len(train_recs), 'n_val': len(val_recs), 'n_test': len(test_recs),
            'class_counts_full_dataset': dict(counts),
            'box_size_histogram': {k: dict(v) for k, v in size_hist.items()},
        },
        timing={'train_time_sec': train_time_sec, 'train_time_min': train_time_sec / 60.0, 'epochs': args.epochs},
        metrics={'best_val_loss': best_val, 'final_train_loss': history[-1].get('train_loss_total') if history else None},
        files={
            'checkpoint': str(best_path), 'onnx': str(onnx_path) if onnx_path else None,
            'history_csv': str(out_dir / 'history_ssdlite_v2.csv'),
            'sample_predictions': str(out_dir / 'sample_predictions_ssdlite_v2.png'),
            'loss_curves': str(out_dir / 'loss_curves_ssdlite_v2.png'),
        },
        notes=[
            "Run bcc_metrics/run_all.py (or detection_eval.py directly) after this to get "
            "mAP@0.5 / mAP@0.5:0.95 / per-class AP on the identical BCCD test split and the "
            "72-image OOD set, comparable to the baseline SSDLite and every other model in "
            "this ablation.",
            f"min_ratio={args.min_ratio} at img_size={args.img_size} puts the smallest anchor "
            f"tier at approximately {args.min_ratio * args.img_size:.0f}px, targeting the "
            "~15-30px platelet size range documented in box_size_histogram_ssdlite_v2.png.",
        ],
    )

    print(f'[INFO] Done. Best val loss: {best_val:.4f}. Total time: {train_time_sec/60:.1f} min.')


if __name__ == '__main__':
    main()
