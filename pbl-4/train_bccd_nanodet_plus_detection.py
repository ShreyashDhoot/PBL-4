#!/usr/bin/env python3
"""
train_bccd_nanodet_plus_detection.py
=====================================
NanoDet-Plus-style detector on BCCD: purpose-built for edge devices,
anchor-free, using a Generalized Focal Loss (GFL) classification head --
exactly the architecture family recommended in the comparison table
("Best latency/RAM of the group").

IMPLEMENTATION NOTE: the upstream `nanodet` repository is not a
pip-installable training library (it is a standalone repo with its own
config/registry system meant to be cloned and run in-place), so importing
"the real NanoDet-Plus" the way `ultralytics`/`effdet` are imported below
for the other models isn't a clean, reliable option here. Rather than add
a fragile `git clone` step to a training script, this file is a faithful,
self-contained from-scratch PyTorch implementation of the NanoDet-Plus
architecture family, matching the SSDLite baseline's own "no exotic
dependency, runs anywhere torch does" convention:

  * Backbone: MobileNetV3-Small (torchvision, ImageNet-pretrained) --
    NanoDet-Plus's own default is ShuffleNetV2 1.0x; MobileNetV3-Small is
    used here because it is a torchvision built-in (no extra dependency)
    at a comparable parameter/latency budget for a Pi-4-class device. The
    3 feature levels are discovered dynamically at strides 8/16/32.
  * Neck: a lightweight 3-level top-down+bottom-up PAN with
    depthwise-separable convs, approximating NanoDet-Plus's Ghost-PAN
    (a full Ghost-module reimplementation was judged not worth the added
    risk of an un-runnable script; the depthwise-separable version keeps
    the same "cheap, edge-friendly fusion" property).
  * Head: anchor-free, per-pyramid-point, shared conv stack -> per-class
    logits + (l,t,r,b) box-distance regression, matching NanoDet-Plus's
    single-scale-assigned, anchor-free head design.
  * Loss: Quality Focal Loss (Li et al. 2020, "Generalized Focal Loss")
    for classification -- the exact loss this model family is named for
    in the comparison table -- plus Smooth-L1 + GIoU for box regression.
    Assignment is FCOS/ATSS-style (point-in-box + a per-level regression
    range so small objects are assigned to the highest-resolution,
    stride-8 level), a common, well-understood substitute for NanoDet-
    Plus's own dynamic soft-label assigner.
  * Small-object upgrades (same as every other model in this repo):
    512px default input, mosaic + platelet copy-paste augmentation,
    cosine LR with warmup, many epochs.

Outputs (under output/nanodet_plus/): history_nanodet_plus.csv,
loss_curves_nanodet_plus.png, sample_predictions_nanodet_plus.png,
nanodet_plus_bccd_best.pth, nanodet_plus_bccd.onnx (+.meta.json),
run_report_nanodet_plus.{json,md}.

Usage:
    python train_bccd_nanodet_plus_detection.py --epochs 120 --img_size 416
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import batched_nms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from bccd_data_utils import (
    CLASSES, download_bccd, build_records, split_records, class_counts, box_size_histogram,
    set_seed, mosaic_augment, platelet_copy_paste, draw_prediction_grid, write_run_report, write_onnx_meta,
)

NUM_FG_CLASSES = 3  # RBC, WBC, Platelets (no background class in an anchor-free GFL head)
STRIDES = (8, 16, 32)
# Per-level max regression distance in pixels ("regress range"): small
# objects (platelets) are forced onto the highest-resolution stride-8 level.
REGRESS_RANGES = ((0, 64), (64, 128), (128, 1e6))


# ----------------------------------------------------------------------------
# Backbone: dynamically discover stride-8/16/32 feature maps so this does
# not depend on hardcoded, version-fragile torchvision layer indices.
# ----------------------------------------------------------------------------
class MobileNetV3SmallMultiLevel(nn.Module):
    def __init__(self, pretrained=True, probe_size=256):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = backbone.features

        with torch.no_grad():
            x = torch.zeros(1, 3, probe_size, probe_size)
            spatial, channels = [], []
            for layer in self.features:
                x = layer(x)
                spatial.append(x.shape[-1])
                channels.append(x.shape[1])
        strides = [probe_size / s for s in spatial]

        self.out_indices = []
        self.out_channels = []
        for target in STRIDES:
            candidates = [i for i, s in enumerate(strides) if abs(s - target) < 1e-3]
            if not candidates:
                # fall back to the closest available stride
                candidates = [int(np.argmin([abs(s - target) for s in strides]))]
            idx = candidates[-1]
            self.out_indices.append(idx)
            self.out_channels.append(channels[idx])
        print(f'[INFO] Backbone feature levels (stride:layer_idx:channels) = '
              f'{list(zip(STRIDES, self.out_indices, self.out_channels))}')

    def forward(self, x):
        outs = []
        out_set = set(self.out_indices)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in out_set:
                outs.append(x)
        return outs  # [C3(stride8), C4(stride16), C5(stride32)]


def dw_sep_conv(in_ch, out_ch, k=3, s=1):
    p = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


class LitePAN(nn.Module):
    """Lightweight top-down + bottom-up feature-pyramid fusion,
    approximating NanoDet-Plus's Ghost-PAN with plain depthwise-separable
    convs (see module docstring for the "why" of this simplification)."""

    def __init__(self, in_channels, out_ch=96):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_channels])
        self.td_convs = nn.ModuleList([dw_sep_conv(out_ch, out_ch) for _ in in_channels[:-1]])
        self.bu_downs = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, 2, 1) for _ in in_channels[:-1]])
        self.bu_convs = nn.ModuleList([dw_sep_conv(out_ch, out_ch) for _ in in_channels[:-1]])
        self.out_channels = out_ch

    def forward(self, feats):
        laterals = [l(f) for l, f in zip(self.laterals, feats)]

        # top-down
        td = [laterals[-1]]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(td[0], size=laterals[i].shape[-2:], mode='nearest')
            td.insert(0, self.td_convs[i](laterals[i] + up))

        # bottom-up
        outs = [td[0]]
        for i in range(len(td) - 1):
            down = self.bu_downs[i](outs[-1])
            outs.append(self.bu_convs[i](td[i + 1] + down))
        return outs  # same order as input feats (stride8, stride16, stride32)


class GFLHead(nn.Module):
    """Anchor-free head: shared conv stack per level -> class logits +
    (l,t,r,b) distance regression (in stride units)."""

    def __init__(self, in_ch, num_classes=NUM_FG_CLASSES, stacked_convs=2, feat_ch=96):
        super().__init__()
        self.num_classes = num_classes
        cls_convs, reg_convs = [], []
        for _ in range(stacked_convs):
            cls_convs.append(dw_sep_conv(in_ch, feat_ch))
            reg_convs.append(dw_sep_conv(in_ch, feat_ch))
            in_ch = feat_ch
        self.cls_convs = nn.Sequential(*cls_convs)
        self.reg_convs = nn.Sequential(*reg_convs)
        self.cls_out = nn.Conv2d(feat_ch, num_classes, 3, padding=1)
        self.reg_out = nn.Conv2d(feat_ch, 4, 3, padding=1)
        nn.init.constant_(self.cls_out.bias, -4.595)  # prior ~0.01 positive rate

    def forward(self, x):
        cls_logits = self.cls_out(self.cls_convs(x))
        reg_dist = F.relu(self.reg_out(self.reg_convs(x)))  # distances must be >= 0
        return cls_logits, reg_dist


class NanoDetPlus(nn.Module):
    def __init__(self, num_classes=NUM_FG_CLASSES, pretrained_backbone=True, feat_ch=96):
        super().__init__()
        self.backbone = MobileNetV3SmallMultiLevel(pretrained=pretrained_backbone)
        self.neck = LitePAN(self.backbone.out_channels, out_ch=feat_ch)
        self.head = GFLHead(feat_ch, num_classes=num_classes, feat_ch=feat_ch)
        self.num_classes = num_classes

    def forward(self, x):
        feats = self.neck(self.backbone(x))
        cls_list, reg_list = [], []
        for f in feats:
            c, r = self.head(f)
            cls_list.append(c)
            reg_list.append(r)
        return cls_list, reg_list  # per level: [B,C,H,W], [B,4,H,W]


# ----------------------------------------------------------------------------
# Point grids, box decode, assignment, losses
# ----------------------------------------------------------------------------
def make_points(feat_shapes, strides, device):
    points, point_strides = [], []
    for (h, w), s in zip(feat_shapes, strides):
        ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        pts = torch.stack([(xs + 0.5) * s, (ys + 0.5) * s], dim=-1).reshape(-1, 2)
        points.append(pts)
        point_strides.append(torch.full((pts.shape[0],), s, device=device, dtype=torch.float32))
    return torch.cat(points, 0), torch.cat(point_strides, 0)


def decode_ltrb(points, ltrb, stride):
    """points[N,2] xy pixel centers, ltrb[N,4] distances in stride units."""
    ltrb_px = ltrb * stride.unsqueeze(-1)
    x1 = points[:, 0] - ltrb_px[:, 0]
    y1 = points[:, 1] - ltrb_px[:, 1]
    x2 = points[:, 0] + ltrb_px[:, 2]
    y2 = points[:, 1] + ltrb_px[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def giou_loss(pred_boxes, target_boxes, eps=1e-7):
    px1, py1, px2, py2 = pred_boxes.unbind(-1)
    tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)
    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    tgt_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)

    ix1, iy1 = torch.maximum(px1, tx1), torch.maximum(py1, ty1)
    ix2, iy2 = torch.minimum(px2, tx2), torch.minimum(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    union = pred_area + tgt_area - inter + eps
    iou = inter / union

    cx1, cy1 = torch.minimum(px1, tx1), torch.minimum(py1, ty1)
    cx2, cy2 = torch.maximum(px2, tx2), torch.maximum(py2, ty2)
    enclose = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + eps
    giou = iou - (enclose - union) / enclose
    return 1.0 - giou


def quality_focal_loss(logits, quality_targets, pos_mask, beta=2.0):
    """QFL (Li et al., GFL): classification target is a soft IoU-quality
    label at positive locations, 0 everywhere else. `quality_targets` and
    `pos_mask` are both [N, C] (class-wise), matching the multi-class QFL
    formulation used by NanoDet-Plus's head."""
    sigma = torch.sigmoid(logits)
    scale = (quality_targets - sigma).abs().pow(beta)
    ce = F.binary_cross_entropy_with_logits(logits, quality_targets, reduction='none')
    return (scale * ce).sum()


def assign_targets(points, point_strides, gt_boxes, gt_labels, regress_ranges_per_point):
    """FCOS/ATSS-style assignment: a point is a positive for a GT box if
    (a) the point lies inside the box, and (b) max(box side) falls within
    that point's pyramid level's regression range (forces small
    platelet-sized boxes onto the high-resolution stride-8 level).
    On ties (multiple GTs claim one point) the smallest-area GT wins, the
    standard FCOS tie-break that favors small objects.
    Returns: pos_mask[N] bool, assigned_gt_idx[N] long (-1 if none),
    """
    n_pts = points.shape[0]
    if len(gt_boxes) == 0:
        return torch.zeros(n_pts, dtype=torch.bool, device=points.device), torch.full((n_pts,), -1, dtype=torch.long, device=points.device)

    gt_boxes_t = torch.as_tensor(gt_boxes, dtype=torch.float32, device=points.device)
    areas = (gt_boxes_t[:, 2] - gt_boxes_t[:, 0]) * (gt_boxes_t[:, 3] - gt_boxes_t[:, 1])

    px, py = points[:, 0:1], points[:, 1:2]  # [N,1]
    x1, y1, x2, y2 = gt_boxes_t[:, 0], gt_boxes_t[:, 1], gt_boxes_t[:, 2], gt_boxes_t[:, 3]
    l = px - x1[None, :]
    t = py - y1[None, :]
    r = x2[None, :] - px
    b = y2[None, :] - py
    inside = (l > 0) & (t > 0) & (r > 0) & (b > 0)  # [N, G]

    max_side = torch.maximum(x2 - x1, y2 - y1)[None, :]  # [1,G]
    lo = regress_ranges_per_point[:, 0:1]
    hi = regress_ranges_per_point[:, 1:2]
    level_ok = (max_side >= lo) & (max_side < hi)  # [N,G]

    valid = inside & level_ok
    areas_masked = areas[None, :].expand(n_pts, -1).clone()
    areas_masked[~valid] = 1e9
    min_area, min_idx = areas_masked.min(dim=1)
    pos_mask = min_area < 1e8
    assigned_gt_idx = torch.where(pos_mask, min_idx, torch.full_like(min_idx, -1))
    return pos_mask, assigned_gt_idx


def compute_losses(cls_list, reg_list, feat_shapes, strides, targets, device, qfl_beta=2.0):
    points, point_strides = make_points(feat_shapes, strides, device)
    n_levels_pts = [h * w for (h, w) in feat_shapes]
    level_of_point = torch.cat([torch.full((n,), i, device=device) for i, n in enumerate(n_levels_pts)])
    regress_ranges = torch.as_tensor(REGRESS_RANGES, dtype=torch.float32, device=device)[level_of_point]

    cls_flat = torch.cat([c.permute(0, 2, 3, 1).reshape(c.shape[0], -1, c.shape[1]) for c in cls_list], dim=1)
    reg_flat = torch.cat([r.permute(0, 2, 3, 1).reshape(r.shape[0], -1, r.shape[1]) for r in reg_list], dim=1)

    total_cls_loss, total_reg_loss, total_pos = 0.0, 0.0, 0
    B = cls_flat.shape[0]
    for b in range(B):
        gt_boxes = targets[b]['boxes']
        gt_labels = targets[b]['labels']  # 0-indexed foreground labels [0..2]
        pos_mask, assigned_idx = assign_targets(points, point_strides, gt_boxes, gt_labels, regress_ranges)

        qfl_target = torch.zeros_like(cls_flat[b])  # [N, C]
        n_pos = int(pos_mask.sum().item())
        if n_pos > 0:
            pos_points = points[pos_mask]
            pos_strides = point_strides[pos_mask]
            pos_gt_idx = assigned_idx[pos_mask]
            gt_boxes_t = torch.as_tensor(gt_boxes, dtype=torch.float32, device=device)[pos_gt_idx]
            gt_labels_t = torch.as_tensor(gt_labels, dtype=torch.long, device=device)[pos_gt_idx]

            pred_ltrb = reg_flat[b][pos_mask]
            pred_boxes = decode_ltrb(pos_points, pred_ltrb, pos_strides)

            # regression target in stride units
            l = (pos_points[:, 0] - gt_boxes_t[:, 0]) / pos_strides
            t = (pos_points[:, 1] - gt_boxes_t[:, 1]) / pos_strides
            r = (gt_boxes_t[:, 2] - pos_points[:, 0]) / pos_strides
            btm = (gt_boxes_t[:, 3] - pos_points[:, 1]) / pos_strides
            target_ltrb = torch.stack([l, t, r, btm], dim=-1).clamp(min=0)

            reg_l1 = F.smooth_l1_loss(pred_ltrb, target_ltrb, reduction='sum')
            with torch.no_grad():
                iou_q = 1.0 - giou_loss(pred_boxes.detach(), gt_boxes_t).clamp(0, 2) / 2.0
                iou_q = iou_q.clamp(0, 1)
            g_loss = giou_loss(pred_boxes, gt_boxes_t).sum()

            qfl_target[pos_mask, gt_labels_t] = iou_q
            total_reg_loss += reg_l1 + g_loss
            total_pos += n_pos

        cls_loss = quality_focal_loss(cls_flat[b], qfl_target, pos_mask, beta=qfl_beta)
        total_cls_loss += cls_loss

    norm = max(1, total_pos)
    return {
        'loss_cls': total_cls_loss / max(1, B),
        'loss_reg': total_reg_loss / norm,
        'loss_total': total_cls_loss / max(1, B) + total_reg_loss / norm,
    }


@torch.no_grad()
def predict_image(model, pil_img, device, img_size, score_thr=0.05, iou_thr=0.5, max_det=300):
    model.eval()
    w0, h0 = pil_img.size
    resized = pil_img.resize((img_size, img_size))
    x = TF.to_tensor(resized).unsqueeze(0).to(device)
    cls_list, reg_list = model(x)
    feat_shapes = [(c.shape[-2], c.shape[-1]) for c in cls_list]
    points, point_strides = make_points(feat_shapes, STRIDES, device)

    cls_flat = torch.cat([c.permute(0, 2, 3, 1).reshape(-1, c.shape[1]) for c in cls_list], dim=0)
    reg_flat = torch.cat([r.permute(0, 2, 3, 1).reshape(-1, r.shape[1]) for r in reg_list], dim=0)

    scores_all = torch.sigmoid(cls_flat)  # [N, C]
    boxes_all = decode_ltrb(points, reg_flat, point_strides)  # [N,4] in img_size space

    scores, labels = scores_all.max(dim=1)
    keep = scores > score_thr
    boxes, scores, labels = boxes_all[keep], scores[keep], labels[keep]
    if boxes.shape[0] == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int), np.zeros((0,))

    keep_idx = batched_nms(boxes, scores, labels, iou_thr)[:max_det]
    boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]

    sx, sy = w0 / img_size, h0 / img_size
    boxes = boxes.cpu().numpy()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    labels = labels.cpu().numpy().astype(int) + 1  # -> 1-indexed w/ background at 0 (matches CLASSES)
    return boxes, labels, scores.cpu().numpy()


# ----------------------------------------------------------------------------
# Dataset (mosaic + platelet copy-paste, same as SSDLite v2)
# ----------------------------------------------------------------------------
class BCCDDatasetGFL(Dataset):
    def __init__(self, records, img_size=416, train=False, mosaic_prob=0.5, copy_paste_prob=0.5, seed=42):
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
        # labels here are 1-indexed with '__background__' at 0 (CLASSES convention);
        # the GFL head is background-free, so shift to 0-indexed foreground labels.
        fg_labels = labels - 1
        target = {'boxes': boxes, 'labels': fg_labels}
        return img_t, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), list(targets)


# ----------------------------------------------------------------------------
# Train / main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train a self-contained NanoDet-Plus-style detector on BCCD.')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--out_dir', default='output/nanodet_plus')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=416)
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

    t0 = time.time()
    voc_dir = download_bccd(Path(args.data_root))
    records = build_records(voc_dir)
    train_recs, val_recs, test_recs = split_records(records)
    print(f'[INFO] Split -> train:{len(train_recs)}  val:{len(val_recs)}  test:{len(test_recs)}')

    counts = class_counts(records)
    size_hist, size_labels = box_size_histogram(records)

    train_ds = BCCDDatasetGFL(train_recs, img_size=args.img_size, train=True,
                               mosaic_prob=args.mosaic_prob, copy_paste_prob=args.copy_paste_prob)
    val_ds = BCCDDatasetGFL(val_recs, img_size=args.img_size, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    model = NanoDetPlus(num_classes=NUM_FG_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    warmup_iters = max(1, args.warmup_epochs)

    def lr_lambda(epoch):
        if epoch < warmup_iters:
            return (epoch + 1) / warmup_iters
        progress = (epoch - warmup_iters) / max(1, args.epochs - warmup_iters)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    best_val = float('inf')
    best_path = out_dir / 'nanodet_plus_bccd_best.pth'

    print(f'[INFO] Starting NanoDet-Plus-style training for {args.epochs} epochs (img_size={args.img_size}) ...')
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_start = time.time()
        train_rows = []
        pbar = tqdm(train_dl, desc=f'Train {epoch}/{args.epochs}', leave=False)
        for images, targets in pbar:
            images = images.to(device)
            cls_list, reg_list = model(images)
            feat_shapes = [(c.shape[-2], c.shape[-1]) for c in cls_list]
            losses = compute_losses(cls_list, reg_list, feat_shapes, STRIDES, targets, device)
            loss = losses['loss_total']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_rows.append({k: float(v) for k, v in losses.items()})
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        model.eval()
        val_rows = []
        with torch.no_grad():
            for images, targets in val_dl:
                images = images.to(device)
                cls_list, reg_list = model(images)
                feat_shapes = [(c.shape[-2], c.shape[-1]) for c in cls_list]
                losses = compute_losses(cls_list, reg_list, feat_shapes, STRIDES, targets, device)
                val_rows.append({k: float(v) for k, v in losses.items()})

        scheduler.step()
        train_avg = pd.DataFrame(train_rows).mean().to_dict() if train_rows else {}
        val_avg = pd.DataFrame(val_rows).mean().to_dict() if val_rows else {}
        row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}
        row.update({f'train_{k}': v for k, v in train_avg.items()})
        row.update({f'val_{k}': v for k, v in val_avg.items()})
        history.append(row)

        val_total = row.get('val_loss_total', float('inf'))
        print(f'[EPOCH {epoch:03d}] train_loss={row.get("train_loss_total", float("nan")):.4f} '
              f'val_loss={val_total:.4f} lr={row["lr"]:.2e} time={time.time()-ep_start:.1f}s')

        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), best_path)
            print(f'[INFO] New best model saved -> {best_path} (val_loss={best_val:.4f})')

    train_time_sec = time.time() - t0
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / 'history_nanodet_plus.csv', index=False)

    try:
        import plotly.express as px
        cols = [c for c in ['train_loss_total', 'val_loss_total'] if c in hist_df.columns]
        if cols:
            fig = px.line(hist_df, x='epoch', y=cols, markers=True, title='NanoDet-Plus-style Detection Loss Curves (QFL + GIoU)')
            fig.write_image(str(out_dir / 'loss_curves_nanodet_plus.png'))
    except Exception as e:
        print(f'[WARN] Could not render loss-curve chart: {e}')

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    def predict_fn(pil_img):
        return predict_image(model, pil_img, device, args.img_size, score_thr=0.05)

    draw_prediction_grid(test_recs, predict_fn, out_dir / 'sample_predictions_nanodet_plus.png', score_thr=args.score_thr)

    onnx_path = out_dir / 'nanodet_plus_bccd.onnx'
    try:
        class _Wrapper(nn.Module):
            def __init__(self, m, img_size, score_thr, iou_thr):
                super().__init__()
                self.m = m
                self.img_size = img_size
                self.score_thr = score_thr
                self.iou_thr = iou_thr

            def forward(self, x):
                cls_list, reg_list = self.m(x)
                feat_shapes = [(c.shape[-2], c.shape[-1]) for c in cls_list]
                points, point_strides = make_points(feat_shapes, STRIDES, x.device)
                cls_flat = torch.cat([c.permute(0, 2, 3, 1).reshape(-1, c.shape[1]) for c in cls_list], dim=0)
                reg_flat = torch.cat([r.permute(0, 2, 3, 1).reshape(-1, r.shape[1]) for r in reg_list], dim=0)
                scores_all = torch.sigmoid(cls_flat)
                boxes_all = decode_ltrb(points, reg_flat, point_strides)
                scores, labels = scores_all.max(dim=1)
                keep = scores > self.score_thr
                boxes, scores, labels = boxes_all[keep], scores[keep], labels[keep]
                keep_idx = batched_nms(boxes, scores, labels, self.iou_thr)
                boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]
                return boxes, scores.unsqueeze(1), (labels.float() + 1).unsqueeze(1)

        wrapper = _Wrapper(model.eval(), args.img_size, 0.05, 0.5).to(device)
        dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        torch.onnx.export(wrapper, dummy, str(onnx_path), input_names=['images'],
                           output_names=['boxes', 'scores', 'labels'], opset_version=12)
        write_onnx_meta(
            onnx_path, input_size=(args.img_size, args.img_size), mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],
            letterbox=False, class_map={"0": "RBC", "1": "WBC", "2": "Platelets"},
            extra={'output_layout': 'separate_boxes_scores_labels'},
        )
    except Exception as e:
        warnings.warn(f'ONNX export failed: {e}. Training outputs are still saved.')
        onnx_path = None

    write_run_report(
        out_dir, model_key='nanodet_plus', model_display_name='NanoDet-Plus-style (self-contained, GFL head)',
        description=(
            "Self-contained, from-scratch PyTorch implementation of the NanoDet-Plus architecture "
            "family (MobileNetV3-Small backbone, lightweight PAN neck, anchor-free Generalized "
            "Focal Loss / Quality Focal Loss head), trained with mosaic + platelet copy-paste "
            "augmentation and a cosine LR schedule. See the script docstring for the documented "
            "simplifications vs. the upstream ShuffleNetV2+GhostPAN+dynamic-soft-label-assigner "
            "NanoDet-Plus (which has no pip-installable training API)."
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
            'history_csv': str(out_dir / 'history_nanodet_plus.csv'),
            'sample_predictions': str(out_dir / 'sample_predictions_nanodet_plus.png'),
        },
        notes=[
            "Architecture is a documented, from-scratch approximation of NanoDet-Plus (see "
            "script docstring): MobileNetV3-Small in place of ShuffleNetV2, a depthwise-separable "
            "PAN in place of Ghost-PAN, and an FCOS/ATSS-style point-in-box assigner in place of "
            "NanoDet-Plus's dynamic soft-label assigner. The classification loss (Quality Focal "
            "Loss) is the genuine GFL formulation the architecture family is named for.",
            "Run bcc_metrics/run_all.py after this for cross-model-comparable mAP/P/R/F1 on the "
            "same BCCD test split and 72-image OOD set used for every other model.",
        ],
    )

    print(f'[INFO] Done. Best val loss: {best_val:.4f}. Total time: {train_time_sec/60:.1f} min.')


if __name__ == '__main__':
    main()
