#!/usr/bin/env python3
import json
import os
import random
import time
import argparse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
import plotly.express as px

CLASSES = ['__background__', 'RBC', 'WBC', 'Platelets']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
COLORS = {1: 'green', 2: 'red', 3: 'blue'}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_bccd(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / 'bccd.zip'
    extracted = root / 'BCCD_Dataset-master'
    voc = extracted / 'BCCD'
    if voc.exists():
        print('[INFO] BCCD already downloaded, skipping.')
        return voc
    print('[INFO] Downloading BCCD dataset...')
    urllib.request.urlretrieve(
        'https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip',
        zip_path
    )
    print('[INFO] Extracting...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(root)
    if not voc.exists():
        raise FileNotFoundError('BCCD VOC folder not found after extraction.')
    print('[INFO] BCCD ready.')
    return voc


def parse_annotation(xml_file, img_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.findtext('filename')
    boxes, labels, areas = [], [], []
    for obj in root.findall('object'):
        name = obj.findtext('name')
        if name not in CLASS_TO_IDX or name == '__background__':
            continue
        bbox = obj.find('bndbox')
        xmin = float(bbox.findtext('xmin'))
        ymin = float(bbox.findtext('ymin'))
        xmax = float(bbox.findtext('xmax'))
        ymax = float(bbox.findtext('ymax'))

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_IDX[name])
        areas.append((xmax - xmin) * (ymax - ymin))
    return {
        'image_path': str(img_dir / filename),
        'filename': filename,
        'boxes': boxes,
        'labels': labels,
        'areas': areas
    }


def build_records(voc_dir: Path):
    ann_dir = voc_dir / 'Annotations'
    img_dir = voc_dir / 'JPEGImages'
    records = [parse_annotation(xml_file, img_dir) for xml_file in sorted(ann_dir.glob('*.xml'))]
    clean = [r for r in records if len(r['boxes']) > 0]
    print(f'[INFO] Loaded {len(clean)} valid images (dropped {len(records)-len(clean)} with no boxes).')
    return clean


def split_records(records, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train = [records[i] for i in idx[:n_train]]
    val = [records[i] for i in idx[n_train:n_train+n_val]]
    test = [records[i] for i in idx[n_train+n_val:]]
    return train, val, test


class BCCDDataset(Dataset):
    def __init__(self, records, train=False):
        self.records = records
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec['image_path']).convert('RGB')

        boxes = torch.as_tensor(rec['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(rec['labels'], dtype=torch.int64)
        area = torch.as_tensor(rec['areas'], dtype=torch.float32)

        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        labels = labels[valid]
        area = area[valid]

        if self.train:
            if random.random() < 0.5:
                img = F.hflip(img)
                w = img.width
                new_boxes = boxes.clone()
                new_boxes[:, 0] = w - boxes[:, 2]
                new_boxes[:, 2] = w - boxes[:, 0]
                v2 = (new_boxes[:, 2] > new_boxes[:, 0]) & (new_boxes[:, 3] > new_boxes[:, 1])
                boxes = new_boxes[v2]
                labels = labels[v2]
                area = area[v2]

            angle = random.uniform(-5, 5)
            img = F.rotate(img, angle)

        img = F.to_tensor(img)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes=4):
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        num_classes=num_classes
    )
    return model


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    running = []
    start = time.time()
    pbar = tqdm(loader, desc=f'Train {epoch}/{total_epochs}', unit='batch', leave=False)

    for batch_idx, (images, targets) in enumerate(pbar, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        row = {'loss_total': float(loss.item())}
        for k, v in loss_dict.items():
            row[k] = float(v.item())
        running.append(row)

        pbar.set_postfix(loss=f'{loss.item():.4f}')

        if batch_idx % 10 == 0:
            elapsed = time.time() - start
            print(f'  [Train E{epoch}] batch {batch_idx}/{len(loader)} | loss={loss.item():.4f} | elapsed={elapsed:.1f}s')

    return pd.DataFrame(running).mean().to_dict()


def eval_loss(model, loader, device, epoch, total_epochs):
    model.train()
    vals = []
    start = time.time()
    pbar = tqdm(loader, desc=f'Val   {epoch}/{total_epochs}', unit='batch', leave=False)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(pbar, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            row = {'loss_total': float(loss.item())}
            for k, v in loss_dict.items():
                row[k] = float(v.item())
            vals.append(row)

            pbar.set_postfix(loss=f'{loss.item():.4f}')

    elapsed = time.time() - start
    avg = pd.DataFrame(vals).mean().to_dict()
    print(f'  [Val   E{epoch}] avg_loss={avg["loss_total"]:.4f} | elapsed={elapsed:.1f}s')
    return avg


def draw_predictions(model, records, device, out_path, score_thr=0.35):
    model.eval()
    chosen = records[:min(6, len(records))]
    canvas = Image.new('RGB', (3 * 320, 2 * 260), 'white')

    for i, rec in enumerate(chosen):
        img = Image.open(rec['image_path']).convert('RGB')
        x = F.to_tensor(img).to(device)
        with torch.no_grad():
            pred = model([x])[0]

        tile = img.resize((320, 220))
        sx, sy = 320 / img.width, 220 / img.height
        draw = ImageDraw.Draw(tile)
        counts = Counter()

        for box, label, score in zip(
            pred['boxes'].cpu().numpy(),
            pred['labels'].cpu().numpy(),
            pred['scores'].cpu().numpy()
        ):
            if score < score_thr:
                continue
            lbl = int(label)
            counts[lbl] += 1
            x1, y1, x2, y2 = box
            color = COLORS.get(lbl, 'yellow')
            draw.rectangle([x1*sx, y1*sy, x2*sx, y2*sy], outline=color, width=2)
            draw.text((x1*sx + 2, y1*sy + 2), f'{CLASSES[lbl]}:{score:.2f}', fill=color)

        board = Image.new('RGB', (320, 260), 'white')
        board.paste(tile, (0, 0))
        ImageDraw.Draw(board).text(
            (8, 228),
            f'RBC:{counts[1]}  WBC:{counts[2]}  PLT:{counts[3]}',
            fill='black'
        )
        canvas.paste(board, ((i % 3) * 320, (i // 3) * 260))

    canvas.save(out_path)
    print(f'[INFO] Saved sample predictions -> {out_path}')


def export_onnx(model, out_path, device):
    class _Wrapper(torch.nn.Module):
        def __init__(self, det):
            super().__init__()
            self.det = det

        def forward(self, x):
            preds = self.det(list(x))
            return (
                preds[0]['boxes'],
                preds[0]['scores'].unsqueeze(1),
                preds[0]['labels'].float().unsqueeze(1)
            )

    wrapper = _Wrapper(model.eval()).to(device)
    dummy = torch.randn(1, 3, 320, 320, device=device)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=['images'],
        output_names=['boxes', 'scores', 'labels'],
        dynamic_axes={'images': {0: 'batch'}},
        opset_version=12
    )
    print(f'[INFO] ONNX model saved -> {out_path}')


def save_charts(hist_df, class_df, split_df, out_dir):
    fig = px.bar(
        class_df, x='class', y='count',
        title='BCCD Box Counts<br><span style="font-size:18px;font-weight:normal;">Source: VOC labels | objects per class</span>'
    )
    fig.update_xaxes(title_text='Cell class')
    fig.update_yaxes(title_text='Box count')
    p = out_dir / 'class_counts_detection.png'
    fig.write_image(str(p))
    with open(str(p) + '.meta.json', 'w') as f:
        json.dump({'caption': 'BCCD class counts', 'description': 'Bounding box counts per class in BCCD.'}, f)

    fig = px.bar(
        split_df, x='split', y='images',
        title='Dataset Split Sizes<br><span style="font-size:18px;font-weight:normal;">Source: random split | images per split</span>'
    )
    fig.update_xaxes(title_text='Split')
    fig.update_yaxes(title_text='Images')
    p = out_dir / 'split_sizes_detection.png'
    fig.write_image(str(p))
    with open(str(p) + '.meta.json', 'w') as f:
        json.dump({'caption': 'Dataset split sizes', 'description': 'Train/val/test image counts.'}, f)

    cols = [c for c in ['train_loss_total', 'val_loss_total'] if c in hist_df.columns]
    if cols:
        fig = px.line(
            hist_df, x='epoch', y=cols, markers=True,
            title='Detection Loss Curves<br><span style="font-size:18px;font-weight:normal;">Source: SSDLite training | total loss by epoch</span>'
        )
        fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5))
        fig.update_xaxes(title_text='Epoch')
        fig.update_yaxes(title_text='Loss')
        p = out_dir / 'loss_curves_detection.png'
        fig.write_image(str(p))
        with open(str(p) + '.meta.json', 'w') as f:
            json.dump({'caption': 'SSDLite detection loss curves', 'description': 'Train and val total loss across epochs.'}, f)

    print('[INFO] Charts saved.')


def main():
    parser = argparse.ArgumentParser(description='Train SSDLite MobileNetV3 on BCCD for blood cell detection')
    parser.add_argument('--data_root', default='data', help='where to download BCCD')
    parser.add_argument('--out_dir', default='output', help='output directory for models and plots')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--score_thr', type=float, default=0.35, help='confidence threshold for prediction visualization')
    parser.add_argument('--num_workers', type=int, default=0, help='set 0 on Windows to avoid DataLoader hangs')
    args = parser.parse_args()

    set_seed(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Device: {device}')

    voc_dir = download_bccd(Path(args.data_root))
    records = build_records(voc_dir)
    train_recs, val_recs, test_recs = split_records(records)
    print(f'[INFO] Split -> train:{len(train_recs)}  val:{len(val_recs)}  test:{len(test_recs)}')

    counts = Counter()
    for r in records:
        for l in r['labels']:
            counts[CLASSES[l]] += 1

    class_df = pd.DataFrame({'class': list(counts.keys()), 'count': list(counts.values())})
    split_df = pd.DataFrame({
        'split': ['train', 'val', 'test'],
        'images': [len(train_recs), len(val_recs), len(test_recs)]
    })

    train_ds = BCCDDataset(train_recs, train=True)
    val_ds = BCCDDataset(val_recs, train=False)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    print(f'[INFO] Train batches: {len(train_dl)} | Val batches: {len(val_dl)}')

    model = build_model(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4
    )

    history = []
    best_val = float('inf')
    best_path = out_dir / 'ssdlite_bccd_best.pth'

    print(f'[INFO] Starting training for {args.epochs} epochs...')
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_stats = train_one_epoch(model, train_dl, optimizer, device, epoch, args.epochs)
        val_stats = eval_loss(model, val_dl, device, epoch, args.epochs)

        row = {'epoch': epoch}
        row.update({f'train_{k}': v for k, v in train_stats.items()})
        row.update({f'val_{k}': v for k, v in val_stats.items()})
        history.append(row)

        val_total = row.get('val_loss_total', float('inf'))
        train_total = row.get('train_loss_total', float('nan'))
        elapsed = time.time() - epoch_start
        print(f'[EPOCH {epoch:02d}] train_loss={train_total:.4f} val_loss={val_total:.4f} time={elapsed:.1f}s')

        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), best_path)
            print(f'[INFO] New best model saved -> {best_path} (val_loss={best_val:.4f})')

    hist_df = pd.DataFrame(history)
    hist_csv = out_dir / 'history_detection.csv'
    hist_df.to_csv(hist_csv, index=False)
    print(f'[INFO] Training history saved -> {hist_csv}')

    save_charts(hist_df, class_df, split_df, out_dir)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    draw_predictions(model, test_recs, device, out_dir / 'sample_predictions_detection.png', score_thr=args.score_thr)
    try:
        export_onnx(model, out_dir / 'ssdlite_bccd.onnx', device)
    except Exception as e:
        print(f'[WARN] ONNX export failed: {e}')
        print('[WARN] Training outputs are saved; skipping ONNX export.')

    print(f'[INFO] Done. Best val loss: {best_val:.4f}')


if __name__ == '__main__':
    main()
        