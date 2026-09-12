#!/usr/bin/env python3
import os
import json
import random
import argparse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import plotly.express as px

CLASSES = ["RBC", "WBC", "Platelets"]
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASSES)}


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
        return voc
    url = 'https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip'
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(root)
    return voc


def parse_voc(voc_dir: Path):
    ann_dir = voc_dir / 'Annotations'
    img_dir = voc_dir / 'JPEGImages'
    rows = []
    img_targets = defaultdict(set)
    for xml_file in sorted(ann_dir.glob('*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.findtext('filename')
        img_path = img_dir / filename
        for obj in root.findall('object'):
            name = obj.findtext('name')
            if name not in CLASS_TO_IDX:
                continue
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.findtext('xmin')))
            ymin = int(float(bbox.findtext('ymin')))
            xmax = int(float(bbox.findtext('xmax')))
            ymax = int(float(bbox.findtext('ymax')))
            rows.append({'image': str(img_path), 'filename': filename, 'label': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
            img_targets[filename].add(CLASS_TO_IDX[name])
    df = pd.DataFrame(rows)
    image_df = pd.DataFrame([{'filename': fn, 'image': str(img_dir / fn), 'labels': sorted(list(lbls))} for fn, lbls in img_targets.items()])
    return df, image_df


class MultiLabelBCCD(Dataset):
    def __init__(self, image_df, tfm=None):
        self.image_df = image_df.reset_index(drop=True)
        self.tfm = tfm
    def __len__(self):
        return len(self.image_df)
    def __getitem__(self, idx):
        row = self.image_df.iloc[idx]
        img = Image.open(row['image']).convert('RGB')
        if self.tfm:
            img = self.tfm(img)
        target = torch.zeros(len(CLASSES), dtype=torch.float32)
        for l in row['labels']:
            target[l] = 1.0
        return img, target


def multilabel_accuracy(logits, targets, thr=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    return (preds.eq(targets).float().mean()).item()


def split_data(image_df, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(image_df))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(0.7*n)
    n_val = int(0.15*n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    return image_df.iloc[train_idx], image_df.iloc[val_idx], image_df.iloc[test_idx]


def train_model(voc_dir: Path, out_dir: Path, epochs=6, batch_size=16, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    box_df, image_df = parse_voc(voc_dir)
    train_df, val_df, test_df = split_data(image_df)

    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = MultiLabelBCCD(train_df, train_tfm)
    val_ds = MultiLabelBCCD(val_df, eval_tfm)
    test_ds = MultiLabelBCCD(test_df, eval_tfm)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    best_val = float('inf')
    best_path = out_dir / 'efficientnet_bccd_best.pth'

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_batches = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += multilabel_accuracy(logits.detach(), y)
            n_batches += 1
        train_loss /= max(1, n_batches)
        train_acc /= max(1, n_batches)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        v_batches = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                val_acc += multilabel_accuracy(logits, y)
                v_batches += 1
        val_loss /= max(1, v_batches)
        val_acc /= max(1, v_batches)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    t_batches = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            test_loss += loss.item()
            test_acc += multilabel_accuracy(logits, y)
            t_batches += 1
    test_loss /= max(1, t_batches)
    test_acc /= max(1, t_batches)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / 'training_history.csv', index=False)
    box_df.to_csv(out_dir / 'bccd_boxes.csv', index=False)

    summary = {
        'num_images': int(len(image_df)),
        'num_boxes': int(len(box_df)),
        'class_counts': box_df['label'].value_counts().to_dict(),
        'split_sizes': {'train': int(len(train_df)), 'val': int(len(val_df)), 'test': int(len(test_df))},
        'best_val_loss': float(best_val),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'task_note': 'EfficientNet-B0 trained for multi-label image-level presence classification by collapsing BCCD box labels into image labels; this is not object detection.'
    }
    with open(out_dir / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    dummy = torch.randn(1, 3, 224, 224, device=device)
    onnx_path = out_dir / 'efficientnet_bccd_best.onnx'
    torch.onnx.export(model, dummy, onnx_path, input_names=['input'], output_names=['logits'], dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=12)

    counts_df = box_df['label'].value_counts().rename_axis('label').reset_index(name='count')
    fig = px.bar(counts_df, x='label', y='count', title='BCCD Class Counts<br><span style="font-size:18px;font-weight:normal;">Source: BCCD annotations | labels per class</span>')
    fig.update_xaxes(title_text='Cell class')
    fig.update_yaxes(title_text='Box count')
    fig.write_image(out_dir / 'class_counts.png')
    with open(out_dir / 'class_counts.png.meta.json', 'w') as f:
        json.dump({'caption': 'BCCD class counts', 'description': 'Bar chart of RBC, WBC, and Platelets annotation counts in BCCD.'}, f)

    split_df = pd.DataFrame({'split': ['train', 'val', 'test'], 'images': [len(train_df), len(val_df), len(test_df)]})
    fig = px.bar(split_df, x='split', y='images', title='Dataset Split Sizes<br><span style="font-size:18px;font-weight:normal;">Source: random split | image counts per split</span>')
    fig.update_xaxes(title_text='Split')
    fig.update_yaxes(title_text='Images')
    fig.write_image(out_dir / 'split_sizes.png')
    with open(out_dir / 'split_sizes.png.meta.json', 'w') as f:
        json.dump({'caption': 'Dataset split sizes', 'description': 'Bar chart of train, validation, and test image counts.'}, f)

    long_hist = hist_df.melt(id_vars='epoch', value_vars=['train_loss','val_loss'], var_name='series', value_name='value')
    fig = px.line(long_hist, x='epoch', y='value', color='series', markers=True, title='Loss Curves by Epoch<br><span style="font-size:18px;font-weight:normal;">Source: training run | lower is better</span>')
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5))
    fig.update_xaxes(title_text='Epoch')
    fig.update_yaxes(title_text='Loss')
    fig.write_image(out_dir / 'loss_curves.png')
    with open(out_dir / 'loss_curves.png.meta.json', 'w') as f:
        json.dump({'caption': 'Training and validation loss', 'description': 'Line chart of loss across epochs for train and validation.'}, f)

    long_acc = hist_df.melt(id_vars='epoch', value_vars=['train_acc','val_acc'], var_name='series', value_name='value')
    fig = px.line(long_acc, x='epoch', y='value', color='series', markers=True, title='Accuracy Curves by Epoch<br><span style="font-size:18px;font-weight:normal;">Source: training run | multi-label accuracy</span>')
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5))
    fig.update_xaxes(title_text='Epoch')
    fig.update_yaxes(title_text='Accuracy')
    fig.write_image(out_dir / 'accuracy_curves.png')
    with open(out_dir / 'accuracy_curves.png.meta.json', 'w') as f:
        json.dump({'caption': 'Training and validation accuracy', 'description': 'Line chart of multi-label accuracy across epochs for train and validation.'}, f)

    samples = test_df.sample(n=min(6, len(test_df)), random_state=42)
    canvas = Image.new('RGB', (3*320, 2*260), (255,255,255))
    with torch.no_grad():
        for i, (_, row) in enumerate(samples.iterrows()):
            img0 = Image.open(row['image']).convert('RGB')
            x = eval_tfm(img0).unsqueeze(0).to(device)
            probs = torch.sigmoid(model(x))[0].cpu().numpy()
            pred = [CLASSES[j] for j,p in enumerate(probs) if p >= 0.5] or [CLASSES[int(np.argmax(probs))]]
            gt = [CLASSES[j] for j in row['labels']]
            img = img0.resize((320, 220))
            tile = Image.new('RGB', (320, 260), (255,255,255))
            tile.paste(img, (0,0))
            ImageDraw.Draw(tile).text((8, 228), f'GT: {", ".join(gt)} | Pred: {", ".join(pred)}', fill=(0,0,0))
            canvas.paste(tile, ((i % 3) * 320, (i // 3) * 260))
    canvas.save(out_dir / 'sample_predictions.png')

    mermaid = """flowchart LR
    A[BCCD VOC XML] --> B[Parse boxes]
    B --> C[Collapse to image labels]
    C --> D[Augment train images]
    D --> E[EfficientNet-B0]
    E --> F[Save .pth]
    F --> G[Export ONNX]
    E --> H[Metrics and plots]
    """
    try:
        from chonkie import create_mermaid_diagram
        create_mermaid_diagram(mermaid, str(out_dir / 'pipeline_diagram.png'), width=1200, height=800)
        with open(out_dir / 'pipeline_diagram.png.meta.json', 'w') as f:
            json.dump({'caption': 'Training pipeline diagram', 'description': 'Flowchart of BCCD parsing, training, checkpointing, and ONNX export.'}, f)
    except Exception:
        with open(out_dir / 'pipeline_diagram.mmd', 'w') as f:
            f.write(mermaid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='output/data')
    parser.add_argument('--out_dir', default='output')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    set_seed(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    voc_dir = download_bccd(Path(args.data_root))
    train_model(voc_dir, out_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

if __name__ == '__main__':
    main()
