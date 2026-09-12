#!/usr/bin/env python3
"""
bccd_data_utils.py
===================
Shared code used by *every* train_bccd_<model>_detection.py script in this
folder, so all models:

  * download / parse the exact same BCCD VOC dataset,
  * use the exact same deterministic 70/15/15 train/val/test split
    (same seed=42, same np.random.default_rng shuffle as the original
    train_bccd_ssdlite_detection.py), so every model in the comparison is
    trained and tested on identical data -- a fair, apples-to-apples
    multi-model ablation,
  * expose the same small-object augmentation building blocks (mosaic,
    platelet-region copy-paste) referenced in the "small object" upgrade
    notes,
  * write the same *standardized, paper-ready* run report (JSON + Markdown)
    so a downstream reader (human or LLM) can generate the Results section
    of the paper directly from output/<model>/run_report_<model>.{json,md}
    without re-deriving numbers from raw logs.

Nothing in this file trains a model; it is pure data/utility code imported
by the train_bccd_*.py scripts and by bcc_metrics/scripts/*.py.
"""

import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ----------------------------------------------------------------------------
# Class conventions -- MUST stay identical across every training script and
# bcc_metrics/scripts/common.py (DET_CLASSES) so labels line up everywhere.
# ----------------------------------------------------------------------------
CLASSES = ['__background__', 'RBC', 'WBC', 'Platelets']  # index 0 = background
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
COLORS = {1: 'green', 2: 'red', 3: 'blue'}
PLATELET_LABEL = CLASS_TO_IDX['Platelets']  # 3

RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Download + VOC parsing (identical logic to the original SSDLite script)
# ----------------------------------------------------------------------------
def download_bccd(root: Path):
    root = Path(root)
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
        'areas': areas,
    }


def build_records(voc_dir: Path):
    voc_dir = Path(voc_dir)
    ann_dir = voc_dir / 'Annotations'
    img_dir = voc_dir / 'JPEGImages'
    records = [parse_annotation(xml_file, img_dir) for xml_file in sorted(ann_dir.glob('*.xml'))]
    clean = [r for r in records if len(r['boxes']) > 0]
    print(f'[INFO] Loaded {len(clean)} valid images (dropped {len(records) - len(clean)} with no boxes).')
    return clean


def split_records(records, seed=RANDOM_SEED):
    """70/15/15 train/val/test split -- IDENTICAL logic (same seed, same
    np.random.default_rng shuffle order) to train_bccd_ssdlite_detection.py
    and bcc_metrics/scripts/voc_data.py::split_bccd_records, so every model
    trained with this file sees the exact same test set as every other
    model AND as the metrics suite's "held-out" evaluation."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train = [records[i] for i in idx[:n_train]]
    val = [records[i] for i in idx[n_train:n_train + n_val]]
    test = [records[i] for i in idx[n_train + n_val:]]
    return train, val, test


def class_counts(records):
    counts = Counter()
    for r in records:
        for l in r['labels']:
            counts[CLASSES[l]] += 1
    return counts


def box_size_histogram(records, edges=(0, 15, 30, 50, 100, 1e9)):
    """Bucket every GT box by max(width, height) in pixels. Directly
    evidences the 'platelets are mostly 15-30px objects' motivation for the
    small-object upgrades (P2 head, lower min_ratio anchors, etc.)."""
    edges = list(edges)
    labels = [f'{int(edges[i])}-{int(edges[i+1]) if edges[i+1] < 1e9 else "inf"}px' for i in range(len(edges) - 1)]
    hist = {cls: Counter() for cls in CLASSES[1:]}
    for r in records:
        for (x1, y1, x2, y2), lbl in zip(r['boxes'], r['labels']):
            size = max(x2 - x1, y2 - y1)
            for i in range(len(edges) - 1):
                if edges[i] <= size < edges[i + 1]:
                    hist[CLASSES[lbl]][labels[i]] += 1
                    break
    return hist, labels


# ----------------------------------------------------------------------------
# VOC -> YOLO (Ultralytics) conversion, shared by the yolov8n / yolo11n /
# rtdetr-nano training scripts so they all read from one disk-cached copy.
# ----------------------------------------------------------------------------
def voc_records_to_yolo_dataset(train_recs, val_recs, test_recs, out_dir: Path, class_names=None):
    """Writes a YOLO-format dataset (images/<split>/*.jpg symlinked or
    copied, labels/<split>/*.txt) plus a data.yaml Ultralytics can consume
    directly. class_names excludes '__background__' (YOLO has no bg class).
    Returns the path to data.yaml.
    """
    import shutil

    out_dir = Path(out_dir)
    class_names = class_names or CLASSES[1:]
    name_to_yolo_idx = {name: i for i, name in enumerate(class_names)}

    for split, recs in (('train', train_recs), ('val', val_recs), ('test', test_recs)):
        img_out = out_dir / 'images' / split
        lbl_out = out_dir / 'labels' / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for rec in recs:
            src = Path(rec['image_path'])
            dst_img = img_out / src.name
            if not dst_img.exists():
                try:
                    os.symlink(src.resolve(), dst_img)
                except (OSError, NotImplementedError):
                    shutil.copy(src, dst_img)

            with Image.open(src) as im:
                w, h = im.size

            lines = []
            for (x1, y1, x2, y2), lbl in zip(rec['boxes'], rec['labels']):
                cls_name = CLASSES[lbl]
                if cls_name not in name_to_yolo_idx:
                    continue
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{name_to_yolo_idx[cls_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            (lbl_out / (src.stem + '.txt')).write_text('\n'.join(lines))

    data_yaml = out_dir / 'data.yaml'
    yaml_text = (
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names)) + "\n"
    )
    data_yaml.write_text(yaml_text)
    print(f'[INFO] YOLO-format dataset written -> {out_dir} ({data_yaml})')
    return data_yaml


# ----------------------------------------------------------------------------
# Small-object augmentation: mosaic + platelet-region copy-paste
# ----------------------------------------------------------------------------
def mosaic_augment(records, index, img_size=512, rng=None):
    """Classic YOLO-style 4-image mosaic. Picks `index` plus 3 random other
    records, tiles them into one img_size x img_size canvas (each source
    resized into one quadrant, boxes remapped), and returns
    (PIL.Image, boxes[N,4] xyxy, labels[N]).

    Purpose (small-object notes): mosaic multiplies the *number and
    diversity of small-object contexts* (platelets) the model sees per
    optimizer step, without needing more raw images, and is one of the two
    requested augmentations ("mosaic augmentation ... to multiply the
    effective number of platelet positives the model sees").
    """
    rng = rng or np.random.default_rng()
    n = len(records)
    idxs = [index] + [int(rng.integers(0, n)) for _ in range(3)]

    s = img_size
    canvas = Image.new('RGB', (s, s), (114, 114, 114))
    cx, cy = s // 2, s // 2
    # quadrant placement: (x1,y1,x2,y2) of the quadrant in the canvas
    quadrants = [(0, 0, cx, cy), (cx, 0, s, cy), (0, cy, cx, s), (cx, cy, s, s)]

    all_boxes, all_labels = [], []
    for qi, ridx in enumerate(idxs):
        rec = records[ridx]
        img = Image.open(rec['image_path']).convert('RGB')
        qx1, qy1, qx2, qy2 = quadrants[qi]
        qw, qh = qx2 - qx1, qy2 - qy1
        sx, sy = qw / img.width, qh / img.height
        tile = img.resize((qw, qh))
        canvas.paste(tile, (qx1, qy1))

        for (x1, y1, x2, y2), lbl in zip(rec['boxes'], rec['labels']):
            nx1 = qx1 + x1 * sx
            ny1 = qy1 + y1 * sy
            nx2 = qx1 + x2 * sx
            ny2 = qy1 + y2 * sy
            if nx2 - nx1 < 1 or ny2 - ny1 < 1:
                continue
            all_boxes.append([nx1, ny1, nx2, ny2])
            all_labels.append(lbl)

    return canvas, np.asarray(all_boxes, dtype=np.float32).reshape(-1, 4), np.asarray(all_labels, dtype=np.int64)


def _boxes_overlap(box, boxes, thr=0.0):
    if len(boxes) == 0:
        return False
    x1, y1, x2, y2 = box
    bx1 = np.maximum(boxes[:, 0], x1)
    by1 = np.maximum(boxes[:, 1], y1)
    bx2 = np.minimum(boxes[:, 2], x2)
    by2 = np.minimum(boxes[:, 3], y2)
    inter = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    return bool((inter > thr).any())


def platelet_copy_paste(image, boxes, labels, donor_pool, rng=None, max_paste=4, jitter_scale=(0.85, 1.2)):
    """Platelet-region copy-paste augmentation (small-object upgrade #2:
    "platelet-region copy-paste augmentation to multiply the effective
    number of platelet positives the model sees").

    Crops real platelet bounding boxes from `donor_pool` (a list of BCCD
    records) and pastes 1..max_paste of them onto `image` at random,
    non-overlapping locations, each with a small random rescale. Returns
    the augmented (PIL.Image, boxes[N,4], labels[N]) with the new platelet
    boxes appended -- these are genuine platelet crops, not synthetic
    blobs, so classifier-relevant texture/color is preserved.
    """
    rng = rng or np.random.default_rng()
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    labels = np.asarray(labels, dtype=np.int64)

    # Collect all platelet crops available in the donor pool once per call
    # (cheap: BCCD platelet boxes are tiny).
    platelet_crops = []
    for rec in donor_pool:
        img_cache = None
        for (x1, y1, x2, y2), lbl in zip(rec['boxes'], rec['labels']):
            if lbl != PLATELET_LABEL:
                continue
            if img_cache is None:
                img_cache = Image.open(rec['image_path']).convert('RGB')
            w, h = x2 - x1, y2 - y1
            if w < 3 or h < 3:
                continue
            platelet_crops.append(img_cache.crop((x1, y1, x2, y2)).copy())
        if len(platelet_crops) > 200:  # cap for speed; plenty of diversity already
            break

    if not platelet_crops:
        return image, boxes, labels

    n_paste = int(rng.integers(1, max_paste + 1))
    out_img = image.copy()
    W, H = out_img.size
    new_boxes, new_labels = [], []

    for _ in range(n_paste):
        crop = platelet_crops[int(rng.integers(0, len(platelet_crops)))]
        scale = rng.uniform(*jitter_scale)
        cw, ch = max(4, int(crop.width * scale)), max(4, int(crop.height * scale))
        crop_r = crop.resize((cw, ch))

        placed = False
        for _try in range(8):
            px = int(rng.integers(0, max(1, W - cw)))
            py = int(rng.integers(0, max(1, H - ch)))
            cand = (px, py, px + cw, py + ch)
            if not _boxes_overlap(cand, boxes) and not _boxes_overlap(cand, np.asarray(new_boxes).reshape(-1, 4) if new_boxes else np.zeros((0, 4))):
                out_img.paste(crop_r, (px, py))
                new_boxes.append([px, py, px + cw, py + ch])
                new_labels.append(PLATELET_LABEL)
                placed = True
                break
        if not placed:
            continue  # skip a paste rather than allow ambiguous overlapping boxes

    if new_boxes:
        boxes = np.concatenate([boxes, np.asarray(new_boxes, dtype=np.float32)], axis=0)
        labels = np.concatenate([labels, np.asarray(new_labels, dtype=np.int64)], axis=0)

    return out_img, boxes, labels


# ----------------------------------------------------------------------------
# Focal loss (small-object upgrade #4: "use focal loss or a class-balanced
# / hard-example-mining scheme so tiny-box positives aren't drowned out")
# ----------------------------------------------------------------------------
def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    """Standard RetinaNet-style sigmoid focal loss (Lin et al., 2017).
    logits/targets: same shape, targets in {0,1}. Implemented with plain
    torch ops (no torchvision-version-dependent import) so it is usable
    from every custom-loop training script in this folder.
    """
    import torch
    import torch.nn.functional as Fnn

    p = torch.sigmoid(logits)
    ce = Fnn.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


class FocalLoss:
    """Thin callable wrapper around sigmoid_focal_loss for use as a
    drop-in classification-loss object in custom training loops."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum'):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, logits, targets):
        return sigmoid_focal_loss(logits, targets, self.alpha, self.gamma, self.reduction)


# ----------------------------------------------------------------------------
# Environment / provenance info -- goes into every run_report so the paper
# can state exactly what hardware/software produced each number.
# ----------------------------------------------------------------------------
def collect_environment_info():
    info = {
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'processor': platform.processor() or platform.machine(),
        'hostname': socket.gethostname(),
        'on_raspberry_pi': os.environ.get('BCC_ON_RASPBERRY_PI', '0') == '1',
    }
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
        info['git_commit'] = out
    except Exception:
        info['git_commit'] = None
    return info


# ----------------------------------------------------------------------------
# Standardized "paper-ready" run report writer
# ----------------------------------------------------------------------------
def write_run_report(out_dir: Path, model_key: str, model_display_name: str, description: str,
                      hyperparams: dict, dataset_stats: dict, timing: dict, metrics: dict,
                      files: dict, notes=None):
    """Writes output/<model_key>/run_report_<model_key>.json and a matching
    .md rendering. This is the single file meant to be handed to an LLM (or
    a human) writing the paper's Methods/Results section for this model --
    it intentionally repeats information already present in the CSV/PNG
    outputs, in prose-adjacent form, so nothing needs to be re-derived.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'model_key': model_key,
        'model_display_name': model_display_name,
        'description': description,
        'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'environment': collect_environment_info(),
        'hyperparameters': hyperparams,
        'dataset_stats': dataset_stats,
        'timing': timing,
        'metrics': metrics,
        'output_files': files,
        'notes': notes or [],
    }

    json_path = out_dir / f'run_report_{model_key}.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    md_lines = [
        f"# Run report: {model_display_name} on BCCD",
        "",
        description,
        "",
        "## Hyperparameters",
        "",
        "| Key | Value |",
        "|---|---|",
    ]
    for k, v in hyperparams.items():
        md_lines.append(f"| {k} | {v} |")
    md_lines += ["", "## Dataset", "", "| Key | Value |", "|---|---|"]
    for k, v in dataset_stats.items():
        md_lines.append(f"| {k} | {v} |")
    md_lines += ["", "## Timing", "", "| Key | Value |", "|---|---|"]
    for k, v in timing.items():
        md_lines.append(f"| {k} | {v} |")
    md_lines += ["", "## Metrics", "", "| Key | Value |", "|---|---|"]
    for k, v in metrics.items():
        md_lines.append(f"| {k} | {v} |")
    md_lines += ["", "## Output files", ""]
    for k, v in files.items():
        md_lines.append(f"- **{k}**: `{v}`")
    if notes:
        md_lines += ["", "## Notes", ""]
        for n in notes:
            md_lines.append(f"- {n}")
    md_lines += ["", "## Environment", "", "```json", json.dumps(report['environment'], indent=2), "```"]

    md_path = out_dir / f'run_report_{model_key}.md'
    md_path.write_text('\n'.join(md_lines))
    print(f'[INFO] Run report written -> {json_path} , {md_path}')
    return report


def write_onnx_meta(onnx_path: Path, input_size, mean, std, letterbox, class_map, extra=None):
    """Sidecar JSON next to a <model>.onnx export describing exactly how to
    preprocess/postprocess it, so bcc_metrics/scripts/models_io.py's
    generic OnnxDetector can run ANY model in this file (YOLO, NanoDet,
    EfficientDet, RT-DETR, RTMDet, SSDLite-v2, ...) through one code path.

    input_size: (H, W) the model expects.
    mean/std:   per-channel normalization applied to a [0,1] float image
                (mean=[0,0,0], std=[1,1,1] if the model's own preprocessing
                is baked into the exported graph, e.g. Ultralytics exports).
    letterbox:  whether preprocessing should letterbox-pad to input_size
                (True for Ultralytics-style models) or plain resize
                (False for torchvision-style SSD/EfficientDet models).
    class_map:  dict mapping the exported model's raw output class index
                (0-indexed, no background) -> DET_CLASSES name, e.g.
                {"0": "RBC", "1": "WBC", "2": "Platelets"}.
    """
    meta = {
        'input_size_hw': list(input_size),
        'mean': list(mean),
        'std': list(std),
        'letterbox': bool(letterbox),
        'class_map': class_map,
    }
    if extra:
        meta.update(extra)
    meta_path = Path(str(onnx_path) + '.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[INFO] ONNX meta written -> {meta_path}')
    return meta_path


# ----------------------------------------------------------------------------
# Shared prediction-grid visualization (matches the look of
# train_bccd_ssdlite_detection.py::draw_predictions so every model's sample
# grid is directly visually comparable in the paper).
# ----------------------------------------------------------------------------
def draw_prediction_grid(records, predict_fn, out_path, score_thr=0.35, n=6):
    """predict_fn(PIL.Image) -> (boxes[N,4] xyxy, labels[N] 1-indexed incl.
    background at 0, scores[N]). Saves a 3x2 grid PNG like the baseline
    script's draw_predictions()."""
    chosen = records[:min(n, len(records))]
    cols = 3
    rows = (len(chosen) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * 320, rows * 260), 'white')

    for i, rec in enumerate(chosen):
        img = Image.open(rec['image_path']).convert('RGB')
        boxes, labels, scores = predict_fn(img)

        tile = img.resize((320, 220))
        sx, sy = 320 / img.width, 220 / img.height
        draw = ImageDraw.Draw(tile)
        counts = Counter()

        for box, label, score in zip(boxes, labels, scores):
            if score < score_thr:
                continue
            lbl = int(label)
            counts[lbl] += 1
            x1, y1, x2, y2 = box
            color = COLORS.get(lbl, 'yellow')
            draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=color, width=2)
            draw.text((x1 * sx + 2, y1 * sy + 2), f'{CLASSES[lbl]}:{score:.2f}', fill=color)

        board = Image.new('RGB', (320, 260), 'white')
        board.paste(tile, (0, 0))
        ImageDraw.Draw(board).text(
            (8, 228), f'RBC:{counts[1]}  WBC:{counts[2]}  PLT:{counts[3]}', fill='black'
        )
        canvas.paste(board, ((i % cols) * 320, (i // cols) * 260))

    canvas.save(out_path)
    print(f'[INFO] Saved sample predictions -> {out_path}')
