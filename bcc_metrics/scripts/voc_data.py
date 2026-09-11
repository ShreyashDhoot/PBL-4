#!/usr/bin/env python3
"""
voc_data.py
===========
All VOC-XML parsing and dataset-building logic lives here so every other
script (detection eval, CV%, Bland-Altman/ICC, method-comparison, robustness)
shares exactly one definition of "how do we read the ground truth".

Two ground-truth sources are read:

1. BCCD_Dataset-master/BCCD  -> the original 364-image BCCD dataset that
   train_bccd_ssdlite_detection.py trains/validates/tests on. We reuse the
   *exact same* deterministic 70/15/15 split (same seed, same shuffle logic)
   so "test set" here means the same test set the model never saw.

2. annotations/annotations   -> 72 images, independently re-labelled by a
   biomed student. Per Comparison_Metrics_Notes.docx, this is treated as a
   held-out, out-of-distribution validation set (never used for training)
   and doubles as the paired (system_count, manual_count) dataset for the
   agreement/method-comparison analysis (Bland-Altman, ICC, Passing-Bablok).
"""

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from common import (
    DET_CLASS_TO_IDX,
    DET_CLASSES,
    CELL_TYPES,
    BCCD_ANNOTATIONS_DIR,
    BCCD_IMAGES_DIR,
    SEVENTYTWO_IMAGES_DIR,
    SEVENTYTWO_ANNOT_DIR,
    RANDOM_SEED,
    log,
)


# ----------------------------------------------------------------------------
# Generic VOC XML -> record parsing
# ----------------------------------------------------------------------------
def parse_voc_xml(xml_path: Path, img_dir: Path):
    """Parse one VOC annotation file into a record dict.
    Mirrors parse_annotation() in train_bccd_ssdlite_detection.py so results
    are directly comparable with what the detector was trained/evaluated on.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename")

    # The 72-set filenames end in .jpeg while the folder that ships with it
    # is literally named "images" (see annotations.zip); handle both cases,
    # and fall back to matching by stem if the extension in the XML doesn't
    # match what's on disk (annotators sometimes typed .jpg vs .jpeg).
    candidate = img_dir / filename
    if not candidate.exists():
        stem = Path(filename).stem
        matches = list(img_dir.glob(stem + ".*"))
        if matches:
            candidate = matches[0]

    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in DET_CLASS_TO_IDX or name == "__background__":
            continue
        bbox = obj.find("bndbox")
        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(DET_CLASS_TO_IDX[name])

    return {
        "image_id": Path(xml_path).stem,
        "xml_path": str(xml_path),
        "image_path": str(candidate),
        "filename": filename,
        "boxes": boxes,
        "labels": labels,
    }


def build_bccd_records():
    """All 364 BCCD images with valid boxes (same filter as training script)."""
    records = [
        parse_voc_xml(xml_file, BCCD_IMAGES_DIR)
        for xml_file in sorted(BCCD_ANNOTATIONS_DIR.glob("*.xml"))
    ]
    clean = [r for r in records if len(r["boxes"]) > 0]
    log(f"BCCD: loaded {len(clean)} valid images (dropped {len(records) - len(clean)} with no boxes).")
    return clean


def split_bccd_records(records, seed=RANDOM_SEED):
    """Reproduce the exact 70/15/15 split used in train_bccd_ssdlite_detection.py
    (same seed, same np.random.default_rng shuffle) so 'test set' metrics here
    refer to images the model never trained/validated on.
    """
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


def build_72_records():
    """The 72 biomed-student-labelled images (independent re-annotation)."""
    records = [
        parse_voc_xml(xml_file, SEVENTYTWO_IMAGES_DIR)
        for xml_file in sorted(SEVENTYTWO_ANNOT_DIR.glob("*.xml"))
    ]
    clean = [r for r in records if len(r["boxes"]) > 0]
    log(f"72-set: loaded {len(clean)} valid images (dropped {len(records) - len(clean)} with no boxes).")
    return clean


def class_counts(records):
    """Counter of box counts per cell type across a list of records."""
    counts = Counter()
    for r in records:
        for l in r["labels"]:
            counts[DET_CLASSES[l]] += 1
    return counts


def per_image_cell_counts(record):
    """{'RBC': n, 'WBC': n, 'Platelets': n} counts for one record."""
    counts = {c: 0 for c in CELL_TYPES}
    for l in record["labels"]:
        name = DET_CLASSES[l]
        if name in counts:
            counts[name] += 1
    return counts


def build_paired_manual_dataset():
    """The core paired dataset for Part 2/6 of the notes:
    for every one of the 72 biomed-labelled images, the *manual* (human)
    per-cell-type counts, keyed by image_id. The *system* counts are filled
    in later (in detection_eval.py / agreement_stats.py) by running the
    trained detector on the same images.

    Returns a list of dicts: {image_id, image_path, manual_RBC, manual_WBC,
    manual_Platelets}
    """
    records = build_72_records()
    rows = []
    for r in records:
        counts = per_image_cell_counts(r)
        row = {"image_id": r["image_id"], "image_path": r["image_path"]}
        for c in CELL_TYPES:
            row[f"manual_{c}"] = counts[c]
        rows.append(row)
    return rows


if __name__ == "__main__":
    # Quick smoke test / summary when run directly.
    bccd = build_bccd_records()
    train, val, test = split_bccd_records(bccd)
    log(f"BCCD split -> train:{len(train)} val:{len(val)} test:{len(test)}")
    log(f"BCCD class counts: {dict(class_counts(bccd))}")

    seventytwo = build_72_records()
    log(f"72-set class counts: {dict(class_counts(seventytwo))}")

    paired = build_paired_manual_dataset()
    log(f"Paired manual dataset rows: {len(paired)}")
    if paired:
        log(f"Sample row: {paired[0]}")
