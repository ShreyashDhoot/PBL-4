#!/usr/bin/env python3
"""
detection_metrics.py
====================
Pure-numpy implementations of the detection-side metrics checklist from the
notes (Section 7):

  - IoU + greedy Hungarian-free matching between predicted and ground-truth
    boxes, per class.
  - Confusion matrix over {RBC, WBC, Platelets, background/missed, extra}.
  - Precision / Recall / F1 per class and overall (micro + macro).
  - Wilson score confidence interval for accuracy (a proportion), since the
    notes flag that a bare percentage over-states precision on a ~72-364
    image test set.
  - mAP@0.5 and mAP@0.5:0.95 (COCO-style, 10 IoU thresholds), computed from
    scratch (no pycocotools dependency) so it runs anywhere torch does.

No script here trains anything; everything operates on (ground_truth, model
prediction) pairs supplied by the caller (see detection_eval.py).
"""

import numpy as np


# ----------------------------------------------------------------------------
# IoU + matching
# ----------------------------------------------------------------------------
def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Vectorized IoU between two sets of xyxy boxes. Returns [N, M]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return iou


def greedy_match(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thr=0.5):
    """Greedy matching: sort predictions by score (desc), match each to the
    highest-IoU unmatched GT box of the *same class* if IoU >= iou_thr.
    Returns:
        matches: list of (pred_idx, gt_idx) for true positives
        fp_idx:  predicted indices with no match (false positives)
        fn_idx:  gt indices never matched (false negatives / missed)
    """
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    matches, fp_idx, fn_idx = [], [], []

    if n_pred == 0:
        return matches, fp_idx, list(range(n_gt))
    if n_gt == 0:
        return matches, list(range(n_pred)), fn_idx

    order = np.argsort(-np.asarray(pred_scores))
    iou = box_iou(np.asarray(pred_boxes), np.asarray(gt_boxes))
    gt_taken = np.zeros(n_gt, dtype=bool)

    for p in order:
        best_j, best_iou = -1, iou_thr
        for j in range(n_gt):
            if gt_taken[j] or pred_labels[p] != gt_labels[j]:
                continue
            if iou[p, j] >= best_iou:
                best_iou = iou[p, j]
                best_j = j
        if best_j >= 0:
            matches.append((int(p), int(best_j)))
            gt_taken[best_j] = True
        else:
            fp_idx.append(int(p))

    fn_idx = [j for j in range(n_gt) if not gt_taken[j]]
    return matches, fp_idx, fn_idx


# ----------------------------------------------------------------------------
# Confusion matrix + PRF1 across a whole dataset
# ----------------------------------------------------------------------------
def accumulate_confusion(all_predictions, all_ground_truths, class_names, iou_thr=0.5, score_thr=0.35):
    """
    all_predictions / all_ground_truths: parallel lists, one entry per image:
        predictions[i] = (boxes[N,4], labels[N], scores[N])
        ground_truths[i] = (boxes[M,4], labels[M])
    labels are 1-indexed class ids matching class_names[label-1]... actually
    we expect labels to already be 0-indexed into class_names for simplicity.

    Returns a (C+1, C+1) confusion matrix where the last row/col is
    'background' (index C): predicted-but-no-GT (row C = FP) and
    GT-but-not-predicted (col C = FN / missed).
    """
    C = len(class_names)
    cm = np.zeros((C + 1, C + 1), dtype=np.int64)  # rows = predicted, cols = truth

    for (p_boxes, p_labels, p_scores), (g_boxes, g_labels) in zip(all_predictions, all_ground_truths):
        keep = np.asarray(p_scores) >= score_thr
        p_boxes = np.asarray(p_boxes)[keep]
        p_labels = np.asarray(p_labels)[keep]
        p_scores = np.asarray(p_scores)[keep]

        matches, fp_idx, fn_idx = greedy_match(p_boxes, p_labels, p_scores, g_boxes, g_labels, iou_thr)

        for p_i, g_i in matches:
            cm[int(p_labels[p_i]), int(g_labels[g_i])] += 1
        for p_i in fp_idx:
            cm[int(p_labels[p_i]), C] += 1  # predicted class, no matching truth -> background column
        for g_i in fn_idx:
            cm[C, int(g_labels[g_i])] += 1  # missed -> background row

    return cm


def prf1_from_confusion(cm: np.ndarray, class_names):
    """Per-class precision/recall/F1 plus micro and macro averages from the
    (C+1)x(C+1) confusion matrix produced by accumulate_confusion.
    """
    C = len(class_names)
    rows = []
    tp_sum = fp_sum = fn_sum = 0

    for c in range(C):
        tp = cm[c, c]
        fp = cm[c, :].sum() - tp  # predicted c, truth != c (incl. background col)
        fn = cm[:, c].sum() - tp  # truth c, predicted != c (incl. background row)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            "class": class_names[c],
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": precision, "recall": recall, "f1": f1,
        })
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    macro_p = np.mean([r["precision"] for r in rows])
    macro_r = np.mean([r["recall"] for r in rows])
    macro_f1 = np.mean([r["f1"] for r in rows])

    accuracy = tp_sum / cm.sum() if cm.sum() > 0 else 0.0  # overall correct-match rate

    return {
        "per_class": rows,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "accuracy": accuracy,
        "n_matched_tp": int(tp_sum),
        "n_total_events": int(cm.sum()),
    }


# ----------------------------------------------------------------------------
# Wilson score interval (accuracy is a proportion; the notes call for this
# instead of a bare percentage given the small n involved)
# ----------------------------------------------------------------------------
def wilson_score_interval(successes: int, n: int, confidence: float = 0.95):
    if n == 0:
        return (0.0, 0.0, 0.0)
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denom = 1 + z ** 2 / n
    center = p_hat + z ** 2 / (2 * n)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return p_hat, max(0.0, lower), min(1.0, upper)


# ----------------------------------------------------------------------------
# mAP@0.5 and mAP@0.5:0.95 (COCO-style), implemented from scratch
# ----------------------------------------------------------------------------
def average_precision_for_class(all_preds_this_class, all_gts_this_class, iou_thr):
    """
    all_preds_this_class: list over images of (boxes[N,4], scores[N])
    all_gts_this_class:   list over images of boxes[M,4]
    Standard 101-point interpolated AP (COCO-style) at a single IoU threshold.
    """
    # Flatten all predictions across images with an image index, sort by score desc.
    flat = []
    n_gt_total = 0
    gt_taken = []
    for img_idx, ((boxes, scores), gts) in enumerate(zip(all_preds_this_class, all_gts_this_class)):
        for b, s in zip(boxes, scores):
            flat.append((img_idx, b, s))
        n_gt_total += len(gts)
        gt_taken.append(np.zeros(len(gts), dtype=bool))

    if n_gt_total == 0:
        return None  # class absent from GT entirely; excluded from mAP average
    if len(flat) == 0:
        return 0.0  # never predicted -> AP is 0

    flat.sort(key=lambda t: -t[2])

    tp = np.zeros(len(flat))
    fp = np.zeros(len(flat))

    for i, (img_idx, box, score) in enumerate(flat):
        gts = all_gts_this_class[img_idx]
        if len(gts) == 0:
            fp[i] = 1
            continue
        ious = box_iou(np.asarray([box]), np.asarray(gts))[0]
        best_j = int(np.argmax(ious)) if len(ious) else -1
        if best_j >= 0 and ious[best_j] >= iou_thr and not gt_taken[img_idx][best_j]:
            tp[i] = 1
            gt_taken[img_idx][best_j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(n_gt_total, 1e-9)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # 101-point interpolation (COCO convention)
    recall_thresholds = np.linspace(0, 1, 101)
    interpolated = np.zeros(101)
    for i, r in enumerate(recall_thresholds):
        mask = recalls >= r
        interpolated[i] = precisions[mask].max() if mask.any() else 0.0
    return float(interpolated.mean())


def compute_map(all_predictions, all_ground_truths, class_names, iou_thresholds):
    """
    all_predictions[i] = (boxes, labels, scores) for image i (0-indexed labels
    into class_names)
    all_ground_truths[i] = (boxes, labels) for image i
    Returns dict: {iou_thr: {class: AP or None}, 'mAP_per_iou': {...},
                   'mAP_0.5': x, 'mAP_0.5:0.95': y, 'per_class_mAP_0.5:0.95': {...}}
    """
    C = len(class_names)
    per_iou_per_class = {}

    for iou_thr in iou_thresholds:
        per_class_ap = {}
        for c in range(C):
            preds_c, gts_c = [], []
            for (p_boxes, p_labels, p_scores), (g_boxes, g_labels) in zip(all_predictions, all_ground_truths):
                p_boxes = np.asarray(p_boxes)
                p_labels = np.asarray(p_labels)
                p_scores = np.asarray(p_scores)
                g_boxes = np.asarray(g_boxes)
                g_labels = np.asarray(g_labels)

                mask_p = p_labels == c
                mask_g = g_labels == c
                preds_c.append((p_boxes[mask_p] if len(p_boxes) else np.zeros((0, 4)),
                                 p_scores[mask_p] if len(p_scores) else np.zeros((0,))))
                gts_c.append(g_boxes[mask_g] if len(g_boxes) else np.zeros((0, 4)))

            ap = average_precision_for_class(preds_c, gts_c, iou_thr)
            per_class_ap[class_names[c]] = ap
        per_iou_per_class[iou_thr] = per_class_ap

    def _mean_ap(per_class_ap):
        vals = [v for v in per_class_ap.values() if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    mAP_per_iou = {iou_thr: _mean_ap(pc) for iou_thr, pc in per_iou_per_class.items()}
    map_50 = mAP_per_iou.get(0.5, None)
    map_50_95 = float(np.mean(list(mAP_per_iou.values())))

    per_class_50_95 = {}
    for c in range(C):
        vals = [per_iou_per_class[t][class_names[c]] for t in iou_thresholds
                if per_iou_per_class[t][class_names[c]] is not None]
        per_class_50_95[class_names[c]] = float(np.mean(vals)) if vals else None

    return {
        "per_iou_per_class": per_iou_per_class,
        "mAP_per_iou": mAP_per_iou,
        "mAP_0.5": map_50,
        "mAP_0.5:0.95": map_50_95,
        "per_class_mAP_0.5:0.95": per_class_50_95,
    }
