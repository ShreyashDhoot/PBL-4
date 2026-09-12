#!/usr/bin/env python3
"""
models_io.py
============
Loads the already-trained checkpoints produced by
train_bccd_ssdlite_detection.py and train_efficientnet_bccd.py, and exposes
a small, uniform inference API used by every metrics script:

    detector = load_ssdlite_detector()
    boxes, labels, scores = detector.predict(pil_image)

    clf = load_efficientnet_classifier()
    probs = clf.predict_proba(pil_image)   # per-class presence probabilities

If a checkpoint is missing, each loader raises a clear RuntimeError with
instructions (run the corresponding train_*.py first) instead of silently
producing nonsense numbers.

Quantization
------------
`quantize_detector(model, mode)` returns a dynamically-quantized copy of the
SSDLite model for the FP32 / INT8 comparison requested in next-step item #1
of the notes ("quantize the model (2-3 levels), re-measure accuracy /
precision / recall / comparison metric"). We implement the three standard,
CPU-portable levels that require no special hardware:

    fp32      - the original trained weights, no quantization
    fp16      - half-precision weights (smaller, same op set)
    dynamic_int8 - torch.quantization.quantize_dynamic on Linear/Conv layers

A true static/QAT INT8 pipeline needs a representative calibration pass and
backend-specific tooling (fbgemm/qnnpack); dynamic quantization is the
standard "2-3 levels, no retraining needed" option and is what actually
ships in practice for edge CPUs like the Raspberry Pi 4, so we use that as
the third level.
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from common import (
    DET_CLASSES,
    SSDLITE_CKPT,
    EFFICIENTNET_CKPT,
    log,
)

CELL_CLASSES = ["RBC", "WBC", "Platelets"]  # EfficientNet head order (train_efficientnet_bccd.py)

_TORCH_IMPORT_ERROR = None
try:
    import torch
    import torch.nn as nn
    from torchvision.transforms import functional as TF
    from torchvision import transforms
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from torchvision.models import MobileNet_V3_Large_Weights, efficientnet_b0, EfficientNet_B0_Weights
except Exception as e:  # pragma: no cover - exercised only when torch is absent
    _TORCH_IMPORT_ERROR = e


def _require_torch():
    if _TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyTorch/torchvision are required to run model inference but are not "
            f"importable in this environment ({_TORCH_IMPORT_ERROR}). Install with:\n"
            "    pip install torch torchvision\n"
            "This only affects scripts that need live model predictions "
            "(detection_eval.py, cv_repeatability.py, agreement_stats.py, "
            "quantization_bench.py, robustness_eval.py, calibration_eval.py). "
            "Scripts that only need ground-truth VOC data (voc_data.py) still work."
        )


class SSDLiteDetector:
    """Thin wrapper around the trained ssdlite320_mobilenet_v3_large model."""

    def __init__(self, device="cpu", quantize_mode="fp32"):
        _require_torch()
        if not SSDLITE_CKPT.exists():
            raise RuntimeError(
                f"SSDLite checkpoint not found at {SSDLITE_CKPT}. "
                "Run `python train_bccd_ssdlite_detection.py` first (from the pbl-4 repo) "
                "to produce output/ssdlite_bccd_best.pth."
            )
        self.device = device
        self.quantize_mode = quantize_mode

        model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        num_classes=len(DET_CLASSES),)
        
        state = torch.load(str(SSDLITE_CKPT), map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        model = self._apply_quantization(model, quantize_mode)
        self.model = model.to(device if quantize_mode != "dynamic_int8" else "cpu")
        # dynamic INT8 quantized modules only run on CPU in current PyTorch.
        if quantize_mode == "dynamic_int8":
            self.device = "cpu"

    @staticmethod
    def _apply_quantization(model, mode):
        if mode == "fp32":
            return model
        if mode == "fp16":
            return model.half()
        if mode == "dynamic_int8":
            # Dynamic quantization of Linear/Conv layers is the standard
            # zero-calibration INT8 path for CPU deployment.
            return torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        raise ValueError(f"Unknown quantize_mode: {mode}")

    def predict(self, image: "Image.Image", score_thr: float = 0.0):
        """Run inference on a single PIL image (RGB).
        Returns (boxes[N,4] xyxy in original pixel coords, labels[N] int,
        scores[N] float), filtered to score >= score_thr.
        """
        x = TF.to_tensor(image)
        if self.quantize_mode == "fp16":
            x = x.half()
        x = x.to(self.device)
        with torch.no_grad():
            pred = self.model([x])[0]
        boxes = pred["boxes"].detach().cpu().float().numpy()
        labels = pred["labels"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().float().numpy()
        keep = scores >= score_thr
        return boxes[keep], labels[keep], scores[keep]

    def count_cells(self, image: "Image.Image", score_thr: float = 0.35):
        """Return {'RBC': n, 'WBC': n, 'Platelets': n} counts for one image."""
        _, labels, _ = self.predict(image, score_thr=score_thr)
        out = {c: 0 for c in CELL_CLASSES}
        for l in labels:
            name = DET_CLASSES[int(l)]
            if name in out:
                out[name] += 1
        return out

    def timed_predict(self, image: "Image.Image", score_thr: float = 0.35):
        """Predict + wall-clock latency in milliseconds (for edge timing table)."""
        t0 = time.perf_counter()
        boxes, labels, scores = self.predict(image, score_thr=score_thr)
        t1 = time.perf_counter()
        return boxes, labels, scores, (t1 - t0) * 1000.0

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def state_dict_size_mb(self):
        """Approximate on-disk size of the model's parameters in MB, useful
        for the FP32 vs FP16 vs INT8 memory-footprint comparison."""
        total_bytes = 0
        for p in self.model.parameters():
            total_bytes += p.numel() * p.element_size()
        for b in self.model.buffers():
            total_bytes += b.numel() * b.element_size()
        return total_bytes / (1024 * 1024)


class EfficientNetClassifier:
    """Thin wrapper around the trained multi-label EfficientNet-B0 head.
    Note (see train_efficientnet_bccd.py): this model does *image-level,
    multi-label presence* classification (does this image contain RBC/WBC/
    Platelets), not per-box classification. It is reported separately from
    the SSDLite detector's per-box accuracy, matching how the original
    training script frames it.
    """

    def __init__(self, device="cpu", quantize_mode="fp32"):
        _require_torch()
        if not EFFICIENTNET_CKPT.exists():
            raise RuntimeError(
                f"EfficientNet-B0 checkpoint not found at {EFFICIENTNET_CKPT}. "
                "Run `python train_efficientnet_bccd.py` first (from the pbl-4 repo) "
                "to produce output/efficientnet_bccd_best.pth."
            )
        self.device = device
        self.quantize_mode = quantize_mode

        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(CELL_CLASSES))
        state = torch.load(str(EFFICIENTNET_CKPT), map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        model = SSDLiteDetector._apply_quantization(model, quantize_mode)
        self.model = model.to(device if quantize_mode != "dynamic_int8" else "cpu")
        if quantize_mode == "dynamic_int8":
            self.device = "cpu"

        self.tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict_proba(self, image: "Image.Image"):
        x = self.tfm(image).unsqueeze(0)
        if self.quantize_mode == "fp16":
            x = x.half()
        x = x.to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits.float())[0].cpu().numpy()
        return probs  # order matches CELL_CLASSES

    def timed_predict_proba(self, image: "Image.Image"):
        t0 = time.perf_counter()
        probs = self.predict_proba(image)
        t1 = time.perf_counter()
        return probs, (t1 - t0) * 1000.0

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def state_dict_size_mb(self):
        total_bytes = 0
        for p in self.model.parameters():
            total_bytes += p.numel() * p.element_size()
        for b in self.model.buffers():
            total_bytes += b.numel() * b.element_size()
        return total_bytes / (1024 * 1024)


def load_ssdlite_detector(device="cpu", quantize_mode="fp32"):
    return SSDLiteDetector(device=device, quantize_mode=quantize_mode)


def load_efficientnet_classifier(device="cpu", quantize_mode="fp32"):
    return EfficientNetClassifier(device=device, quantize_mode=quantize_mode)


class SSDLiteV2Detector:
    """Native loader for the small-object-optimized SSDLite v2 model (see
    pbl-4/train_bccd_ssdlite_v2_detection.py). Reuses that script's own
    `build_model_v2()` so the reconstructed architecture (custom anchor
    generator, 512px size, focal-loss head) is guaranteed to match exactly
    what the checkpoint was trained with, instead of duplicating that
    construction logic here."""

    def __init__(self, device="cpu", quantize_mode="fp32", img_size=512):
        _require_torch()
        from common import PBL4_DIR
        ckpt = PBL4_DIR / "output" / "ssdlite_v2" / "ssdlite_v2_bccd_best.pth"
        if not ckpt.exists():
            raise RuntimeError(
                f"SSDLite v2 checkpoint not found at {ckpt}. Run "
                "`python train_bccd_ssdlite_v2_detection.py` first (from the pbl-4 repo)."
            )
        import sys
        if str(PBL4_DIR) not in sys.path:
            sys.path.insert(0, str(PBL4_DIR))
        from train_bccd_ssdlite_v2_detection import build_model_v2  # noqa: E402

        self.device = device
        self.quantize_mode = quantize_mode
        self.img_size = img_size

        model = build_model_v2(pretrained_backbone=False)
        state = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        model = SSDLiteDetector._apply_quantization(model, quantize_mode)
        self.model = model.to(device if quantize_mode != "dynamic_int8" else "cpu")
        if quantize_mode == "dynamic_int8":
            self.device = "cpu"

    def predict(self, image: "Image.Image", score_thr: float = 0.0):
        w0, h0 = image.width, image.height
        resized = image.resize((self.img_size, self.img_size))
        x = TF.to_tensor(resized)
        if self.quantize_mode == "fp16":
            x = x.half()
        x = x.to(self.device)
        with torch.no_grad():
            pred = self.model([x])[0]
        boxes = pred["boxes"].detach().cpu().float().numpy()
        labels = pred["labels"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().float().numpy()
        if len(boxes):
            sx, sy = w0 / self.img_size, h0 / self.img_size
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
        keep = scores >= score_thr
        return boxes[keep], labels[keep], scores[keep]

    def count_cells(self, image, score_thr=0.35):
        _, labels, _ = self.predict(image, score_thr=score_thr)
        out = {c: 0 for c in CELL_CLASSES}
        for l in labels:
            name = DET_CLASSES[int(l)]
            if name in out:
                out[name] += 1
        return out

    def timed_predict(self, image, score_thr=0.35):
        t0 = time.perf_counter()
        boxes, labels, scores = self.predict(image, score_thr=score_thr)
        t1 = time.perf_counter()
        return boxes, labels, scores, (t1 - t0) * 1000.0

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def state_dict_size_mb(self):
        total_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        total_bytes += sum(b.numel() * b.element_size() for b in self.model.buffers())
        return total_bytes / (1024 * 1024)


def load_ssdlite_v2_detector(device="cpu", quantize_mode="fp32"):
    return SSDLiteV2Detector(device=device, quantize_mode=quantize_mode)


# ----------------------------------------------------------------------------
# Generic ONNX runtime detector -- one code path for every non-torchvision
# model exported by the train_bccd_*.py scripts (YOLOv8n-P2, YOLO11n-P2,
# RT-DETR, NanoDet-Plus-style, EfficientDet-Lite0, and optionally
# RTMDet-tiny if an mmdeploy export exists). Reads the `<onnx>.meta.json`
# sidecar each training script writes via bccd_data_utils.write_onnx_meta()
# to know how to preprocess (letterbox vs. plain resize, normalization) and
# how to parse the output (two supported layouts, see meta['output_layout']).
# ----------------------------------------------------------------------------
_ORT_IMPORT_ERROR = None
try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover
    _ORT_IMPORT_ERROR = e


def _require_onnxruntime():
    if _ORT_IMPORT_ERROR is not None:
        raise RuntimeError(
            "onnxruntime is required to evaluate the ONNX-exported models (everything except "
            f"the two native torchvision SSDLite models) but is not importable ({_ORT_IMPORT_ERROR}). "
            "Install with:\n    pip install onnxruntime\n"
        )


def _letterbox(image: "Image.Image", size):
    """Resize preserving aspect ratio, pad with gray (114,114,114) to
    `size` (H,W) -- matches Ultralytics' own preprocessing convention, so
    ONNX exports with letterbox=true in their meta.json decode correctly.
    Returns (padded_image, scale, pad_left, pad_top)."""
    H, W = size
    w0, h0 = image.width, image.height
    scale = min(W / w0, H / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    resized = image.resize((nw, nh))
    canvas = Image.new("RGB", (W, H), (114, 114, 114))
    pad_left, pad_top = (W - nw) // 2, (H - nh) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, scale, pad_left, pad_top


def _nms_numpy(boxes, scores, labels, iou_thr=0.55):
    """Plain-numpy class-aware NMS, used only for the 'raw_yolo_head'
    fallback layout (installed ultralytics too old for export(nms=True))."""
    keep_all = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        b, s = boxes[idx], scores[idx]
        order = s.argsort()[::-1]
        picked = []
        while order.size > 0:
            i = order[0]
            picked.append(idx[i])
            if order.size == 1:
                break
            xx1 = np.maximum(b[i, 0], b[order[1:], 0])
            yy1 = np.maximum(b[i, 1], b[order[1:], 1])
            xx2 = np.minimum(b[i, 2], b[order[1:], 2])
            yy2 = np.minimum(b[i, 3], b[order[1:], 3])
            w = np.clip(xx2 - xx1, 0, None)
            h = np.clip(yy2 - yy1, 0, None)
            inter = w * h
            area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
            area_o = (b[order[1:], 2] - b[order[1:], 0]) * (b[order[1:], 3] - b[order[1:], 1])
            iou = inter / np.maximum(area_i + area_o - inter, 1e-9)
            order = order[1:][iou <= iou_thr]
        keep_all.extend(picked)
    return np.array(sorted(keep_all), dtype=int)


class OnnxDetector:
    """Generic onnxruntime-based detector matching the same predict() /
    count_cells() / timed_predict() interface as SSDLiteDetector, driven
    entirely by the `<onnx_path>.meta.json` sidecar (see
    bccd_data_utils.write_onnx_meta in pbl-4/)."""

    def __init__(self, onnx_path, device="cpu"):
        _require_onnxruntime()
        onnx_path = Path(onnx_path)
        meta_path = Path(str(onnx_path) + ".meta.json")
        if not onnx_path.exists():
            raise RuntimeError(f"ONNX file not found at {onnx_path}. Train/export that model first.")
        if not meta_path.exists():
            raise RuntimeError(f"ONNX meta sidecar not found at {meta_path} (expected alongside the .onnx file).")

        with open(meta_path) as f:
            self.meta = json.load(f)

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.size_hw = tuple(self.meta["input_size_hw"])
        self.mean = np.asarray(self.meta.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.std = np.asarray(self.meta.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)
        self.letterbox = bool(self.meta.get("letterbox", False))
        self.output_layout = self.meta.get("output_layout", "separate_boxes_scores_labels")
        self.onnx_path = onnx_path

    def _preprocess(self, image: "Image.Image"):
        H, W = self.size_hw
        if self.letterbox:
            canvas, scale, pad_left, pad_top = _letterbox(image, (H, W))
            transform_info = ("letterbox", scale, pad_left, pad_top)
        else:
            canvas = image.resize((W, H))
            sx, sy = image.width / W, image.height / H
            transform_info = ("resize", sx, sy)

        arr = np.asarray(canvas).astype(np.float32) / 255.0  # HWC, RGB, [0,1]
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)  # 1,C,H,W
        return arr, transform_info

    def _undo_transform(self, boxes, transform_info):
        if len(boxes) == 0:
            return boxes
        boxes = boxes.copy()
        if transform_info[0] == "letterbox":
            _, scale, pad_left, pad_top = transform_info
            boxes[:, [0, 2]] -= pad_left
            boxes[:, [1, 3]] -= pad_top
            boxes /= scale
        else:
            _, sx, sy = transform_info
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
        return boxes

    def predict(self, image: "Image.Image", score_thr: float = 0.0):
        arr, transform_info = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: arr})

        if self.output_layout == "single_array_xyxy_conf_cls":
            det = outputs[0]
            det = np.asarray(det).reshape(-1, det.shape[-1])  # [N,6]
            keep = det[:, 4] > 1e-6  # exported graph already NMS'd + conf-filtered
            det = det[keep]
            boxes, scores = det[:, :4], det[:, 4]
            labels = det[:, 5].astype(int) + 1  # -> 1-indexed w/ background at 0

        elif self.output_layout == "raw_yolo_head":
            raw = np.asarray(outputs[0])[0]  # [4+nc, num_anchors] (Ultralytics raw export convention)
            boxes_xywh, cls_scores = raw[:4, :].T, raw[4:, :].T
            labels_raw = cls_scores.argmax(axis=1)
            scores = cls_scores.max(axis=1)
            keep0 = scores > 0.05
            boxes_xywh, scores, labels_raw = boxes_xywh[keep0], scores[keep0], labels_raw[keep0]
            cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
            if len(boxes):
                keep_idx = _nms_numpy(boxes, scores, labels_raw, iou_thr=0.55)
                boxes, scores, labels_raw = boxes[keep_idx], scores[keep_idx], labels_raw[keep_idx]
            labels = labels_raw.astype(int) + 1

        else:  # "separate_boxes_scores_labels" -- our own torchvision-style export contract
            boxes, scores, labels = outputs[0], outputs[1].reshape(-1), outputs[2].reshape(-1).astype(int)

        boxes = self._undo_transform(np.asarray(boxes, dtype=np.float32), transform_info)
        keep = scores >= score_thr
        return boxes[keep], np.asarray(labels)[keep], np.asarray(scores)[keep]

    def count_cells(self, image, score_thr=0.35):
        _, labels, _ = self.predict(image, score_thr=score_thr)
        out = {c: 0 for c in CELL_CLASSES}
        for l in labels:
            idx = int(l)
            if 0 <= idx < len(DET_CLASSES):
                name = DET_CLASSES[idx]
                if name in out:
                    out[name] += 1
        return out

    def timed_predict(self, image, score_thr=0.35):
        t0 = time.perf_counter()
        boxes, labels, scores = self.predict(image, score_thr=score_thr)
        t1 = time.perf_counter()
        return boxes, labels, scores, (t1 - t0) * 1000.0

    def num_parameters(self):
        return None  # not directly available from an ONNX graph without extra tooling

    def state_dict_size_mb(self):
        return self.onnx_path.stat().st_size / (1024 * 1024)


def load_detector(model_key, device="cpu", quantize_mode="fp32"):
    """Single dispatch point used by every bcc_metrics eval script: given a
    MODEL_REGISTRY key, returns an object exposing predict() / count_cells()
    / timed_predict() / num_parameters() / state_dict_size_mb(), regardless
    of whether the underlying model is a native torchvision SSD or an
    ONNX-exported model from any other framework."""
    from common import MODEL_REGISTRY

    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'. Known keys: {list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_key]

    if spec["kind"] == "native_ssd":
        return load_ssdlite_detector(device=device, quantize_mode=quantize_mode)
    if spec["kind"] == "native_ssd_v2":
        return load_ssdlite_v2_detector(device=device, quantize_mode=quantize_mode)
    if spec["kind"] == "native_only":
        raise RuntimeError(
            f"Model '{model_key}' has no ONNX export ({spec['display_name']}), and this bcc_metrics "
            "suite only runs generic evaluation through ONNX for non-torchvision models. See that "
            "model's own train_bccd_*.py run_report for its native evaluation numbers instead."
        )
    # "onnx"
    if quantize_mode != "fp32":
        return load_onnx_detector_quantized(spec["onnx"], device=device, quantize_mode=quantize_mode)
    return OnnxDetector(spec["onnx"], device=device)


def load_onnx_detector_quantized(onnx_path, device="cpu", quantize_mode="fp16"):
    """FP16 / dynamic-INT8 variants of an ONNX model, produced on first use
    via onnxruntime's own quantization tooling and cached next to the
    original .onnx file, so quantization_bench.py's 3-level comparison
    works for every ONNX-based model too, not just the native SSDLite."""
    _require_onnxruntime()
    onnx_path = Path(onnx_path)
    if quantize_mode == "fp32":
        return OnnxDetector(onnx_path, device=device)

    cache_path = onnx_path.with_suffix(f".{quantize_mode}.onnx")
    meta_src = Path(str(onnx_path) + ".meta.json")
    meta_dst = Path(str(cache_path) + ".meta.json")
    if not meta_dst.exists() and meta_src.exists():
        meta_dst.write_text(meta_src.read_text())

    if not cache_path.exists():
        if quantize_mode == "fp16":
            from onnxruntime.transformers.float16 import convert_float_to_float16
            import onnx
            model = onnx.load(str(onnx_path))
            model_fp16 = convert_float_to_float16(model)
            onnx.save(model_fp16, str(cache_path))
        elif quantize_mode == "dynamic_int8":
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(str(onnx_path), str(cache_path), weight_type=QuantType.QInt8)
        else:
            raise ValueError(f"Unknown quantize_mode: {quantize_mode}")
    return OnnxDetector(cache_path, device=device)


# ----------------------------------------------------------------------------
# Simulated re-imaging perturbations (Protocol B in the notes: "pure
# algorithmic reproducibility under simulated re-capture" and the robustness
# / corruption-severity curve for next-step item #5).
# ----------------------------------------------------------------------------
def perturb_image(image: "Image.Image", rng: np.random.Generator, severity: str = "mild"):
    """Apply a small, realistic re-imaging perturbation: rotation, brightness/
    contrast jitter, and a small crop shift. `severity` controls magnitude and
    is reused by robustness_eval.py for the corruption-severity curve.
    """
    from PIL import ImageEnhance

    severity_cfg = {
        "none": dict(rot=0.0, bright=0.0, contrast=0.0, shift=0),
        "mild": dict(rot=2.5, bright=0.08, contrast=0.08, shift=2),
        "moderate": dict(rot=5.0, bright=0.18, contrast=0.18, shift=6),
        "severe": dict(rot=8.0, bright=0.35, contrast=0.35, shift=12),
    }
    cfg = severity_cfg[severity]

    img = image
    if cfg["rot"] > 0:
        angle = rng.uniform(-cfg["rot"], cfg["rot"])
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))

    if cfg["bright"] > 0:
        factor = 1.0 + rng.uniform(-cfg["bright"], cfg["bright"])
        img = ImageEnhance.Brightness(img).enhance(max(0.1, factor))

    if cfg["contrast"] > 0:
        factor = 1.0 + rng.uniform(-cfg["contrast"], cfg["contrast"])
        img = ImageEnhance.Contrast(img).enhance(max(0.1, factor))

    if cfg["shift"] > 0:
        dx = int(rng.integers(-cfg["shift"], cfg["shift"] + 1))
        dy = int(rng.integers(-cfg["shift"], cfg["shift"] + 1))
        img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=(255, 255, 255))

    return img


def simulate_stain_color_shift(image: "Image.Image", rng: np.random.Generator, severity: str = "mild"):
    """Recolor an image to imitate a different Romanowsky-type stain
    (Giemsa/Wright/Leishman blends look different) for next-step item #5:
    'obscure/recolor images to imitate dyes', framed as a robustness curve
    against corruption severity.
    """
    arr = np.asarray(image).astype(np.float32)
    severity_cfg = {"none": 0.0, "mild": 0.08, "moderate": 0.20, "severe": 0.40}
    strength = severity_cfg[severity]
    if strength == 0.0:
        return image

    # Random per-channel gain + a hue-ish rotation approximated via channel mixing.
    gains = 1.0 + rng.uniform(-strength, strength, size=3)
    mix = np.eye(3) + rng.uniform(-strength * 0.5, strength * 0.5, size=(3, 3))
    arr = arr @ mix.T
    arr = arr * gains
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def obscure_image(image: "Image.Image", rng: np.random.Generator, severity: str = "mild"):
    """Gaussian-blur-based 'obscuration' (stand-in for out-of-focus / poorly
    prepared smears), the blur half of next-step item #5.
    """
    from PIL import ImageFilter

    severity_cfg = {"none": 0.0, "mild": 1.0, "moderate": 2.5, "severe": 5.0}
    radius = severity_cfg[severity]
    if radius == 0.0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=radius))
