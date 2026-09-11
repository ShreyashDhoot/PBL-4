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
