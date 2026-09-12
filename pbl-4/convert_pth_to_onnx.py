#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

CLASSES = ["__background__", "RBC", "WBC", "Platelets"]


def build_model(num_classes: int):
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        num_classes=num_classes,
    )
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, pth_path: Path, device: str) -> None:
    ckpt = torch.load(str(pth_path), map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)


class DetectionExportWrapper(torch.nn.Module):
    def __init__(self, detector: torch.nn.Module):
        super().__init__()
        self.detector = detector

    def forward(self, x: torch.Tensor):
        preds = self.detector(list(x))
        return (
            preds[0]["boxes"],
            preds[0]["scores"],
            preds[0]["labels"],
        )


def export_to_onnx(model: torch.nn.Module, onnx_path: Path, opset: int, image_size: int, device: str) -> None:
    wrapper = DetectionExportWrapper(model).to(device).eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    export_kwargs = dict(
        args=(dummy,),
        f=str(onnx_path),
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )

    # Prefer legacy exporter to avoid torch.export dynamic-shape guard failures
    # seen with detection post-processing in newer exporters.
    try:
        torch.onnx.export(wrapper, **export_kwargs, dynamo=False)
    except TypeError:
        torch.onnx.export(wrapper, **export_kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SSDLite .pth weights to ONNX")
    parser.add_argument("--pth", default="output/ssdlite_bccd_best.pth", help="Path to .pth weights")
    parser.add_argument("--onnx", default="output/ssdlite_bccd.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--image_size", type=int, default=320, help="Square input size used for export")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Export device")
    return parser.parse_args()


def main():
    args = parse_args()

    pth_path = Path(args.pth)
    onnx_path = Path(args.onnx)

    if not pth_path.exists():
        raise FileNotFoundError(f".pth file not found: {pth_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model weights: {pth_path}")
    model = build_model(num_classes=len(CLASSES)).to(args.device)
    load_checkpoint(model, pth_path, args.device)

    print(f"[INFO] Exporting ONNX to: {onnx_path}")
    export_to_onnx(model, onnx_path, args.opset, args.image_size, args.device)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Export complete: {onnx_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
