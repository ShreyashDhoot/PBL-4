#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import onnxruntime as ort
except Exception:
    ort = None

CLASSES = ["__background__", "RBC", "WBC", "Platelets"]
COLORS = {
    1: (46, 204, 113),   # RBC - green
    2: (231, 76, 60),    # WBC - red
    3: (52, 152, 219),   # Platelets - blue
}


def capture_with_libcamera(output_path: Path, width: int, height: int, timeout_ms: int) -> None:
    cmd_name = None
    if shutil.which("libcamera-still"):
        cmd_name = "libcamera-still"
    elif shutil.which("rpicam-still"):
        cmd_name = "rpicam-still"

    if cmd_name is None:
        raise RuntimeError(
            "Camera capture command not found. Install libcamera tools (libcamera-still/rpicam-still)."
        )

    # Auto exposure/white-balance lets the OV5647 + IR-cut setup adapt between day/night scenes.
    cmd = [
        cmd_name,
        "-n",
        "-o",
        str(output_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--timeout",
        str(timeout_ms),
        "--awb",
        "auto",
        "--metering",
        "average",
    ]
    subprocess.run(cmd, check=True)


class OnnxDetector:
    def __init__(self, model_path: Path):
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed in this Python environment. "
                "Install it first on Raspberry Pi, then rerun."
            )

        self.model_path = model_path
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)

        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(f"Expected 1 model input, got {len(inputs)}")
        self.input_name = inputs[0].name

        outputs = self.session.get_outputs()
        if len(outputs) < 3:
            raise RuntimeError(
                "Expected at least 3 model outputs (boxes, scores, labels). "
                f"Found {len(outputs)} outputs."
            )
        self.output_names = [o.name for o in outputs]

    @staticmethod
    def preprocess(image: Image.Image) -> np.ndarray:
        resized = image.resize((320, 320), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def infer(self, image: Image.Image):
        inp = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: inp})

        boxes = np.array(outputs[0])[0]
        scores = np.array(outputs[1])[0]
        labels = np.array(outputs[2])[0]

        return boxes, scores, labels


def draw_detections(image: Image.Image, boxes, scores, labels, score_thr: float) -> Image.Image:
    canvas = image.resize((320, 320), Image.BILINEAR).copy()
    draw = ImageDraw.Draw(canvas)

    for box, score, label in zip(boxes, scores, labels):
        lbl = int(label)
        conf = float(score)
        if conf < score_thr or lbl <= 0 or lbl >= len(CLASSES):
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        color = COLORS.get(lbl, (255, 255, 0))
        name = CLASSES[lbl]
        tag = f"{name} {conf:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text_y = y1 - 14 if y1 > 16 else y1 + 2
        draw.text((x1 + 2, text_y), tag, fill=color)

    return canvas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture image from Raspberry Pi camera and run ONNX blood-cell detection"
    )
    parser.add_argument(
        "--model",
        default="output/ssdlite_bccd.onnx",
        help="Path to ONNX model exported from training",
    )
    parser.add_argument(
        "--capture_dir",
        default="output/captures",
        help="Directory to store captured and annotated images",
    )
    parser.add_argument(
        "--input_image",
        default=None,
        help="Optional existing image path. If set, camera capture is skipped.",
    )
    parser.add_argument("--score_thr", type=float, default=0.35, help="Score threshold")
    parser.add_argument("--width", type=int, default=1920, help="Camera capture width")
    parser.add_argument("--height", type=int, default=1080, help="Camera capture height")
    parser.add_argument(
        "--timeout_ms",
        type=int,
        default=1200,
        help="Camera warm-up timeout in milliseconds",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] ONNX model not found: {model_path}")
        print("[HINT] Export or place your ONNX model at the path above.")
        sys.exit(1)

    out_dir = Path(args.capture_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")

    if args.input_image:
        input_path = Path(args.input_image)
        if not input_path.exists():
            print(f"[ERROR] Input image not found: {input_path}")
            sys.exit(1)
    else:
        input_path = out_dir / f"capture_{ts}.jpg"
        print("[INFO] Capturing image from Raspberry Pi camera...")
        try:
            capture_with_libcamera(input_path, args.width, args.height, args.timeout_ms)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Camera capture command failed: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    print(f"[INFO] Using image: {input_path}")
    image = Image.open(input_path).convert("RGB")

    try:
        detector = OnnxDetector(model_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        print("[HINT] On Raspberry Pi 32-bit, install a compatible onnxruntime build for your Python version.")
        sys.exit(1)

    boxes, scores, labels = detector.infer(image)
    vis = draw_detections(image, boxes, scores, labels, args.score_thr)

    output_path = out_dir / f"bbox_{ts}.jpg"
    vis.save(output_path, quality=95)

    kept = sum(
        1
        for s, l in zip(scores, labels)
        if float(s) >= args.score_thr and 0 < int(l) < len(CLASSES)
    )

    print(f"[INFO] Detections kept: {kept}")
    print(f"[INFO] Annotated output: {output_path}")


if __name__ == "__main__":
    main()
