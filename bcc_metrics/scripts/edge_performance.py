#!/usr/bin/env python3
"""
edge_performance.py
=====================
Notes reference: next-step item #6, last bullet ("Report power draw /
inference time per image on the Pi 4 itself alongside the quantization
results -- ties directly into the 'resource-limited settings' framing").
Also underlies the paper's existing Section V-C ("Edge Performance
Analysis": <1s/image inference, <200MB memory).

Measures, for both SSDLite (detector) and EfficientNet-B0 (classifier):
  - per-image inference latency (mean/median/p95/std), CPU-forced
  - peak resident memory during a batch of inferences (via psutil if
    available, else skipped with a clear note)
  - power draw: instrumented via an optional external reading. On a real
    Raspberry Pi you'd log a USB power meter or `vcgencmd measure_volts`+
    current-sense HAT reading during the run; this script exposes a single
    hook (`read_power_watts()`) that returns None unless
    BCC_POWER_METER_CMD is set to a shell command that prints a single
    float (watts) to stdout, sampled during inference. This keeps the
    script honest: it never fabricates a power number.

Outputs:
  output/tables/edge_performance.csv
  output/figures/edge_latency_boxplot.png
  output/reports/edge_performance.json
"""

import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import TABLES_DIR, FIGURES_DIR, REPORTS_DIR, log
from voc_data import build_bccd_records, split_bccd_records
from models_io import load_ssdlite_detector, load_efficientnet_classifier

N_IMAGES = 30

try:
    import psutil
except Exception:
    psutil = None


def read_power_watts():
    """Returns a float (watts) from an external power-meter command if
    BCC_POWER_METER_CMD is set (e.g. a script reading a USB power meter or
    an INA219 current-sense HAT on the Pi), else None. Never fabricated.
    """
    cmd = os.environ.get("BCC_POWER_METER_CMD")
    if not cmd:
        return None
    try:
        out = subprocess.check_output(cmd, shell=True, timeout=5).decode().strip()
        return float(out)
    except Exception as e:
        log(f"Power meter command failed ({e}); continuing without a power reading.", tag="WARN")
        return None


def measure_model(name, predict_fn, images, n_warmup=3):
    for img in images[:n_warmup]:
        predict_fn(img)  # warm up (lazy init, caches, etc.)

    ram_before = psutil.Process().memory_info().rss / (1024 * 1024) if psutil else None
    power_before = read_power_watts()

    latencies_ms = []
    peak_ram_mb = ram_before or 0.0
    for img in images:
        t0 = time.perf_counter()
        predict_fn(img)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        if psutil:
            cur = psutil.Process().memory_info().rss / (1024 * 1024)
            peak_ram_mb = max(peak_ram_mb, cur)

    power_after = read_power_watts()
    power_watts = None
    if power_before is not None and power_after is not None:
        power_watts = (power_before + power_after) / 2.0
    elif power_after is not None:
        power_watts = power_after

    latencies_ms = np.array(latencies_ms)
    return {
        "model": name,
        "n_images": len(images),
        "mean_latency_ms": float(latencies_ms.mean()),
        "median_latency_ms": float(np.median(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "std_latency_ms": float(latencies_ms.std()),
        "peak_ram_mb": peak_ram_mb if psutil else None,
        "power_watts": power_watts,
        "on_target_hardware_raspberry_pi4": os.environ.get("BCC_ON_RASPBERRY_PI", "0") == "1",
    }, latencies_ms


def plot_latency_boxplot(latency_map, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = list(latency_map.keys())
    data = [latency_map[k] for k in labels]
    # matplotlib >=3.9 renamed boxplot's `labels` kwarg to `tick_labels`;
    # try the new name first and fall back for older matplotlib installs.
    try:
        ax.boxplot(data, tick_labels=labels)
    except TypeError:
        ax.boxplot(data, labels=labels)
    ax.set_ylabel("Latency (ms/image)")
    ax.set_title("Per-image inference latency (CPU)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Saved {out_path}")


def main():
    bccd = build_bccd_records()
    _, _, test = split_bccd_records(bccd)
    sample_records = test[:min(N_IMAGES, len(test))]
    images = [Image.open(r["image_path"]).convert("RGB") for r in sample_records]

    if not psutil:
        log("psutil not installed -- peak RAM will be reported as null. `pip install psutil` to enable.", tag="WARN")
    if not os.environ.get("BCC_POWER_METER_CMD"):
        log("BCC_POWER_METER_CMD not set -- power_watts will be reported as null. "
            "Set this to a command that prints instantaneous watts (e.g. from a USB "
            "power meter or INA219 HAT on the Pi) to populate real numbers.", tag="WARN")

    results = {}
    latency_map = {}

    detector = load_ssdlite_detector(device="cpu")
    res, lat = measure_model("SSDLite (detector)", lambda im: detector.predict(im, score_thr=0.35), images)
    results["ssdlite"] = res
    latency_map["SSDLite"] = lat

    try:
        clf = load_efficientnet_classifier(device="cpu")
        res, lat = measure_model("EfficientNet-B0 (classifier)", clf.predict_proba, images)
        results["efficientnet"] = res
        latency_map["EfficientNet-B0"] = lat
    except Exception as e:
        log(f"EfficientNet edge measurement skipped: {e}", tag="WARN")

    df = pd.DataFrame(list(results.values()))
    df.to_csv(TABLES_DIR / "edge_performance.csv", index=False)
    plot_latency_boxplot(latency_map, FIGURES_DIR / "edge_latency_boxplot.png")

    with open(REPORTS_DIR / "edge_performance.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    for k, v in results.items():
        log(f"{v['model']}: mean={v['mean_latency_ms']:.1f}ms  p95={v['p95_latency_ms']:.1f}ms  "
            f"RAM={v['peak_ram_mb']}  power_W={v['power_watts']}")
    log("Edge performance measurement complete. See output/tables/edge_performance.csv")


if __name__ == "__main__":
    main()
