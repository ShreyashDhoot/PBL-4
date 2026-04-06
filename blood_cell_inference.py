#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║       🩸  Blood Cell Classifier — Raspberry Pi 4        ║
║       Architecture : MobileNetV2 (CPU-optimised)        ║
║       Interface    : Full-screen Tkinter dashboard      ║
╚══════════════════════════════════════════════════════════╝

HOW IT WORKS
────────────
  • The script watches a "hot-folder" for new images dropped by your
    separate image-capture process (microscope camera, PiCamera script,
    USB microscope grabber, etc.).
  • As soon as a new file appears the model runs inference and the
    beautiful dashboard updates in real-time.
  • You can also drag-and-drop / paste a path manually via the GUI.

INTEGRATION WITH YOUR CAPTURE CODE
────────────────────────────────────
  Your image-capture script just needs to SAVE a file to the watch
  folder.  Example (PiCamera2 snippet):

      WATCH_DIR = "/tmp/blood_cells/incoming"   # must match below
      camera.capture_file(f"{WATCH_DIR}/capture_{timestamp}.jpg")

  That's it — this script picks it up automatically.

USAGE
──────
  python3 blood_cell_inference.py --weights mobilenet_blood.pth
  python3 blood_cell_inference.py --weights mobilenet_blood.pth --watch /tmp/blood_cells/incoming
  python3 blood_cell_inference.py --weights mobilenet_blood.pth --image /path/to/single.jpg

REQUIREMENTS  (all available via pip on Raspberry Pi OS 64-bit)
  pip install torch torchvision pillow numpy matplotlib
  (tkinter ships with Python on Raspberry Pi OS)
"""

# ─── stdlib ───────────────────────────────────────────────────────────────────
import argparse
import os
import sys
import time
import threading
import queue
import json
import datetime
from pathlib import Path
from collections import deque

# ─── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")                 # off-screen rendering for Pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ─── tkinter ──────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — edit these to match your training setup
# ══════════════════════════════════════════════════════════════════════════════

# Cell-type names must match the order used during training.
# The dataset (draaslan/blood-cell-detection) has 4 classes:
CLASS_NAMES = [
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
]

# Accent colour per class (for the GUI bars)
CLASS_COLORS = {
    "Eosinophil":  "#E74C3C",   # red
    "Lymphocyte":  "#3498DB",   # blue
    "Monocyte":    "#2ECC71",   # green
    "Neutrophil":  "#F39C12",   # orange
}

# Short clinical descriptions shown on the dashboard
CLASS_DESC = {
    "Eosinophil":  "Combats parasites & allergic reactions. ~1–4 % of WBCs.",
    "Lymphocyte":  "Key adaptive-immunity cell (T/B/NK). ~20–40 % of WBCs.",
    "Monocyte":    "Largest WBC; differentiates into macrophages. ~2–8 %.",
    "Neutrophil":  "First-line bacterial defence. ~55–70 % of WBCs.",
}

IMG_SIZE    = 224
NUM_CLASSES = len(CLASS_NAMES)
WATCH_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ImageNet normalisation (same as training)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ──── UI palette ──────────────────────────────────────────────────────────────
BG_DARK     = "#0D1117"
BG_CARD     = "#161B22"
BG_CARD2    = "#1C2128"
ACCENT      = "#58A6FF"
ACCENT2     = "#3FB950"
TEXT_MAIN   = "#E6EDF3"
TEXT_SUB    = "#8B949E"
TEXT_WARN   = "#F0883E"
BORDER      = "#30363D"
RED_ALERT   = "#F85149"


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_mobilenetv2(num_classes: int) -> nn.Module:
    """Exactly the same architecture used in training."""
    model = tv_models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


def load_model(weights_path: str, num_classes: int, device: torch.device) -> nn.Module:
    model = build_mobilenetv2(num_classes)
    state = torch.load(weights_path, map_location=device)
    # Handle both raw state-dict and checkpoint dicts
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    def __init__(self, weights_path: str):
        self.device = torch.device("cpu")   # Pi 4 has no CUDA
        print(f"[Engine] Loading weights from: {weights_path}")
        self.model  = load_model(weights_path, NUM_CLASSES, self.device)
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        # Warm-up run to initialise BN statistics on eval
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            _ = self.model(dummy)
        print("[Engine] Model ready.")

    @torch.no_grad()
    def predict(self, img_path: str):
        """
        Returns:
            pred_class  : str   — top predicted class name
            confidence  : float — softmax probability of top class
            probs       : dict  — {class_name: probability}
            inf_ms      : float — inference time in milliseconds
        """
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        logits = self.model(tensor)
        t1 = time.perf_counter()

        probs_t     = torch.softmax(logits, dim=1)[0]
        probs       = {CLASS_NAMES[i]: float(probs_t[i]) for i in range(NUM_CLASSES)}
        pred_idx    = int(probs_t.argmax())
        pred_class  = CLASS_NAMES[pred_idx]
        confidence  = float(probs_t[pred_idx])
        inf_ms      = (t1 - t0) * 1000.0

        return pred_class, confidence, probs, inf_ms, img


# ══════════════════════════════════════════════════════════════════════════════
#  FILE WATCHER (background thread)
# ══════════════════════════════════════════════════════════════════════════════

class FolderWatcher(threading.Thread):
    """Polls a directory for new image files and pushes them to a queue."""

    def __init__(self, watch_dir: str, result_queue: queue.Queue, interval: float = 0.5):
        super().__init__(daemon=True)
        self.watch_dir    = Path(watch_dir)
        self.result_queue = result_queue
        self.interval     = interval
        self._seen        = set()
        self._stop_event  = threading.Event()

    def run(self):
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Watcher] Monitoring: {self.watch_dir}")
        while not self._stop_event.is_set():
            try:
                current = {
                    p for p in self.watch_dir.iterdir()
                    if p.suffix.lower() in WATCH_EXTS
                }
                new_files = current - self._seen
                for f in sorted(new_files, key=lambda p: p.stat().st_mtime):
                    self.result_queue.put(str(f))
                self._seen = current
            except Exception as e:
                print(f"[Watcher] Error: {e}")
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()


# ══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB PROBABILITY BAR CHART  →  PIL Image
# ══════════════════════════════════════════════════════════════════════════════

def make_prob_chart(probs: dict, pred_class: str, width_px=460, height_px=220) -> Image.Image:
    dpi   = 100
    fw    = width_px  / dpi
    fh    = height_px / dpi

    fig, ax = plt.subplots(figsize=(fw, fh), dpi=dpi)
    fig.patch.set_facecolor(BG_CARD2)
    ax.set_facecolor(BG_CARD2)

    names  = list(probs.keys())
    values = [probs[n] * 100 for n in names]
    colors = [CLASS_COLORS.get(n, "#58A6FF") for n in names]
    # Highlight predicted bar
    edge_colors = ["#FFFFFF" if n == pred_class else "none" for n in names]
    edge_widths = [2.0       if n == pred_class else 0.0   for n in names]

    bars = ax.barh(names, values, color=colors, edgecolor=edge_colors,
                   linewidth=edge_widths, height=0.55, zorder=3)

    for bar, val, name in zip(bars, values, names):
        ax.text(min(val + 1.5, 99), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                color="#FFFFFF" if name == pred_class else TEXT_SUB,
                fontsize=8, fontweight="bold" if name == pred_class else "normal")

    ax.set_xlim(0, 108)
    ax.set_xlabel("Confidence (%)", color=TEXT_SUB, fontsize=8)
    ax.tick_params(colors=TEXT_MAIN, labelsize=9)
    ax.spines[:].set_color(BORDER)
    ax.xaxis.label.set_color(TEXT_SUB)
    ax.grid(axis="x", color=BORDER, linewidth=0.5, zorder=0)
    ax.set_title("Class Probabilities", color=TEXT_MAIN, fontsize=9, pad=6)

    plt.tight_layout(pad=0.4)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf  = canvas.buffer_rgba()
    img  = Image.frombytes("RGBA", canvas.get_width_height(), buf)
    plt.close(fig)
    return img.convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
#  TKINTER DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

class BloodCellDashboard(tk.Tk):

    HISTORY_LEN = 8   # recent-inference strip

    def __init__(self, engine: InferenceEngine, watch_dir: str):
        super().__init__()
        self.engine    = engine
        self.img_queue = queue.Queue()
        self.history   = deque(maxlen=self.HISTORY_LEN)   # (thumb_tk, pred, conf)

        # ── window setup ──────────────────────────────────────────────────────
        self.title("🩸 Blood Cell Classifier — MobileNetV2")
        self.configure(bg=BG_DARK)
        self.geometry("1024x700")
        try:
            self.state("zoomed")          # fullscreen on Pi
        except Exception:
            pass
        self.resizable(True, True)

        # ── watcher ───────────────────────────────────────────────────────────
        self.watcher = FolderWatcher(watch_dir, self.img_queue)
        self.watcher.start()

        self._build_ui()
        self._poll_queue()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG_CARD, pady=10)
        hdr.pack(fill="x", padx=0, pady=0)

        tk.Label(hdr, text="🩸", font=("Helvetica", 22), bg=BG_CARD,
                 fg=RED_ALERT).pack(side="left", padx=(18, 6))
        tk.Label(hdr, text="Blood Cell Classifier",
                 font=("Helvetica", 18, "bold"), bg=BG_CARD,
                 fg=TEXT_MAIN).pack(side="left")
        tk.Label(hdr, text="  MobileNetV2 · Raspberry Pi 4",
                 font=("Helvetica", 11), bg=BG_CARD,
                 fg=TEXT_SUB).pack(side="left", pady=(4, 0))

        # status pill
        self.status_var = tk.StringVar(value="● Watching for images…")
        tk.Label(hdr, textvariable=self.status_var,
                 font=("Helvetica", 10), bg=BG_CARD,
                 fg=ACCENT2).pack(side="right", padx=20)

        # ── toolbar ───────────────────────────────────────────────────────────
        tb = tk.Frame(self, bg=BG_DARK, pady=6)
        tb.pack(fill="x", padx=14)

        self._btn(tb, "📂  Open Image", self._open_file).pack(side="left", padx=(0, 8))
        self._btn(tb, "📋  Classify Clipboard Path",
                  self._paste_path).pack(side="left", padx=(0, 8))
        self._btn(tb, "💾  Save Result", self._save_result,
                  color=ACCENT2).pack(side="left", padx=(0, 8))

        self.watch_lbl = tk.Label(
            tb,
            text=f"👁  Watch: {self.watcher.watch_dir}",
            font=("Helvetica", 9), bg=BG_DARK, fg=TEXT_SUB
        )
        self.watch_lbl.pack(side="right")

        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill="x", padx=0, pady=2)

        # ── main content ──────────────────────────────────────────────────────
        content = tk.Frame(self, bg=BG_DARK)
        content.pack(fill="both", expand=True, padx=14, pady=8)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        # LEFT: image + result card
        left = tk.Frame(content, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._build_left(left)

        # RIGHT: confidence chart + info + history
        right = tk.Frame(content, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_right(right)

    def _build_left(self, parent):
        # Image display card
        img_card = self._card(parent, title="Input Image")
        img_card.pack(fill="both", expand=True, pady=(0, 8))

        self.img_label = tk.Label(img_card, bg=BG_CARD2,
                                  text="No image loaded\nDrop a file in the watch folder\nor click 'Open Image'",
                                  font=("Helvetica", 11), fg=TEXT_SUB,
                                  justify="center")
        self.img_label.pack(fill="both", expand=True, padx=8, pady=8)

        # Result card
        res_card = self._card(parent, title="Prediction")
        res_card.pack(fill="x")

        row1 = tk.Frame(res_card, bg=BG_CARD)
        row1.pack(fill="x", padx=10, pady=(6, 2))

        self.pred_label = tk.Label(row1, text="—",
                                   font=("Helvetica", 28, "bold"),
                                   bg=BG_CARD, fg=ACCENT)
        self.pred_label.pack(side="left")

        self.conf_badge = tk.Label(row1, text="",
                                   font=("Helvetica", 13),
                                   bg=BG_CARD, fg=TEXT_SUB)
        self.conf_badge.pack(side="left", padx=(14, 0), pady=(8, 0))

        self.desc_label = tk.Label(res_card, text="",
                                   font=("Helvetica", 10), bg=BG_CARD,
                                   fg=TEXT_SUB, wraplength=380, justify="left")
        self.desc_label.pack(anchor="w", padx=10, pady=(2, 4))

        row2 = tk.Frame(res_card, bg=BG_CARD)
        row2.pack(fill="x", padx=10, pady=(0, 8))

        self.inf_lbl = tk.Label(row2, text="⏱  —",
                                font=("Helvetica", 9), bg=BG_CARD, fg=TEXT_SUB)
        self.inf_lbl.pack(side="left")

        self.ts_lbl = tk.Label(row2, text="",
                               font=("Helvetica", 9), bg=BG_CARD, fg=TEXT_SUB)
        self.ts_lbl.pack(side="right")

    def _build_right(self, parent):
        # Probability chart card
        chart_card = self._card(parent, title="Confidence Distribution")
        chart_card.pack(fill="x", pady=(0, 8))

        self.chart_label = tk.Label(chart_card, bg=BG_CARD2,
                                    text="Run inference to see probabilities",
                                    font=("Helvetica", 10), fg=TEXT_SUB)
        self.chart_label.pack(fill="both", expand=True, padx=6, pady=6)

        # Class info card
        info_card = self._card(parent, title="Cell Reference")
        info_card.pack(fill="x", pady=(0, 8))
        self._build_class_legend(info_card)

        # History strip card
        hist_card = self._card(parent, title="Recent Inferences")
        hist_card.pack(fill="x")
        self.hist_frame = tk.Frame(hist_card, bg=BG_CARD)
        self.hist_frame.pack(fill="x", padx=6, pady=6)

    def _build_class_legend(self, parent):
        for cls in CLASS_NAMES:
            row = tk.Frame(parent, bg=BG_CARD)
            row.pack(fill="x", padx=8, pady=2)
            dot = tk.Label(row, text="●", font=("Helvetica", 11),
                           bg=BG_CARD, fg=CLASS_COLORS.get(cls, ACCENT))
            dot.pack(side="left", padx=(0, 6))
            tk.Label(row, text=cls, font=("Helvetica", 10, "bold"),
                     bg=BG_CARD, fg=TEXT_MAIN, width=12,
                     anchor="w").pack(side="left")
            tk.Label(row, text=CLASS_DESC.get(cls, ""),
                     font=("Helvetica", 8), bg=BG_CARD,
                     fg=TEXT_SUB, wraplength=220,
                     justify="left", anchor="w").pack(side="left", fill="x")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _card(self, parent, title=""):
        frame = tk.Frame(parent, bg=BG_CARD,
                         highlightbackground=BORDER, highlightthickness=1)
        if title:
            tk.Label(frame, text=title, font=("Helvetica", 9, "bold"),
                     bg=BG_CARD, fg=TEXT_SUB).pack(anchor="w", padx=10, pady=(6, 2))
            sep = tk.Frame(frame, bg=BORDER, height=1)
            sep.pack(fill="x", padx=6)
        return frame

    def _btn(self, parent, text, cmd, color=ACCENT):
        return tk.Button(
            parent, text=text, command=cmd,
            bg=BG_CARD2, fg=color,
            activebackground=BORDER, activeforeground=TEXT_MAIN,
            relief="flat", font=("Helvetica", 10),
            padx=10, pady=4, cursor="hand2",
            highlightbackground=BORDER, highlightthickness=1,
        )

    # ── queue polling (runs in main thread via after()) ───────────────────────

    def _poll_queue(self):
        try:
            img_path = self.img_queue.get_nowait()
            self._run_inference(img_path)
        except queue.Empty:
            pass
        self.after(300, self._poll_queue)

    # ── inference ─────────────────────────────────────────────────────────────

    def _run_inference(self, img_path: str):
        self.status_var.set("⚙  Running inference…")
        self.update_idletasks()
        try:
            pred, conf, probs, inf_ms, pil_img = self.engine.predict(img_path)
        except Exception as e:
            self.status_var.set(f"❌  Error: {e}")
            messagebox.showerror("Inference Error", str(e))
            return

        # ── update image panel ────────────────────────────────────────────────
        display_img = pil_img.copy()
        display_img.thumbnail((420, 340), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(display_img)
        self.img_label.configure(image=tk_img, text="")
        self.img_label.image = tk_img    # keep reference

        # ── update result card ────────────────────────────────────────────────
        self.pred_label.configure(text=pred,
                                  fg=CLASS_COLORS.get(pred, ACCENT))
        self.conf_badge.configure(text=f"{conf * 100:.1f}% confidence")
        self.desc_label.configure(text=CLASS_DESC.get(pred, ""))
        self.inf_lbl.configure(text=f"⏱  {inf_ms:.1f} ms  ·  MobileNetV2")
        self.ts_lbl.configure(
            text=datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

        # ── update chart ──────────────────────────────────────────────────────
        chart_pil = make_prob_chart(probs, pred, width_px=420, height_px=230)
        chart_tk  = ImageTk.PhotoImage(chart_pil)
        self.chart_label.configure(image=chart_tk, text="")
        self.chart_label.image = chart_tk

        # ── update history strip ──────────────────────────────────────────────
        thumb = pil_img.copy()
        thumb.thumbnail((60, 60), Image.LANCZOS)
        thumb_tk = ImageTk.PhotoImage(thumb)
        self.history.appendleft((thumb_tk, pred, conf))
        self._redraw_history()

        # ── status ────────────────────────────────────────────────────────────
        self.status_var.set(
            f"✅  {Path(img_path).name}  →  {pred}  ({conf*100:.1f}%)")

        # store last result for saving
        self._last = dict(img_path=img_path, pred=pred, conf=conf,
                          probs=probs, inf_ms=inf_ms, pil_img=pil_img,
                          chart_pil=chart_pil)

    def _redraw_history(self):
        for w in self.hist_frame.winfo_children():
            w.destroy()
        for thumb_tk, pred, conf in list(self.history):
            cell = tk.Frame(self.hist_frame, bg=BG_CARD2, padx=3, pady=3,
                            highlightbackground=CLASS_COLORS.get(pred, BORDER),
                            highlightthickness=2)
            cell.pack(side="left", padx=3)
            lbl = tk.Label(cell, image=thumb_tk, bg=BG_CARD2)
            lbl.image = thumb_tk
            lbl.pack()
            tk.Label(cell, text=pred[:4], font=("Helvetica", 7, "bold"),
                     bg=BG_CARD2, fg=CLASS_COLORS.get(pred, ACCENT)).pack()
            tk.Label(cell, text=f"{conf*100:.0f}%", font=("Helvetica", 7),
                     bg=BG_CARD2, fg=TEXT_SUB).pack()

    # ── toolbar actions ───────────────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select Blood Cell Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                       ("All files", "*.*")])
        if path:
            self._run_inference(path)

    def _paste_path(self):
        try:
            path = self.clipboard_get().strip().strip('"').strip("'")
        except Exception:
            path = ""
        if path and Path(path).exists():
            self._run_inference(path)
        else:
            messagebox.showwarning("Path not found",
                                   f"Clipboard contents is not a valid file path:\n{path}")

    def _save_result(self):
        if not hasattr(self, "_last"):
            messagebox.showinfo("Nothing to save", "Run inference first.")
            return
        d    = self._last
        dest = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"result_{d['pred']}_{int(time.time())}.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not dest:
            return

        # Compose a result image: original | chart side by side
        orig = d["pil_img"].copy()
        orig.thumbnail((480, 380), Image.LANCZOS)
        chart = d["chart_pil"].copy()
        w  = orig.width + chart.width + 20
        h  = max(orig.height, chart.height) + 80
        out = Image.new("RGB", (w, h), color=BG_DARK[1:].zfill(6))

        # Convert hex to RGB tuple
        def hex2rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        out = Image.new("RGB", (w, h), hex2rgb(BG_DARK))
        out.paste(orig,  (0,   40))
        out.paste(chart, (orig.width + 20, 40))

        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(out)
        try:
            font_big  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
            font_sm   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except Exception:
            font_big = font_sm = ImageFont.load_default()

        draw.text((10, 8),  f"Blood Cell Classification — {d['pred']}",
                  fill=hex2rgb(TEXT_MAIN), font=font_big)
        draw.text((10, h - 28),
                  f"Confidence: {d['conf']*100:.1f}%  |  Inference: {d['inf_ms']:.1f} ms  |  "
                  f"Model: MobileNetV2  |  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                  fill=hex2rgb(TEXT_SUB), font=font_sm)

        out.save(dest)
        self.status_var.set(f"💾  Saved → {Path(dest).name}")

    def on_close(self):
        self.watcher.stop()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Blood Cell Classifier — Raspberry Pi 4 Inference UI")
    p.add_argument(
        "--weights", default="mobilenet_blood.pth",
        help="Path to the saved MobileNetV2 .pth weights file (default: mobilenet_blood.pth)")
    p.add_argument(
        "--watch", default="/tmp/blood_cells/incoming",
        help="Folder to monitor for new images from your capture script "
             "(default: /tmp/blood_cells/incoming)")
    p.add_argument(
        "--image", default=None,
        help="Classify a single image immediately on startup, then keep the GUI open")
    p.add_argument(
        "--classes", nargs="+", default=None,
        help="Override class names (space-separated, must match training order)")
    return p.parse_args()


def main():
    args = parse_args()

    # Allow class override from CLI
    global CLASS_NAMES, NUM_CLASSES
    if args.classes:
        CLASS_NAMES  = args.classes
        NUM_CLASSES  = len(CLASS_NAMES)

    # Validate weights
    if not Path(args.weights).exists():
        print(f"\n⚠  Weights file not found: {args.weights}")
        print("   Please copy your trained .pth file and pass it via --weights\n"
              "   Example:\n"
              "     python3 blood_cell_inference.py --weights /home/pi/mobilenet_blood.pth\n")
        sys.exit(1)

    # Build engine
    engine = InferenceEngine(args.weights)

    # Launch GUI
    app = BloodCellDashboard(engine, watch_dir=args.watch)
    app.protocol("WM_DELETE_WINDOW", app.on_close)

    # If a single image was supplied, classify it after the GUI is up
    if args.image:
        app.after(500, lambda: app._run_inference(args.image))

    app.mainloop()


if __name__ == "__main__":
    main()
