#!/usr/bin/env python3
"""
sss_train.py — Train the SSS spectrogram engine.

Reads a labels.tsv of the form
    filename<TAB>color_label<TAB>shape_label<TAB>face_label
Loads each image (PPM, PNG, JPG — anything PIL handles), computes
row-FFT (Y) and column-FFT (X) spectrograms, then aggregates per
label:

    COLOR cell  ← mean of low-band Y amplitudes per RGB channel
    FACE  cell  ← mean of high-band Y amplitudes per RGB channel
    SHAPE cell  ← mean of full-band X amplitudes (grayscale)

Phases are NOT averaged — that destroys structure. Instead, the
phase from the first image labelled with that token is stored as
a reference, and the C runtime jitters it during generation.

Outputs a v8 .sss binary that the C runtime (sss_io.c) can load.
The file format and the fingerprint() function below MUST stay in
lock-step with ce_core/sss_rowvae.{h,c}.
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SSS_MAGIC = 0x53535838
SSS_VERSION = 8
SSS_FP_LEN = 256

CE_COLOR, CE_SHAPE, CE_FACE = 1, 2, 3


def fingerprint(text: str) -> np.ndarray:
    """Byte-histogram fingerprint. Mirror of sss_fingerprint() in C."""
    fp = np.zeros(SSS_FP_LEN, dtype=np.float32)
    data = text.encode("utf-8")
    for i, b in enumerate(data):
        fp[b] += 1.0
        n = (b + 1 + i) & 0xFF
        fp[n] += 0.3
    s = float(np.sqrt(np.sum(fp * fp)))
    if s > 1e-9:
        fp /= s
    return fp


def load_image(path: Path, size: int) -> np.ndarray:
    """Load any image as RGB float32 [0,1] resized to (size, size)."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def row_fft(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row real FFT. img: (H, W, 3) → amp/phase (H, NF, 3)."""
    spec = np.fft.rfft(img, axis=1)
    return np.abs(spec).astype(np.float32), np.angle(spec).astype(np.float32)


def col_fft(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column real FFT of grayscale. img: (H, W, 3) → amp/phase (W, NF)."""
    gray = img.mean(axis=2)
    spec = np.fft.rfft(gray, axis=0)        # (NF, W)
    spec = spec.T                            # (W, NF)
    return np.abs(spec).astype(np.float32), np.angle(spec).astype(np.float32)


def write_cell(f, ctype: int, label: str, fp: np.ndarray,
               amp: np.ndarray, phase: np.ndarray) -> None:
    label_b = label.encode("utf-8")
    f.write(struct.pack("<I", ctype))
    f.write(struct.pack("<I", len(label_b)))
    f.write(label_b)
    f.write(fp.astype(np.float32).tobytes())
    amp_flat = amp.astype(np.float32).ravel()
    f.write(struct.pack("<I", amp_flat.size))
    f.write(amp_flat.tobytes())
    phase_flat = phase.astype(np.float32).ravel()
    f.write(struct.pack("<I", phase_flat.size))
    f.write(phase_flat.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the SSS spectrogram model.")
    ap.add_argument("--labels", required=True, help="path to labels.tsv")
    ap.add_argument("--root",   default=".",  help="image root directory")
    ap.add_argument("--out",    required=True, help="output .sss path")
    ap.add_argument("--size",   type=int, default=64,
                    help="square training resolution (must be even, default 64)")
    args = ap.parse_args()

    if args.size % 2 != 0:
        print("--size must be even", file=sys.stderr)
        return 1

    H = W = args.size
    NF = W // 2 + 1
    NF_LOW = NF // 3
    NF_HIGH = NF - NF_LOW

    color_amp = defaultdict(list)
    color_phase_ref: dict[str, np.ndarray] = {}
    shape_amp = defaultdict(list)
    shape_phase_ref: dict[str, np.ndarray] = {}
    face_amp = defaultdict(list)
    face_phase_ref: dict[str, np.ndarray] = {}

    root = Path(args.root)
    n_images = 0
    with open(args.labels, encoding="utf-8") as fp_in:
        for lineno, line in enumerate(fp_in, 1):
            line = line.rstrip("\n").rstrip("\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                print(f"[{args.labels}:{lineno}] need 4 fields, got {len(parts)}",
                      file=sys.stderr)
                continue
            fname, c_lbl, s_lbl, f_lbl = parts[:4]
            ipath = root / fname
            if not ipath.exists():
                print(f"[{args.labels}:{lineno}] missing image: {ipath}",
                      file=sys.stderr)
                continue

            img = load_image(ipath, args.size)
            y_amp, y_phase = row_fft(img)        # (H, NF, 3)
            x_amp, x_phase = col_fft(img)        # (W, NF)

            color_amp[c_lbl].append(y_amp[:, :NF_LOW, :])
            color_phase_ref.setdefault(c_lbl, y_phase[:, :NF_LOW, :].copy())

            face_amp[f_lbl].append(y_amp[:, NF_LOW:, :])
            face_phase_ref.setdefault(f_lbl, y_phase[:, NF_LOW:, :].copy())

            shape_amp[s_lbl].append(x_amp)
            shape_phase_ref.setdefault(s_lbl, x_phase.copy())

            n_images += 1

    if n_images == 0:
        print("no usable images found", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cells: list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for label, stack in color_amp.items():
        amp = np.mean(np.stack(stack, axis=0), axis=0)
        cells.append((CE_COLOR, label, fingerprint(label), amp, color_phase_ref[label]))
    for label, stack in shape_amp.items():
        amp = np.mean(np.stack(stack, axis=0), axis=0)
        cells.append((CE_SHAPE, label, fingerprint(label), amp, shape_phase_ref[label]))
    for label, stack in face_amp.items():
        amp = np.mean(np.stack(stack, axis=0), axis=0)
        cells.append((CE_FACE, label, fingerprint(label), amp, face_phase_ref[label]))

    with open(out_path, "wb") as f:
        f.write(struct.pack("<IIIIIII",
                            SSS_MAGIC, SSS_VERSION,
                            H, W, NF, NF_LOW,
                            len(cells)))
        for ctype, label, fp_vec, amp, phase in cells:
            write_cell(f, ctype, label, fp_vec, amp, phase)

    bytes_written = out_path.stat().st_size
    print(f"wrote {out_path}  ({bytes_written} bytes, {len(cells)} cells "
          f"from {n_images} images, H=W={args.size}, NF={NF}, NF_LOW={NF_LOW})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
