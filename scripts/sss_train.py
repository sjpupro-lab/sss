#!/usr/bin/env python3
"""
sss_train.py — Train the SSS spectrogram engine.

Reads a labels.tsv in either of two formats (auto-detected per row):

    Legacy (4-column, hand-classified):
        filename<TAB>color_label<TAB>shape_label<TAB>face_label
        → one COLOR / SHAPE / FACE cell per *label* with amplitudes
          averaged across all matching images (current behaviour).

    Free-form (2-column, descriptive):
        filename<TAB>"red circle smile"
        → splits the description on whitespace; for every (image,
          word) pair we emit one COLOR / SHAPE / FACE cell carrying
          *that image's* amplitudes and phase. The fingerprint is
          computed from the bare word (so multiple images of "red"
          all share the same fp), but the cell labels are suffixed
          (`red_0`, `red_1`, …) so they don't collide on disk.
          The C generator picks among same-fp cells via a seed-
          based tie-break (see find_cells_per_token).

Both formats can coexist in one labels.tsv. FFT separates COLOR vs
FACE by frequency band, so describing an image as "red circle" lets
"red" land in the colour band while "circle" lands in shape.

Phases are stored *as written* — for legacy averaged labels we keep
the first image's phase (averaging phases across images destroys
structure); for per-image cells we keep that image's actual phase,
which is the whole point of dropping the average.

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

SSS_MAGIC = 0x53535839       # "SSX9" — adds per-cell ce_key
SSS_VERSION = 9
SSS_FP_LEN = 256

CE_COLOR, CE_SHAPE, CE_FACE = 1, 2, 3


def fingerprint(text: str) -> np.ndarray:
    """Byte-histogram fingerprint. Mirror of sss_fingerprint() in C.
    Kept alongside ce_key for backward compatibility — the C runtime
    no longer searches with it, but the field still ships in every
    cell so legacy callers (sss_search, fp_distance) keep working."""
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


def ce_key_bytes(text: str) -> bytes:
    """Compute the 64-byte CEUnit ce_key for `text` via the C bridge.
    The trainer prefers the C function so the runtime's ce_distance
    search keys line up byte-for-byte with what the .sss file ships.

    Falls back to a raw 64-byte zero block if libsss_pybridge.so isn't
    available — older builds will still produce a loadable file (the
    runtime's v8 path regenerates ce_key from the label on demand)."""
    try:
        # When invoked via `python3 scripts/sss_train.py`, sys.path[0]
        # is `scripts/`, not the repo root, so `tools.sss_memory` is
        # not importable until we add the parent dir.
        import sys
        repo_root = str(Path(__file__).resolve().parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from tools.sss_memory import ce_feed_bytes
    except Exception:
        return b"\x00" * 64
    try:
        return ce_feed_bytes(text)
    except Exception:
        return b"\x00" * 64


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
               amp: np.ndarray, phase: np.ndarray,
               ce_key: bytes) -> None:
    label_b = label.encode("utf-8")
    if len(ce_key) != 64:
        raise ValueError(f"ce_key must be 64 bytes, got {len(ce_key)}")
    f.write(struct.pack("<I", ctype))
    f.write(struct.pack("<I", len(label_b)))
    f.write(label_b)
    f.write(ce_key)                      # v9 addition
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

    # ── Legacy (4-column) accumulators: average amps across images
    #    per label, keep the first image's phase as the reference.
    color_amp = defaultdict(list)
    color_phase_ref: dict[str, np.ndarray] = {}
    shape_amp = defaultdict(list)
    shape_phase_ref: dict[str, np.ndarray] = {}
    face_amp = defaultdict(list)
    face_phase_ref: dict[str, np.ndarray] = {}

    # ── Free-form (2-column) per-(image, word) cells. One entry per
    #    cell that will be written as-is, no averaging. Tuple shape:
    #    (ctype, label, fp, ce_key, amp, phase).
    individual_cells: list[tuple[
        int, str, np.ndarray, bytes, np.ndarray, np.ndarray]] = []
    word_counts: dict[str, int] = defaultdict(int)

    root = Path(args.root)
    n_images = 0
    with open(args.labels, encoding="utf-8") as fp_in:
        for lineno, line in enumerate(fp_in, 1):
            line = line.rstrip("\n").rstrip("\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            # Auto-detect format. Legacy ≥ 4 fields (filename + 3
            # labels); free-form is exactly 2 fields. Anything else
            # is a malformed row.
            if len(parts) >= 4:
                fname, c_lbl, s_lbl, f_lbl = parts[:4]
                mode = "legacy"
            elif len(parts) == 2:
                fname, desc = parts
                mode = "freeform"
            else:
                print(f"[{args.labels}:{lineno}] expected 2 or ≥4 TAB-separated"
                      f" fields, got {len(parts)}", file=sys.stderr)
                continue

            ipath = root / fname
            if not ipath.exists():
                print(f"[{args.labels}:{lineno}] missing image: {ipath}",
                      file=sys.stderr)
                continue

            img = load_image(ipath, args.size)
            y_amp, y_phase = row_fft(img)        # (H, NF, 3)
            x_amp, x_phase = col_fft(img)        # (W, NF)

            if mode == "legacy":
                color_amp[c_lbl].append(y_amp[:, :NF_LOW, :])
                color_phase_ref.setdefault(c_lbl, y_phase[:, :NF_LOW, :].copy())
                face_amp[f_lbl].append(y_amp[:, NF_LOW:, :])
                face_phase_ref.setdefault(f_lbl, y_phase[:, NF_LOW:, :].copy())
                shape_amp[s_lbl].append(x_amp)
                shape_phase_ref.setdefault(s_lbl, x_phase.copy())
            else:
                # Free-form: every word becomes its own per-image cell.
                # FFT bands separate COLOR (low Y) from FACE (high Y),
                # so a word that means "red" naturally settles into the
                # colour band and "circle" into the shape band — even
                # though we apply all three. fp + ce_key come from the
                # bare word so all sibling cells share both keys; the
                # `_N` suffix only disambiguates the human-readable
                # label on disk.
                words = [w for w in desc.split() if w]
                for word in words:
                    idx = word_counts[word]
                    word_counts[word] += 1
                    label = f"{word}_{idx}"
                    fp = fingerprint(word)
                    key = ce_key_bytes(word)
                    individual_cells.append((
                        CE_COLOR, label, fp, key,
                        y_amp[:, :NF_LOW, :].copy(),
                        y_phase[:, :NF_LOW, :].copy()))
                    individual_cells.append((
                        CE_FACE, label, fp, key,
                        y_amp[:, NF_LOW:, :].copy(),
                        y_phase[:, NF_LOW:, :].copy()))
                    individual_cells.append((
                        CE_SHAPE, label, fp, key,
                        x_amp.copy(),
                        x_phase.copy()))

            n_images += 1

    if n_images == 0:
        print("no usable images found", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cells: list[tuple[
        int, str, np.ndarray, bytes, np.ndarray, np.ndarray]] = []
    # Legacy mode: averaged amps + first-image phase per label.
    for label, stack in color_amp.items():
        amp = np.mean(np.stack(stack, axis=0), axis=0)
        cells.append((CE_COLOR, label, fingerprint(label),
                      ce_key_bytes(label), amp, color_phase_ref[label]))
    for label, stack in shape_amp.items():
        amp = np.mean(np.stack(stack, axis=0), axis=0)
        cells.append((CE_SHAPE, label, fingerprint(label),
                      ce_key_bytes(label), amp, shape_phase_ref[label]))
    for label, stack in face_amp.items():
        amp = np.mean(np.stack(stack, axis=0), axis=0)
        cells.append((CE_FACE, label, fingerprint(label),
                      ce_key_bytes(label), amp, face_phase_ref[label]))
    # Free-form mode: per-(image, word) cells written as-is.
    cells.extend(individual_cells)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<IIIIIII",
                            SSS_MAGIC, SSS_VERSION,
                            H, W, NF, NF_LOW,
                            len(cells)))
        for ctype, label, fp_vec, ce_key, amp, phase in cells:
            write_cell(f, ctype, label, fp_vec, amp, phase, ce_key)

    bytes_written = out_path.stat().st_size
    print(f"wrote {out_path}  ({bytes_written} bytes, {len(cells)} cells "
          f"from {n_images} images, H=W={args.size}, NF={NF}, NF_LOW={NF_LOW})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
