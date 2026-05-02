#!/usr/bin/env python3
"""
sss_train.py — Train the SSS sculpt engine.

Reads a labels.tsv of the form
    filename<TAB>color_label<TAB>shape_label<TAB>face_label
and groups each image into three pools:

    COLOR cell  ← pool of images sharing a color label
    SHAPE cell  ← pool of images sharing a shape label
    FACE  cell  ← pool of images sharing a face label

Images are stored *as full row patterns* — not averaged, not
spectrogrammed. The C runtime walks each cell's pool row by row
to "sculpt" a new image at generate time.

Outputs a v9 .sss binary that the C runtime (sss_io.c) can load.
The file format and the fingerprint() function below MUST stay in
lock-step with ce_core/sss_rowvae.{h,c}.
"""
from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SSS_MAGIC = 0x53535839      # 'SSX9'
SSS_VERSION = 9
SSS_FP_LEN = 256
CHANNELS = 3

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


def write_cell(f, ctype: int, label: str, fp: np.ndarray,
               imgs: np.ndarray) -> None:
    """imgs: (num_imgs, H, W, 3) float32 in [0,1]."""
    label_b = label.encode("utf-8")
    f.write(struct.pack("<I", ctype))
    f.write(struct.pack("<I", len(label_b)))
    f.write(label_b)
    f.write(fp.astype(np.float32).tobytes())
    n = imgs.shape[0]
    f.write(struct.pack("<I", n))
    f.write(imgs.astype(np.float32).ravel().tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the SSS sculpt model.")
    ap.add_argument("--labels", required=True, help="path to labels.tsv")
    ap.add_argument("--root",   default=".",  help="image root directory")
    ap.add_argument("--out",    required=True, help="output .sss path")
    ap.add_argument("--size",   type=int, default=64,
                    help="square training resolution (must be even, default 64)")
    ap.add_argument("--max-per-cell", type=int, default=32,
                    help="cap pool size per cell (default 32; extras are dropped)")
    args = ap.parse_args()

    if args.size % 2 != 0:
        print("--size must be even", file=sys.stderr)
        return 1

    H = W = args.size

    color_pool: dict[str, list[np.ndarray]] = defaultdict(list)
    shape_pool: dict[str, list[np.ndarray]] = defaultdict(list)
    face_pool:  dict[str, list[np.ndarray]] = defaultdict(list)

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
            color_pool[c_lbl].append(img)
            shape_pool[s_lbl].append(img)
            face_pool[f_lbl].append(img)
            n_images += 1

    if n_images == 0:
        print("no usable images found", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cells: list[tuple[int, str, np.ndarray, np.ndarray]] = []

    def collect(ctype: int, pool: dict[str, list[np.ndarray]]):
        for label, imgs in pool.items():
            if not imgs:
                continue
            stack = np.stack(imgs[: args.max_per_cell], axis=0)
            cells.append((ctype, label, fingerprint(label), stack))

    collect(CE_COLOR, color_pool)
    collect(CE_SHAPE, shape_pool)
    collect(CE_FACE,  face_pool)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<IIIIIII",
                            SSS_MAGIC, SSS_VERSION,
                            H, W, CHANNELS,
                            len(cells), 0))
        for ctype, label, fp_vec, imgs in cells:
            write_cell(f, ctype, label, fp_vec, imgs)

    bytes_written = out_path.stat().st_size
    total_imgs_stored = sum(c[3].shape[0] for c in cells)
    print(f"wrote {out_path}  ({bytes_written} bytes, {len(cells)} cells, "
          f"{total_imgs_stored} stored row-patterns from {n_images} source images, "
          f"H=W={args.size})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
