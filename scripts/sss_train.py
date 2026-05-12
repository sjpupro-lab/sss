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
          *that image's* amplitudes. The fingerprint is computed
          from the bare word (so multiple images of "red" all share
          the same fp), but the cell labels are suffixed
          (`red_0`, `red_1`, …) so they don't collide on disk.
          The C generator picks among same-fp cells via a seed-
          based tie-break (see find_cells_per_token).

Both formats can coexist in one labels.tsv. FFT separates COLOR vs
FACE by frequency band, so describing an image as "red circle" lets
"red" land in the colour band while "circle" lands in shape.

Phase data is NOT written (Phase 1 amp-only radio tuning, see
`ce_core/sss_rowvae.c`). The on-disk `phase_len` field stays in the
v9 schema for backward compatibility — we just emit 0 floats — so
existing v9 readers still parse the file. The generator overlays a
per-seed random phase at noise-init time, which is the dominant
source of per-seed output diversity now that the trained-side phase
is gone.

Outputs a v9 .sss binary that the C runtime (sss_io.c) can load.
The file format and the fingerprint() function below MUST stay in
lock-step with ce_core/sss_rowvae.{h,c}.
"""
from __future__ import annotations

import argparse
import functools
import math
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# Wire `tools.sss_memory.ce_feed_bytes` (which routes to the C bridge)
# once at import time. `python3 scripts/sss_train.py` puts `scripts/`
# on sys.path[0]; we add the repo root so `from tools…` resolves.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    from tools.sss_memory import ce_feed_bytes as _ce_feed_bytes_impl
    _CE_BRIDGE_ERROR: str | None = None
except Exception as _err:                    # noqa: BLE001 — keep raw msg
    _ce_feed_bytes_impl = None               # type: ignore[assignment]
    _CE_BRIDGE_ERROR = str(_err)

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


@functools.lru_cache(maxsize=None)
def ce_key_bytes(text: str) -> bytes:
    """Compute the 64-byte CEUnit ce_key for `text` via the C bridge.
    The trainer prefers the C function so the runtime's ce_distance
    search keys line up byte-for-byte with what the .sss file ships.

    Memoised: each distinct token crosses the Python↔ctypes boundary
    exactly once, even when many cells reference the same word.

    Raises RuntimeError if libsss_pybridge.so isn't loadable. We
    intentionally do NOT silently fall back to a 64-byte zero key —
    that would write a v9 file whose ce_distance search returns
    nonsense, producing models that look fine on disk but search
    incorrectly at generate time. Rebuild with `make pybridge` and
    retry."""
    if _ce_feed_bytes_impl is None:
        raise RuntimeError(
            "tools.sss_memory.ce_feed_bytes is unreachable: "
            f"{_CE_BRIDGE_ERROR}. Rebuild with `make pybridge` so the "
            "trainer can derive each cell's ce_key from C ce_feed.")
    return _ce_feed_bytes_impl(text)


def load_image(path: Path, size: int) -> np.ndarray:
    """Load any image as RGB float32 [0,1] resized to (size, size)."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def row_fft(img: np.ndarray) -> np.ndarray:
    """Per-row real FFT amplitude. img: (H, W, 3) → amp (H, NF, 3).

    Phase is intentionally not returned (Phase 1 amp-only). The
    generator regenerates phase as a per-seed random uniform at
    noise-init time; storing the training-side phase would only
    inflate the model without changing generated output."""
    spec = np.fft.rfft(img, axis=1)
    return np.abs(spec).astype(np.float32)


def col_fft(img: np.ndarray) -> np.ndarray:
    """Per-column real FFT amplitude of grayscale.
    img: (H, W, 3) → amp (W, NF). See row_fft re: phase omission."""
    gray = img.mean(axis=2)
    spec = np.fft.rfft(gray, axis=0)        # (NF, W)
    spec = spec.T                            # (W, NF)
    return np.abs(spec).astype(np.float32)


def write_cell(f, ctype: int, label: str, fp: np.ndarray,
               amp: np.ndarray, ce_key: bytes) -> None:
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
    # Phase 1 amp-only: emit phase_len = 0 (no phase floats follow).
    # Readers still parse the u32 — the layout is intentionally a
    # backward-compatible subset of v9.
    f.write(struct.pack("<I", 0))


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
    #    per label. Phase used to be kept as a per-label reference,
    #    but Phase 1 dropped it (see write_cell docstring).
    color_amp = defaultdict(list)
    shape_amp = defaultdict(list)
    face_amp = defaultdict(list)

    # ── Free-form (2-column) per-(image, word) cells. One entry per
    #    cell that will be written as-is, no averaging. Tuple shape:
    #    (ctype, label, fp, ce_key, amp).
    individual_cells: list[tuple[
        int, str, np.ndarray, bytes, np.ndarray]] = []
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
            y_amp = row_fft(img)        # (H, NF, 3)
            x_amp = col_fft(img)        # (W, NF)

            if mode == "legacy":
                color_amp[c_lbl].append(y_amp[:, :NF_LOW, :])
                face_amp[f_lbl].append(y_amp[:, NF_LOW:, :])
                shape_amp[s_lbl].append(x_amp)
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
                        y_amp[:, :NF_LOW, :].copy()))
                    individual_cells.append((
                        CE_FACE, label, fp, key,
                        y_amp[:, NF_LOW:, :].copy()))
                    individual_cells.append((
                        CE_SHAPE, label, fp, key,
                        x_amp.copy()))

            n_images += 1

    if n_images == 0:
        print("no usable images found", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cells: list[tuple[
        int, str, np.ndarray, bytes, np.ndarray]] = []
    # Legacy mode: averaged amps per label (phase is no longer kept;
    # see write_cell docstring).
    for label, stack in color_amp.items():
        amp = np.mean(np.stack(stack, axis=0), axis=0)
        cells.append((CE_COLOR, label, fingerprint(label),
                      ce_key_bytes(label), amp))
    for label, stack in shape_amp.items():
        amp = np.mean(np.stack(stack, axis=0), axis=0)
        cells.append((CE_SHAPE, label, fingerprint(label),
                      ce_key_bytes(label), amp))
    for label, stack in face_amp.items():
        amp = np.mean(np.stack(stack, axis=0), axis=0)
        cells.append((CE_FACE, label, fingerprint(label),
                      ce_key_bytes(label), amp))
    # Free-form mode: per-(image, word) cells written as-is.
    cells.extend(individual_cells)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<IIIIIII",
                            SSS_MAGIC, SSS_VERSION,
                            H, W, NF, NF_LOW,
                            len(cells)))
        for ctype, label, fp_vec, ce_key, amp in cells:
            write_cell(f, ctype, label, fp_vec, amp, ce_key)

    bytes_written = out_path.stat().st_size
    print(f"wrote {out_path}  ({bytes_written} bytes, {len(cells)} cells "
          f"from {n_images} images, H=W={args.size}, NF={NF}, NF_LOW={NF_LOW})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
