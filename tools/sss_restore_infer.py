"""SSS memory-driven image restoration / inference CLI.

End-to-end glue around the new subband cross-attention path:

    python3 tools/sss_restore_infer.py damaged.png \
        --label "anime girl" \
        --ref data/sanrio \
        --out restored.png

Pipeline (all numpy / sss_image_io — no cv2, no Pillow, no scipy):

    1. Load and resize the damaged image to 256×256 (BGR uint8).
    2. Auto-detect damaged rows / columns (or load --mask PNG).
    3. Lazily build a CEMemory and ingest --ref images / dirs into it
       so the cross-attention path has a real candidate pool. The
       ingest call writes 25%×4 subbands per cell (see
       tools.sss_ingest._block_subband_25x4).
    4. For each ROW_BLOCK row slab:
       - Slice damaged_block + mask_block.
       - SculptGenerator._block_from_subband_attention picks the
         winning candidates by row/column attention + LL/LH/HL/HH
         consistency, inverse-Haars them back, and fills *only the
         damaged pixels*.
    5. Write the restored 256×256 image as PNG.

Same `--label` is used both as the ingest tag (so reference cells
land on the right concept) and as the prompt-tag bias passed into the
attention scorer.

Mobile / low-memory path: each block is processed independently, the
candidate pool is filtered by tag before retrieval, and the inverse
Haar synthesis allocates one (bh, W, 3) intermediate per block.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

# When this file is launched directly (python3 tools/sss_restore_infer.py
# ...), `tools` isn't on sys.path yet. Insert the repo root so the
# package-qualified imports below work in both test and CLI mode.
if __package__ in (None, "") and __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

from tools import sss_image_io
from tools.sss_ingest import (
    H, W, ROW_BLOCKS, _read_image_any, _resize_nn, ingest_labeled_image,
)
from tools.sss_unified import CEMemory, SculptGenerator
from tools.sss_restore import auto_detect_damage


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _load_image_bgr(path: str) -> np.ndarray:
    """BGR uint8, resized to 256×256, 3-channel guaranteed. Routes
    PNG / PPM through sss_image_io and everything else through
    ffmpeg (matches sss_ingest._read_image_any)."""
    img = _read_image_any(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[..., :3]
    img = _resize_nn(img, H, W)
    return np.ascontiguousarray(img, dtype=np.uint8)


def _load_mask(path: Optional[str], shape) -> Optional[np.ndarray]:
    """Mask PNG/PPM → (H, W) float in {0, 1}. Anything brighter than
    127 (mean across channels) counts as damaged."""
    if not path:
        return None
    raw = _read_image_any(path)
    if raw.ndim == 3:
        raw = raw.mean(axis=2)
    raw = _resize_nn(raw[..., None], shape[0], shape[1])[..., 0]
    return (raw > 127).astype(np.float32)


def _ingest_refs(ref_paths, label: str, memory: CEMemory) -> int:
    """Ingest one or more reference images / directories into `memory`
    so the attention scorer has candidates. Returns the number of
    images ingested."""
    files = []
    for p in ref_paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                full = os.path.join(p, name)
                ext  = os.path.splitext(name)[1].lower()
                if os.path.isfile(full) and ext in (
                    ".png", ".ppm", ".jpg", ".jpeg", ".bmp"):
                    files.append(full)
        elif os.path.isfile(p):
            files.append(p)

    n_ok = 0
    for f in files:
        try:
            res = ingest_labeled_image(f, memory, label, source=f)
            if isinstance(res, dict) and not res.get("error"):
                n_ok += 1
        except Exception:
            # Don't let one bad file stop the whole batch — restore
            # only needs *some* candidates to work.
            continue
    return n_ok


# ─────────────────────────────────────────────────────────────
# Restoration core
# ─────────────────────────────────────────────────────────────
def restore_image(damaged: np.ndarray,
                  mask: Optional[np.ndarray],
                  memory: CEMemory,
                  generator: SculptGenerator,
                  label: str = "") -> np.ndarray:
    """Apply per-block subband cross-attention restoration to a 256×256
    BGR image. Damaged pixels (mask == 1) are replaced from the
    memory's best-matching subband candidates; clean pixels are copied
    untouched."""
    if damaged.shape != (H, W, 3):
        raise ValueError(f"restore_image: expected (H, W, 3); got {damaged.shape}")

    if mask is None:
        mask = auto_detect_damage(damaged)
    mask = mask.astype(np.float32)
    if mask.shape != (H, W):
        mask = _resize_nn(mask[..., None], H, W)[..., 0]

    tags = [w.lower() for w in (label or "").split() if len(w) > 1]
    out = damaged.copy()
    for bname, (s, e) in ROW_BLOCKS.items():
        y0, y1 = int(H * s), int(H * e)
        if mask[y0:y1].max() < 0.5:
            # Block has no damage — leave it alone.
            continue
        block      = damaged[y0:y1]
        block_mask = mask[y0:y1]

        # Candidate pool: prefer tag-matched cells of this block.
        if tags:
            candidates = memory.get_by_tags(bname, tags, n=8)
        else:
            candidates = memory.cells.get(bname, [])
        # If no tag match, fall back to every cell in this block.
        if not candidates:
            candidates = memory.cells.get(bname, [])
        if not candidates:
            # No memory at all — leave the block as-is; the caller
            # gets damaged pixels untouched, which is more honest than
            # interpolating from nothing.
            continue
        out[y0:y1] = generator._block_from_subband_attention(
            block, block_mask, candidates, prompt_tags=tags)
    return out


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def _cli():
    ap = argparse.ArgumentParser(
        description="SSS subband-attention image restoration")
    ap.add_argument("input", help="damaged image (PNG / PPM / anything ffmpeg reads)")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--label", default="",
                    help="text label for retrieval (e.g. 'anime girl')")
    ap.add_argument("--mask", default=None,
                    help="optional damage mask (PNG, white = damaged); "
                         "omitted → auto-detect via row/column MSE")
    ap.add_argument("--ref", nargs="*", default=None,
                    help="reference image files / directories to ingest "
                         "before restoring (the cross-attention pool). "
                         "Required unless an existing CEMemory has been "
                         "pre-populated by another tool in the same run.")
    ap.add_argument("--db",  default=None,
                    help="CEMemory db_path (defaults to a fresh temp dir)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    damaged = _load_image_bgr(args.input)
    mask = _load_mask(args.mask, (H, W))
    if mask is None:
        mask = auto_detect_damage(damaged)
    t_load = time.perf_counter() - t0

    # Build memory + generator. Use an isolated db_path so the CLI
    # doesn't pollute the unified pipeline's process-global memory.
    import tempfile
    db_path = args.db or tempfile.mkdtemp(prefix="sss_restore_infer_")
    memory = CEMemory(db_path)
    generator = SculptGenerator(memory)

    n_ingested = 0
    if args.ref:
        t1 = time.perf_counter()
        n_ingested = _ingest_refs(args.ref, args.label or "ref", memory)
        t_ingest = time.perf_counter() - t1
        print(f"[refs] ingested {n_ingested} image(s) in {t_ingest:.2f}s "
              f"→ {memory.total_cells()} cells")
    else:
        print("[refs] no --ref given; using whatever cells the memory "
              "already carries (likely zero — pass --ref dir/ to fix)")

    t2 = time.perf_counter()
    restored = restore_image(damaged, mask, memory, generator,
                             label=args.label)
    t_rest = time.perf_counter() - t2

    sss_image_io.write_png(args.out, restored)

    # Quick metrics for the operator. We don't have ground truth, so
    # report only the diff between input and output.
    diff = restored.astype(np.float32) - damaged.astype(np.float32)
    mse  = float(np.mean(diff * diff))
    print(f"[done] in={args.input!r}  out={args.out!r}  "
          f"shape={restored.shape}  damaged_pixels={int(mask.sum())}  "
          f"changed_MSE={mse:.1f}")
    print(f"[time] load={t_load*1000:.1f}ms  restore={t_rest*1000:.1f}ms  "
          f"total={(time.perf_counter()-t0)*1000:.1f}ms")


if __name__ == "__main__":
    _cli()
