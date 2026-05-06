#!/usr/bin/env python3
"""SSS Unified Pipeline v2 — No Noise.

Bundles the scattered SSS logic (parse → memory search → sculpt →
motion → evaluate → store) into one entry point: `run_sss_pipeline()`.

Generation is fully deterministic: blank canvas → CE cell sculpt /
recombine. No random noise. Variations come from fixed hue/saturation/
brightness curves selected by index.

Memory is backed by the C bridge (ce_storage_add_typed) so every
`memory.add(...)` lands in real CEStorage — same store the engine's
search/decode operates on. Build the bridge once with::

    make pybridge

then::

    python3 tools/sss_unified.py            # quick demo run
    python3 tools/test_sss_unified.py       # smoke test
"""
from __future__ import annotations

import os
import random
import subprocess
from collections import defaultdict
from typing import Optional

import numpy as np

try:
    import cv2  # native OpenCV
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None  # populated below by the fallback shim

# Bridge-backed CEMemory (ce_storage_add_typed under the hood).
from tools.sss_memory import CEMemory as _BridgedCEMemory

# Stdlib-only PNG / PPM encoder. Used for `imwrite` so the demo writes
# images even when cv2 isn't installed.
from tools import sss_image_io


# ────────────────────────────────────────────────────────────
# 0a. cv2 fallback — numpy-only implementations of the OpenCV
#     surface this module uses. Activated only if `import cv2`
#     fails. Intentionally minimal: not a complete cv2 stand-in,
#     just enough for sss_unified's deterministic generate / eval
#     path (cvtColor, Canny, resize, filter2D, simple drawing
#     primitives, addWeighted, inRange, imwrite-as-PPM).
# ────────────────────────────────────────────────────────────
if not HAS_CV2:
    class _CV2Fallback:
        # Codes used by this module. Values are arbitrary — only
        # equality with the constants below matters.
        COLOR_BGR2RGB  = 4
        COLOR_BGR2GRAY = 6
        COLOR_GRAY2BGR = 8
        COLOR_BGR2HSV  = 40
        COLOR_HSV2BGR  = 54

        # ---- color conversions --------------------------------
        @staticmethod
        def cvtColor(img, code):
            arr = np.asarray(img)
            if code == _CV2Fallback.COLOR_BGR2RGB:
                return arr[..., ::-1].copy()
            if code == _CV2Fallback.COLOR_BGR2GRAY:
                # B=29/256, G=150/256, R=77/256  ≈ 0.114/0.587/0.299
                a = arr.astype(np.uint16)
                gray = (a[..., 0] * 29 + a[..., 1] * 150
                        + a[..., 2] * 77) >> 8
                return gray.astype(np.uint8)
            if code == _CV2Fallback.COLOR_GRAY2BGR:
                if arr.ndim == 2:
                    return np.stack([arr, arr, arr], axis=-1)
                return arr
            if code == _CV2Fallback.COLOR_BGR2HSV:
                return _bgr2hsv(arr)
            if code == _CV2Fallback.COLOR_HSV2BGR:
                return _hsv2bgr(arr)
            raise ValueError(f"cvtColor: unsupported code {code}")

        # ---- gradient / blur ----------------------------------
        @staticmethod
        def Canny(gray, t1, t2):
            # Sobel magnitude + threshold. Not Canny in the strict
            # sense (no NMS / hysteresis) but produces a binary edge
            # map that Evaluator and the cell-canny stash can use.
            g = gray.astype(np.int32)
            gx = np.zeros_like(g); gy = np.zeros_like(g)
            gx[:, 1:-1] = (g[:, 2:] - g[:, :-2])
            gy[1:-1, :] = (g[2:, :] - g[:-2, :])
            mag = np.abs(gx) + np.abs(gy)
            return ((mag >= int(t2)).astype(np.uint8)) * 255

        @staticmethod
        def resize(img, dsize, interpolation=None):
            # Nearest neighbour. dsize is (W, H) per cv2 convention.
            nw, nh = int(dsize[0]), int(dsize[1])
            h, w = img.shape[:2]
            yi = (np.arange(nh) * h // max(1, nh)).astype(np.int64)
            xi = (np.arange(nw) * w // max(1, nw)).astype(np.int64)
            yi = np.clip(yi, 0, h - 1)
            xi = np.clip(xi, 0, w - 1)
            if img.ndim == 2:
                return img[yi[:, None], xi[None, :]]
            return img[yi[:, None], xi[None, :], :]

        @staticmethod
        def GaussianBlur(img, ksize, sigmaX, sigmaY=0):
            # Simple separable box blur of width ksize[0] — close
            # enough for the deterministic-blur use case here.
            kw = max(1, int(ksize[0]) | 1)
            pad = kw // 2
            kernel = np.ones(kw, np.float32) / kw
            f = img.astype(np.float32)
            if f.ndim == 2:
                f = np.pad(f, ((pad, pad), (pad, pad)), mode="edge")
                tmp = np.zeros_like(f)
                for i in range(kw):
                    tmp += np.roll(f, i - pad, axis=0) * kernel[i]
                tmp2 = np.zeros_like(tmp)
                for i in range(kw):
                    tmp2 += np.roll(tmp, i - pad, axis=1) * kernel[i]
                return np.clip(
                    tmp2[pad:-pad, pad:-pad], 0, 255).astype(np.uint8)
            f = np.pad(f, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            tmp = np.zeros_like(f)
            for i in range(kw):
                tmp += np.roll(f, i - pad, axis=0) * kernel[i]
            tmp2 = np.zeros_like(tmp)
            for i in range(kw):
                tmp2 += np.roll(tmp, i - pad, axis=1) * kernel[i]
            return np.clip(
                tmp2[pad:-pad, pad:-pad, :], 0, 255).astype(np.uint8)

        @staticmethod
        def filter2D(img, ddepth, kernel):
            kh, kw = kernel.shape
            ph, pw = kh // 2, kw // 2
            f = img.astype(np.float32)
            if f.ndim == 2:
                padded = np.pad(f, ((ph, ph), (pw, pw)), mode="edge")
                out = np.zeros_like(f)
                for y in range(kh):
                    for x in range(kw):
                        out += (padded[y:y + f.shape[0],
                                       x:x + f.shape[1]]
                                * float(kernel[y, x]))
            else:
                padded = np.pad(f, ((ph, ph), (pw, pw), (0, 0)),
                                mode="edge")
                out = np.zeros_like(f)
                for y in range(kh):
                    for x in range(kw):
                        out += (padded[y:y + f.shape[0],
                                       x:x + f.shape[1], :]
                                * float(kernel[y, x]))
            return np.clip(out, 0, 255).astype(img.dtype if img.dtype
                                               != np.bool_ else np.uint8)

        # ---- pixel arithmetic ---------------------------------
        @staticmethod
        def addWeighted(a, wa, b, wb, gamma, dst=None):
            out = np.clip(
                a.astype(np.float32) * float(wa)
                + b.astype(np.float32) * float(wb)
                + float(gamma), 0, 255).astype(np.uint8)
            if dst is not None:
                dst[...] = out
            return out

        @staticmethod
        def inRange(img, lo, hi):
            lo = np.asarray(lo); hi = np.asarray(hi)
            mask = np.all((img >= lo) & (img <= hi), axis=-1)
            return mask.astype(np.uint8) * 255

        # ---- drawing primitives -------------------------------
        @staticmethod
        def circle(img, center, r, color, thickness=1):
            _draw_circle(img, center, int(r), color, int(thickness))

        @staticmethod
        def rectangle(img, p1, p2, color, thickness=1):
            _draw_rect(img, p1, p2, color, int(thickness))

        @staticmethod
        def line(img, p1, p2, color, thickness=1):
            _draw_line(img, p1, p2, color, int(thickness))

        @staticmethod
        def ellipse(img, center, axes, angle, start, end,
                    color, thickness=1):
            _draw_ellipse(img, center, axes, float(angle),
                          float(start), float(end), color,
                          int(thickness))

        @staticmethod
        def fillPoly(img, pts_list, color):
            for pts in pts_list:
                _fill_poly(img, np.asarray(pts).reshape(-1, 2), color)

        # ---- IO ----------------------------------------------
        @staticmethod
        def imwrite(path, img):
            # Write a PPM next to whatever path was requested. We do
            # not try to encode PNG/JPEG by hand. The demo's ffmpeg
            # step won't work without cv2 either; that's expected.
            base, _ext = os.path.splitext(path)
            rgb = np.ascontiguousarray(img[..., ::-1].astype(np.uint8))
            h, w = rgb.shape[:2]
            with open(base + ".ppm", "wb") as f:
                f.write(f"P6\n{w} {h}\n255\n".encode())
                f.write(rgb.tobytes())
            return True

    # ---- numpy helpers used by the shim above -----------------
    def _bgr2hsv(arr):
        a = arr.astype(np.float32)
        b, g, r = a[..., 0], a[..., 1], a[..., 2]
        cmax = np.maximum(np.maximum(b, g), r)
        cmin = np.minimum(np.minimum(b, g), r)
        delta = cmax - cmin
        safe = np.where(delta == 0, 1.0, delta)
        h = np.zeros_like(cmax)
        rmax = (cmax == r) & (delta > 0)
        gmax = (cmax == g) & (delta > 0) & ~rmax
        bmax = (cmax == b) & (delta > 0) & ~rmax & ~gmax
        # OpenCV scales H to [0, 180); each 60° segment becomes 30 units.
        h = np.where(rmax, ((g - b) / safe) * 30.0, h)
        h = np.where(gmax, ((b - r) / safe) * 30.0 + 60.0, h)
        h = np.where(bmax, ((r - g) / safe) * 30.0 + 120.0, h)
        h = np.mod(h, 180.0)
        s = np.where(cmax > 0, (delta / np.where(cmax == 0, 1.0, cmax))
                     * 255.0, 0.0)
        v = cmax
        return np.stack([h, s, v], axis=-1).clip(0, 255).astype(np.uint8)

    def _hsv2bgr(arr):
        a = arr.astype(np.float32)
        h = a[..., 0] * 2.0   # → 0..360
        s = a[..., 1] / 255.0
        v = a[..., 2]
        c = v * s
        h60 = h / 60.0
        x = c * (1.0 - np.abs(np.mod(h60, 2.0) - 1.0))
        m = v - c
        seg = np.mod(h60.astype(np.int32), 6)
        zero = np.zeros_like(v)
        r = np.select(
            [seg == 0, seg == 1, seg == 2, seg == 3, seg == 4, seg == 5],
            [c, x, zero, zero, x, c], default=zero)
        g = np.select(
            [seg == 0, seg == 1, seg == 2, seg == 3, seg == 4, seg == 5],
            [x, c, c, x, zero, zero], default=zero)
        b = np.select(
            [seg == 0, seg == 1, seg == 2, seg == 3, seg == 4, seg == 5],
            [zero, zero, x, c, c, x], default=zero)
        out = np.stack([b + m, g + m, r + m], axis=-1)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _coerce_color(img, color):
        n = img.shape[-1] if img.ndim == 3 else 1
        if np.isscalar(color):
            return np.full(n, int(color), dtype=img.dtype)
        c = np.asarray(color).flatten()
        if c.size < n:
            c = np.concatenate([c, np.zeros(n - c.size)])
        return c[:n].astype(img.dtype)

    def _draw_circle(img, center, r, color, thickness):
        cx, cy = int(center[0]), int(center[1])
        h, w = img.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        col = _coerce_color(img, color)
        if thickness < 0:
            mask = d2 <= r * r
        else:
            outer = (r + max(thickness, 1) / 2.0) ** 2
            inner = max(0.0, r - max(thickness, 1) / 2.0) ** 2
            mask = (d2 <= outer) & (d2 >= inner)
        if img.ndim == 2:
            img[mask] = col
        else:
            img[mask] = col

    def _draw_rect(img, p1, p2, color, thickness):
        x0, y0 = int(p1[0]), int(p1[1])
        x1, y1 = int(p2[0]), int(p2[1])
        if x0 > x1: x0, x1 = x1, x0
        if y0 > y1: y0, y1 = y1, y0
        h, w = img.shape[:2]
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(w - 1, x1); y1 = min(h - 1, y1)
        col = _coerce_color(img, color)
        if thickness < 0:
            img[y0:y1 + 1, x0:x1 + 1] = col
            return
        t = max(1, thickness)
        img[y0:min(y0 + t, h), x0:x1 + 1] = col
        img[max(0, y1 - t + 1):y1 + 1, x0:x1 + 1] = col
        img[y0:y1 + 1, x0:min(x0 + t, w)] = col
        img[y0:y1 + 1, max(0, x1 - t + 1):x1 + 1] = col

    def _draw_line(img, p1, p2, color, thickness):
        x0, y0 = int(p1[0]), int(p1[1])
        x1, y1 = int(p2[0]), int(p2[1])
        n = max(abs(x1 - x0), abs(y1 - y0)) + 1
        xs = np.linspace(x0, x1, n).round().astype(np.int64)
        ys = np.linspace(y0, y1, n).round().astype(np.int64)
        h, w = img.shape[:2]
        col = _coerce_color(img, color)
        t = max(1, thickness)
        for dy in range(-(t // 2), (t // 2) + 1):
            for dx in range(-(t // 2), (t // 2) + 1):
                yy = np.clip(ys + dy, 0, h - 1)
                xx = np.clip(xs + dx, 0, w - 1)
                img[yy, xx] = col

    def _draw_ellipse(img, center, axes, angle, start, end,
                      color, thickness):
        cx, cy = int(center[0]), int(center[1])
        a, b = max(1, int(axes[0])), max(1, int(axes[1]))
        h, w = img.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        rad = np.deg2rad(angle)
        ca, sa = np.cos(rad), np.sin(rad)
        dx = xx - cx; dy = yy - cy
        rx = dx * ca + dy * sa
        ry = -dx * sa + dy * ca
        norm = (rx * rx) / (a * a) + (ry * ry) / (b * b)
        if thickness < 0:
            shape = norm <= 1.0
        else:
            band = max(thickness, 1) / float(max(a, b))
            outer = (1.0 + band) ** 2
            inner = max(0.0, 1.0 - band) ** 2
            shape = (norm <= outer) & (norm >= inner)
        # Sweep gate. atan2(ry, rx) in degrees, normalised to 0..360.
        sweep = (end - start) % 360
        if sweep == 0 and start != end:
            arc = np.ones_like(shape)
        else:
            theta = np.mod(np.degrees(np.arctan2(ry, rx)) - start, 360.0)
            arc = theta <= max(sweep, 1.0)
        col = _coerce_color(img, color)
        img[shape & arc] = col

    def _fill_poly(img, pts, color):
        # Even-odd ray casting via numpy. Slow per-edge but fine for
        # the seed-foundation triangle (3 vertices).
        h, w = img.shape[:2]
        yy, xx = np.mgrid[:h, :w]
        inside = np.zeros((h, w), dtype=bool)
        n = len(pts)
        j = n - 1
        for i in range(n):
            xi, yi = float(pts[i][0]), float(pts[i][1])
            xj, yj = float(pts[j][0]), float(pts[j][1])
            denom = (yj - yi) if (yj - yi) != 0 else 1e-9
            cond = ((yi > yy) != (yj > yy)) & (
                xx < (xj - xi) * (yy - yi) / denom + xi)
            inside ^= cond
            j = i
        col = _coerce_color(img, color)
        img[inside] = col

    cv2 = _CV2Fallback()


# ────────────────────────────────────────────────────────────
# 0b. Tick integer math — mirrors ce_core/slig_tick_math.h
#     Replaces the float blends and trig used in the generation
#     and motion paths so behaviour matches the C engine. The
#     score / evaluator code stays in float (per spec).
# ────────────────────────────────────────────────────────────
TICK_SIN_TABLE = np.array([
    128, 131, 134, 137, 140, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174,
    177, 179, 182, 185, 188, 191, 193, 196, 199, 201, 204, 206, 209, 211, 213, 216,
    218, 220, 222, 224, 226, 228, 230, 232, 234, 235, 237, 239, 240, 241, 243, 244,
    245, 246, 248, 249, 250, 250, 251, 252, 253, 253, 254, 254, 254, 255, 255, 255,
    255, 255, 255, 255, 254, 254, 254, 253, 253, 252, 251, 250, 250, 249, 248, 246,
    245, 244, 243, 241, 240, 239, 237, 235, 234, 232, 230, 228, 226, 224, 222, 220,
    218, 216, 213, 211, 209, 206, 204, 201, 199, 196, 193, 191, 188, 185, 182, 179,
    177, 174, 171, 168, 165, 162, 159, 156, 153, 150, 147, 144, 140, 137, 134, 131,
    128, 125, 122, 119, 116, 112, 109, 106, 103, 100,  97,  94,  91,  88,  85,  82,
     79,  77,  74,  71,  68,  65,  63,  60,  57,  55,  52,  50,  47,  45,  43,  40,
     38,  36,  34,  32,  30,  28,  26,  24,  22,  21,  19,  17,  16,  15,  13,  12,
     11,  10,   8,   7,   6,   6,   5,   4,   3,   3,   2,   2,   2,   1,   1,   1,
      1,   1,   1,   1,   2,   2,   2,   3,   3,   4,   5,   6,   6,   7,   8,  10,
     11,  12,  13,  15,  16,  17,  19,  21,  22,  24,  26,  28,  30,  32,  34,  36,
     38,  40,  43,  45,  47,  50,  52,  55,  57,  60,  63,  65,  68,  71,  74,  77,
     79,  82,  85,  88,  91,  94,  97, 100, 103, 106, 109, 112, 116, 119, 122, 125,
], dtype=np.uint8)

TICK_COS_TABLE = np.array([
    255, 255, 255, 255, 254, 254, 254, 253, 253, 252, 251, 250, 250, 249, 248, 246,
    245, 244, 243, 241, 240, 239, 237, 235, 234, 232, 230, 228, 226, 224, 222, 220,
    218, 216, 213, 211, 209, 206, 204, 201, 199, 196, 193, 191, 188, 185, 182, 179,
    177, 174, 171, 168, 165, 162, 159, 156, 153, 150, 147, 144, 140, 137, 134, 131,
    128, 125, 122, 119, 116, 112, 109, 106, 103, 100,  97,  94,  91,  88,  85,  82,
     79,  77,  74,  71,  68,  65,  63,  60,  57,  55,  52,  50,  47,  45,  43,  40,
     38,  36,  34,  32,  30,  28,  26,  24,  22,  21,  19,  17,  16,  15,  13,  12,
     11,  10,   8,   7,   6,   6,   5,   4,   3,   3,   2,   2,   2,   1,   1,   1,
      1,   1,   1,   1,   2,   2,   2,   3,   3,   4,   5,   6,   6,   7,   8,  10,
     11,  12,  13,  15,  16,  17,  19,  21,  22,  24,  26,  28,  30,  32,  34,  36,
     38,  40,  43,  45,  47,  50,  52,  55,  57,  60,  63,  65,  68,  71,  74,  77,
     79,  82,  85,  88,  91,  94,  97, 100, 103, 106, 109, 112, 116, 119, 122, 125,
    128, 131, 134, 137, 140, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174,
    177, 179, 182, 185, 188, 191, 193, 196, 199, 201, 204, 206, 209, 211, 213, 216,
    218, 220, 222, 224, 226, 228, 230, 232, 234, 235, 237, 239, 240, 241, 243, 244,
    245, 246, 248, 249, 250, 250, 251, 252, 253, 253, 254, 254, 254, 255, 255, 255,
], dtype=np.uint8)

# Convert radians → tick-256 phase index. 256 entries == 2π.
TICK_PHASE_SCALE = 256.0 / (2.0 * np.pi)


def tick_sin_signed(idx):
    """Lookup signed sin in [-128, 127] from TICK_SIN_TABLE."""
    return int(TICK_SIN_TABLE[idx & 0xFF]) - 128


def tick_cos_signed(idx):
    return int(TICK_COS_TABLE[idx & 0xFF]) - 128


def tick_phase_idx(rad):
    """Float radians → tick-table index (0..255)."""
    return int(rad * TICK_PHASE_SCALE) & 0xFF


W, H = 256, 256
ROW_BLOCKS = {
    "hair":  (0.00, 0.15),
    "face":  (0.15, 0.40),
    "upper": (0.40, 0.65),
    "lower": (0.65, 0.88),
    "bg":    (0.88, 1.00),
}

BASE_DIR = os.environ.get("SSS_UNIFIED_DIR",
                          os.path.join(os.path.expanduser("~"), "sss_unified"))


def write_ppm(path, img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(rgb.tobytes())


# ────────────────────────────────────────────────────────────
# 1. CEMemory  (bridge-backed; layered with tags / pending / index)
# ────────────────────────────────────────────────────────────
class CEMemory(_BridgedCEMemory):
    """CEMemory with tag search, pending queue, and prompt index.

    Inherits the bridge-backed `add_cell` so every `add()` routes
    through ce_storage_add_typed. The Python-side cell dict carries
    the extra fields (`tags`, `uses`) that the pipeline needs but
    that don't belong inside CEStorage entries.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(BASE_DIR, "memory")
        super().__init__(db_path)
        self.pending = []
        self.prompt_index = defaultdict(list)
        # Pose / radar sidecar — populated by tools.sss_ingest when
        # tools.sss_pose_radar produces motion + dirty-zone hints. Lives
        # on the same memory instance so it's saved/searched alongside
        # the row-block cells, but doesn't enter CEStorage (the bridge
        # is keyed on row blocks). MotionEngine reads this when a
        # prompt's tags match an entry.
        self.pose_cells = []
        # ROW_BLOCKS adds two slots beyond the four the bridge knows
        # about (hair/face/upper/lower/bg vs. top/mid_top/mid_bot/bot).
        # Re-map to keep CEStorage slots stable across versions.
        self._slot_for_block = {
            "hair":  "top",
            "face":  "mid_top",
            "upper": "mid_bot",
            "lower": "bot",
            "bg":    "bot",  # bg shares bot bucket; block_idx still unique
        }

    # ── primary API used by the pipeline ────────────────────────

    def add(self, block, canny, color_avg, quality, tags, source="",
            *, amp=None, phase=None, fp=None):
        """Append a cell. Routes through bridge -> ce_storage_add_typed.

        Optional spectrogram fields (kept Python-side; CEStorage only
        sees the canny + color bytes via the bridge):
            amp   — list of per-(row, channel) rfft amplitudes
                    (length = block_height * 3, each entry shape (NF,))
            phase — same shape as amp; per-(row, channel) phases
            fp    — 256-bin sss-fingerprint of the label text (or None)
        """
        bridge_block = self._slot_for_block.get(block, "bot")
        cid = super().add_cell(bridge_block, canny, None, color_avg,
                               quality=float(quality), source=source)
        # super().add_cell also appends to self.cells[bridge_block] (its
        # own sidecar). We layer our own per-pipeline-block sidecar on
        # top, so drop the parent's row to avoid double-counting in
        # total_cells / avg_quality.
        parent_list = self.cells.get(bridge_block)
        if parent_list:
            parent_list.pop()
            if not parent_list:
                # Keep keys disjoint between the two views.
                self.cells.pop(bridge_block, None)

        cells_for_block = self.cells.setdefault(block, [])
        local_id = len(cells_for_block)
        cell = {
            "id": local_id,
            "ce_id": cid,
            "ce_block": bridge_block,
            "canny": canny,
            "color": (color_avg.tolist()
                      if isinstance(color_avg, np.ndarray) else list(color_avg)),
            "quality": float(quality),
            "tags": list(tags),
            "source": source,
            "gen": self.generation,
            "uses": 0,
            "amp": amp,
            "phase": phase,
            "fp": fp,
        }
        cells_for_block.append(cell)
        for t in tags:
            self.prompt_index[t].append((block, local_id))
        return local_id

    def search(self, tags, block=None):
        hits = []
        blocks = [block] if block else list(ROW_BLOCKS.keys())
        for b in blocks:
            for cell in self.cells.get(b, []):
                match = sum(1 for t in tags if t in cell.get("tags", []))
                if match > 0:
                    hits.append((match, cell, b))
        hits.sort(key=lambda x: (x[0], x[1]["quality"]), reverse=True)
        return hits

    def add_pending(self, data, reason):
        self.pending.append({"data": data, "reason": reason,
                             "gen": self.generation})

    def get_best(self, block, n=3):
        return sorted(self.cells.get(block, []),
                      key=lambda c: c["quality"], reverse=True)[:n]

    def get_by_tags(self, block, tags, n=5):
        cands = []
        for cell in self.cells.get(block, []):
            match = sum(1 for t in tags if t in cell.get("tags", []))
            if match > 0:
                cands.append((match * cell["quality"], cell))
        cands.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in cands[:n]]

    def update_quality(self, block, cid, new_q):
        cell = self.cells[block][cid]
        # Tick blend: q*0.7 + new*0.3 → (q*179 + new*77) >> 8.
        # 179 + 77 = 256 so the divide is a clean right shift.
        old256 = int(max(0.0, min(1.0, cell["quality"])) * 256)
        new256 = int(max(0.0, min(1.0, float(new_q))) * 256)
        blended = (old256 * 179 + new256 * 77) >> 8
        cell["quality"] = max(0.0, min(1.0, blended / 256.0))
        cell["uses"] += 1

    # ── pose / radar sidecar ────────────────────────────────
    def add_pose(self, pose_data, tags, *, source: str = "",
                 quality: Optional[float] = None) -> int:
        """Append one pose / radar entry to the sidecar.

        Stays out of CEStorage and out of `self.cells` so callers that
        count row-block cells (test_sss_ingest, total_cells) are
        unaffected. Returned index is the slot in `self.pose_cells`.
        """
        q = float(quality) if quality is not None else float(
            pose_data.get("quality", 0.5) if isinstance(pose_data, dict) else 0.5)
        entry = {
            "id": len(self.pose_cells),
            "tags": list(tags),
            "data": pose_data,
            "source": source,
            "quality": q,
            "gen": self.generation,
            "uses": 0,
        }
        self.pose_cells.append(entry)
        for t in tags:
            self.prompt_index[t].append(("pose", entry["id"]))
        return entry["id"]

    def find_pose(self, tags, n: int = 4):
        """Pose entries ranked by tag-match × quality. Returns at most
        `n` entries; empty list when nothing matches or the sidecar is
        empty."""
        if not self.pose_cells or not tags:
            return []
        scored = []
        for e in self.pose_cells:
            match = sum(1 for t in tags if t in e.get("tags", []))
            if match > 0:
                scored.append((match * max(0.05, e.get("quality", 0.0)), e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:n]]

    def total_pose_cells(self) -> int:
        return len(self.pose_cells)

    def total_cells(self):
        return sum(len(v) for v in self.cells.values())

    def avg_quality(self):
        all_q = [c["quality"]
                 for cells in self.cells.values() for c in cells]
        return float(np.mean(all_q)) if all_q else 0.0


# ────────────────────────────────────────────────────────────
# 2. Planner
# ────────────────────────────────────────────────────────────
class Planner:
    COLOR_MAP = {
        "빨간": "red", "파란": "blue", "초록": "green", "노란": "yellow",
        "하얀": "white", "검은": "black", "분홍": "pink", "주황": "orange",
        "red": "red", "blue": "blue", "green": "green",
    }
    EXPR_MAP = {
        "웃는": "smile", "슬픈": "sad", "놀란": "surprise",
        "눈감은": "blink", "말하는": "talk", "화난": "angry",
    }
    MOTION_MAP = {
        "걷는": "walk", "뛰는": "run", "흔드는": "wave",
        "떨어지는": "fall", "날리는": "blow",
        "꽃잎": "petal", "바람": "wind",
    }

    def parse(self, prompt):
        intent = {
            "needs_chat": True, "needs_image": False,
            "needs_video": False, "needs_learn": False,
            "tags": [], "colors": [], "expressions": [], "motions": [],
            "raw": prompt,
        }
        p = prompt.lower()
        for kw in ["그려", "만들어", "생성", "이미지", "캐릭터", "그림",
                   "draw", "image", "generate", "character", "make"]:
            if kw in p:
                intent["needs_image"] = True
        for kw in ["영상", "동영상", "움직", "모션", "애니",
                   "video", "motion", "animate", "movie"]:
            if kw in p:
                intent["needs_video"] = True
                intent["needs_image"] = True
        for kw in ["학습", "업그레이드", "개선", "upgrade", "learn", "improve"]:
            if kw in p:
                intent["needs_learn"] = True
        for kw, tag in self.COLOR_MAP.items():
            if kw in p:
                intent["colors"].append(tag)
                intent["tags"].append(tag)
        for kw, tag in self.EXPR_MAP.items():
            if kw in p:
                intent["expressions"].append(tag)
                intent["tags"].append(tag)
        for kw, tag in self.MOTION_MAP.items():
            if kw in p:
                intent["motions"].append(tag)
                intent["tags"].append(tag)
        if not intent["tags"]:
            intent["tags"] = ["default"]
        return intent

    def search_memory(self, memory, intent):
        hits = memory.search(intent["tags"])
        if hits:
            best_match = hits[0][0]
            total_tags = len(intent["tags"])
            ratio = best_match / total_tags if total_tags > 0 else 0
            return {
                "hit": ratio > 0.3,
                "ratio": ratio,
                "cells": hits[:8],
                "strategy": "recombine" if ratio > 0.5 else "sculpt_hybrid",
            }
        return {"hit": False, "ratio": 0, "cells": [], "strategy": "sculpt_new"}


# ────────────────────────────────────────────────────────────
# 3. SculptGenerator  (deterministic, no random noise)
# ────────────────────────────────────────────────────────────
class SculptGenerator:
    COLOR_BGR = {
        "red":   (50, 50, 210),  "blue":   (210, 70, 50),
        "green": (50, 170, 50),  "yellow": (40, 210, 210),
        "white": (235, 235, 235), "black":  (25, 25, 25),
        "pink":  (175, 145, 215), "orange": (40, 130, 220),
        "skin":  (190, 210, 230), "hair_dark": (45, 50, 65),
        "hair_blonde": (120, 195, 220),
    }

    def __init__(self, memory):
        self.memory = memory
        self.params = {name: {
            "edge_strength": 0.35,
            "blend_sharpness": 0.8,
            "boundary_fade": 8,
        } for name in ROW_BLOCKS}

    def generate(self, intent, search_result):
        canvas = np.zeros((H, W, 3), np.uint8)
        strategy = search_result["strategy"]

        for name, (start, end) in ROW_BLOCKS.items():
            y0, y1 = int(H * start), int(H * end)
            bh = y1 - y0

            if strategy == "recombine":
                block = self._recombine(name, bh, intent, search_result["cells"])
            elif strategy == "sculpt_hybrid":
                block = self._sculpt_hybrid(name, bh, intent, search_result["cells"])
            else:
                block = self._sculpt_new(name, bh, intent)

            block = self._sculpt_color(block, intent, name)
            if name == "face" and intent["expressions"]:
                block = self._sculpt_expression(block, intent["expressions"][0])

            canvas[y0:y1] = np.clip(block, 0, 255).astype(np.uint8)

        return self._fade_boundaries(canvas)

    def generate_variation(self, base_image, intent, variation_idx=0):
        varied = base_image.copy()
        hsv = cv2.cvtColor(varied, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Hue shifts expressed as tick-256 indices (256/360 ≈ 0.711
        # ticks/deg). Original spec: ±5° → ±3 ticks, ±3° → ±2 ticks,
        # +8° → +6 ticks, -2° → -1 tick.
        hue_tick_shifts = [2, -2, 3, -3, 6, -1]
        sat_factors = [1.05, 0.95, 1.10, 0.90, 1.03, 0.97]
        bright_factors = [1.02, 0.98, 1.04, 0.96, 1.01, 0.99]

        # Convert hue from cv2's 0..179 (180-step) space into
        # tick-256 (256-step), apply the shift in tick space, convert
        # back. Mirrors how the C engine tracks angles.
        hue_180 = hsv[:, :, 0].astype(np.int32)
        hue_tick = ((hue_180 * 256) // 180) & 0xFF
        hue_tick = (hue_tick + hue_tick_shifts[variation_idx
                                               % len(hue_tick_shifts)]) & 0xFF
        hsv[:, :, 0] = ((hue_tick * 180) // 256).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factors[variation_idx % len(sat_factors)], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * bright_factors[variation_idx % len(bright_factors)], 0, 255)

        varied = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if variation_idx % 3 == 0:
            kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
            varied = cv2.filter2D(varied, -1, kernel)
            varied = np.clip(varied, 0, 255).astype(np.uint8)
        return varied

    def _block_from_amp_phase(self, cells, bh):
        """Row-partition multiple cells' (row × channel × NF) rfft
        spectrograms back into a (bh, W, 3) float32 block.

        Each cell's `amp` / `phase` are contiguous float32 ndarrays of
        shape `(cell_bh, 3, NF)` (see tools/sss_ingest._block_fft).
        Each output row picks the cell that owns its region; rows of a
        cell trained at a different block height map by nearest-
        neighbour. Returns None if no cell carries spectrograms."""
        with_spec = [c for c in cells
                     if c.get("amp") is not None and c.get("phase") is not None]
        if not with_spec:
            return None
        block = np.zeros((bh, W, 3), np.float32)
        n_cells = len(with_spec)
        for r in range(bh):
            region = (r * n_cells) // max(1, bh)
            if region >= n_cells:
                region = n_cells - 1
            cell = with_spec[region]
            amps = np.asarray(cell["amp"])
            phases = np.asarray(cell["phase"])
            if amps.ndim != 3 or phases.shape != amps.shape:
                continue
            cell_bh = amps.shape[0]
            sr = min(int(r * cell_bh / max(1, bh)), cell_bh - 1)
            for c in range(3):
                spec = amps[sr, c] * np.exp(1j * phases[sr, c])
                row_recon = np.fft.irfft(spec, n=W)
                block[r, :, c] = np.clip(row_recon * 255.0, 0, 255)
        return block

    def _recombine(self, block_name, bh, intent, hits):
        block = np.zeros((bh, W, 3), np.float32)
        block_cells = [(s, c) for s, c, b in hits if b == block_name]

        # When any matching cell carries an FFT spectrogram (the new
        # labeled-ingest path), prefer that reconstruction over the
        # old colour/canny blend — the spectrogram is the actual
        # trained pattern.
        spec_block = self._block_from_amp_phase(
            [c for _, c in block_cells], bh)
        if spec_block is not None:
            return np.clip(spec_block, 0, 255)

        if len(block_cells) >= 2:
            _, struct_cell = max(block_cells, key=lambda x: x[1]["quality"])
            _, color_cell = max(block_cells, key=lambda x: x[0])
            color = np.array(color_cell["color"], np.float32)
            block[:] = color
            if struct_cell["canny"] is not None:
                edge = struct_cell["canny"]
                if edge.shape[0] != bh:
                    edge = cv2.resize(edge, (W, bh))
                edge3 = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR).astype(np.float32)
                es = self.params[block_name]["edge_strength"]
                # Tick scale: edge3 * 0.8 → (edge3 * 204) >> 8.
                edge_204 = ((edge3.astype(np.uint16) * 204) >> 8).astype(np.float32)
                block = block * (1 - es) + (block + edge_204) * es
        elif block_cells:
            _, cell = block_cells[0]
            color = np.array(cell["color"], np.float32)
            block[:] = color
            if cell["canny"] is not None:
                edge = cell["canny"]
                if edge.shape[0] != bh:
                    edge = cv2.resize(edge, (W, bh))
                edge3 = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR).astype(np.float32)
                block += edge3 * self.params[block_name]["edge_strength"]
        else:
            block = self._sculpt_new(block_name, bh, intent)
        return np.clip(block, 0, 255)

    def _sculpt_hybrid(self, block_name, bh, intent, hits):
        cell_block = self._recombine(block_name, bh, intent, hits)
        new_block = self._sculpt_new(block_name, bh, intent)
        block_cells = [(s, c) for s, c, b in hits if b == block_name]
        if block_cells:
            best_q = max(c["quality"] for _, c in block_cells)
            cell_weight = min(0.85, 0.4 + best_q)
        else:
            cell_weight = 0.3
        return np.clip(cell_block * cell_weight + new_block * (1 - cell_weight), 0, 255)

    def _sculpt_new(self, block_name, bh, intent):
        # Prefer spectrogram reconstruction when memory has any cells
        # for this block that carry trained amp/phase. Falls back to
        # the colour-blend init when none do (legacy seed cells, text
        # cells, video frames).
        all_block_cells = self.memory.cells.get(block_name, [])
        if all_block_cells:
            best = self.memory.get_best(block_name, 4)
            spec_block = self._block_from_amp_phase(best, bh)
            if spec_block is not None:
                return np.clip(spec_block, 0, 255)

        block = np.zeros((bh, W, 3), np.float32)
        base_colors = {
            "hair":  self.COLOR_BGR.get("hair_dark", (50, 55, 70)),
            "face":  self.COLOR_BGR.get("skin", (190, 210, 230)),
            "upper": (180, 190, 200),
            "lower": (160, 170, 180),
            "bg":    (200, 210, 190),
        }
        base = np.array(base_colors.get(block_name, (150, 150, 150)), np.float32)
        block[:] = base

        best = self.memory.get_best(block_name, 3)
        if best:
            weighted_color = np.zeros(3, np.float32)
            total_w = 0.0
            for cell in best:
                c = np.array(cell["color"], np.float32)
                q = cell["quality"]
                weighted_color += c * q
                total_w += q
            if total_w > 0:
                weighted_color /= total_w
                # Tick blend: base*0.6 + weighted*0.4
                #          → (base*153 + weighted*102) >> 8.
                base_u  = np.clip(base, 0, 255).astype(np.uint16)
                wcol_u  = np.clip(weighted_color, 0, 255).astype(np.uint16)
                block[:] = ((base_u * 153 + wcol_u * 102) >> 8
                            ).astype(np.float32)
            top_cell = best[0]
            if top_cell["canny"] is not None:
                edge = top_cell["canny"]
                if edge.shape[0] != bh:
                    edge = cv2.resize(edge, (W, bh))
                edge3 = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR).astype(np.float32)
                es = self.params[block_name]["edge_strength"]
                block = block + edge3 * es
        return np.clip(block, 0, 255)

    def _sculpt_color(self, block, intent, block_name):
        if not intent["colors"]:
            return block
        block = block.astype(np.float32)
        for c in intent["colors"]:
            if c in self.COLOR_BGR:
                tint = np.array(self.COLOR_BGR[c], np.float32)
                weights = {"hair": 0.5, "face": 0.3, "upper": 0.45,
                           "lower": 0.35, "bg": 0.2}
                w = weights.get(block_name, 0.3)
                block = block * (1 - w) + tint * w
        return np.clip(block, 0, 255)

    def _sculpt_expression(self, block, expr):
        bh, bw = block.shape[:2]
        block = np.clip(block, 0, 255).astype(np.uint8)
        cy, cx = bh // 2, bw // 2

        eye_params = {
            "smile":    {"w": 8, "h": 5,  "pupil": 3, "highlight": True,  "squint": True},
            "sad":      {"w": 8, "h": 7,  "pupil": 3, "highlight": False, "squint": False},
            "surprise": {"w": 9, "h": 11, "pupil": 4, "highlight": True,  "squint": False},
            "blink":    {"w": 8, "h": 1,  "pupil": 0, "highlight": False, "squint": True},
            "talk":     {"w": 8, "h": 6,  "pupil": 3, "highlight": True,  "squint": False},
            "neutral":  {"w": 8, "h": 6,  "pupil": 3, "highlight": False, "squint": False},
        }
        ep = eye_params.get(expr, eye_params["neutral"])
        for ex in [cx - 25, cx + 25]:
            ey = cy - 15
            if ep["h"] <= 1:
                cv2.line(block, (ex - 8, ey), (ex + 8, ey), (40, 40, 50), 2)
            else:
                cv2.ellipse(block, (ex, ey), (ep["w"], ep["h"]), 0, 0, 360, (255, 255, 255), -1)
                if ep["pupil"] > 0:
                    cv2.circle(block, (ex, ey), ep["pupil"], (35, 35, 45), -1)
                if ep["highlight"]:
                    cv2.circle(block, (ex - 1, ey - 2), 1, (255, 255, 255), -1)
                if ep["squint"]:
                    cv2.ellipse(block, (ex, ey - ep["h"] + 1),
                                (ep["w"], 2), 0, 0, 360,
                                block[max(0, ey - 20), ex].tolist(), -1)

        brow_params = {"smile": 2, "sad": -3, "surprise": 5,
                       "talk": 1, "neutral": 0, "blink": 0}
        brow_lift = brow_params.get(expr, 0)
        for bx in [cx - 25, cx + 25]:
            by = cy - 28 - brow_lift
            cv2.line(block,
                     (bx - 10, by + brow_lift // 2),
                     (bx + 10, by - brow_lift // 2),
                     (60, 50, 45), 2)

        my = cy + 22
        if expr == "smile":
            cv2.ellipse(block, (cx, my), (16, 9), 0, 10, 170, (90, 70, 120), 2)
            cv2.ellipse(block, (cx, my + 2), (12, 5), 0, 0, 180, (130, 100, 140), -1)
        elif expr == "sad":
            cv2.ellipse(block, (cx, my + 6), (14, 7), 0, 190, 350, (90, 70, 120), 2)
        elif expr == "surprise":
            cv2.ellipse(block, (cx, my), (10, 12), 0, 0, 360, (80, 60, 110), -1)
        elif expr == "talk":
            cv2.ellipse(block, (cx, my), (11, 7), 0, 0, 360, (90, 70, 120), -1)
        else:
            cv2.line(block, (cx - 12, my), (cx + 12, my), (90, 70, 120), 2)

        if expr in ("smile", "talk"):
            for bx in [cx - 30, cx + 30]:
                overlay = block.copy()
                cv2.circle(overlay, (bx, cy + 8), 10, (200, 180, 220), -1)
                cv2.addWeighted(overlay, 0.15, block, 0.85, 0, block)

        for i in range(0, bw, 3):
            length = 8 + (i % 7) * 2
            cv2.line(block, (i, 0),
                     (i + (i % 5) - 2, min(length, bh - 1)),
                     (50, 55, 70), 1)

        return block.astype(np.float32)

    def _fade_boundaries(self, canvas):
        result = canvas.astype(np.float32)
        block_list = list(ROW_BLOCKS.items())
        for i in range(len(block_list) - 1):
            _, (_, end1) = block_list[i]
            y_boundary = int(H * end1)
            fade = self.params[block_list[i][0]]["boundary_fade"]
            for dy in range(1, fade + 1):
                t = dy / fade
                y_above = max(0, y_boundary - dy)
                y_below = min(H - 1, y_boundary + dy)
                if y_above < H and y_below < H:
                    result[y_boundary] = (result[y_above] * (1 - t * 0.3)
                                          + result[y_below] * (t * 0.3))
        return np.clip(result, 0, 255).astype(np.uint8)

    def update_params(self, block_name, score_data):
        bp = self.params[block_name]
        block_score = score_data.get(block_name, 0)
        if block_score < 0.3:
            bp["edge_strength"] = min(0.7, bp["edge_strength"] + 0.03)
        elif block_score > 0.6:
            bp["blend_sharpness"] = min(1.0, bp["blend_sharpness"] + 0.02)


# ────────────────────────────────────────────────────────────
# 4. MotionEngine
# ────────────────────────────────────────────────────────────
class MotionEngine:
    def __init__(self, memory=None):
        # Optional. When set, generate_frames() prefers pose-cell
        # dirty_zones / motion rules over the deterministic fallback.
        self.memory = memory

    def _find_pose_for_intent(self, intent):
        if self.memory is None or not hasattr(self.memory, "find_pose"):
            return None
        tags = list(intent.get("tags") or [])
        if not tags:
            return None
        try:
            entries = self.memory.find_pose(tags, n=1)
        except Exception:
            return None
        if not entries:
            return None
        data = entries[0].get("data") or {}
        if not isinstance(data, dict):
            return None
        return data

    @staticmethod
    def _zone_rows(dirty, name, fallback):
        """Clip a dirty-zone span into a [r0, r1) tuple usable for
        slicing. `fallback` is `(r0, r1)`; used when the zone is
        missing or degenerate."""
        if not isinstance(dirty, dict):
            return fallback
        span = dirty.get(name)
        if not span or len(span) < 2:
            return fallback
        a, b = int(span[0]), int(span[1])
        if b < a:
            a, b = b, a
        a = max(0, min(H - 1, a))
        b = max(0, min(H, b + 1))
        if b <= a:
            return fallback
        return (a, b)

    def generate_frames(self, base_image, intent, n_frames=24):
        # Pose-cell hints take priority for row band placement when a
        # matching entry exists; falls back to ROW_BLOCKS percentages.
        pose_data = self._find_pose_for_intent(intent)
        dirty = pose_data.get("dirty_zones") if pose_data else None
        pose_motions = pose_data.get("motions") if pose_data else []

        frames = []
        has_petal = any(m in intent.get("motions", [])
                        for m in ["petal", "fall", "blow"])
        has_wind = any(m in intent.get("motions", [])
                       for m in ["wind", "blow", "wave"])
        has_pose_motions = isinstance(pose_motions, list) and bool(pose_motions)

        petals = []
        if has_petal:
            for i in range(12):
                petals.append({
                    "x": 20 + (i * 37) % (W - 40),
                    "y": -(i * 13) % 30,
                    "vy": 1.5 + (i % 4) * 0.8,
                    "vx_base": ((i % 5) - 2) * 0.4,
                    "size": 4 + (i % 5),
                    "hue": 165 + (i % 6) * 3,
                    "phase": i * 0.7,
                })

        # Resolve row bands. When a matching pose cell exists we use
        # its detected dirty zones so wind/face/arm motions follow the
        # actual silhouette instead of the fixed 15%/40% percentages.
        hair_band  = self._zone_rows(dirty, "hair",  (0,            int(H * 0.15)))
        face_band  = self._zone_rows(dirty, "face",  (int(H * 0.15), int(H * 0.40)))
        arm_band   = self._zone_rows(dirty, "arms",  (int(H * 0.40), int(H * 0.65)))
        cloth_band = self._zone_rows(dirty, "cloth", (int(H * 0.65), int(H * 0.88)))

        for fi in range(n_frames):
            frame = base_image.copy()

            if has_wind:
                hair_lo, hair_hi = hair_band
                for row in range(hair_lo, hair_hi):
                    # tick sin → signed [-128,127]; scale to ±3 px shift.
                    sin_s = tick_sin_signed(
                        tick_phase_idx(row * 0.06 + fi * 0.18))
                    shift = (sin_s * 3) // 128
                    frame[row] = np.roll(frame[row], shift, axis=0)

            if intent.get("expressions"):
                fy0, fy1 = face_band
                # tick sin scaled to ±0.02 brightness factor.
                sin_s = tick_sin_signed(tick_phase_idx(fi * 0.25))
                brightness = 1.0 + (sin_s * 0.02) / 128.0
                frame[fy0:fy1] = np.clip(
                    frame[fy0:fy1].astype(np.float32) * brightness,
                    0, 255).astype(np.uint8)

            # Pose-cell motion rules: arm sway and cloth sway both run
            # as bounded horizontal row rolls scaled by the cell's
            # `amplitude`. They only fire when a pose cell matched the
            # prompt — the deterministic fallback is just no extra
            # motion (keeps every existing test deterministic).
            if has_pose_motions:
                for m in pose_motions:
                    name = m.get("name", "")
                    amp = float(m.get("amplitude", 0.0))
                    freq = float(m.get("frequency", 0.12))
                    if amp <= 0.0:
                        continue
                    if name == "arm_sway":
                        rlo, rhi = arm_band
                    elif name == "cloth_sway":
                        rlo, rhi = cloth_band
                    else:
                        continue
                    if rhi <= rlo:
                        continue
                    sin_s = tick_sin_signed(tick_phase_idx(fi * freq))
                    shift = int(round(amp * sin_s / 128.0))
                    if shift == 0:
                        continue
                    frame[rlo:rhi] = np.roll(
                        frame[rlo:rhi], shift, axis=1)

            if has_petal:
                for p in petals:
                    p["y"] += p["vy"]
                    sin_s = tick_sin_signed(
                        tick_phase_idx(p["phase"] + fi * 0.12))
                    p["x"] += p["vx_base"] + (sin_s * 0.6) / 128.0
                    if p["y"] >= H:
                        p["y"] -= (H + 20)
                    cy = int(np.clip(p["y"], 0, H - 1))
                    cx = int(np.clip(p["x"], 0, W - 1))
                    s = p["size"]
                    y0m, y1m = max(0, cy - s), min(H, cy + s)
                    x0m, x1m = max(0, cx - s), min(W, cx + s)
                    if y1m > y0m and x1m > x0m:
                        roi = frame[y0m:y1m, x0m:x1m].astype(np.float32)
                        petal_color = np.array(cv2.cvtColor(
                            np.uint8([[[p["hue"], 120, 230]]]),
                            cv2.COLOR_HSV2BGR)[0, 0], np.float32)
                        tint = np.full_like(roi, petal_color)
                        frame[y0m:y1m, x0m:x1m] = np.clip(
                            roi * 0.55 + tint * 0.45, 0, 255).astype(np.uint8)

            frames.append(frame)
        return frames


# ────────────────────────────────────────────────────────────
# 5. Evaluator
# ────────────────────────────────────────────────────────────
class Evaluator:
    COLOR_RANGES = {
        "red":    ((0,   50, 50), (10,  255, 255)),
        "blue":   ((100, 50, 50), (130, 255, 255)),
        "green":  ((35,  50, 50), (85,  255, 255)),
        "yellow": ((20,  50, 50), (35,  255, 255)),
        "pink":   ((140, 30, 50), (175, 255, 255)),
        "orange": ((10,  50, 50), (25,  255, 255)),
    }

    def evaluate(self, image, intent, frames=None):
        scores = {}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        scores["edge"] = round(float(np.mean(edges > 0)), 4)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        scores["color_var"] = round(
            float(np.std(hsv[:, :, 0]) / 180 + np.mean(hsv[:, :, 1]) / 255 * 0.5), 4)
        scores["non_trivial"] = round(
            min(1.0, float(np.std(image.astype(float))) / 50), 4)

        means = [np.mean(image[int(H * s):int(H * e)])
                 for _, (s, e) in ROW_BLOCKS.items()]
        diffs = [abs(means[i] - means[i + 1]) for i in range(len(means) - 1)]
        scores["coherence"] = round(max(0, 1.0 - np.mean(diffs) / 100), 4)

        quality = (scores["edge"] * 0.25
                   + scores["color_var"] * 0.15
                   + scores["non_trivial"] * 0.25
                   + scores["coherence"] * 0.2)

        prompt_score = 0.0
        checks = 0
        if intent["colors"]:
            for c in intent["colors"]:
                checks += 1
                if c in self.COLOR_RANGES:
                    lo, hi = self.COLOR_RANGES[c]
                    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
                    prompt_score += min(1.0, np.mean(mask > 0) * 8)
                elif c == "white":
                    prompt_score += float(np.mean(hsv[:, :, 2]) / 255)
                elif c == "black":
                    prompt_score += float(1 - np.mean(hsv[:, :, 2]) / 255)
        if intent["expressions"]:
            checks += 1
            fy0, fy1 = int(H * 0.15), int(H * 0.40)
            face_edge = np.mean(cv2.Canny(
                cv2.cvtColor(image[fy0:fy1], cv2.COLOR_BGR2GRAY),
                50, 150) > 0)
            prompt_score += min(1.0, face_edge * 4)
        if frames and len(frames) > 1 and intent["motions"]:
            checks += 1
            ds = [np.mean(np.abs(frames[i].astype(float)
                                 - frames[i - 1].astype(float)))
                  for i in range(1, min(5, len(frames)))]
            prompt_score += min(1.0, np.mean(ds) / 12)

        prompt_match = prompt_score / checks if checks > 0 else 0.5
        scores["prompt_match"] = round(prompt_match, 4)
        scores["quality"] = round(quality, 4)
        scores["final"] = round(quality * 0.4 + prompt_match * 0.6, 4)

        scores["blocks"] = {}
        for name, (s, e) in ROW_BLOCKS.items():
            block = image[int(H * s):int(H * e)]
            be = np.mean(cv2.Canny(cv2.cvtColor(block, cv2.COLOR_BGR2GRAY),
                                   50, 150) > 0)
            bv = np.std(block.astype(float)) / 80
            scores["blocks"][name] = round(min(1.0, be * 0.4 + bv * 0.6), 4)

        scores["weak_points"] = []
        if scores["edge"] < 0.03:
            scores["weak_points"].append("edge_low")
        if scores["coherence"] < 0.5:
            scores["weak_points"].append("discontinuity")
        if prompt_match < 0.3:
            scores["weak_points"].append("prompt_mismatch")
        return scores


# ────────────────────────────────────────────────────────────
# 6. Pipeline orchestrator
# ────────────────────────────────────────────────────────────
class SSSPipeline:
    """Stateful orchestrator. Holds memory + planner + generator + evaluator
    so the same instance can be re-used across many prompts."""

    def __init__(self, db_path: Optional[str] = None):
        os.makedirs(BASE_DIR, exist_ok=True)
        self.memory = CEMemory(db_path)
        self.planner = Planner()
        self.generator = SculptGenerator(self.memory)
        # MotionEngine reads memory.pose_cells for prompt-matching
        # dirty-row / motion hints when present.
        self.motion = MotionEngine(memory=self.memory)
        self.evaluator = Evaluator()
        # SSS_MODEL_PATH points at a .sss file produced by
        # scripts/sss_train.py. When present, run() routes through the
        # C sss_rowvae generator first; otherwise it stays on the
        # Python sculpt path.
        self._sss_model_path = os.environ.get("SSS_MODEL_PATH", "")
        self.run_count = 0

    # ── seeding ─────────────────────────────────────────────
    def seed_foundation(self):
        shapes = [
            ("circle", lambda img, r: cv2.circle(img, (128, 128), r, (220, 220, 230), -1)),
            ("rect",   lambda img, r: cv2.rectangle(img,
                                                    (128 - r, 128 - r // 2),
                                                    (128 + r, 128 + r // 2),
                                                    (220, 220, 230), -1)),
            ("tri",    lambda img, r: cv2.fillPoly(
                img,
                [np.array([[128, 128 - r], [128 - r, 128 + r], [128 + r, 128 + r]])],
                (220, 220, 230))),
        ]
        for sname, fn in shapes:
            for size, r in [("s", 25), ("m", 50), ("l", 80)]:
                img = np.zeros((H, W, 3), np.uint8)
                fn(img, r)
                for bname, (s, e) in ROW_BLOCKS.items():
                    y0, y1 = int(H * s), int(H * e)
                    block = img[y0:y1]
                    canny = cv2.Canny(cv2.cvtColor(block, cv2.COLOR_BGR2GRAY), 50, 150)
                    color = np.mean(block, axis=(0, 1))
                    self.memory.add(bname, canny, color, 0.40,
                                    [sname, size], f"found_{sname}_{size}")

        for cname, bgr in [("red", (50, 50, 210)), ("blue", (210, 70, 50)),
                           ("green", (50, 170, 50)), ("pink", (175, 145, 215)),
                           ("skin", (190, 210, 230)), ("yellow", (40, 210, 210))]:
            for bname in ROW_BLOCKS:
                self.memory.add(bname, None, np.array(bgr, float), 0.42,
                                [cname], f"found_{cname}")

        for expr in ["smile", "sad", "neutral", "surprise", "blink", "talk"]:
            self.memory.add("face", None, np.array([190, 210, 230], float),
                            0.45, [expr, "face_expr"], f"found_{expr}")
            for bname in ROW_BLOCKS:
                if bname != "face":
                    self.memory.add(bname, None, np.array([180, 190, 200], float),
                                    0.38, [expr], f"found_{expr}")

    def _try_c_generate(self, prompt):
        """Run the C sss_rowvae generator at SSS_W×SSS_W. Returns a BGR
        uint8 ndarray on success, None on any failure (no model path,
        missing file, missing C symbol, runtime error). Resampled with
        nearest-neighbour to (H, W) so the rest of the pipeline can
        compose with it as if it came from SculptGenerator."""
        path = self._sss_model_path
        if not path or not os.path.exists(path):
            return None
        try:
            from tools.sss_memory import sss_generate as _c_generate
            img = _c_generate(path, prompt, seed=int(self.run_count) or 1,
                              detail=1.0, steps=24, size=W)
            if img is None or img.shape != (H, W, 3):
                return None
            return img
        except Exception:
            return None

    # ── single run ─────────────────────────────────────────
    def run(self, prompt):
        self.run_count += 1
        result = {"run": self.run_count, "prompt": prompt}

        intent = self.planner.parse(prompt)
        search = self.planner.search_memory(self.memory, intent)
        result["search"] = {
            "hit": search["hit"],
            "ratio": round(search["ratio"], 3),
            "strategy": search["strategy"],
            "n_hits": len(search["cells"]),
        }

        image = frames = None
        if intent["needs_image"] or intent["needs_video"]:
            # Prefer the C sss_rowvae generator when a trained .sss
            # model is available (env var or default path). Falls back
            # to the Python sculpt path for any failure: missing model
            # file, missing C symbol on an older libsss_pybridge.so, or
            # any runtime error from sss_pybridge_generate.
            image = self._try_c_generate(prompt)
            if image is None:
                image = self.generator.generate(intent, search)
            candidates = [image]
            for vi in range(3):
                candidates.append(self.generator.generate_variation(image, intent, vi))
            best_score = -1.0
            for cand in candidates:
                sc = self.evaluator.evaluate(cand, intent)
                if sc["final"] > best_score:
                    best_score = sc["final"]
                    image = cand

        if intent["needs_video"] and image is not None:
            frames = self.motion.generate_frames(image, intent, n_frames=24)
            result["frames"] = len(frames)

        score = {"final": 0, "quality": 0, "prompt_match": 0}
        if image is not None:
            score = self.evaluator.evaluate(image, intent, frames)
            result["score"] = score

            threshold = 0.20 + self.memory.generation * 0.008
            if score["final"] >= threshold:
                tags = list(intent["tags"])
                for bname, (s, e) in ROW_BLOCKS.items():
                    y0, y1 = int(H * s), int(H * e)
                    block = image[y0:y1]
                    canny = cv2.Canny(cv2.cvtColor(block, cv2.COLOR_BGR2GRAY),
                                      50, 150)
                    color = np.mean(block, axis=(0, 1))
                    bq = score["blocks"].get(bname, score["final"])
                    self.memory.add(bname, canny, color, bq, tags,
                                    f"run{self.run_count}")
                    self.generator.update_params(bname, score["blocks"])
                self.memory.generation += 1
                result["stored"] = "accepted"
            else:
                self.memory.add_pending(
                    {"prompt": prompt, "score": score["final"]},
                    score.get("weak_points", []))
                result["stored"] = "pending"

        parts = []
        if intent["needs_image"]:
            parts.append("이미지 생성 완료")
        if intent["needs_video"]:
            parts.append("영상 생성 완료")
        parts.append(
            f"품질={score['quality']:.3f} prompt={score['prompt_match']:.3f} "
            f"최종={score['final']:.3f}")
        parts.append(f"메모리 {self.memory.total_cells()}셀")
        result["chat"] = " | ".join(parts)
        result["memory"] = {
            "cells": self.memory.total_cells(),
            "avg_q": round(self.memory.avg_quality(), 4),
            "ce_storage": self.memory.ce_storage_count(),
        }
        return result, image, frames


# ────────────────────────────────────────────────────────────
# 7. Module-level convenience (for quick scripting)
# ────────────────────────────────────────────────────────────
_pipeline: Optional[SSSPipeline] = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = SSSPipeline()
        _pipeline.seed_foundation()
    return _pipeline


def run_sss_pipeline(prompt):
    """One-call entry point. Lazily seeds foundation on first use."""
    return _get_pipeline().run(prompt)


# ────────────────────────────────────────────────────────────
# 7b. Self-upgrade loop
#     - cycles a fixed prompt set (shape × color × expression)
#     - cells are sampled from memory with quality-weighted random
#       picking, so the same prompt can produce different combos
#     - cells are NEVER deleted; bad results just lower the cell's
#       quality (= selection probability), good results raise it
#     - no pixel-level random noise is ever added
# ────────────────────────────────────────────────────────────
_UPGRADE_COMBOS = [
    ("circle", "red",    "smile"),
    ("circle", "blue",   "sad"),
    ("rect",   "green",  "surprise"),
    ("tri",    "yellow", "neutral"),
    ("circle", "pink",   "talk"),
    ("rect",   "orange", "smile"),
    ("tri",    "red",    "surprise"),
    ("circle", "blue",   "smile"),
]


def _build_intent(shape, color, expr):
    return {
        "needs_chat": True,
        "needs_image": True,
        "needs_video": False,
        "needs_learn": True,
        "tags": [shape, color, expr],
        "colors": [color],
        "expressions": [expr],
        "motions": [],
        "raw": f"{color} {shape} {expr}",
    }


def _weighted_sample_hits(memory, intent, rng, per_block=3):
    """Pick cells from each block via quality-weighted random sampling.

    Returns the same `(score, cell, block)` triple shape that
    `Planner.search_memory` produces, so the generator can consume it
    unchanged. Tag-matching cells are favoured but unmatched cells can
    still be drawn — that's where combinatorial diversity comes from.
    """
    hits = []
    tags = intent.get("tags", []) or []
    for block_name in ROW_BLOCKS:
        cells = memory.cells.get(block_name, [])
        if not cells:
            continue
        weights = []
        for cell in cells:
            match = sum(1 for t in tags if t in cell.get("tags", []))
            q = max(0.05, float(cell.get("quality", 0.0)))
            weights.append((match + 0.1) * q)
        n_pick = min(per_block, len(cells))
        chosen_idx = set()
        for _ in range(n_pick * 2):
            if len(chosen_idx) >= n_pick:
                break
            i = rng.choices(range(len(cells)), weights=weights, k=1)[0]
            chosen_idx.add(i)
        for i in chosen_idx:
            cell = cells[i]
            match = sum(1 for t in tags if t in cell.get("tags", []))
            hits.append((max(1, match), cell, block_name))
    return hits


def run_upgrade_loop(n_cycles=5, seed=None):
    """Drive `n_cycles` of generate → evaluate → quality-update.

    No noise is ever added to the rendered pixels; diversity only comes
    from picking different memory cells per call. Cells are kept in
    memory across the whole loop — only their `quality` shifts (which
    is also the selection-probability weight).

    `seed=None` (default) reseeds from OS entropy on every call so
    repeated calls against the same memory state still pick different
    cell combos. Pass an explicit int for reproducible runs (tests).

    Returns a list of per-cycle dicts:
        {"cycle", "generated", "accepted", "avg_score",
         "best_score", "total_cells"}
    """
    pipeline = _get_pipeline()
    rng = random.Random(seed)

    cycles_report = []
    for cycle_idx in range(int(n_cycles)):
        scores_this_cycle = []
        accepted = 0
        for shape, color, expr in _UPGRADE_COMBOS:
            intent = _build_intent(shape, color, expr)

            # Quality-weighted random pick — same prompt → different
            # cell combos across calls.
            hits = _weighted_sample_hits(pipeline.memory, intent, rng)
            if hits:
                strategy = "recombine" if len(hits) >= len(ROW_BLOCKS) else "sculpt_hybrid"
            else:
                strategy = "sculpt_new"
            search = {
                "hit": bool(hits),
                "ratio": min(1.0, len(hits) / max(1, len(ROW_BLOCKS) * 2)),
                "cells": hits,
                "strategy": strategy,
            }

            image = pipeline.generator.generate(intent, search)
            score = pipeline.evaluator.evaluate(image, intent)
            final = float(score.get("final", 0.0))
            scores_this_cycle.append(final)

            # Update quality of the cells we drew. update_quality blends
            # 0.7 * old + 0.3 * new, so good results nudge up, bad
            # results nudge down — but the cell stays in memory either
            # way.
            for _match, cell, b in hits:
                cells_in_block = pipeline.memory.cells.get(b, [])
                try:
                    idx = cells_in_block.index(cell)
                except ValueError:
                    continue
                pipeline.memory.update_quality(b, idx, final)

            # Accepted runs add fresh cells (more variety, not noise).
            threshold = 0.20 + pipeline.memory.generation * 0.008
            if final >= threshold:
                tags = list(intent["tags"])
                for bname, (s, e) in ROW_BLOCKS.items():
                    y0, y1 = int(H * s), int(H * e)
                    block = image[y0:y1]
                    canny = cv2.Canny(
                        cv2.cvtColor(block, cv2.COLOR_BGR2GRAY), 50, 150)
                    avg = np.mean(block, axis=(0, 1))
                    bq = score.get("blocks", {}).get(bname, final)
                    pipeline.memory.add(
                        bname, canny, avg, bq, tags,
                        f"upgrade_c{cycle_idx + 1}")
                pipeline.memory.generation += 1
                accepted += 1

        cycles_report.append({
            "cycle": cycle_idx + 1,
            "generated": len(_UPGRADE_COMBOS),
            "accepted": accepted,
            "avg_score": round(
                float(np.mean(scores_this_cycle)) if scores_this_cycle else 0.0,
                4),
            "best_score": round(
                float(np.max(scores_this_cycle)) if scores_this_cycle else 0.0,
                4),
            "total_cells": pipeline.memory.total_cells(),
        })
    return cycles_report


# ────────────────────────────────────────────────────────────
# 8. Demo runner
# ────────────────────────────────────────────────────────────
def run_demo():
    print("=" * 60)
    print("  SSS Unified Pipeline v2 — No Noise")
    print("=" * 60)

    pipeline = SSSPipeline()
    pipeline.seed_foundation()
    print(f"\n  Foundation: {pipeline.memory.total_cells()} cells "
          f"(CEStorage: {pipeline.memory.ce_storage_count()})")

    rounds = [
        ("Round 1 (첫 생성)", [
            "빨간 캐릭터가 웃는 이미지 그려줘",
            "파란 캐릭터가 슬픈 이미지 만들어줘",
            "분홍 캐릭터가 꽃잎 속에서 웃는 영상 만들어줘",
        ]),
        ("Round 2 (기억 재사용)", [
            "빨간 캐릭터가 웃는 이미지 그려줘",
            "파란 캐릭터가 슬픈 이미지 만들어줘",
            "분홍 캐릭터가 꽃잎 속에서 웃는 영상 만들어줘",
        ]),
        ("Round 3 (개선 확인)", [
            "빨간 캐릭터가 웃는 이미지 그려줘",
            "파란 캐릭터가 슬픈 이미지 만들어줘",
            "분홍 캐릭터가 꽃잎 속에서 웃는 영상 만들어줘",
            "초록 캐릭터가 놀란 이미지 생성해줘",
        ]),
    ]

    output_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    tracking = defaultdict(list)

    for rname, prompts in rounds:
        print(f"\n{'━' * 60}")
        print(f"  {rname}")
        print(f"{'━' * 60}")
        for prompt in prompts:
            result, image, frames = pipeline.run(prompt)
            sc = result.get("score", {})
            ms = result.get("search", {})
            print(f"\n  \"{prompt}\"")
            print(f"  strategy={ms.get('strategy', '?')} "
                  f"hit={ms.get('hit', False)}({ms.get('ratio', 0):.2f}) "
                  f"n_hits={ms.get('n_hits', 0)}")
            print(f"  quality={sc.get('quality', 0):.3f} "
                  f"prompt={sc.get('prompt_match', 0):.3f} "
                  f"final={sc.get('final', 0):.3f}")
            print(f"  stored={result.get('stored', '?')} "
                  f"memory={result['memory']['cells']}셀 "
                  f"ce={result['memory']['ce_storage']} "
                  f"avg_q={result['memory']['avg_q']:.3f}")
            if sc.get("weak_points"):
                print(f"  weak: {sc['weak_points']}")

            tracking[prompt].append(sc.get("final", 0))

            if image is not None:
                sss_image_io.imwrite(
                    os.path.join(output_dir,
                                 f"run{result['run']:02d}.png"), image)
            if frames and len(frames) > 1:
                fd = os.path.join(output_dir, f"fr_{result['run']:02d}")
                os.makedirs(fd, exist_ok=True)
                for fi, fr in enumerate(frames):
                    sss_image_io.imwrite(
                        os.path.join(fd, f"f_{fi:03d}.png"), fr)
                vp = os.path.join(output_dir, f"vid_{result['run']:02d}.mp4")
                subprocess.run(
                    ["ffmpeg", "-y", "-framerate", "24",
                     "-i", os.path.join(fd, "f_%03d.png"),
                     "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-crf", "22", "-preset", "fast", vp],
                    capture_output=True)
                for f in os.listdir(fd):
                    os.remove(os.path.join(fd, f))
                os.rmdir(fd)

    print(f"\n{'━' * 60}")
    print("  개선 추적")
    print(f"{'━' * 60}")
    for prompt, scores in tracking.items():
        if len(scores) >= 2:
            delta = scores[-1] - scores[0]
            d = "↑" if delta > 0.003 else ("↓" if delta < -0.003 else "→")
            print(f"\n  \"{prompt[:35]}\"")
            vals = " → ".join(f"{s:.3f}" for s in scores)
            bar_first = "█" * int(scores[0] * 50) + "░" * (50 - int(scores[0] * 50))
            bar_last = "█" * int(scores[-1] * 50) + "░" * (50 - int(scores[-1] * 50))
            print(f"    {vals}  {d} ({delta:+.3f})")
            print(f"    R1: {bar_first}")
            print(f"    R3: {bar_last}")

    print(f"\n{'━' * 60}")
    print(f"  최종: {pipeline.run_count}회 실행, "
          f"{pipeline.memory.total_cells()}셀 "
          f"(CEStorage: {pipeline.memory.ce_storage_count()}), "
          f"세대 {pipeline.memory.generation}")
    print(f"  avg_q={pipeline.memory.avg_quality():.4f}, "
          f"pending={len(pipeline.memory.pending)}건")


if __name__ == "__main__":
    run_demo()
