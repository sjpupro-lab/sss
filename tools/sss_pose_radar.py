"""SSS pose / radar perception — silhouette → joints → motion → dirty zones.

Lightweight wave/radar-style perception that pairs with the existing
labeled-image ingest path. Given an image + label it produces:

    silhouette mask
    row / column density fields
    radar / edge resonance map
    estimated body joints (head, neck, shoulders, elbows, hands,
                           waist, hips, knees, feet)
    motion placement rules (hair wave, blink, arm sway, cloth, bounce)
    dirty-row zones (hair / face / arms / cloth / body)

The result feeds into SSS memory so the engine learns "where it can
move" and "which rows need updating" alongside "what an image looks
like".

Pure numpy + tools.sss_image_io. No cv2 / Pillow / Mediapipe / Torch.

CLI:
    python3 tools/sss_pose_radar.py input.png --label "..." --out debug/

Public API:
    analyze_pose_radar(filepath_or_image, label, *,
                       out_dir=None, filename=None) -> dict
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Union

import numpy as np

# When this file is launched directly (`python3 tools/sss_pose_radar.py
# ...`), `tools` isn't on sys.path yet. Insert the repo root so the
# package-qualified imports below work in both test and CLI mode.
if __package__ in (None, "") and __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

from tools import sss_image_io

W, H = 256, 256


# ────────────────────────────────────────────────────────────
# 1. image loading + 256×256 resize
# ────────────────────────────────────────────────────────────
def _load_image(filepath: str) -> np.ndarray:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".png":
        img = sss_image_io.read_png(filepath)
    elif ext == ".ppm":
        img = sss_image_io.read_ppm(filepath)
    else:
        # delegate to ingest's ffmpeg path for everything else
        from tools.sss_ingest import _read_image_any
        img = _read_image_any(filepath)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    return img


def _resize_nn(img, target_h=H, target_w=W):
    """Same linspace-endpoint NN resize the ingest pipeline uses, so a
    pose run on the same image lines up with its row blocks."""
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    yi = np.linspace(0, max(0, h - 1), target_h).round().astype(np.int64)
    xi = np.linspace(0, max(0, w - 1), target_w).round().astype(np.int64)
    yi = np.clip(yi, 0, max(0, h - 1))
    xi = np.clip(xi, 0, max(0, w - 1))
    if img.ndim == 2:
        return img[yi[:, None], xi[None, :]]
    return img[yi[:, None], xi[None, :], :]


# ────────────────────────────────────────────────────────────
# 2. background removal + largest-component extraction
# ────────────────────────────────────────────────────────────
def _bg_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Foreground mask (uint8 {0, 255}). Removes a uniform-ish border
    colour, very-bright frames, and the alternating-grey checker that
    PNG viewers render for transparent pixels."""
    h, w = img_bgr.shape[:2]
    border = np.concatenate([
        img_bgr[:4].reshape(-1, 3),
        img_bgr[-4:].reshape(-1, 3),
        img_bgr[4:-4, :4].reshape(-1, 3),
        img_bgr[4:-4, -4:].reshape(-1, 3),
    ], axis=0).astype(np.int32)
    bg_color = np.median(border, axis=0).astype(np.int32)

    diff = np.abs(img_bgr.astype(np.int32)
                  - bg_color[None, None, :]).max(axis=-1)
    mask = (diff > 35).astype(np.uint8) * 255

    # Drop near-white pixels regardless of bg_color — common when a
    # white frame surrounds an asset on a non-white border colour.
    very_bright = np.all(img_bgr >= 232, axis=-1)
    mask[very_bright] = 0

    # PNG checker: alternating mid-grey tones with very low chroma. If
    # the image carries a noticeable grey-low-chroma population, drop
    # pixels that match that population.
    rgb = img_bgr.astype(np.int32)
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    grey = chroma < 6
    if int(grey.sum()) > int(h * w * 0.10):
        grey_tone = img_bgr[grey].mean(axis=0)
        t_diff = np.abs(img_bgr.astype(np.int32)
                        - grey_tone.astype(np.int32)).max(axis=-1)
        mask[(t_diff < 25) & grey] = 0
    return mask


def _erode_cross(m: np.ndarray) -> np.ndarray:
    """3-px cross-shaped erosion of a {0,1} uint8 mask."""
    out = m.copy()
    out[1:]    &= m[:-1]
    out[:-1]   &= m[1:]
    out[:, 1:]  &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    return out


def _dilate_cross(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:]    |= m[:-1]
    out[:-1]   |= m[1:]
    out[:, 1:]  |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def _morph_clean(mask: np.ndarray, k: int = 2) -> np.ndarray:
    """Open then close with cross-shaped 3-px kernel. `k` controls
    the strength of each pass (default 2)."""
    m = (mask > 0).astype(np.uint8)
    for _ in range(k):
        m = _erode_cross(m)
    for _ in range(2 * k):
        m = _dilate_cross(m)
    for _ in range(k):
        m = _erode_cross(m)
    return (m * 255).astype(np.uint8)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Largest 4-connected component as uint8 {0, 255}.

    Iterative dilation from the centroid seed within the mask. Each
    iteration is one numpy boolean pass; the loop terminates when the
    region stops growing. On a 256×256 silhouette this runs in well
    under 100 iterations even for diagonal extents.
    """
    bm = (mask > 0)
    if not bm.any():
        return mask
    ys, xs = np.where(bm)
    cy = int(np.median(ys))
    cx = int(np.median(xs))
    if not bm[cy, cx]:
        # nearest-on-mask fallback: pick the closest True pixel.
        d = (ys - cy) ** 2 + (xs - cx) ** 2
        i = int(np.argmin(d))
        cy, cx = int(ys[i]), int(xs[i])

    region = np.zeros_like(bm)
    region[cy, cx] = True
    while True:
        prev_count = int(region.sum())
        nxt = region.copy()
        nxt[1:]    |= region[:-1]
        nxt[:-1]   |= region[1:]
        nxt[:, 1:]  |= region[:, :-1]
        nxt[:, :-1] |= region[:, 1:]
        region = nxt & bm
        if int(region.sum()) == prev_count:
            break
    return (region.astype(np.uint8)) * 255


# ────────────────────────────────────────────────────────────
# 3. radar / wave perception fields
# ────────────────────────────────────────────────────────────
def _radar_fields(img_bgr, mask):
    """Return (row_density, col_density, edge_map, fft_response).

    row_density[y]   ∈ [0, 1]   — fraction of mask pixels at row y
    col_density[x]   ∈ [0, 1]   — fraction of mask pixels at col x
    edge_map         uint8       — Sobel magnitude inside the mask
    fft_response                 — per-row rfft amplitude (low-bin slice)
    """
    bm = (mask > 0)
    h, w = bm.shape
    row_density = bm.sum(axis=1).astype(np.float32) / max(1, w)
    col_density = bm.sum(axis=0).astype(np.float32) / max(1, h)

    a = img_bgr.astype(np.uint16)
    gray = ((a[..., 0] * 29 + a[..., 1] * 150 + a[..., 2] * 77) >> 8
            ).astype(np.int32)
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.clip(np.abs(gx) + np.abs(gy), 0, 255).astype(np.uint8)
    edge_map = mag * bm.astype(np.uint8)

    rows = (img_bgr.astype(np.float32).mean(axis=-1) / 255.0)
    rows = rows * bm.astype(np.float32)
    spec = np.fft.rfft(rows, axis=1)
    amp = np.abs(spec).astype(np.float32)
    n_keep = min(16, amp.shape[1])
    fft_response = np.ascontiguousarray(amp[:, :n_keep])
    return row_density, col_density, edge_map, fft_response


# ────────────────────────────────────────────────────────────
# 4. silhouette → joints (deterministic, explainable)
# ────────────────────────────────────────────────────────────
def _row_first_last(bm: np.ndarray):
    """Per-row leftmost/rightmost mask column. Vectorised via argmax on
    bool: argmax returns the first True column; flipping handles the
    last True. Rows with no True get (w, -1) so callers can detect."""
    h, w = bm.shape
    rows_have = bm.any(axis=1)
    left = np.argmax(bm, axis=1).astype(np.int32)
    left = np.where(rows_have, left, np.int32(w))
    right = (w - 1 - np.argmax(bm[:, ::-1], axis=1)).astype(np.int32)
    right = np.where(rows_have, right, np.int32(-1))
    return left, right


def _estimate_joints(mask: np.ndarray) -> dict:
    """Estimate ~15 joints from a silhouette.

    Reads structure off the silhouette itself (per-row width, leftmost /
    rightmost protrusion, narrow-band detection) so a thin character and
    a stocky character both get sensible placements. Falls back to
    centerline-percentage anchors only when the silhouette doesn't span
    enough rows for a feature.
    """
    bm = (mask > 0)
    h, w = bm.shape
    if not bm.any():
        return {}
    ys, xs = np.where(bm)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    left, right = _row_first_last(bm)
    width = np.where(right >= 0, right - left + 1, 0).astype(np.int32)
    center = np.where(right >= 0, (left + right) // 2, w // 2).astype(np.int32)

    height = max(1, y1 - y0 + 1)

    # Head: top of the silhouette (a few rows in for noise margin).
    head_y = y0 + int(height * 0.04)
    head_y = max(y0, min(y1, head_y))
    head_x = int(center[head_y]) if width[head_y] > 0 else (x0 + x1) // 2

    # Neck: narrowest row in [y0+10%, y0+30%]. Char models tend to dip
    # at the neck and broaden again at the shoulders.
    neck_lo = y0 + int(height * 0.10)
    neck_hi = y0 + int(height * 0.30)
    cand = np.arange(max(y0, neck_lo), min(y1, neck_hi) + 1)
    cand = cand[width[cand] > 0]
    if cand.size:
        neck_y = int(cand[np.argmin(width[cand])])
    else:
        neck_y = y0 + int(height * 0.18)
    neck_x = int(center[neck_y]) if width[neck_y] > 0 else head_x

    # Shoulders: a few rows below neck where width re-expands.
    sh_y = min(y1, neck_y + max(2, height // 30))
    if width[sh_y] > 0:
        sh_l = (int(left[sh_y]),  int(sh_y))
        sh_r = (int(right[sh_y]), int(sh_y))
    else:
        sh_l = (head_x - height // 8, sh_y)
        sh_r = (head_x + height // 8, sh_y)

    # Hands: leftmost/rightmost protrusions in the arm zone
    # ([sh_y, y0+60%]). The widest left/right reach in that band is the
    # hand; the row that owns that reach is the hand row.
    arm_top = sh_y
    arm_bot = min(y1, y0 + int(height * 0.60))
    arm_rows = np.arange(arm_top, arm_bot + 1)
    arm_rows = arm_rows[width[arm_rows] > 0]
    if arm_rows.size:
        i_l = int(arm_rows[np.argmin(left[arm_rows])])
        i_r = int(arm_rows[np.argmax(right[arm_rows])])
        hand_l = (int(left[i_l]),  i_l)
        hand_r = (int(right[i_r]), i_r)
    else:
        hand_l = (sh_l[0] - 6, sh_l[1] + height // 4)
        hand_r = (sh_r[0] + 6, sh_r[1] + height // 4)

    # Elbows: midpoint between shoulder and hand on each side.
    elb_l = ((sh_l[0] + hand_l[0]) // 2, (sh_l[1] + hand_l[1]) // 2)
    elb_r = ((sh_r[0] + hand_r[0]) // 2, (sh_r[1] + hand_r[1]) // 2)

    # Waist: narrowest row in [shoulder + 10%, y0 + 70%].
    waist_lo = sh_y + int(height * 0.10)
    waist_hi = min(y1, y0 + int(height * 0.70))
    cand = np.arange(max(sh_y, waist_lo), waist_hi + 1)
    cand = cand[width[cand] > 0]
    if cand.size:
        waist_y = int(cand[np.argmin(width[cand])])
    else:
        waist_y = y0 + int(height * 0.55)
    waist_x = int(center[waist_y]) if width[waist_y] > 0 else neck_x

    # Hips: widest row below waist (capped at 85%).
    hip_lo = waist_y
    hip_hi = min(y1, y0 + int(height * 0.85))
    cand = np.arange(hip_lo, hip_hi + 1)
    cand = cand[width[cand] > 0]
    if cand.size:
        hip_y = int(cand[np.argmax(width[cand])])
        hip_l = (int(left[hip_y]),  hip_y)
        hip_r = (int(right[hip_y]), hip_y)
    else:
        hip_y = y0 + int(height * 0.70)
        hip_l = (waist_x - 12, hip_y)
        hip_r = (waist_x + 12, hip_y)

    # Feet: bottom-most occupied row's leftmost/rightmost endpoints.
    foot_band = np.arange(max(y1 - max(2, height // 20), y0), y1 + 1)
    foot_band = foot_band[width[foot_band] > 0]
    if foot_band.size:
        f_y = int(foot_band[-1])
        foot_l = (int(left[f_y]),  f_y)
        foot_r = (int(right[f_y]), f_y)
    else:
        foot_l = (hip_l[0], min(h - 1, y1))
        foot_r = (hip_r[0], min(h - 1, y1))

    knee_l = ((hip_l[0] + foot_l[0]) // 2,
              (hip_l[1] + foot_l[1]) // 2)
    knee_r = ((hip_r[0] + foot_r[0]) // 2,
              (hip_r[1] + foot_r[1]) // 2)

    def _pt(p):
        return [int(p[0]), int(p[1])]

    return {
        "head":       [int(head_x), int(y0)],
        "neck":       [int(neck_x), int(neck_y)],
        "shoulder_l": _pt(sh_l),
        "shoulder_r": _pt(sh_r),
        "elbow_l":    _pt(elb_l),
        "elbow_r":    _pt(elb_r),
        "hand_l":     _pt(hand_l),
        "hand_r":     _pt(hand_r),
        "waist":      [int(waist_x), int(waist_y)],
        "hip_l":      _pt(hip_l),
        "hip_r":      _pt(hip_r),
        "knee_l":     _pt(knee_l),
        "knee_r":     _pt(knee_r),
        "foot_l":     _pt(foot_l),
        "foot_r":     _pt(foot_r),
    }


# ────────────────────────────────────────────────────────────
# 5. dirty zones + motion rules
# ────────────────────────────────────────────────────────────
def _clip_range(a: int, b: int, h: int):
    a = max(0, min(h - 1, int(a)))
    b = max(0, min(h - 1, int(b)))
    if b < a:
        a, b = b, a
    return [a, b]


def _dirty_zones(joints: dict, bbox: list, h: int = H) -> dict:
    x0, y0, x1, y1 = bbox
    head_y  = joints.get("head",       [0, y0])[1]
    neck_y  = joints.get("neck",       [0, y0 + (y1 - y0) // 5])[1]
    sh_y    = joints.get("shoulder_l", [0, neck_y + 2])[1]
    waist_y = joints.get("waist",      [0, y0 + (y1 - y0) // 2])[1]
    hip_y   = joints.get("hip_l",      [0, y0 + (2 * (y1 - y0)) // 3])[1]
    foot_y  = joints.get("foot_l",     [0, y1])[1]

    return {
        "hair":  _clip_range(y0 - 2, max(y0 + 2, neck_y - 4), h),
        "face":  _clip_range(head_y + max(2, (neck_y - head_y) // 3),
                             max(neck_y, sh_y), h),
        "arms":  _clip_range(sh_y, max(sh_y + 2, waist_y), h),
        "cloth": _clip_range(max(sh_y, waist_y - 4),
                             max(waist_y + 2, hip_y), h),
        "body":  _clip_range(y0, min(y1, foot_y), h),
    }


def _motion_rules(joints: dict, dirty: dict, bbox: list,
                  quality: float) -> list:
    if not joints:
        return []
    head  = joints.get("head",       [(bbox[0] + bbox[2]) // 2, bbox[1]])
    neck  = joints.get("neck",       [head[0], head[1] + 30])
    sh_l  = joints.get("shoulder_l", [head[0] - 20, neck[1] + 4])
    sh_r  = joints.get("shoulder_r", [head[0] + 20, neck[1] + 4])
    waist = joints.get("waist",      [head[0], (sh_l[1] + bbox[3]) // 2])
    bbox_cx = (bbox[0] + bbox[2]) // 2
    bbox_cy = (bbox[1] + bbox[3]) // 2

    def _amp(scale: float) -> float:
        return float(round(max(0.4, min(4.0, scale * (0.6 + 0.6 * quality))), 3))

    def _mk(name, origin, vec, scale, freq, zone):
        return {
            "name": str(name),
            "origin": [int(origin[0]), int(origin[1])],
            "vector": [float(vec[0]), float(vec[1])],
            "amplitude": _amp(scale),
            "frequency": float(freq),
            "dirty_rows": list(dirty[zone]),
        }

    return [
        _mk("hair_wave",  head,                                  (1, 0), 2.0, 0.18, "hair"),
        _mk("blink",      [head[0], (head[1] + neck[1]) // 2],   (0, 1), 0.8, 0.10, "face"),
        _mk("arm_sway",   [(sh_l[0] + sh_r[0]) // 2, (sh_l[1] + sh_r[1]) // 2],
                                                                 (1, 0), 2.5, 0.14, "arms"),
        _mk("cloth_sway", waist,                                 (1, 0), 2.0, 0.11, "cloth"),
        _mk("body_bounce",[bbox_cx, bbox_cy],                    (0, 1), 1.4, 0.08, "body"),
    ]


# ────────────────────────────────────────────────────────────
# 6. debug rendering helpers (numpy-only)
# ────────────────────────────────────────────────────────────
def _line(img, p0, p1, color, thickness=1):
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    h, w = img.shape[:2]
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    half = thickness // 2
    while True:
        for j in range(-half, half + 1):
            for i in range(-half, half + 1):
                xx, yy = x0 + i, y0 + j
                if 0 <= xx < w and 0 <= yy < h:
                    img[yy, xx] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 < dx:
            err += dx; y0 += sy


def _disk(img, p, r, color):
    h, w = img.shape[:2]
    x, y = int(p[0]), int(p[1])
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    sub = img[y0:y1, x0:x1]
    sub[m] = color


def _draw_skeleton(img, joints, bbox):
    canvas = img.copy()
    pairs = [
        ("head", "neck"),
        ("neck", "shoulder_l"), ("neck", "shoulder_r"),
        ("shoulder_l", "elbow_l"), ("elbow_l", "hand_l"),
        ("shoulder_r", "elbow_r"), ("elbow_r", "hand_r"),
        ("neck", "waist"),
        ("waist", "hip_l"), ("waist", "hip_r"),
        ("hip_l", "knee_l"), ("knee_l", "foot_l"),
        ("hip_r", "knee_r"), ("knee_r", "foot_r"),
    ]
    edge_color = np.array([0, 255, 0], np.uint8)
    joint_color = np.array([0, 0, 255], np.uint8)
    for a, b in pairs:
        if a in joints and b in joints:
            _line(canvas, joints[a], joints[b], edge_color, thickness=2)
    for _name, p in joints.items():
        _disk(canvas, p, 3, joint_color)

    x0, y0, x1, y1 = bbox
    h, w = canvas.shape[:2]
    bb = np.array([255, 255, 0], np.uint8)
    x0 = max(0, min(w - 1, x0)); x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0)); y1 = max(0, min(h - 1, y1))
    canvas[y0, x0:x1 + 1] = bb
    canvas[y1, x0:x1 + 1] = bb
    canvas[y0:y1 + 1, x0] = bb
    canvas[y0:y1 + 1, x1] = bb
    return canvas


def _draw_dirty_rows(img, dirty):
    canvas = img.copy()
    palette = {
        "hair":  np.array([0, 200, 255], np.uint8),
        "face":  np.array([0, 255, 200], np.uint8),
        "arms":  np.array([200, 0, 255], np.uint8),
        "cloth": np.array([255, 200, 0], np.uint8),
        "body":  np.array([255, 0, 200], np.uint8),
    }
    h = canvas.shape[0]
    for name, span in dirty.items():
        a, b = int(span[0]), int(span[1])
        a = max(0, min(h - 1, a)); b = max(0, min(h - 1, b))
        if b < a:
            a, b = b, a
        canvas[a:b + 1, 0:6] = palette.get(
            name, np.array([255, 255, 255], np.uint8))
    return canvas


def _compose_radar(img, mask, rd, cd, edge_map):
    canvas = img.copy()
    h, w = canvas.shape[:2]
    bm = (mask > 0)
    if bm.any():
        tint = np.array([60, 220, 60], np.int32)
        canvas[bm] = ((canvas[bm].astype(np.int32) * 6 + tint * 4) // 10
                      ).astype(np.uint8)
    em = edge_map.astype(np.uint16)
    canvas[..., 2] = np.clip(canvas[..., 2].astype(np.uint16) + em // 2,
                             0, 255).astype(np.uint8)
    rd_max = float(rd.max()) if rd.size else 0.0
    cd_max = float(cd.max()) if cd.size else 0.0
    if rd_max > 1e-6:
        bar_w = max(4, w // 12)
        rd_n = (rd / rd_max) * bar_w
        for y in range(h):
            b = int(rd_n[y])
            if b > 0:
                canvas[y, :b] = np.array([20, 60, 220], np.uint8)
    if cd_max > 1e-6:
        bar_h = max(4, h // 12)
        cd_n = (cd / cd_max) * bar_h
        for x in range(w):
            b = int(cd_n[x])
            if b > 0:
                canvas[:b, x] = np.array([220, 60, 20], np.uint8)
    return canvas


# ────────────────────────────────────────────────────────────
# 7. public entry
# ────────────────────────────────────────────────────────────
def analyze_pose_radar(filepath_or_image: Union[str, np.ndarray, "os.PathLike"],
                       label: str,
                       *,
                       out_dir: Optional[str] = None,
                       filename: Optional[str] = None) -> dict:
    """Run the full pose / radar pass on one image.

    Returns a JSON-friendly dict with `bbox`, `joints`, `motions`,
    `dirty_zones`, `quality`, `label`, `source` plus a non-JSON
    `_arrays` sidecar carrying the mask, density fields, edge map and
    fft response. Callers that serialise the dict should pop `_arrays`
    first (the CLI and the JSON file writer both do).
    """
    if isinstance(filepath_or_image, np.ndarray):
        img = filepath_or_image
        src_name = filename or "<array>"
    else:
        path = str(filepath_or_image)
        img = _load_image(path)
        src_name = filename or os.path.basename(path)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    img = _resize_nn(img, H, W)
    img = np.ascontiguousarray(img, dtype=np.uint8)

    raw_mask = _bg_mask(img)
    cleaned = _morph_clean(raw_mask, k=2)
    sil = _largest_component(cleaned)
    bm = (sil > 0)

    if bm.any():
        ys, xs = np.where(bm)
        bbox = [int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max())]
    else:
        bbox = [0, 0, W - 1, H - 1]

    rd, cd, edge_map, fft_resp = _radar_fields(img, sil)
    joints = _estimate_joints(sil) if bm.any() else {}
    dirty = _dirty_zones(joints, bbox, h=H) if joints else {
        "hair":  _clip_range(bbox[1], bbox[1] + max(1, (bbox[3] - bbox[1]) // 6), H),
        "face":  _clip_range(bbox[1] + (bbox[3] - bbox[1]) // 6,
                             bbox[1] + (bbox[3] - bbox[1]) // 3, H),
        "arms":  _clip_range(bbox[1] + (bbox[3] - bbox[1]) // 3,
                             bbox[1] + (bbox[3] - bbox[1]) // 2, H),
        "cloth": _clip_range(bbox[1] + (bbox[3] - bbox[1]) // 2,
                             bbox[1] + (2 * (bbox[3] - bbox[1])) // 3, H),
        "body":  _clip_range(bbox[1], bbox[3], H),
    }

    area_frac = float(bm.sum()) / float(W * H)
    area_score = 1.0 if 0.04 < area_frac < 0.7 else 0.4
    joint_score = min(1.0, len(joints) / 15.0)
    edge_density = float((edge_map > 30).mean())
    edge_score = min(1.0, edge_density * 6.0)
    quality = float(np.clip(0.42 + 0.10 * area_score
                            + 0.05 * joint_score
                            + 0.05 * edge_score, 0.42, 0.62))
    motions = _motion_rules(joints, dirty, bbox, quality)

    result = {
        "type": "pose_radar",
        "label": label or "",
        "source": src_name,
        "bbox": [int(v) for v in bbox],
        "joints": {k: [int(v[0]), int(v[1])] for k, v in joints.items()},
        "motions": motions,
        "dirty_zones": {k: list(v) for k, v in dirty.items()},
        "quality": float(round(quality, 4)),
        "area_frac": float(round(area_frac, 4)),
    }
    result["_arrays"] = {
        "mask": sil,
        "row_density": rd,
        "col_density": cd,
        "edge_map": edge_map,
        "fft_response": fft_resp,
        "image": img,
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        sss_image_io.write_png(os.path.join(out_dir, "mask.png"), sil)
        radar = _compose_radar(img, sil, rd, cd, edge_map)
        sss_image_io.write_png(os.path.join(out_dir, "radar.png"), radar)
        skel = _draw_skeleton(img, joints, bbox)
        sss_image_io.write_png(
            os.path.join(out_dir, "skeleton_overlay.png"), skel)
        dr = _draw_dirty_rows(img, dirty)
        sss_image_io.write_png(os.path.join(out_dir, "dirty_rows.png"), dr)
        json_payload = {k: v for k, v in result.items() if k != "_arrays"}
        with open(os.path.join(out_dir, "pose_motion.json"),
                  "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2, ensure_ascii=False)

    return result


# ────────────────────────────────────────────────────────────
# 8. CLI
# ────────────────────────────────────────────────────────────
def _cli():
    ap = argparse.ArgumentParser(
        description="SSS pose / radar perception")
    ap.add_argument("input", help="image path (PNG, PPM, or any "
                                  "ffmpeg-readable format)")
    ap.add_argument("--label", default="",
                    help="text label for tagging / source attribution")
    ap.add_argument("--out", default=None,
                    help="optional output / debug directory")
    args = ap.parse_args()

    result = analyze_pose_radar(args.input, args.label, out_dir=args.out)
    result.pop("_arrays", None)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        print(f"\nDebug files written to: {args.out}")


if __name__ == "__main__":
    _cli()
