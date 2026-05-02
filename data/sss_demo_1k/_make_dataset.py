"""Generate a ~1000-image synthetic training set for the SSS engine.

Same vocabulary as data/sss_demo (3 colors × 3 shapes × 2 faces = 18 combos),
but for each combination we render N=56 variants by perturbing position,
size, hue jitter, and per-pixel noise. That gives 18 × 56 = 1008 images,
which exercises the trainer at scale and tests label-mean stability.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
SIZE = 64
PER_COMBO = 56
RNG = np.random.default_rng(20260502)

COLORS = {
    "red":   (1.00, 0.15, 0.15),
    "green": (0.15, 0.85, 0.25),
    "blue":  (0.15, 0.25, 0.95),
}


def shape_circle(s, cy, cx, r):
    yy, xx = np.mgrid[0:s, 0:s]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def shape_square(s, cy, cx, r):
    yy, xx = np.mgrid[0:s, 0:s]
    return (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)


def shape_triangle(s, cy, cx, r):
    yy, xx = np.mgrid[0:s, 0:s]
    apex_y = cy - r
    base_y = cy + r
    span = max(1.0, base_y - apex_y)
    half = (yy - apex_y) / span * (r * 1.4)
    return (yy >= apex_y) & (yy <= base_y) & (np.abs(xx - cx) <= half)


SHAPE_FNS = {"circle": shape_circle, "square": shape_square, "triangle": shape_triangle}


def face_smile(img, s, cy, cx, scale):
    yy, xx = np.mgrid[0:s, 0:s]
    eye_off_x = scale * 0.18
    eye_off_y = scale * 0.18
    eye_r = max(1.0, scale * 0.06)
    for ex in (cx - eye_off_x, cx + eye_off_x):
        m = ((yy - (cy - eye_off_y)) ** 2 + (xx - ex) ** 2) <= eye_r * eye_r
        img[m] = (img[m] * 0.25)
    arc = (((yy - (cy + scale * 0.20)) ** 2) / max(1.0, (scale * 0.08) ** 2)
           + ((xx - cx) ** 2) / max(1.0, (scale * 0.28) ** 2)) <= 1.0
    arc &= yy > cy + scale * 0.15
    img[arc] = (img[arc] * 0.30)
    return img


def face_sad(img, s, cy, cx, scale):
    yy, xx = np.mgrid[0:s, 0:s]
    eye_off_x = scale * 0.18
    eye_off_y = scale * 0.18
    eye_r = max(1.0, scale * 0.06)
    for ex in (cx - eye_off_x, cx + eye_off_x):
        m = ((yy - (cy - eye_off_y)) ** 2 + (xx - ex) ** 2) <= eye_r * eye_r
        img[m] = (img[m] * 0.25)
    arc = (((yy - (cy + scale * 0.40)) ** 2) / max(1.0, (scale * 0.08) ** 2)
           + ((xx - cx) ** 2) / max(1.0, (scale * 0.28) ** 2)) <= 1.0
    arc &= yy < cy + scale * 0.40
    img[arc] = (img[arc] * 0.30)
    return img


FACE_FNS = {"smile": face_smile, "sad": face_sad}


def render_variant(color_rgb, shape_fn, face_fn, idx):
    s = SIZE
    bg_brightness = float(RNG.uniform(0.88, 0.98))
    bg = np.full((s, s, 3), bg_brightness, dtype=np.float32)
    cy = float(RNG.uniform(s * 0.42, s * 0.58))
    cx = float(RNG.uniform(s * 0.42, s * 0.58))
    r  = float(RNG.uniform(s * 0.22, s * 0.34))
    mask = shape_fn(s, cy, cx, r)
    hue_jitter = RNG.normal(0.0, 0.04, size=3).astype(np.float32)
    rgb = np.clip(np.array(color_rgb, dtype=np.float32) + hue_jitter, 0.0, 1.0)
    bg[mask] = rgb
    bg = face_fn(bg, s, cy, cx, r)
    bg += RNG.normal(0.0, 0.015, size=bg.shape).astype(np.float32)
    return np.clip(bg, 0.0, 1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cn, cv in COLORS.items():
        for sn, sfn in SHAPE_FNS.items():
            for fn, ffn in FACE_FNS.items():
                for k in range(PER_COMBO):
                    img = render_variant(cv, sfn, ffn, k)
                    fname = f"{cn}_{sn}_{fn}_{k:03d}.ppm"
                    Image.fromarray((img * 255).astype(np.uint8)).save(OUT / fname)
                    rows.append(f"{fname}\t{cn}\t{sn}\t{fn}")
    (OUT / "labels.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} images + labels.tsv into {OUT}")


if __name__ == "__main__":
    main()
