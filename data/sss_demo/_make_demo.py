"""Generate a tiny synthetic dataset for the SSS spectrogram engine smoke test.

Produces PPM images and a labels.tsv with the columns the trainer expects:
    filename<TAB>color<TAB>shape<TAB>face
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
SIZE = 64

COLORS = {
    "red":   (1.0, 0.15, 0.15),
    "green": (0.15, 0.85, 0.25),
    "blue":  (0.15, 0.25, 0.95),
}

# Shapes are functions returning a 0/1 mask.
def shape_circle(s):
    yy, xx = np.mgrid[0:s, 0:s]
    cy = cx = s / 2.0
    r = s * 0.32
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r

def shape_square(s):
    yy, xx = np.mgrid[0:s, 0:s]
    cy = cx = s / 2.0
    r = s * 0.28
    return (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)

def shape_triangle(s):
    yy, xx = np.mgrid[0:s, 0:s]
    cx = s / 2.0
    base_y = s * 0.85
    apex_y = s * 0.15
    half = (yy - apex_y) / max(1.0, (base_y - apex_y)) * (s * 0.45)
    return (yy >= apex_y) & (yy <= base_y) & (np.abs(xx - cx) <= half)

SHAPES = {
    "circle":   shape_circle,
    "square":   shape_square,
    "triangle": shape_triangle,
}

# Faces are little inscribed details — eyes/mouth — that show up only in
# the high-frequency band. Returns an additive RGB delta in [-1, 1].
def face_smile(s):
    img = np.zeros((s, s, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:s, 0:s]
    # eyes
    for ex in (s * 0.40, s * 0.60):
        m = ((yy - s * 0.42) ** 2 + (xx - ex) ** 2) <= (s * 0.04) ** 2
        img[m] = [-0.7, -0.7, -0.7]
    # mouth (smile arc)
    arc = ((yy - s * 0.65) ** 2 / (s * 0.05) ** 2
           + (xx - s * 0.50) ** 2 / (s * 0.18) ** 2) <= 1.0
    arc &= yy > s * 0.62
    img[arc] = [-0.6, -0.6, -0.6]
    return img

def face_sad(s):
    img = np.zeros((s, s, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:s, 0:s]
    for ex in (s * 0.40, s * 0.60):
        m = ((yy - s * 0.42) ** 2 + (xx - ex) ** 2) <= (s * 0.04) ** 2
        img[m] = [-0.7, -0.7, -0.7]
    arc = ((yy - s * 0.78) ** 2 / (s * 0.05) ** 2
           + (xx - s * 0.50) ** 2 / (s * 0.18) ** 2) <= 1.0
    arc &= yy < s * 0.78
    img[arc] = [-0.6, -0.6, -0.6]
    return img

FACES = {
    "smile": face_smile,
    "sad":   face_sad,
}

def render(color, shape_fn, face_fn, s=SIZE):
    bg = np.full((s, s, 3), 0.95, dtype=np.float32)
    mask = shape_fn(s)
    bg[mask] = color
    bg += face_fn(s)
    return np.clip(bg, 0.0, 1.0)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cn, cv in COLORS.items():
        for sn, sfn in SHAPES.items():
            for fn, ffn in FACES.items():
                img = render(cv, sfn, ffn)
                fname = f"img_{cn}_{sn}_{fn}.ppm"
                Image.fromarray((img * 255).astype(np.uint8)).save(OUT / fname)
                rows.append(f"{fname}\t{cn}\t{sn}\t{fn}")
    (OUT / "labels.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} images + labels.tsv into {OUT}")

if __name__ == "__main__":
    main()
