"""Build PNG figures from the SSS 1k-model outputs for the README."""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "build" / "sss_1k_out"
OUT  = ROOT / "docs" / "sss_assets"
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 4   # 64 → 256 for visibility

def load(name):
    return Image.open(SRC / name).convert("RGB")

def upscale(im):
    return im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)

def grid(ims, cols, gap=4, bg=(255, 255, 255)):
    rows = (len(ims) + cols - 1) // cols
    w, h = ims[0].size
    W = cols * w + (cols + 1) * gap
    H = rows * h + (rows + 1) * gap
    canvas = Image.new("RGB", (W, H), bg)
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        canvas.paste(im, (gap + c * (w + gap), gap + r * (h + gap)))
    return canvas

# Combo grid (3×3-ish layout of distinct prompts).
combo_files = [
    "red_circle_smile.ppm",  "blue_circle_smile.ppm",   "green_circle_smile.ppm",
    "red_square_smile.ppm",  "blue_square_sad.ppm",     "red_triangle_sad.ppm",
    "blue_triangle_smile.ppm","green_triangle_smile.ppm",
]
combo_imgs = [upscale(load(n)) for n in combo_files]
grid(combo_imgs, cols=4).save(OUT / "samples_grid.png")

# Seed variation row.
seed_imgs = [upscale(load(f"seed_{i}.ppm")) for i in (1, 2, 3, 4)]
grid(seed_imgs, cols=4).save(OUT / "seed_variation.png")

# Convergence row: 1, 4, 8, 16, 24, 48 steps.
conv_imgs = [upscale(load(f"conv_{k}.ppm")) for k in (1, 4, 8, 16, 24, 48)]
grid(conv_imgs, cols=6).save(OUT / "convergence.png")

# Single training-data sample for visual baseline.
src_train = ROOT / "data" / "sss_demo_1k" / "red_circle_smile_000.ppm"
if src_train.exists():
    upscale(Image.open(src_train).convert("RGB")).save(OUT / "training_sample.png")

print("wrote", sorted(p.name for p in OUT.glob("*.png")))
