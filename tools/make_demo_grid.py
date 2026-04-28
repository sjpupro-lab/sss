#!/usr/bin/env python3
"""Compose two side-by-side grids for the README demo:

  docs/demo/dataset_grid.png   — the 10 training images (4 colours + 6 fruits)
  docs/demo/generated_grid.png — the 6 generated outputs paired with their prompts

Each image is 256x256, downscaled to 128x128 in the grid for compactness.
Captions are drawn under each tile using the default PIL font.
"""
import os
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(REPO, "docs", "demo")
TILE = 128
CAPTION_H = 20
PADDING = 4
BG = (245, 245, 240)


def font():
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
    except Exception:
        return ImageFont.load_default()


def make_grid(items, cols, out_path):
    """items: list of (image_path, caption)."""
    rows = (len(items) + cols - 1) // cols
    cell_w = TILE + 2 * PADDING
    cell_h = TILE + CAPTION_H + 2 * PADDING
    W = cell_w * cols
    H = cell_h * rows
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    f = font()
    for i, (path, caption) in enumerate(items):
        r, c = divmod(i, cols)
        x = c * cell_w + PADDING
        y = r * cell_h + PADDING
        img = Image.open(path).resize((TILE, TILE))
        canvas.paste(img, (x, y))
        # caption
        try:
            tw = draw.textlength(caption, font=f)
        except AttributeError:
            tw, _ = draw.textsize(caption, font=f)
        tx = x + (TILE - tw) // 2
        ty = y + TILE + 2
        draw.text((tx, ty), caption, fill=(40, 40, 40), font=f)
    canvas.save(out_path, "PNG", optimize=True)
    print(f"wrote {out_path}")


def main():
    dataset = [
        (os.path.join(DEMO, "dataset", "red.png"),    "red"),
        (os.path.join(DEMO, "dataset", "green.png"),  "green"),
        (os.path.join(DEMO, "dataset", "blue.png"),   "blue"),
        (os.path.join(DEMO, "dataset", "yellow.png"), "yellow"),
        (os.path.join(DEMO, "dataset", "apple.png"),     "apple"),
        (os.path.join(DEMO, "dataset", "banana.png"),    "banana"),
        (os.path.join(DEMO, "dataset", "grape.png"),     "grape"),
        (os.path.join(DEMO, "dataset", "lime.png"),      "lime"),
        (os.path.join(DEMO, "dataset", "blueberry.png"), "blueberry"),
        (os.path.join(DEMO, "dataset", "orange.png"),    "orange"),
    ]
    generated = [
        (os.path.join(DEMO, "generated", "red_apple.png"),       '"red apple"'),
        (os.path.join(DEMO, "generated", "yellow_banana.png"),   '"yellow banana"'),
        (os.path.join(DEMO, "generated", "purple_grape.png"),    '"purple grape"'),
        (os.path.join(DEMO, "generated", "green_lime.png"),      '"green lime"'),
        (os.path.join(DEMO, "generated", "blue_blueberry.png"),  '"blue blueberry"'),
        (os.path.join(DEMO, "generated", "orange_fruit.png"),    '"orange fruit"'),
    ]
    make_grid(dataset,   cols=5, out_path=os.path.join(DEMO, "dataset_grid.png"))
    make_grid(generated, cols=3, out_path=os.path.join(DEMO, "generated_grid.png"))


if __name__ == "__main__":
    main()
