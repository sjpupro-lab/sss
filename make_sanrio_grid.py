#!/usr/bin/env python3
"""생성된 Sanrio 캐릭터 이미지를 PNG 변환 + 원본과 비교 그리드 생성"""
from PIL import Image, ImageDraw
import os
import numpy as np

GEN_DIR    = "build/sanrio_gen"
SRC_DIR    = "data/sanrio"
OUT_DIR    = "build/sanrio_gen"
SCALE      = 4   # 64 → 256px
PAD        = 6
LABEL_H    = 18

os.makedirs(OUT_DIR, exist_ok=True)

# 1. PPM → PNG 변환
for f in sorted(os.listdir(GEN_DIR)):
    if f.endswith('.ppm'):
        img = Image.open(os.path.join(GEN_DIR, f))
        img.save(os.path.join(OUT_DIR, f.replace('.ppm', '.png')))

# 2. 원본 샘플 PNG 변환 (각 캐릭터 첫 번째)
originals = {}
for char in ['keroppi', 'kitty', 'mymelody']:
    for f in sorted(os.listdir(SRC_DIR)):
        if f.startswith(char) and f.endswith('.ppm'):
            img = Image.open(os.path.join(SRC_DIR, f))
            png_path = os.path.join(OUT_DIR, f"orig_{char}.png")
            img.save(png_path)
            originals[char] = png_path
            break

# 3. 비교 그리드 생성
# 행: keroppi / kitty / mymelody
# 열: 원본 | seed1 | seed2 | seed3
chars = ['keroppi', 'kitty', 'mymelody']
cols = 4  # 원본 + seed 3개
rows = len(chars)

cell_w = 64 * SCALE + PAD * 2
cell_h = 64 * SCALE + PAD * 2 + LABEL_H
header_h = 30

grid_w = cols * cell_w
grid_h = rows * cell_h + header_h

grid = Image.new('RGB', (grid_w, grid_h), (20, 20, 30))
draw = ImageDraw.Draw(grid)

# 헤더
headers = ['원본 (학습 데이터)', 'seed=1', 'seed=42', 'seed=99 (detail=1.2)']
for ci, h in enumerate(headers):
    draw.text((ci * cell_w + PAD, 6), h, fill=(180, 180, 255))

for ri, char in enumerate(chars):
    y_base = header_h + ri * cell_h

    # 원본
    if char in originals:
        orig = Image.open(originals[char]).resize((64 * SCALE, 64 * SCALE), Image.NEAREST)
        grid.paste(orig, (PAD, y_base + PAD))
        draw.text((PAD, y_base + PAD + 64 * SCALE + 2), f"[원본] {char}", fill=(255, 220, 100))

    # 생성 이미지 s1, s2, s3
    for si, seed_tag in enumerate(['s1', 's2', 's3']):
        png_path = os.path.join(OUT_DIR, f"{char}_{seed_tag}.png")
        if os.path.exists(png_path):
            gen_img = Image.open(png_path).resize((64 * SCALE, 64 * SCALE), Image.NEAREST)
            x = (si + 1) * cell_w + PAD
            grid.paste(gen_img, (x, y_base + PAD))
            draw.text((x, y_base + PAD + 64 * SCALE + 2), f"[생성] {char} {seed_tag}", fill=(150, 255, 150))

# 구분선
for ri in range(1, rows):
    y = header_h + ri * cell_h
    draw.line([(0, y), (grid_w, y)], fill=(60, 60, 80), width=2)
for ci in range(1, cols):
    x = ci * cell_w
    draw.line([(x, 0), (x, grid_h)], fill=(60, 60, 80), width=2)

grid_path = os.path.join(OUT_DIR, "sanrio_result_grid.png")
grid.save(grid_path)
print(f"그리드 저장: {grid_path} ({grid_w}x{grid_h})")
