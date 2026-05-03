#!/usr/bin/env python3
"""
캐릭터 이미지에서 sin 주파수 기반 포즈 변형 학습 데이터 생성

각 캐릭터 이미지를 N개의 포즈로 변형:
  - 상하 bob (bouncing): sin으로 캐릭터를 위아래로 이동
  - 좌우 sway (흔들기): sin으로 좌우 기울기
  - 스케일 breathe (숨쉬기): sin으로 크기 변화

labels.tsv 구조:
  fname \t color \t shape \t face(=pose_XX)
"""
import os
import json
import math
import numpy as np
from PIL import Image, ImageTransform

UPLOAD_DIR = "/home/ubuntu/upload"
OUT_DIR    = "/home/ubuntu/sss/data/sanrio_anim"
LABELS_JSON = "/home/ubuntu/character_labels.json"
SIZE = 64
N_POSES = 16   # sin 주기 1바퀴 = 16 포즈 프레임

CHAR_LABELS = {
    "keroppi":  ("green", "frog"),
    "kitty":    ("white", "cat"),
    "mymelody": ("pink",  "rabbit"),
}

os.makedirs(OUT_DIR, exist_ok=True)

with open(LABELS_JSON) as f:
    labels = json.load(f)

tsv_lines = []
counts = {c: 0 for c in CHAR_LABELS}

def make_pose_frame(img_arr, pose_idx, n_poses, char):
    """
    pose_idx / n_poses → t ∈ [0, 2π)
    세 가지 움직임을 합성:
      1. bob: 상하 이동 (최대 ±5px)
      2. sway: 좌우 기울기 shear (최대 ±0.08)
      3. breathe: 크기 변화 (±3%)
    """
    H, W = img_arr.shape[:2]
    t = 2.0 * math.pi * pose_idx / n_poses

    # 1. bob: 상하 이동
    bob = int(round(5.0 * math.sin(t)))

    # 2. sway: 좌우 shear
    sway = 0.06 * math.sin(t + math.pi / 4)

    # 3. breathe: 스케일
    scale = 1.0 + 0.04 * math.sin(t * 2)

    img = Image.fromarray(img_arr)

    # 배경 (흰색 또는 캐릭터에 맞는 배경)
    bg_colors = {
        "keroppi":  (240, 255, 240),
        "kitty":    (255, 255, 255),
        "mymelody": (255, 240, 250),
    }
    bg = bg_colors.get(char, (240, 240, 240))
    canvas = Image.new("RGB", (W, H), bg)

    # 스케일 적용
    new_w = int(W * scale)
    new_h = int(H * scale)
    img_scaled = img.resize((new_w, new_h), Image.LANCZOS)

    # 중앙 정렬 후 bob 적용
    paste_x = (W - new_w) // 2
    paste_y = (H - new_h) // 2 + bob

    # 범위 클리핑
    src_x = max(0, -paste_x)
    src_y = max(0, -paste_y)
    dst_x = max(0, paste_x)
    dst_y = max(0, paste_y)
    crop_w = min(new_w - src_x, W - dst_x)
    crop_h = min(new_h - src_y, H - dst_y)

    if crop_w > 0 and crop_h > 0:
        region = img_scaled.crop((src_x, src_y, src_x + crop_w, src_y + crop_h))
        canvas.paste(region, (dst_x, dst_y))

    # sway: affine shear 변환
    # shear matrix: [1, sway, 0, 0, 1, 0]
    shear_mat = (1, sway, -sway * H / 2,
                 0, 1,    0)
    canvas = canvas.transform((W, H), Image.AFFINE, shear_mat,
                               resample=Image.BILINEAR,
                               fillcolor=bg)

    return np.array(canvas, dtype=np.uint8)


def save_ppm(arr, path):
    H, W = arr.shape[:2]
    with open(path, 'wb') as f:
        f.write(f"P6\n{W} {H}\n255\n".encode('ascii'))
        f.write(arr.tobytes())


print(f"포즈 데이터셋 생성: {N_POSES}포즈 × 각 캐릭터")

for fname, char in labels.items():
    if char not in CHAR_LABELS:
        continue
    src = os.path.join(UPLOAD_DIR, fname)
    if not os.path.exists(src):
        continue

    img = Image.open(src).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    img_arr = np.array(img, dtype=np.uint8)

    color, shape = CHAR_LABELS[char]
    img_idx = counts[char]

    for pose_idx in range(N_POSES):
        pose_arr = make_pose_frame(img_arr, pose_idx, N_POSES, char)
        ppm_name = f"{char}_{img_idx:03d}_pose{pose_idx:02d}.ppm"
        save_ppm(pose_arr, os.path.join(OUT_DIR, ppm_name))

        face_lbl = f"pose{pose_idx:02d}"
        tsv_lines.append(f"{ppm_name}\t{color}\t{shape}\t{face_lbl}")

    counts[char] += 1
    print(f"  {char}[{img_idx}]: {N_POSES}포즈 생성")

# labels.tsv 저장
tsv_path = os.path.join(OUT_DIR, "labels.tsv")
with open(tsv_path, 'w') as f:
    f.write('\n'.join(tsv_lines) + '\n')

total_imgs = sum(counts.values())
print(f"\n완료!")
print(f"  총 {total_imgs}개 원본 × {N_POSES}포즈 = {len(tsv_lines)}개 PPM")
print(f"  저장: {OUT_DIR}/labels.tsv")
print(f"\nTSV 샘플:")
for line in tsv_lines[:4]:
    print(f"  {line}")
