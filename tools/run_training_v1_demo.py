"""End-to-end demo:

1. 새로 추가된 `data/training_v1` 학습데이터를 SSSPipeline 메모리에 ingest
2. 한국어/영어 키워드를 섞은 다양한 프롬프트로 이미지 생성
3. 짧은 모션 시퀀스(걷기/점프/스핀) 생성
4. 결과를 격자 형태 PNG 로 정리

Run:
    cd /home/ubuntu/work/sss && python3 tools/run_training_v1_demo.py
"""
from __future__ import annotations
import os
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.sss_unified import SSSPipeline
from tools import sss_ingest

_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_FONT = ImageFont.truetype(_FONT_PATH, 22)

OUT_DIR = ROOT / "results" / "training_v1_demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ingest_training_v1(pipeline: SSSPipeline) -> int:
    base = ROOT / "data" / "training_v1"
    labels_path = base / "images" / "labels.tsv"
    n_added = 0
    with open(labels_path, "r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            fname, label = parts
            img_path = base / "images" / fname
            if not img_path.exists():
                continue
            try:
                sss_ingest.ingest_labeled_image(
                    str(img_path), pipeline.memory, label_text=label,
                    source=f"training_v1:{fname}",
                )
                n_added += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  warn: {fname}: {exc}")
    # text ingest
    for text_file in ("korean_train.txt", "descriptive_en.txt"):
        try:
            sss_ingest.ingest_file(
                str(base / text_file), pipeline.memory, filename=text_file)
        except Exception as exc:  # noqa: BLE001
            print(f"  warn text {text_file}: {exc}")
    return n_added


def upscale(img: np.ndarray, factor: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor),
                      interpolation=cv2.INTER_CUBIC)


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    h, w = img.shape[:2]
    bar_h = 36
    canvas = np.full((h + bar_h, w, 3), 20, np.uint8)
    canvas[:h] = img
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((10, h + 6), text, font=_FONT, fill=(240, 240, 240))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def build_grid(images, cols: int) -> np.ndarray:
    rows = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]
    canvas = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return canvas


PROMPTS = [
    # Korean prompts trigger Planner's needs_image=True via "그려/이미지/캐릭터"
    "빨간 원이 웃는 이미지 그려줘",
    "파란 별이 놀란 표정 이미지 만들어",
    "노란 삼각형이 슬픈 그림 그려",
    "초록 사각형이 무표정 이미지 생성",
    "분홍 캐릭터가 웃는 이미지 그려",
    "주황 캐릭터가 놀란 이미지 만들어",
    "보라 별이 빛나는 이미지 그려",
    "하얀 토끼 캐릭터 이미지 그려",
    # English fallbacks (also trigger needs_image)
    "draw a red circle smile character",
    "generate a pink star surprise image",
    "make a green triangle sad image",
    "draw a yellow square neutral image",
]

MOTION_PROMPTS = [
    ("walk", "파란 캐릭터가 걷는 영상 그려"),
    ("wave", "노란 캐릭터가 손 흔드는 모션 그려"),
    ("petal", "분홍 꽃잎이 바람에 떨어지는 영상 그려"),
]


def main():
    pipeline = SSSPipeline()
    pipeline.seed_foundation()
    print(f"foundation seeded: {pipeline.memory.total_cells()} cells")

    n_added = ingest_training_v1(pipeline)
    print(f"training_v1 ingested rows: {n_added}, "
          f"total memory cells: {pipeline.memory.total_cells()}")

    # ---------- still images ----------
    tiles = []
    for prompt in PROMPTS:
        result, image, _ = pipeline.run(prompt)
        if image is None:
            print(f"  ! no image for: {prompt}")
            continue
        score = result.get("score", {}).get("final", 0)
        big = upscale(image, 3)
        tile = annotate(big, f"{prompt}  q={score:.2f}")
        tiles.append(tile)
        out_name = (
            "img_"
            + "".join(ch if ch.isalnum() else "_" for ch in prompt)[:60]
            + ".png"
        )
        cv2.imwrite(str(OUT_DIR / out_name), big)
    grid = build_grid(tiles, cols=4)
    cv2.imwrite(str(OUT_DIR / "grid_images.png"), grid)
    print(f"saved {len(tiles)} images and grid_images.png")

    # ---------- motion frames ----------
    motion_tiles_all = []
    for kind, prompt in MOTION_PROMPTS:
        result, image, frames = pipeline.run(prompt)
        if image is None:
            print(f"  ! no motion image for: {prompt}")
            continue
        # use first 6 frames, evenly spaced
        if frames is None or len(frames) == 0:
            frames = [image]
        idxs = np.linspace(0, len(frames) - 1, 6).astype(int)
        row = []
        for i, idx in enumerate(idxs):
            big = upscale(frames[idx], 3)
            row.append(annotate(big, f"{kind} frame {int(idx):02d}"))
        motion_tiles_all.append(row)

    if motion_tiles_all:
        per_w = motion_tiles_all[0][0].shape[1]
        per_h = motion_tiles_all[0][0].shape[0]
        rows_n = len(motion_tiles_all)
        cols_n = 6
        canvas = np.full((rows_n * per_h, cols_n * per_w, 3), 255, np.uint8)
        for r, row in enumerate(motion_tiles_all):
            for c, tile in enumerate(row):
                canvas[r * per_h:(r + 1) * per_h,
                       c * per_w:(c + 1) * per_w] = tile
        cv2.imwrite(str(OUT_DIR / "grid_motion.png"), canvas)
        print(f"saved grid_motion.png with {rows_n} sequences")

    print(f"done. output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
