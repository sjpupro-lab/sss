#!/bin/bash
# make_all_videos.sh — SSS 테스트 영상 전체 생성
set -e
cd /home/ubuntu/sss

MODEL=build/models/sanrio.sss
SCALE=8
FPS=12

mkdir -p build/videos

# ─────────────────────────────────────────────────────────────
# 테스트② seed 변화: keroppi seed 1/2/3 이어붙이기
# 같은 프롬프트 → 다른 결과
# ─────────────────────────────────────────────────────────────
echo "=== 테스트② seed 변화 영상 ==="

for seed in 1 2 3; do
  mkdir -p build/frames/seed_${seed}
  SSS_FRAME_DIR=build/frames/seed_${seed} \
    ./build/sss_gen $MODEL "green frog keroppi" /dev/null $seed 1.0 2>/dev/null
done

# 3개 seed 프레임을 PNG로 변환 후 이어붙이기
python3 - << 'PYEOF'
import os
from PIL import Image

scale = 8
seeds = [1, 2, 3]
out_dir = "build/frames/seed_concat"
os.makedirs(out_dir, exist_ok=True)

frame_idx = 0
for seed in seeds:
    src = f"build/frames/seed_{seed}"
    files = sorted(f for f in os.listdir(src) if f.endswith('.ppm'))
    for f in files:
        img = Image.open(os.path.join(src, f))
        big = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        # 라벨 추가
        from PIL import ImageDraw
        draw = ImageDraw.Draw(big)
        draw.text((4, 4), f"seed={seed}", fill=(255, 255, 0))
        big.save(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"))
        frame_idx += 1

print(f"seed concat: {frame_idx}개 프레임 → {out_dir}")
PYEOF

ffmpeg -y -framerate $FPS -i build/frames/seed_concat/frame_%04d.png \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 \
  build/videos/test2_seed_variation.mp4 2>&1 | tail -3
echo "완성: build/videos/test2_seed_variation.mp4"

# ─────────────────────────────────────────────────────────────
# 테스트③-A 프롬프트 전환: keroppi → kitty (행 32에서 전환)
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== 테스트③-A 프롬프트 전환 영상 ==="

mkdir -p build/frames/morph
SSS_FRAME_DIR=build/frames/morph \
  SSS_ANIM_MODE=morph \
  SSS_PROMPT2="white cat kitty" \
  ./build/sss_gen $MODEL "green frog keroppi" build/sanrio_gen/morph_keroppi_kitty.ppm 1 1.0 2>&1 | tail -1

python3 - << 'PYEOF'
import os
from PIL import Image, ImageDraw

scale = 8
src = "build/frames/morph"
out_dir = "build/frames/morph_png"
os.makedirs(out_dir, exist_ok=True)

files = sorted(f for f in os.listdir(src) if f.endswith('.ppm'))
total = len(files)
for i, f in enumerate(files):
    img = Image.open(os.path.join(src, f))
    big = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(big)
    label = "keroppi" if i < total // 2 else "→ kitty"
    color = (100, 255, 100) if i < total // 2 else (255, 180, 255)
    draw.text((4, 4), label, fill=color)
    big.save(os.path.join(out_dir, f"frame_{i:04d}.png"))

print(f"morph: {len(files)}개 프레임 → {out_dir}")
PYEOF

ffmpeg -y -framerate $FPS -i build/frames/morph_png/frame_%04d.png \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 \
  build/videos/test3a_prompt_morph.mp4 2>&1 | tail -3
echo "완성: build/videos/test3a_prompt_morph.mp4"

# ─────────────────────────────────────────────────────────────
# 테스트③-B 주파수 애니메이션: detail이 sin 파형으로 진동
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== 테스트③-B 주파수 애니메이션 영상 ==="

mkdir -p build/frames/freq
SSS_FRAME_DIR=build/frames/freq \
  SSS_ANIM_MODE=freq \
  ./build/sss_gen $MODEL "pink rabbit mymelody" build/sanrio_gen/freq_mymelody.ppm 1 1.0 2>&1 | tail -1

python3 - << 'PYEOF'
import os, math
from PIL import Image, ImageDraw

scale = 8
src = "build/frames/freq"
out_dir = "build/frames/freq_png"
os.makedirs(out_dir, exist_ok=True)

files = sorted(f for f in os.listdir(src) if f.endswith('.ppm'))
total = len(files)
for i, f in enumerate(files):
    img = Image.open(os.path.join(src, f))
    big = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(big)
    t = i / max(total - 1, 1)
    d = 0.2 + 0.8 * (0.5 + 0.5 * math.sin(t * math.pi * 3.0))
    draw.text((4, 4), f"detail={d:.2f}", fill=(255, 220, 80))
    big.save(os.path.join(out_dir, f"frame_{i:04d}.png"))

print(f"freq: {len(files)}개 프레임 → {out_dir}")
PYEOF

ffmpeg -y -framerate $FPS -i build/frames/freq_png/frame_%04d.png \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 \
  build/videos/test3b_freq_anim.mp4 2>&1 | tail -3
echo "완성: build/videos/test3b_freq_anim.mp4"

# ─────────────────────────────────────────────────────────────
# 최종 요약
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== 전체 영상 목록 ==="
ls -lh build/videos/
