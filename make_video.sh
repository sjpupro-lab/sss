#!/bin/bash
# make_video.sh — SSS 프레임 → mp4 변환 헬퍼
# 사용법: bash make_video.sh <frames_dir> <output.mp4> [fps] [scale]
#
# 예: bash make_video.sh build/frames/keroppi build/videos/keroppi.mp4 12 8

FRAMES_DIR="${1:-build/frames/keroppi}"
OUTPUT="${2:-build/videos/out.mp4}"
FPS="${3:-12}"
SCALE="${4:-8}"   # 64px → 512px (8배)

mkdir -p "$(dirname "$OUTPUT")"

# PPM을 업스케일 PNG로 변환 (python)
TMP_PNG="${FRAMES_DIR}_png"
mkdir -p "$TMP_PNG"

python3 -c "
import os, sys
from PIL import Image

src = sys.argv[1]
dst = sys.argv[2]
scale = int(sys.argv[3])

files = sorted(f for f in os.listdir(src) if f.endswith('.ppm'))
for i, f in enumerate(files):
    img = Image.open(os.path.join(src, f))
    # NEAREST: 픽셀 선명하게 업스케일
    big = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    out_name = f'frame_{i:04d}.png'
    big.save(os.path.join(dst, out_name))
print(f'변환 완료: {len(files)}개 프레임 → {dst}')
" "$FRAMES_DIR" "$TMP_PNG" "$SCALE"

# ffmpeg으로 mp4 생성
ffmpeg -y \
  -framerate "$FPS" \
  -i "${TMP_PNG}/frame_%04d.png" \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -crf 18 \
  "$OUTPUT" 2>&1 | tail -5

echo "영상 완성: $OUTPUT"
rm -rf "$TMP_PNG"
