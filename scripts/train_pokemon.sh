#!/bin/bash
# ============================================================
# 포켓몬 데이터셋 SSS 학습 파이프라인
# ============================================================
# 사용법:
#   1. pokemon.csv를 data/pokemon_sss/pokemon.csv에 위치
#   2. 이 스크립트 실행: bash scripts/train_pokemon.sh
#
# 파이프라인:
#   Python으로 포켓몬 CSV → PPM 이미지 + labels.tsv 변환
#   → sss_train.py로 .sss 모델 학습
#   → sss_gen으로 이미지 생성 테스트
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

echo "=== SSS 포켓몬 학습 파이프라인 ==="
echo ""

# 1. sss_gen 빌드
echo "[1/3] sss_gen 빌드..."
make sss_gen 2>&1 | tail -2

# 2. 학습
echo "[2/3] sss_train.py 학습 (data/pokemon_sss/)..."
python3 scripts/sss_train.py \
  --labels data/pokemon_sss/labels.tsv \
  --root   data/pokemon_sss \
  --out    build/models/pokemon.sss \
  --size   64 \
  --max-per-cell 32

# 3. 생성 테스트
echo "[3/3] 이미지 생성 테스트..."
mkdir -p build/sss_pokemon

PROMPTS=(
  "fire solo gen1"
  "water solo gen1"
  "grass poison gen1"
  "electric solo gen1"
  "psychic gen1_legendary"
  "dragon flying gen1"
  "dark ghost gen2"
  "steel gen7_legendary"
  "ice water gen2"
  "normal gen1"
)

for prompt in "${PROMPTS[@]}"; do
  fname="$(echo "$prompt" | tr ' ' '_').ppm"
  ./build/sss_gen build/models/pokemon.sss "$prompt" "build/sss_pokemon/$fname" 1 1.0
done

echo ""
echo "=== 완료! ==="
echo "생성 이미지: build/sss_pokemon/"
echo "모델: build/models/pokemon.sss"
