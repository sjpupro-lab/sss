#!/bin/bash
# run_practical_test.sh — end-to-end CANVAS training smoke test.
#
# Builds the engine, runs stream_train with a configurable clause
# budget, and points the user at the visualizer.
#
# Usage:
#   ./tools/run_practical_test.sh             # default: 1000 clauses
#   ./tools/run_practical_test.sh 5000
#   ./tools/run_practical_test.sh 25000 data/my_corpus.txt

set -e

MAX_CLAUSES="${1:-1000}"
INPUT="${2:-}"
CKPT_EVERY=$(( MAX_CLAUSES >= 5000 ? 5000 : MAX_CLAUSES / 2 ))
if [ "$CKPT_EVERY" -lt 1 ]; then CKPT_EVERY=0; fi

echo "=== CANVAS Practical Test (${MAX_CLAUSES} clauses) ==="

# 1. 빌드
echo "[1/5] Building engine + stream_train..."
make all >/dev/null
make stream >/dev/null
echo "      OK"

# 2. 데이터 선택
if [ -z "$INPUT" ]; then
    for candidate in data/kaggle_train.txt data/sample_en.txt data/sample_ko.txt; do
        if [ -f "$candidate" ]; then INPUT="$candidate"; break; fi
    done
fi
if [ ! -f "$INPUT" ]; then
    echo "ERROR: no training data found. Expected one of:"
    echo "    data/kaggle_train.txt"
    echo "    data/sample_en.txt"
    echo "  Or pass an explicit path as the second argument."
    exit 1
fi
LINES=$(wc -l < "$INPUT" | awk '{print $1}')
SIZE=$(du -h "$INPUT" | awk '{print $1}')
echo "[2/5] Data: $INPUT ($LINES lines, $SIZE)"

# 3. 기존 모델 정리 (체크포인트 보관은 명시적으로)
OUT_DIR="build/models"
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
SAVE_PATH="$OUT_DIR/practical_${MAX_CLAUSES}.spai"
echo "[3/5] Training ${MAX_CLAUSES} clauses (checkpoint every ${CKPT_EVERY})..."
./build/stream_train \
    --input "$INPUT" \
    --max "$MAX_CLAUSES" \
    --save "$SAVE_PATH" \
    --checkpoint "$CKPT_EVERY" \
    --verify

# 4. 생성된 산출물 목록
echo ""
echo "[4/5] Checkpoints in $OUT_DIR:"
ls -lh "$OUT_DIR"/*.spai 2>/dev/null | awk '{print "     ", $5, $9}'

# 5. 시각화 안내
echo ""
echo "[5/5] Done. Final model: $SAVE_PATH"
echo ""
echo "Next: run the visualizer (Pillow + numpy + ffmpeg required):"
echo "  python3 tools/visualize_training.py $OUT_DIR"
echo ""
echo "It produces:"
echo "  $OUT_DIR/viz/frame_*.png"
echo "  $OUT_DIR/training_evolution.mp4"
