#!/bin/bash
# SSS Forge UI — Termux 원커맨드 런처
# Usage: ./ui/start.sh [port]

set -e
cd "$(dirname "$0")/.."  # cd to sss/ root

PORT=${1:-8080}

echo "╔═══════════════════════════════════════╗"
echo "║          SSS Forge Launcher           ║"
echo "╚═══════════════════════════════════════╝"
echo ""

# ── 1. 빌드 확인 ──────────────────────────────
if [ ! -f build/gen_image_ce ] || [ ! -f build/train_demo ] || [ ! -f build/chat ]; then
    echo "[1/3] Building SSS engine..."
    make all
    make demo_tools
    make chat
    make stream
    echo "      Build complete."
else
    echo "[1/3] Engine binaries found."
fi

# ── 2. 데모 모델 확인 ─────────────────────────
mkdir -p build/models
if [ ! -f build/models/demo.ces ]; then
    echo "[2/3] Training demo model..."
    ./build/train_demo data/demo build/models/demo
    echo "      Demo model ready."
else
    echo "[2/3] Demo model found."
fi

# ── 3. 서버 시작 ──────────────────────────────
echo "[3/3] Starting server on port $PORT..."
echo ""
echo "  Open in browser:"
echo "    http://localhost:$PORT"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

python3 ui/server.py $PORT
