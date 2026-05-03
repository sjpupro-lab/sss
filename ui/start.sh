#!/bin/bash
# SSS Forge UI launcher: ensures binaries + demo model exist, then starts the
# Python bridge server.
set -e
cd "$(dirname "$0")/.."
PORT=${1:-8080}

# 1. Make sure the C binaries are present.
if [ ! -f build/gen_image_ce ] \
   || [ ! -f build/train_demo ] \
   || [ ! -f build/chat ] \
   || [ ! -f build/stream_train ]; then
    echo "[start.sh] building SSS engine binaries…"
    make all
    make demo_tools
    make chat
    make stream
fi

# 2. Make sure there is a demo model to play with.
mkdir -p build/models
if [ ! -f build/models/demo.ces ]; then
    echo "[start.sh] training demo model from data/demo …"
    ./build/train_demo data/demo build/models/demo
fi

# 3. Launch the UI server.
echo "[start.sh] launching UI on http://0.0.0.0:${PORT}"
exec python3 ui/server.py "$PORT"
