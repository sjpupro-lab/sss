#!/bin/bash
# SSS Unified UI — Termux 원커맨드 런처
# Usage: ./ui/start.sh [port]
#
# Old binary-backed server (gen_image_ce / train_demo / chat) lives in
# ui/server.py and is no longer the default. Launch it directly with
# `python3 ui/server.py <port>` if you still need it.

set -e
cd "$(dirname "$0")/.."  # cd to sss/ root

PORT=${1:-8090}

echo "╔═══════════════════════════════════════╗"
echo "║      SSS Unified Pipeline Launcher    ║"
echo "╚═══════════════════════════════════════╝"
echo ""
echo "  Open in browser:"
echo "    http://localhost:$PORT"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

exec python3 ui/unified_server.py "$PORT"
