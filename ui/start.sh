#!/bin/bash
# SSS Unified UI — Termux 원커맨드 런처
# Usage: ./ui/start.sh [port]
#
# Phase 5 retired the legacy gen_image_ce / sss_animate generators.
# ui/server.py still exists for compatibility and now routes
# /api/generate + /api/viz-generate through scripts/sss_gen.py. The
# unified_server.py below is the default and adds the /api/sss_gen
# endpoint described in docs/migration_phase5.md.

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
