#!/bin/bash
# SSS Forge UI launcher: ensures binaries + demo model exist, then starts the
# Python bridge server.
#
# Usage:  ./ui/start.sh [port]            # default port 8080, binds 127.0.0.1
#         HOST=0.0.0.0 ./ui/start.sh 8080 # explicitly expose to LAN
set -e
cd "$(dirname "$0")/.."
PORT=${1:-8080}
HOST=${HOST:-127.0.0.1}

# Validate the port up front so a typo gives a clear error before we build.
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
    echo "[start.sh] error: invalid port '$PORT' (must be an integer 1..65535)" >&2
    exit 2
fi

# 1. Build (always — make is incremental, so this is cheap when up to date and
#    avoids serving a stale binary if sources changed since the last build).
echo "[start.sh] running incremental build…"
make all
make demo_tools
make chat
make stream

# 2. Make sure there is a demo model to play with.
mkdir -p build/models
if [ ! -f build/models/demo.ces ]; then
    echo "[start.sh] training demo model from data/demo …"
    ./build/train_demo data/demo build/models/demo
fi

# 3. Launch the UI server.
echo "[start.sh] launching UI on http://${HOST}:${PORT}"
exec python3 ui/server.py "$PORT" --host "$HOST"
