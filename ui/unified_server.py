#!/usr/bin/env python3
"""SSS Unified UI server.

Small stdlib HTTP bridge for tools/sss_unified.py.
Run from repo root:
    python3 ui/unified_server.py 8090
Then open:
    http://127.0.0.1:8090
"""
from __future__ import annotations

import base64
import http.server
import json
import os
import sys
import traceback
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"
OUT_DIR = ROOT / "build" / "unified_ui"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ensure repo root import works when launched from ui/ or Termux shortcuts.
sys.path.insert(0, str(ROOT))

PIPELINE = None


def _json(handler, obj, status=200):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler):
    n = int(handler.headers.get("Content-Length", "0") or "0")
    if n <= 0:
        return {}
    raw = handler.rfile.read(n)
    return json.loads(raw.decode("utf-8"))


def _png_data_uri(img):
    """Accept numpy BGR/RGB image or path and return data:image/png URI."""
    try:
        import cv2
        import numpy as np
    except Exception:
        cv2 = None
        np = None

    if img is None:
        return None
    if isinstance(img, (str, os.PathLike)):
        p = Path(img)
        if not p.exists():
            return None
        ext = p.suffix.lower()
        data = p.read_bytes()
        if ext == ".png":
            mime = "image/png"
        elif ext in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        else:
            return None
        return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    if cv2 is None or np is None:
        return None
    arr = img
    if not hasattr(arr, "shape"):
        return None
    arr = arr.copy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _as_plain(x, depth=0):
    """Convert numpy/path-rich result into JSON-safe light summary."""
    if depth > 4:
        return str(type(x).__name__)
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in ("image", "frames", "base_image", "canvas", "result_image"):
                continue
            out[str(k)] = _as_plain(v, depth + 1)
        return out
    if isinstance(x, (list, tuple)):
        return [_as_plain(v, depth + 1) for v in x[:20]]
    if hasattr(x, "shape"):
        return {"type": "ndarray", "shape": list(x.shape), "dtype": str(x.dtype)}
    return str(x)


def _get_pipeline_module():
    import tools.sss_unified as unified
    return unified


def _run_unified(prompt: str):
    """Run tools.sss_unified with flexible API compatibility."""
    global PIPELINE
    unified = _get_pipeline_module()

    if hasattr(unified, "run_sss_pipeline"):
        return unified.run_sss_pipeline(prompt)

    if PIPELINE is None:
        if not hasattr(unified, "SSSPipeline"):
            raise RuntimeError("tools.sss_unified has no run_sss_pipeline or SSSPipeline")
        PIPELINE = unified.SSSPipeline()

    for method in ("run", "generate", "__call__"):
        fn = getattr(PIPELINE, method, None)
        if callable(fn):
            return fn(prompt)
    raise RuntimeError("SSSPipeline exists but has no run/generate/__call__")


def _normalize_result(result):
    """Make UI payload from whatever tools.sss_unified returns."""
    image = None
    frames = []
    if isinstance(result, dict):
        for k in ("image", "result", "output", "final", "canvas", "base_image"):
            if k in result:
                image = result[k]
                break
        frames = result.get("frames") or result.get("video") or []
    else:
        image = result

    image_uri = _png_data_uri(image)
    frame_uris = []
    if isinstance(frames, (list, tuple)):
        for f in frames[:12]:
            uri = _png_data_uri(f)
            if uri:
                frame_uris.append(uri)

    plain = _as_plain(result)
    scores = {}
    intent = {}
    memory = {}
    strategy = "unknown"
    if isinstance(result, dict):
        scores = result.get("scores") or result.get("score") or result.get("eval") or {}
        intent = result.get("intent") or {}
        memory = result.get("memory") or result.get("memory_stats") or {}
        strategy = result.get("strategy") or result.get("mode") or strategy

    return {
        "image": image_uri,
        "frames": frame_uris,
        "scores": _as_plain(scores),
        "intent": _as_plain(intent),
        "memory": _as_plain(memory),
        "strategy": strategy,
        "raw": plain,
    }


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[sss-unified-ui] " + fmt % args + "\n")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/unified.html"):
            p = UI_DIR / "unified.html"
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/health":
            _json(self, {"ok": True, "root": str(ROOT)})
            return
        _json(self, {"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/unified":
            try:
                body = _read_json(self)
                prompt = str(body.get("prompt") or "red smile character")
                result = _run_unified(prompt)
                payload = _normalize_result(result)
                payload["ok"] = True
                payload["prompt"] = prompt
                _json(self, payload)
            except Exception as e:
                _json(self, {
                    "ok": False,
                    "error": str(e),
                    "trace": traceback.format_exc().splitlines()[-12:],
                }, 500)
            return
        _json(self, {"error": "not found"}, 404)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    host = os.environ.get("HOST", "127.0.0.1")
    print("SSS Unified UI")
    print(f"root: {ROOT}")
    print(f"open: http://{host}:{port}")
    http.server.ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
