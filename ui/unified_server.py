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
import threading
import traceback
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"
OUT_DIR = ROOT / "build" / "unified_ui"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ensure repo root import works when launched from ui/ or Termux shortcuts.
sys.path.insert(0, str(ROOT))

# Stdlib-only PNG encoder lives in tools/. No cv2 / Pillow dependency.
# numpy is a hard requirement (tools.sss_unified and tools.sss_image_io
# both import it unconditionally), so don't pretend otherwise here.
import numpy as np  # noqa: E402
from tools.sss_image_io import numpy_to_png_data_uri  # noqa: E402

PIPELINE = None

# ThreadingHTTPServer can deliver overlapping requests, but the
# tools.sss_unified pipeline keeps a process-wide CEMemory instance and
# is not thread-safe. Serialise every pipeline call through this lock
# so /api/unified and /api/upgrade can't race each other.
_PIPELINE_LOCK = threading.Lock()


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
    """Return a data: URI for a numpy image or a file path.

    - numpy ndarray → encoded as PNG (`data:image/png;base64,...`).
    - existing .png path → passed through as `data:image/png;base64,...`.
    - existing .jpg/.jpeg path → passed through as `data:image/jpeg;base64,...`.
    Anything else (None, missing path, unsupported extension, non-image
    object) returns None.

    PNG encoding is handled by tools/sss_image_io and uses only stdlib
    + numpy — no cv2, no Pillow."""
    if img is None:
        return None

    # File-path branch: pass the file's own bytes through. The MIME
    # reflects what's actually on disk so we don't claim PNG for a JPEG.
    if isinstance(img, (str, os.PathLike)):
        p = Path(img)
        if not p.exists():
            return None
        ext = p.suffix.lower()
        if ext == ".png":
            mime = "image/png"
        elif ext in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        else:
            return None
        data = p.read_bytes()
        return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")

    # ndarray branch.
    arr = np.asarray(img)
    if arr.ndim < 2:
        return None
    return numpy_to_png_data_uri(arr)


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


def _merge_tuple_result(out):
    """Pipeline returns (result_dict, image, frames). Fold image/frames
    into the dict so _normalize_result can find them on the usual keys."""
    if isinstance(out, tuple):
        result = out[0] if len(out) > 0 else {}
        image = out[1] if len(out) > 1 else None
        frames = out[2] if len(out) > 2 else None
        if isinstance(result, dict):
            if image is not None:
                result["image"] = image
            if frames is not None:
                result["frames"] = frames
        return result
    return out


def _run_unified(prompt: str):
    """Run tools.sss_unified with flexible API compatibility."""
    global PIPELINE
    unified = _get_pipeline_module()

    if hasattr(unified, "run_sss_pipeline"):
        return _merge_tuple_result(unified.run_sss_pipeline(prompt))

    if PIPELINE is None:
        if not hasattr(unified, "SSSPipeline"):
            raise RuntimeError("tools.sss_unified has no run_sss_pipeline or SSSPipeline")
        PIPELINE = unified.SSSPipeline()

    for method in ("run", "generate", "__call__"):
        fn = getattr(PIPELINE, method, None)
        if callable(fn):
            return _merge_tuple_result(fn(prompt))
    raise RuntimeError("SSSPipeline exists but has no run/generate/__call__")


def _run_upgrade(cycles: int):
    unified = _get_pipeline_module()
    if not hasattr(unified, "run_upgrade_loop"):
        raise RuntimeError("tools.sss_unified has no run_upgrade_loop")
    return unified.run_upgrade_loop(int(cycles))


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
                with _PIPELINE_LOCK:
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
        if path == "/api/upgrade":
            try:
                body = _read_json(self)
                cycles = int(body.get("cycles") or 5)
                cycles = max(1, min(50, cycles))
                with _PIPELINE_LOCK:
                    report = _run_upgrade(cycles)
                _json(self, {"ok": True, "cycles": cycles, "report": report})
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
