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
import tempfile
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

# cv2 is optional. When present we use it for PNG encoding (it's a few
# times faster than the pure-Python encoder); when missing we fall back
# to sss_image_io. Both produce identical PNG bytes from the same input.
try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    from tools.sss_image_io import numpy_to_png_data_uri as _sss_png_uri
except ImportError:
    _sss_png_uri = None

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

    - numpy ndarray → encoded as PNG (`data:image/png;base64,...`),
      via cv2 when available else tools/sss_image_io.
    - existing .png path → passed through as `data:image/png;base64,...`.
    - existing .jpg/.jpeg path → passed through as `data:image/jpeg;base64,...`.
    Anything else (None, missing path, unsupported extension, non-image
    object) returns None."""
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
    # Coerce to uint8 — cv2.imencode and sss_image_io both want uint8.
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Prefer cv2 when present (faster encoder).
    if cv2 is not None:
        ok, buf = cv2.imencode(".png", arr)
        if ok:
            return ("data:image/png;base64,"
                    + base64.b64encode(buf.tobytes()).decode("ascii"))

    # Fall back to the stdlib + numpy encoder.
    if _sss_png_uri is not None:
        return _sss_png_uri(arr)

    return None


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


# ── ingest plumbing ──────────────────────────────────────────
INGEST_MAX_BYTES = 50 * 1024 * 1024     # 50 MB hard cap


class _BadRequest(ValueError):
    """Client sent a malformed request — translate to HTTP 400."""


class _PayloadTooLarge(ValueError):
    """Body exceeds the configured cap — translate to HTTP 413."""


def _parse_multipart(handler, max_size: int = INGEST_MAX_BYTES):
    """Stdlib-only multipart/form-data parser. Returns a list of
    (field_name, filename, bytes) for every part that carries a file.
    Avoids `cgi.FieldStorage` (deprecated in 3.13).

    Splits on the CRLF-anchored boundary `\\r\\n--<boundary>`, not on
    a bare `--<boundary>`, so the literal byte sequence appearing
    inside an uploaded binary can't truncate the file.
    """
    ct = handler.headers.get("Content-Type", "")
    if not ct.lower().startswith("multipart/form-data"):
        raise _BadRequest("Content-Type must be multipart/form-data")
    # Pull the boundary out of the Content-Type header.
    boundary = None
    for token in ct.split(";"):
        token = token.strip()
        if token.lower().startswith("boundary="):
            boundary = token.split("=", 1)[1].strip().strip('"')
            break
    if not boundary:
        raise _BadRequest("multipart: missing boundary")

    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        raise _BadRequest("multipart: empty body")
    if length > max_size:
        raise _PayloadTooLarge(
            f"upload too large ({length} bytes; max {max_size})")
    body = handler.rfile.read(length)

    # RFC 7578: every boundary line is preceded by CRLF except the very
    # first one (which sits at byte 0). Prepend CRLF so all delimiters
    # share the same shape, then split on the canonical \r\n--boundary.
    delim = b"\r\n--" + boundary.encode("ascii")
    parts = (b"\r\n" + body).split(delim)

    files = []
    # parts[0] is the preamble (typically empty), discard.
    for chunk in parts[1:]:
        if chunk.startswith(b"--"):
            break                         # closing delimiter --boundary--
        if not chunk.startswith(b"\r\n"):
            continue                      # malformed; skip
        chunk = chunk[2:]                 # strip the leading CRLF
        sep = chunk.find(b"\r\n\r\n")
        if sep < 0:
            continue
        hdr_block = chunk[:sep].decode("latin-1", errors="replace")
        # Split consumed the trailing \r\n that separates the body from
        # the next boundary, so `data` is exact — no further trimming.
        data = chunk[sep + 4:]

        # We only care about parts with a `filename=` (i.e. real files).
        name = None; filename = None
        for line in hdr_block.split("\r\n"):
            if not line.lower().startswith("content-disposition:"):
                continue
            for piece in line.split(";"):
                piece = piece.strip()
                if piece.startswith("name="):
                    name = piece[5:].strip().strip('"')
                elif piece.startswith("filename="):
                    filename = piece[9:].strip().strip('"')
        if filename:
            files.append((name, filename, data))
    return files


def _run_ingest(filepath: str, filename: str):
    """Drive tools.sss_ingest against the lazily-seeded global pipeline's
    memory. Acquired under _PIPELINE_LOCK by the caller."""
    unified = _get_pipeline_module()
    pipeline = unified._get_pipeline()    # ensures foundation seed
    from tools import sss_ingest
    return sss_ingest.ingest_file(filepath, pipeline.memory,
                                  filename=filename)


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
        # Serve the legacy SSS Forge UI on /forge for users who still
        # want it. Returns 404 if ui/index.html isn't shipped (it's
        # part of the older binary-driven flow and may be removed).
        if path == "/forge":
            p = UI_DIR / "index.html"
            if not p.exists():
                _json(self, {"error": "ui/index.html not found"}, 404)
                return
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
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
        if path == "/api/ingest":
            try:
                files = _parse_multipart(self)
                if not files:
                    raise _BadRequest("no file part in upload")
                # We accept multiple files in one POST but the UI sends
                # one per request, so this loop usually has length 1.
                results = []
                for _name, filename, data in files:
                    # Persist to a temp path so ingest can use ffmpeg/path-
                    # based decoders unchanged. Clean up after.
                    suffix = os.path.splitext(filename)[1] or ".bin"
                    fd, tmp_path = tempfile.mkstemp(suffix=suffix,
                                                   prefix="sss_ingest_")
                    try:
                        with os.fdopen(fd, "wb") as fp:
                            fp.write(data)
                        with _PIPELINE_LOCK:
                            summary = _run_ingest(tmp_path, filename)
                        results.append({
                            "ok": True,
                            "filename": filename,
                            "bytes": len(data),
                            "result": summary,
                        })
                    except Exception as e:
                        results.append({
                            "ok": False,
                            "filename": filename,
                            "error": str(e),
                        })
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                # If a single file was sent, flatten the response so the
                # UI can read `result` / `error` directly.
                if len(results) == 1:
                    _json(self, {**results[0]})
                else:
                    _json(self, {"ok": True, "files": results})
            except _PayloadTooLarge as e:
                _json(self, {"ok": False, "error": str(e)}, 413)
            except _BadRequest as e:
                _json(self, {"ok": False, "error": str(e)}, 400)
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
