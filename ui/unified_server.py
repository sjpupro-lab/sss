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
from typing import Optional
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

# tools.sss_image_io is a hard dependency. We always need at least one
# working PNG encoder, so importing this unconditionally fails fast at
# server startup (clear ImportError) instead of degrading silently to a
# blank image at request time.
from tools.sss_image_io import numpy_to_png_data_uri as _sss_png_uri  # noqa: E402

# cv2 is optional — when present it tends to be a few times faster than
# the pure-Python encoder. Both produce *equivalent* PNGs (the decoded
# image matches), but the on-disk bytes differ because each encoder
# picks its own filter / compression strategy.
try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

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

    # Fall back to the stdlib + numpy encoder. Imported at module load,
    # so this branch always returns a real data URI.
    return _sss_png_uri(arr)


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
    """Stdlib-only multipart/form-data parser. Returns
        (files, fields)
    where `files` is a list of (field_name, filename, bytes) for every
    part that carries a filename, and `fields` is a {name: str} dict
    of every plain text part (e.g. the `label` form input).

    Avoids `cgi.FieldStorage` (deprecated in 3.13). Splits on the
    CRLF-anchored boundary `\\r\\n--<boundary>`, not a bare
    `--<boundary>`, so the literal byte sequence inside an uploaded
    binary can't truncate the file.
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
    fields = {}
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
        elif name:
            # Plain form field (e.g. `label`). Decode best-effort
            # utf-8 — browsers always send form fields as utf-8 today.
            try:
                fields[name] = data.decode("utf-8")
            except UnicodeDecodeError:
                fields[name] = data.decode("latin-1", errors="replace")
    return files, fields


def _run_ingest(filepath: str, filename: str, *,
                label: str = "", base_dir: str = ""):
    """Drive tools.sss_ingest against the lazily-seeded global pipeline's
    memory. Acquired under _PIPELINE_LOCK by the caller. `label` is the
    text the user typed in the UI; `base_dir` resolves CSV relative
    paths."""
    unified = _get_pipeline_module()
    pipeline = unified._get_pipeline()    # ensures foundation seed
    from tools import sss_ingest
    return sss_ingest.ingest_file(filepath, pipeline.memory,
                                  filename=filename,
                                  label=label, base_dir=base_dir)


def _run_restore(src_path: str, mask_path: Optional[str], *,
                 mode: str, prompt: str, strength: float, scale: int,
                 target: str) -> dict:
    """Drive tools.sss_restore. Returns
        {"src": np.ndarray, "out": np.ndarray, "mask"?: np.ndarray,
         "metrics": {"mse": float, "psnr": float, "elapsed_ms": float}}
    The mask is included in the response only for `restore` mode so the
    UI can show what region was repaired (auto-detected or supplied)."""
    import time
    from tools import sss_restore as _restore
    from tools.sss_ingest import _read_image_any
    src = _read_image_any(src_path)
    mask = None
    if mask_path is not None and os.path.exists(mask_path):
        mask_img = _read_image_any(mask_path)
        # White → damaged, dark → clean. Threshold at 127 on the mean
        # of the three channels.
        mask = (mask_img.mean(axis=2) > 127).astype(np.float32)

    rest = _restore.SSSRestore(prompt=prompt)
    t0 = time.perf_counter()
    if mode == "restore":
        if mask is None:
            mask = _restore.auto_detect_damage(src)
        out = rest.restore(src, mask)
    elif mode == "upscale":
        out = rest.upscale(src, scale=max(1, min(4, int(scale))))
    elif mode == "sharpen":
        out = rest.sharpen(src, strength=strength)
    elif mode == "denoise":
        out = rest.denoise(src, strength=strength)
    elif mode in ("color", "color_correct"):
        out = rest.color_correct(src, target_prompt=target or prompt)
    elif mode == "enhance":
        out = rest.enhance(src, prompt=prompt)
    else:
        raise ValueError(f"unsupported mode: {mode!r}")
    elapsed = (time.perf_counter() - t0) * 1000.0

    # MSE / PSNR are only well-defined when in/out share a shape (so
    # not for upscale). Report 0 / inf in that case so the UI doesn't
    # blow up trying to format `nan`.
    if out.shape == src.shape:
        diff = out.astype(np.float32) - src.astype(np.float32)
        mse  = float(np.mean(diff * diff))
        psnr = float("inf") if mse < 1e-9 else 10.0 * float(np.log10(255.0 * 255.0 / mse))
    else:
        mse  = 0.0
        psnr = 0.0

    metrics = {
        "mse":   round(mse, 2),
        "psnr":  round(psnr, 2) if psnr != float("inf") else 99.0,
        "elapsed_ms": round(elapsed, 1),
    }
    out_payload = {"src": src, "out": out, "metrics": metrics}
    if mask is not None and mode == "restore":
        # Render the mask as a grayscale BGR overlay so _png_data_uri
        # sends back something the browser can show next to the
        # before / after pair.
        m_u8 = (mask * 255.0).clip(0, 255).astype(np.uint8)
        out_payload["mask"] = np.stack([m_u8, m_u8, m_u8], axis=-1)
    return out_payload


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
        # /forge — the legacy SSS Forge UI (`ui/index.html`) calls a
        # different set of API endpoints (`/api/stream/<jobId>`,
        # `/api/job/<jobId>`, `/api/generate`, `/api/chat`, …) than the
        # unified server exposes. Serving the HTML here would load a UI
        # whose buttons all 404. Instead we return a small explainer
        # page that tells the user how to launch the legacy server and
        # links back to the unified UI.
        if path == "/forge":
            forge_html = (
                "<!DOCTYPE html>"
                "<html lang=\"ko\"><head><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
                "<title>SSS Forge UI</title>"
                "<style>body{background:#0f1117;color:#e6edf3;"
                "font-family:monospace;margin:0;padding:24px;line-height:1.6}"
                ".wrap{max-width:720px;margin:0 auto}"
                "h1{font-size:18px;margin-bottom:16px}"
                "code,pre{background:#161b22;border:1px solid #30363d;"
                "border-radius:6px;padding:2px 6px}"
                "pre{padding:12px;overflow:auto}"
                "a{color:#58a6ff;text-decoration:none}"
                "a:hover{text-decoration:underline}"
                ".note{color:#8b949e;font-size:13px;margin-top:18px}"
                "</style></head><body><div class=\"wrap\">"
                "<h1>SSS Forge UI</h1>"
                "<p>Forge UI는 별도 서버(<code>ui/server.py</code>)로 동작합니다."
                " 이 통합 서버(<code>ui/unified_server.py</code>)는 Forge가 호출하는"
                " <code>/api/stream</code>, <code>/api/job</code>, <code>/api/generate</code> 등을"
                " 구현하지 않으므로 같은 포트로는 띄울 수 없습니다.</p>"
                "<p>Forge를 띄우려면 다른 터미널에서:</p>"
                "<pre>python3 ui/server.py 8080</pre>"
                "<p>그 다음 <a href=\"http://localhost:8080/\">http://localhost:8080/</a> 를 열면 됩니다.</p>"
                "<p class=\"note\">"
                "← <a href=\"/\">SSS Unified Pipeline UI로 돌아가기</a>"
                "</p></div></body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(forge_html)))
            self.end_headers()
            self.wfile.write(forge_html)
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
                files, fields = _parse_multipart(self)
                if not files:
                    raise _BadRequest("no file part in upload")
                label = (fields.get("label") or "").strip()

                # Stage every uploaded file into one temp dir so a CSV
                # can resolve sibling images by relative path. Single-
                # file uploads still hit the same path; the dir is
                # cleaned up at the end.
                staging = tempfile.mkdtemp(prefix="sss_ingest_")
                staged_paths = []
                try:
                    for _name, filename, data in files:
                        # Sanitise: drop any directory component the
                        # browser may have sent (it shouldn't, but
                        # belt-and-braces against path traversal).
                        safe = os.path.basename(filename) or "upload.bin"
                        path_on_disk = os.path.join(staging, safe)
                        # If two uploads share a name, suffix the later
                        # ones — preserves CSV path matching.
                        if os.path.exists(path_on_disk):
                            stem, ext = os.path.splitext(safe)
                            i = 1
                            while os.path.exists(path_on_disk):
                                path_on_disk = os.path.join(
                                    staging, f"{stem}_{i}{ext}")
                                i += 1
                        with open(path_on_disk, "wb") as fp:
                            fp.write(data)
                        staged_paths.append((filename, path_on_disk, len(data)))

                    # If a .csv is part of the batch, treat the rest as
                    # sibling references for path resolution and only
                    # ingest the CSV(s). Otherwise iterate every file.
                    csv_paths = [t for t in staged_paths
                                 if t[0].lower().endswith(".csv")]
                    to_ingest = csv_paths if csv_paths else staged_paths

                    results = []
                    for filename, path_on_disk, nbytes in to_ingest:
                        try:
                            with _PIPELINE_LOCK:
                                summary = _run_ingest(
                                    path_on_disk, filename,
                                    label=label, base_dir=staging)
                            # ingest_file returns {"error": ...} for
                            # client-fault cases (missing label, etc.)
                            # without raising. Surface as not-ok.
                            if isinstance(summary, dict) and summary.get("error"):
                                results.append({
                                    "ok": False,
                                    "filename": filename,
                                    "bytes": nbytes,
                                    "error": summary["error"],
                                    "result": summary,
                                })
                            else:
                                results.append({
                                    "ok": True,
                                    "filename": filename,
                                    "bytes": nbytes,
                                    "result": summary,
                                })
                        except Exception as e:
                            results.append({
                                "ok": False,
                                "filename": filename,
                                "bytes": nbytes,
                                "error": str(e),
                            })
                finally:
                    try:
                        for _, p, _ in staged_paths:
                            try: os.unlink(p)
                            except OSError: pass
                        os.rmdir(staging)
                    except OSError:
                        pass

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
        if path == "/api/restore":
            try:
                files, fields = _parse_multipart(self)
                image_data = None
                image_filename = ""
                mask_data = None
                mask_filename = ""
                for name, filename, data in files:
                    if name == "image":
                        image_data = data
                        image_filename = filename or ""
                    elif name == "mask":
                        mask_data = data
                        mask_filename = filename or ""
                if image_data is None:
                    raise _BadRequest("no `image` part in upload")

                mode      = (fields.get("mode") or "enhance").strip().lower()
                prompt    = (fields.get("prompt") or "").strip()
                target    = (fields.get("target") or "").strip()
                strength  = float(fields.get("strength") or 0.5)
                scale     = int(fields.get("scale") or 2)

                # Stage the upload to a temp file so the loader can use
                # the same ffmpeg-aware path tools.sss_ingest uses for
                # "anything not PNG/PPM".
                staging = tempfile.mkdtemp(prefix="sss_restore_")
                try:
                    # Preserve the upload's extension so the loader can
                    # route by format (PNG/PPM via sss_image_io,
                    # everything else via ffmpeg). Default to .ppm when
                    # the browser didn't send a filename, since PPM is
                    # the only format the stdlib loader handles without
                    # the extension hint.
                    img_ext = os.path.splitext(image_filename)[1].lower() or ".png"
                    src_path = os.path.join(staging, "input" + img_ext)
                    with open(src_path, "wb") as fp:
                        fp.write(image_data)
                    mask_path = None
                    if mask_data is not None:
                        mask_ext = os.path.splitext(mask_filename)[1].lower() or ".png"
                        mask_path = os.path.join(staging, "mask" + mask_ext)
                        with open(mask_path, "wb") as fp:
                            fp.write(mask_data)

                    with _PIPELINE_LOCK:
                        result = _run_restore(
                            src_path, mask_path,
                            mode=mode, prompt=prompt,
                            strength=strength, scale=scale,
                            target=target)
                    payload = {
                        "ok": True,
                        "mode": mode,
                        "image": _png_data_uri(result["out"]),
                        "input": _png_data_uri(result["src"]),
                        "shape_in":  list(result["src"].shape),
                        "shape_out": list(result["out"].shape),
                        "metrics":   result["metrics"],
                    }
                    if "mask" in result:
                        payload["mask"] = _png_data_uri(result["mask"])
                    _json(self, payload)
                finally:
                    try:
                        for fn in os.listdir(staging):
                            try: os.unlink(os.path.join(staging, fn))
                            except OSError: pass
                        os.rmdir(staging)
                    except OSError:
                        pass
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
