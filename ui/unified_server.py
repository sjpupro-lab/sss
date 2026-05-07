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
import collections
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
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


# ── Forge bridge (ported from ui/server.py) ──────────────────
# The deprecated ui/server.py + ui/index.html UI ran a parallel HTTP
# bridge for the C engine binaries (gen_image_ce, sss_gen, train_demo,
# stream_train, chat). The unified server now hosts the same surface
# under /api/generate /api/train* /api/chat /api/models /api/upload
# /api/browse /api/job /api/stream — the UI drops the standalone Forge
# server. ui/server.py + ui/index.html move to legacy_deprecated/ so
# older docs that referenced port 8080 still find a working binary.

BUILD_DIR    = ROOT / "build"
MODELS_DIR   = BUILD_DIR / "models"
DATA_DIR     = ROOT / "data"
UPLOADS_DIR  = DATA_DIR / "uploads"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# job_id → {"proc": Popen|None, "log": [str], "done": bool,
#           "exit_code": int|None, "result": dict, "started_at": float}
# Held in-process. The original Forge server's structure 1:1 — UI code
# already speaks this shape.
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()


def _safe_under_root(p) -> Optional[Path]:
    """Resolve `p` and reject anything outside ROOT. Returns the
    resolved Path or None when `p` escapes the repo. Used by /api/browse
    so the file picker can't traverse the host filesystem."""
    try:
        target = Path(p).expanduser().resolve()
    except (OSError, ValueError):
        return None
    root = ROOT.resolve()
    try:
        target.relative_to(root)
        return target
    except ValueError:
        return None


def _safe_output_path(p, *, default_dir: Path,
                      allowed_suffixes=None) -> Optional[Path]:
    """Validate a user-supplied training-output path.

    Accepts either:
      - a bare filename (str without separators) → joined under
        `default_dir`, or
      - an absolute / relative path → resolved and required to live
        under ROOT (rejects ../ traversal and absolute paths outside
        the repo).

    `allowed_suffixes` (e.g. {".sss", ".spai", ".ces"}) is enforced when
    set so a request can't write a `.sh` over `data/`. Returns the
    resolved Path or None when the request is unsafe."""
    if not p:
        return None
    p_str = str(p).strip()
    if not p_str:
        return None
    # Bare filename ("custom.sss") → land inside default_dir. Same rule
    # as ce_storage_io: a basename never traverses, so we can skip the
    # ROOT relativisation in this branch.
    if "/" not in p_str and "\\" not in p_str:
        candidate = default_dir / p_str
    else:
        try:
            cand = Path(p_str).expanduser()
            if not cand.is_absolute():
                cand = (ROOT / cand)
            candidate = cand.resolve()
        except (OSError, ValueError):
            return None
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError:
            return None
    if allowed_suffixes is not None:
        if candidate.suffix.lower() not in allowed_suffixes:
            return None
    return candidate


def _scan_models():
    """Catalogue build/models/*.{ces,spai,sss} for the model picker."""
    out = []
    if not MODELS_DIR.exists():
        return out
    for f in sorted(MODELS_DIR.iterdir()):
        if f.suffix in (".ces", ".spai", ".sss"):
            out.append({
                "name":    f.stem,
                "type":    f.suffix[1:],
                "path":    str(f),
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return out


def _ppm_to_png_uri(path: str) -> Optional[str]:
    """Convert a P6 PPM at `path` to a `data:image/png;base64,...` URI.
    Reuses the unified server's _png_data_uri ndarray path so we share
    one PNG encoder (cv2 when present, sss_image_io otherwise)."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        magic = f.readline().strip()
        if magic not in (b"P6", b"P3"):
            return None
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        try:
            w, h = (int(t) for t in line.split())
        except ValueError:
            return None
        try:
            maxval = int(f.readline().strip())
        except ValueError:
            return None
        # PPM allows maxval up to 65535 (16-bit). The P6 binary path
        # stores 16-bit samples as two big-endian bytes, but every
        # caller in this server emits 8-bit P6 PPMs, so the 16-bit
        # branch only matters if a user uploads one — reject it
        # explicitly instead of silently returning a corrupted PNG.
        if magic == b"P6":
            if maxval > 255:
                return None
            data = f.read()
            samples = list(data[:w * h * 3])
        else:
            # P3 ASCII path: parse to ints first, rescale to 0..255,
            # *then* coerce to a bytes-compatible range. Doing
            # `bytes(int(t) ...)` directly raises on maxval>255.
            try:
                samples = [int(t) for t in f.read().split()]
            except ValueError:
                return None
            if maxval <= 0:
                return None
            if maxval != 255:
                # Round-half-up rescale so 0 stays 0 and maxval stays 255.
                scale = 255.0 / float(maxval)
                samples = [max(0, min(255, int(round(s * scale))))
                           for s in samples]
            else:
                samples = [max(0, min(255, s)) for s in samples]
            data = bytes(samples)
    if magic == b"P6" and maxval != 255:
        # P6 8-bit with non-255 maxval: rescale per byte. No overflow
        # because both sides fit in 0..255.
        scale = 255.0 / float(maxval)
        data = bytes(max(0, min(255, int(round(b * scale)))) for b in data)
    if len(data) < w * h * 3:
        return None
    arr = np.frombuffer(data[:w * h * 3], dtype=np.uint8).reshape(h, w, 3)
    return _png_data_uri(arr)


def _analyze_ppm(path: str) -> dict:
    """16×16 block-wise energy + edge maps + 16-bin RGB histograms +
    channel dominance. Used by /api/viz-generate to drive the Forge
    "Signal" tab. Pure-Python integer arithmetic; no extra deps."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        f.readline()                          # magic (P6 only path)
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = (int(t) for t in line.split())
        f.readline()                          # maxval
        data = f.read()
    arr = np.frombuffer(data[:w * h * 3], dtype=np.uint8).reshape(h, w, 3)
    bw, bh = 16, 16
    energy = np.zeros((bh, bw), dtype=np.float32)
    edge   = np.zeros((bh, bw), dtype=np.float32)
    # Proportional partition — each block covers
    #   [int(idx*size/N), int((idx+1)*size/N)).
    # Floor division (`w // 16`) would drop the rightmost / bottom
    # pixels whenever w or h aren't multiples of 16 (e.g. w=65 → 5
    # pixels never sampled). Proportional partitioning sweeps every
    # pixel exactly once and degrades gracefully for w,h < 16 too.
    for by in range(bh):
        y0 = (by * h) // bh
        y1 = ((by + 1) * h) // bh
        if y1 <= y0:
            continue
        for bx in range(bw):
            x0 = (bx * w) // bw
            x1 = ((bx + 1) * w) // bw
            if x1 <= x0:
                continue
            blk = arr[y0:y1, x0:x1]
            energy[by, bx] = float(blk.mean())
            if blk.shape[1] >= 2:
                grad = np.abs(blk[:, 1:].astype(np.int32) - blk[:, :-1].astype(np.int32))
                edge[by, bx] = float(grad.mean())
    hist_r, _ = np.histogram(arr[..., 0], bins=16, range=(0, 256))
    hist_g, _ = np.histogram(arr[..., 1], bins=16, range=(0, 256))
    hist_b, _ = np.histogram(arr[..., 2], bins=16, range=(0, 256))
    total = float(arr.shape[0] * arr.shape[1]) or 1.0
    sum_r = float(arr[..., 0].sum())
    sum_g = float(arr[..., 1].sum())
    sum_b = float(arr[..., 2].sum())
    s = sum_r + sum_g + sum_b + 1.0
    return {
        "width":  int(w),
        "height": int(h),
        "hist_r": [round(v / total, 4) for v in hist_r.tolist()],
        "hist_g": [round(v / total, 4) for v in hist_g.tolist()],
        "hist_b": [round(v / total, 4) for v in hist_b.tolist()],
        "energy_map": [[round(v, 1) for v in row] for row in energy.tolist()],
        "edge_map":   [[round(v, 1) for v in row] for row in edge.tolist()],
        "dominance":  {
            "r": round(sum_r / s, 3),
            "g": round(sum_g / s, 3),
            "b": round(sum_b / s, 3),
        },
    }


# Per-job log cap. The Forge predecessor stored every stdout line
# unbounded; long training runs (stream_train, sss_train) would eat
# memory until the server died. The deque caps in-place — older lines
# fall off the back but the SSE stream never blocks because the
# producer keeps writing.
_JOB_LOG_MAX        = 2000      # ~200 KB worst case (100B / line)
# Completed-job retention. Entries hang around long enough for a slow
# poll-only client to see the result, then get pruned. Cleanup runs on
# every _job_start so a running server with traffic stays bounded;
# idle servers naturally accumulate < _JOB_MAX_TOTAL jobs.
_JOB_TTL_SECONDS    = 60 * 60   # 1h after exit
_JOB_MAX_TOTAL      = 64        # hard cap regardless of TTL


def _prune_jobs() -> None:
    """Drop completed jobs older than _JOB_TTL_SECONDS, then enforce the
    _JOB_MAX_TOTAL cap by evicting the oldest done jobs first. Must be
    called under _JOBS_LOCK."""
    now = time.time()
    expired = [
        jid for jid, j in _JOBS.items()
        if j.get("done")
        and j.get("finished_at", j.get("started_at", now)) + _JOB_TTL_SECONDS < now
    ]
    for jid in expired:
        _JOBS.pop(jid, None)
    if len(_JOBS) > _JOB_MAX_TOTAL:
        # Sort done jobs by finished_at ascending and pop until we fit.
        # Live jobs are always preserved; if every slot is live, we let
        # the dict grow this once — caller wins.
        done_sorted = sorted(
            (jid for jid, j in _JOBS.items() if j.get("done")),
            key=lambda jid: _JOBS[jid].get("finished_at",
                                           _JOBS[jid].get("started_at", 0)),
        )
        excess = len(_JOBS) - _JOB_MAX_TOTAL
        for jid in done_sorted[:excess]:
            _JOBS.pop(jid, None)


def _job_log_append(job: dict, line: str) -> None:
    """Append one log line, bumping the monotonic counter SSE consumers
    use to detect dropped (rolled-out-of-deque) lines."""
    job["log"].append(line)
    job["log_total"] = job.get("log_total", 0) + 1


def _job_run(job_id: str, cmd, cwd=None, on_done=None) -> None:
    """Background thread: run `cmd`, append every stdout line to
    _JOBS[job_id]['log']. Survives crashes (ERROR row appended).
    `on_done` is invoked with the job dict after exit."""
    job = _JOBS[job_id]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=cwd or str(ROOT), text=True, bufsize=1,
        )
        job["proc"] = proc
        for line in proc.stdout:
            _job_log_append(job, line.rstrip("\n"))
        proc.wait()
        job["exit_code"] = proc.returncode
    except Exception as e:
        _job_log_append(job, f"[ERROR] {e}")
        job["exit_code"] = -1
    finally:
        job["finished_at"] = time.time()
        job["done"] = True
        if on_done is not None:
            try:
                on_done(job)
            except Exception as e:
                _job_log_append(job, f"[on_done ERROR] {e}")


def _job_start(cmd, cwd=None, on_done=None) -> str:
    """Allocate a job slot and kick off the worker thread. Returns the
    short hex id the UI uses for the SSE stream."""
    job_id = uuid.uuid4().hex[:8]
    with _JOBS_LOCK:
        # Garbage-collect stale jobs before adding a new slot. Cheap —
        # the dict tops out at _JOB_MAX_TOTAL.
        _prune_jobs()
        _JOBS[job_id] = {
            "proc": None,
            "log":  collections.deque(maxlen=_JOB_LOG_MAX),
            "log_total": 0,
            "done": False, "exit_code": None,
            "result": {}, "started_at": time.time(),
            "finished_at": None,
        }
    t = threading.Thread(target=_job_run,
                         args=(job_id, cmd, cwd, on_done),
                         daemon=True)
    t.start()
    return job_id


# ── Atmos: scene decompose + motion preview ───────────────────
# New production hook — exposed on /api/atmos so the UI can drive
# tools.sss_atmos.AtmosScene without re-shelling sss_animate. Returns
# the per-object summary plus N PNG-data-URI animation frames.
def _run_atmos_decompose(image_bytes: bytes, *, n_frames: int = 24,
                         frame_size: int = 256, base_id: int = 0,
                         filename: str = "") -> dict:
    """Read a PNG/PPM upload, run ce_scene_build_from_rgba, then
    render `n_frames` ticks at `frame_size` × `frame_size`.
    Returns {"objects": [...], "frames": [data_uri, ...]}."""
    n_frames  = max(0, min(120, int(n_frames)))
    frame_size = max(32, min(1024, int(frame_size)))

    # Persist upload so tools.sss_ingest._read_image_any can route
    # PNG/PPM/etc. by extension. Cleaned up at the end. Preserve the
    # original extension so the loader hits the right decoder — PNG and
    # PPM read paths are completely separate inside sss_image_io. When
    # the upload arrived without a filename (curl --data-binary, or a
    # client that forgot to set Content-Disposition: filename=) sniff
    # the magic header so we don't always default to .png and silently
    # mis-route PPM uploads.
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        head = image_bytes[:8]
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            ext = ".png"
        elif head.startswith(b"P6") or head.startswith(b"P3"):
            ext = ".ppm"
        elif head.startswith(b"\xff\xd8\xff"):
            ext = ".jpg"
        elif head[:6] in (b"GIF87a", b"GIF89a"):
            ext = ".gif"
        elif head.startswith(b"BM"):
            ext = ".bmp"
        else:
            # Last-resort: keep the historical .png fallback so the
            # error message comes out of the PNG decoder rather than
            # _read_image_any choking on an unknown extension.
            ext = ".png"
    staging = tempfile.mkdtemp(prefix="sss_atmos_")
    try:
        path = os.path.join(staging, "input" + ext)
        with open(path, "wb") as fp:
            fp.write(image_bytes)
        from tools.sss_ingest import _read_image_any
        from tools.sss_atmos import AtmosScene, MOTION_NAMES  # noqa: F401
        src = _read_image_any(path)            # (H, W, 3) uint8
    finally:
        try:
            for fn in os.listdir(staging):
                try: os.unlink(os.path.join(staging, fn))
                except OSError: pass
            os.rmdir(staging)
        except OSError:
            pass

    h, w, _ = src.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = src
    rgba[..., 3]  = 255

    objects = []
    frames  = []
    with AtmosScene.from_rgba(rgba, w, h, base_id=base_id) as scene:
        for o in scene.objects:
            objects.append({
                "id":     o.id,
                "motion": o.motion_name,
                "cx":     o.base_cx,
                "cy":     o.base_cy,
                "color":  [o.color_r, o.color_g, o.color_b],
            })
        for _ in range(n_frames):
            frame = scene.render(width=frame_size, height=frame_size)
            frames.append(_png_data_uri(frame))
            scene.tick()
    return {
        "object_count": len(objects),
        "objects":      objects,
        "frames":       frames,
        "size":         frame_size,
    }


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
    atmos_objects = None
    if isinstance(result, dict):
        scores = result.get("scores") or result.get("score") or result.get("eval") or {}
        intent = result.get("intent") or {}
        memory = result.get("memory") or result.get("memory_stats") or {}
        strategy = result.get("strategy") or result.get("mode") or strategy
        # Surface the SSS_ATMOS=1 decomposition that
        # tools.sss_unified.SSSPipeline.run attaches to the result dict
        # so the UI can render the Atmos panel without a second call.
        atmos_objects = result.get("atmos_objects")

    payload = {
        "image": image_uri,
        "frames": frame_uris,
        "scores": _as_plain(scores),
        "intent": _as_plain(intent),
        "memory": _as_plain(memory),
        "strategy": strategy,
        "raw": plain,
    }
    if atmos_objects is not None:
        payload["atmos_objects"] = _as_plain(atmos_objects)
    return payload


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

        # ── Forge: model picker + filesystem browser ─────────
        if path == "/api/models":
            _json(self, {"models": _scan_models()})
            return

        if path == "/api/browse":
            from urllib.parse import parse_qs
            qs = parse_qs(urlparse(self.path).query)
            req_dir = qs.get("dir", [str(ROOT / "data")])[0]
            target = _safe_under_root(req_dir)
            if target is None:
                _json(self, {"error": "access denied"}, 403); return
            if not target.exists() or not target.is_dir():
                _json(self, {"error": f"not a directory: {req_dir}"}, 400)
                return
            entries = []
            try:
                items = sorted(target.iterdir(),
                               key=lambda p: (not p.is_dir(), p.name.lower()))
            except PermissionError:
                _json(self, {"error": "permission denied"}, 403); return
            for item in items:
                try:
                    is_dir = item.is_dir()
                    size_kb = (round(item.stat().st_size / 1024, 1)
                               if not is_dir else 0)
                except OSError:
                    continue
                entries.append({
                    "name": item.name,
                    "is_dir": is_dir,
                    "size_kb": size_kb,
                    "rel": str(item.relative_to(ROOT)),
                })
            root = ROOT.resolve()
            parent = None
            if target != root:
                parent_path = target.parent.resolve()
                try:
                    parent_path.relative_to(root)
                    parent = str(parent_path)
                except ValueError:
                    parent = None
            _json(self, {
                "path": str(target),
                "rel_path": (str(target.relative_to(root))
                             if target != root else ""),
                "parent": parent,
                "root":   str(root),
                "entries": entries,
            })
            return

        # ── Forge: SSE log stream (and fallback /api/job/<id>) ──
        if path.startswith("/api/stream/"):
            job_id = path.split("/")[-1]
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
            if job is None:
                _json(self, {"error": "job not found"}, 404); return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            # `sent` is a logical line number, not a deque index. The
            # ring-buffer log can drop older lines, so we compare against
            # the producer's monotonic counter and only emit
            # currently-retained tail entries.
            sent = 0
            try:
                while True:
                    log_total = job.get("log_total", 0)
                    if sent < log_total:
                        snapshot = list(job["log"])
                        retained_start = log_total - len(snapshot)
                        if sent < retained_start:
                            # Lines fell off the back. Tell the client
                            # so it doesn't think the gap is a server
                            # bug; jump forward to the oldest retained.
                            dropped = retained_start - sent
                            self.wfile.write(
                                ("data: " + json.dumps(
                                    {"line": f"[…{dropped} earlier "
                                             "log line(s) dropped — "
                                             "ring buffer rolled]"}
                                ) + "\n\n").encode())
                            sent = retained_start
                        for i in range(sent - retained_start, len(snapshot)):
                            self.wfile.write(
                                ("data: " + json.dumps(
                                    {"line": snapshot[i]}) + "\n\n").encode())
                            self.wfile.flush()
                            sent += 1
                    if job["done"] and sent >= job.get("log_total", 0):
                        result = {"done": True, "exit_code": job["exit_code"]}
                        result.update(job.get("result", {}))
                        self.wfile.write(
                            f"data: {json.dumps(result)}\n\n".encode())
                        self.wfile.flush()
                        break
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass                          # client navigated away
            return

        if path.startswith("/api/job/"):
            job_id = path.split("/")[-1]
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
            if job is None:
                _json(self, {"error": "job not found"}, 404); return
            # `job["log"]` is a deque; islice gives the last 50 retained
            # lines without paying for a full list copy. log_total tells
            # the client how many lines were ever produced (for
            # reconciliation against the SSE stream).
            from itertools import islice
            log_dq = job["log"]
            tail_n = min(50, len(log_dq))
            tail = list(islice(log_dq, len(log_dq) - tail_n, len(log_dq)))
            _json(self, {
                "done":         job["done"],
                "exit_code":    job["exit_code"],
                "log":          tail,
                "log_total":    job.get("log_total", 0),
                "log_retained": len(log_dq),
                "result":       job.get("result", {}),
                "started_at":   job.get("started_at"),
                "finished_at":  job.get("finished_at"),
            })
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

        # ── Forge: image generation (sss_gen / gen_image_ce) ──
        if path == "/api/generate" or path == "/api/viz-generate":
            try:
                body = _read_json(self)
                model = body.get("model", str(MODELS_DIR / "demo.sss"))
                prompt = body.get("prompt", "red")
                seed = str(body.get("seed", 1))
                if not Path(model).exists():
                    _json(self, {"error": f"model not found: {model}"}, 400)
                    return

                out_name = f"gen_{uuid.uuid4().hex[:6]}.ppm"
                out_path = BUILD_DIR / out_name

                if model.endswith(".sss"):
                    bin_path = BUILD_DIR / "sss_gen"
                    if not bin_path.exists():
                        _json(self, {"error": "sss_gen binary not found — "
                                              "run `make sss_gen`"}, 400)
                        return
                    detail = f"{float(body.get('detail', 1.5)):.3f}"
                    steps  = str(int(body.get("steps", 120)))
                    cmd = [str(bin_path), model, prompt, str(out_path),
                           seed, detail, steps]
                    if body.get("atmos"):
                        cmd.append("--atmos")
                    if body.get("out_w"):
                        cmd += ["--out-w", str(int(body["out_w"]))]
                    if body.get("out_h"):
                        cmd += ["--out-h", str(int(body["out_h"]))]
                else:
                    bin_path = BUILD_DIR / "gen_image_ce"
                    if not bin_path.exists():
                        _json(self, {"error": "gen_image_ce binary not "
                                              "found — run `make legacy_demo` "
                                              "(legacy .ces path)"}, 400)
                        return
                    steps = str(int(body.get("steps", 50)))
                    wave  = str(int(body.get("wave_iters", 200)))
                    hybrid   = bool(body.get("hybrid", False))
                    guidance = f"{float(body.get('guidance', 1.0)):.3f}"
                    cmd = [str(bin_path), model, prompt, str(out_path),
                           seed, steps, wave]
                    if hybrid:
                        cmd += ["--hybrid", "--guidance", guidance]

                want_viz = (path == "/api/viz-generate")

                def on_done(job, _out=out_path, _viz=want_viz):
                    # Encode the PPM to PNG, run analysis if requested,
                    # then delete the on-disk PPM. The Forge predecessor
                    # left these in build/ forever — over a long-running
                    # server those add up to gigabytes. The data URI we
                    # already embedded is the durable artefact.
                    if _out.exists():
                        try:
                            job["result"]["image"] = _ppm_to_png_uri(str(_out))
                            if _viz:
                                try:
                                    job["result"]["analysis"] = _analyze_ppm(str(_out))
                                except Exception as e:
                                    job["result"]["analysis_error"] = str(e)
                        finally:
                            try:
                                _out.unlink()
                            except OSError:
                                pass
                        for line in list(job["log"]):
                            if line.startswith("wrote ") or "mean RGB" in line:
                                job["result"]["info"] = line

                job_id = _job_start(cmd, on_done=on_done)
                _json(self, {"job_id": job_id})
            except Exception as e:
                _json(self, {"error": str(e),
                             "trace": traceback.format_exc().splitlines()[-8:]
                            }, 500)
            return

        # ── Forge: training ───────────────────────────────────
        if path == "/api/train":
            try:
                body = _read_json(self)
                dataset_in  = body.get("dataset",  str(ROOT / "data" / "demo"))
                out_base_in = body.get("out_base", "demo")
                masked_epochs = int(body.get("masked_epochs", 0))
                bin_path = BUILD_DIR / "train_demo"
                if not bin_path.exists():
                    _json(self, {"error": "train_demo binary not found — "
                                          "run `make legacy_demo`"}, 400)
                    return
                # Path safety: refuse to read datasets or write outputs
                # outside the repo root — a malformed `out_base` could
                # otherwise overwrite arbitrary files when the server is
                # bound to a non-local interface.
                dataset_p = _safe_under_root(dataset_in)
                if dataset_p is None or not dataset_p.exists():
                    _json(self, {"error": "dataset must be a path under "
                                          f"{ROOT}"}, 400)
                    return
                out_base_p = _safe_output_path(
                    out_base_in, default_dir=MODELS_DIR)
                if out_base_p is None:
                    _json(self, {"error": "out_base must be a basename or "
                                          "a path under the repo root"}, 400)
                    return
                out_base_p.parent.mkdir(parents=True, exist_ok=True)
                cmd = [str(bin_path), str(dataset_p), str(out_base_p)]
                if masked_epochs > 0:
                    cmd += ["--masked-epochs", str(masked_epochs)]

                def on_done(job, _base=str(out_base_p)):
                    ces  = Path(_base + ".ces")
                    spai = Path(_base + ".spai")
                    job["result"]["ces_exists"]  = ces.exists()
                    job["result"]["spai_exists"] = spai.exists()
                    if ces.exists():
                        job["result"]["ces_size_kb"] = round(
                            ces.stat().st_size / 1024, 1)

                _json(self, {"job_id": _job_start(cmd, on_done=on_done)})
            except Exception as e:
                _json(self, {"error": str(e)}, 500)
            return

        if path == "/api/train-sss":
            try:
                body = _read_json(self)
                labels_in    = body.get("labels")
                root_dir_in  = body.get("root")
                out_model_in = body.get("out", "custom.sss")
                size = str(int(body.get("size", 64)))
                if not labels_in or not root_dir_in:
                    _json(self, {"error": "labels and root are required"}, 400)
                    return
                # Same path-safety pattern as /api/train above. The
                # spec-train script is also invoked locally so we want
                # `labels` and `root` to live inside the repo too.
                labels_p = _safe_under_root(labels_in)
                if labels_p is None or not labels_p.exists():
                    _json(self, {"error": f"labels must be a path under "
                                          f"{ROOT}"}, 400)
                    return
                root_p = _safe_under_root(root_dir_in)
                if root_p is None or not root_p.is_dir():
                    _json(self, {"error": f"root must be a directory under "
                                          f"{ROOT}"}, 400)
                    return
                out_model_p = _safe_output_path(
                    out_model_in, default_dir=MODELS_DIR,
                    allowed_suffixes={".sss"})
                if out_model_p is None:
                    # Tolerate users typing "name" (no extension) by
                    # auto-appending .sss before validating.
                    fallback = (out_model_in if out_model_in.endswith(".sss")
                                else out_model_in + ".sss")
                    out_model_p = _safe_output_path(
                        fallback, default_dir=MODELS_DIR,
                        allowed_suffixes={".sss"})
                if out_model_p is None:
                    _json(self, {"error": "out must be a *.sss basename or "
                                          "a path under the repo root"}, 400)
                    return
                out_model_p.parent.mkdir(parents=True, exist_ok=True)
                out_model = str(out_model_p)
                cmd = [
                    "python3", str(ROOT / "scripts" / "sss_train.py"),
                    "--labels", str(labels_p),
                    "--root",   str(root_p),
                    "--out",    out_model,
                    "--size",   size,
                ]

                def on_done(job, _path=out_model):
                    p = Path(_path)
                    job["result"]["sss_exists"] = p.exists()
                    if p.exists():
                        job["result"]["sss_size_kb"] = round(
                            p.stat().st_size / 1024, 1)
                        job["result"]["sss_path"] = str(p)

                _json(self, {"job_id": _job_start(cmd, on_done=on_done)})
            except Exception as e:
                _json(self, {"error": str(e)}, 500)
            return

        if path == "/api/train-text":
            try:
                body = _read_json(self)
                input_in    = body.get("input",
                                       str(ROOT / "data" / "wiki5k.txt"))
                max_lines   = str(int(body.get("max", 5000)))
                out_model_in = body.get("out", "text_model.spai")
                bin_path = BUILD_DIR / "stream_train"
                if not bin_path.exists():
                    _json(self, {"error": "stream_train binary not found — "
                                          "run `make stream`"}, 400)
                    return
                # Path-safety: input corpus must be under the repo;
                # output must land in MODELS_DIR (basename) or under the
                # repo root, with a .spai suffix.
                input_p = _safe_under_root(input_in)
                if input_p is None or not input_p.exists():
                    _json(self, {"error": f"input must be a path under "
                                          f"{ROOT}"}, 400)
                    return
                out_p = _safe_output_path(
                    out_model_in, default_dir=MODELS_DIR,
                    allowed_suffixes={".spai"})
                if out_p is None:
                    fallback = (out_model_in if out_model_in.endswith(".spai")
                                else out_model_in + ".spai")
                    out_p = _safe_output_path(
                        fallback, default_dir=MODELS_DIR,
                        allowed_suffixes={".spai"})
                if out_p is None:
                    _json(self, {"error": "out must be a *.spai basename or "
                                          "a path under the repo root"}, 400)
                    return
                out_p.parent.mkdir(parents=True, exist_ok=True)
                cmd = [str(bin_path), "--input", str(input_p),
                       "--max", max_lines, "--save", str(out_p), "--verify"]
                _json(self, {"job_id": _job_start(cmd)})
            except Exception as e:
                _json(self, {"error": str(e)}, 500)
            return

        # ── Forge: chat (single-turn) ─────────────────────────
        if path == "/api/chat":
            try:
                body = _read_json(self)
                model   = body.get("model", "")
                message = body.get("message", "")
                bin_path = BUILD_DIR / "chat"
                if not bin_path.exists():
                    _json(self, {"error": "chat binary not found — "
                                          "run `make chat`"}, 400)
                    return
                if not model or not Path(model).exists():
                    _json(self, {"error": "model path required"}, 400); return
                # Single-turn: pipe message + :q. The Forge UI treats this
                # as one request → one reply, so we don't keep state.
                input_text = f"{message}\n:q\n"
                result = subprocess.run(
                    [str(bin_path), "--load", model],
                    input=input_text, capture_output=True,
                    text=True, timeout=30, cwd=str(ROOT),
                )
                lines = result.stdout.strip().split("\n")
                response_lines = []
                capture = False
                for line in lines:
                    # The chat REPL echoes the user prompt back inline; flip
                    # `capture` after we see it so we only return the reply.
                    if ">" in line and message in line:
                        capture = True; continue
                    if (capture and not line.startswith("[chat]")
                            and line.strip() != ":q"):
                        if line.strip():
                            response_lines.append(line)
                _json(self, {
                    "response": ("\n".join(response_lines)
                                 if response_lines else "(no response)"),
                    "raw":      lines[-20:],
                })
            except subprocess.TimeoutExpired:
                _json(self, {"error": "timeout"}, 500)
            except Exception as e:
                _json(self, {"error": str(e)}, 500)
            return

        # ── Forge: dataset upload (multipart, save to data/uploads) ──
        if path == "/api/upload":
            try:
                files, fields = _parse_multipart(self)
                session = "sess_" + uuid.uuid4().hex[:8]
                session_name = (fields.get("session") or "").strip()
                if session_name:
                    safe = "".join(c for c in session_name
                                   if c.isalnum() or c in "._-")[:40]
                    if safe:
                        session = safe
                target_dir = UPLOADS_DIR / session
                target_dir.mkdir(parents=True, exist_ok=True)
                saved = []
                labels_path = None
                for name, fname, data in files:
                    if not fname:
                        continue
                    fname = Path(fname).name
                    if not fname:
                        continue
                    dst = target_dir / fname
                    with open(dst, "wb") as f:
                        f.write(data)
                    saved.append({"name": fname, "size": len(data)})
                    if (name == "labels" or fname.endswith(".tsv")
                            or fname == "labels.txt"):
                        labels_path = str(dst)
                _json(self, {
                    "dir":        str(target_dir),
                    "rel_dir":    str(target_dir.relative_to(ROOT)),
                    "labels":     labels_path,
                    "rel_labels": (str(Path(labels_path).relative_to(ROOT))
                                   if labels_path else None),
                    "files":      saved,
                })
            except _PayloadTooLarge as e:
                _json(self, {"error": str(e)}, 413)
            except _BadRequest as e:
                _json(self, {"error": str(e)}, 400)
            except Exception as e:
                _json(self, {"error": str(e)}, 500)
            return

        # ── Atmos: scene decompose + motion preview (NEW) ─────
        if path == "/api/atmos":
            try:
                files, fields = _parse_multipart(self)
                if not files:
                    raise _BadRequest("no `image` part in upload")
                # Take the first uploaded file regardless of field name —
                # the UI uses `image`, but cURL examples often hand-roll
                # the form and forget to set it.
                _name, _filename, data = files[0]
                n_frames   = int(fields.get("n_frames")   or 24)
                frame_size = int(fields.get("frame_size") or 256)
                base_id    = int(fields.get("base_id")    or 0)
                result = _run_atmos_decompose(
                    data, n_frames=n_frames,
                    frame_size=frame_size, base_id=base_id,
                    filename=_filename or "")
                result["ok"] = True
                _json(self, result)
            except _BadRequest as e:
                _json(self, {"ok": False, "error": str(e)}, 400)
            except _PayloadTooLarge as e:
                _json(self, {"ok": False, "error": str(e)}, 413)
            except Exception as e:
                _json(self, {
                    "ok": False,
                    "error": str(e),
                    "trace": traceback.format_exc().splitlines()[-8:],
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
