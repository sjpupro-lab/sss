#!/usr/bin/env python3
"""
SSS Forge UI bridge server.

Runs the SSS C binaries (gen_image_ce, train_demo, stream_train, chat) as
subprocesses on demand and exposes them through a small HTTP/JSON API plus
Server-Sent Events for live stdout streaming. Pure Python 3 stdlib — no
external packages, no Pillow. PPMs are encoded to PNG inline using zlib.
"""

import argparse
import base64
import http.server
import json
import os
import re
import socketserver
import struct
import subprocess
import sys
import threading
import time
import uuid
import zlib
from collections import OrderedDict
from urllib.parse import urlparse

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD_DIR  = os.path.join(ROOT, "build")
MODELS_DIR = os.path.join(BUILD_DIR, "models")
UI_DIR     = os.path.join(ROOT, "ui")
TMP_DIR    = os.path.join(BUILD_DIR, "ui_tmp")
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# job_id -> dict(proc, log[list[str]], done, exit_code, result, cv)
# Bounded LRU: when we exceed JOBS_MAX, the oldest *finished* job is evicted.
JOBS      = OrderedDict()
JOBS_LOCK = threading.Lock()
JOBS_MAX  = 64


# ─────────────────────────────────────────────────────────────────────────────
# PPM → PNG (pure stdlib)
# ─────────────────────────────────────────────────────────────────────────────

def _read_ppm(path):
    """Parse a P6 PPM. Returns (w, h, rgb_bytes). Raises on invalid input."""
    with open(path, "rb") as f:
        data = f.read()
    # Tokenize header — supports comments and arbitrary whitespace.
    i = 0
    def skip_ws_and_comments():
        nonlocal i
        while i < len(data):
            c = data[i:i+1]
            if c in (b" ", b"\t", b"\n", b"\r"):
                i += 1
            elif c == b"#":
                while i < len(data) and data[i:i+1] != b"\n":
                    i += 1
            else:
                return
    def next_token():
        nonlocal i
        skip_ws_and_comments()
        start = i
        while i < len(data) and data[i:i+1] not in (b" ", b"\t", b"\n", b"\r"):
            i += 1
        return data[start:i]
    magic = next_token()
    if magic != b"P6":
        raise ValueError("only P6 PPM supported, got %r" % magic)
    w = int(next_token())
    h = int(next_token())
    maxv = int(next_token())
    # exactly one whitespace byte after maxval per spec
    i += 1
    pixels = data[i:i + w * h * 3]
    if len(pixels) != w * h * 3:
        raise ValueError("truncated PPM body: have %d need %d" % (len(pixels), w*h*3))
    if maxv != 255:
        # rescale to 8-bit
        scale = 255.0 / maxv
        pixels = bytes(int(b * scale) & 0xff for b in pixels)
    return w, h, pixels


def _png_chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))


def ppm_to_png_bytes(path):
    """Encode a P6 PPM at `path` to PNG bytes."""
    w, h, rgb = _read_ppm(path)
    # Pre-pend filter byte (0 = None) per scanline.
    stride = w * 3
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw.extend(rgb[y * stride:(y + 1) * stride])
    compressed = zlib.compress(bytes(raw), 9)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (sig
            + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", compressed)
            + _png_chunk(b"IEND", b""))


def ppm_to_data_uri(path):
    return "data:image/png;base64," + base64.b64encode(ppm_to_png_bytes(path)).decode("ascii")


# ─────────────────────────────────────────────────────────────────────────────
# PPM analysis (energy / edge / histograms / dominance)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_ppm(path, grid=16):
    """Compute lightweight visualization stats for a PPM file."""
    w, h, rgb = _read_ppm(path)
    bx = max(1, w // grid)
    by = max(1, h // grid)
    energy = [[0.0] * grid for _ in range(grid)]
    edges  = [[0.0] * grid for _ in range(grid)]
    hist_r = [0] * 16
    hist_g = [0] * 16
    hist_b = [0] * 16
    sum_r = sum_g = sum_b = 0

    # Per-pixel scan. 256x256 = 65k px, fine in pure Python.
    stride = w * 3
    for y in range(h):
        row = rgb[y * stride:(y + 1) * stride]
        gy = min(grid - 1, y // by)
        prev_lum = None
        for x in range(w):
            r = row[x * 3]
            g = row[x * 3 + 1]
            b = row[x * 3 + 2]
            sum_r += r; sum_g += g; sum_b += b
            hist_r[r >> 4] += 1
            hist_g[g >> 4] += 1
            hist_b[b >> 4] += 1
            lum = (r + g + b) / 3.0
            gx = min(grid - 1, x // bx)
            energy[gy][gx] += lum
            if prev_lum is not None:
                edges[gy][gx] += abs(lum - prev_lum)
            prev_lum = lum

    cell_px = float(bx * by)
    for gy in range(grid):
        for gx in range(grid):
            energy[gy][gx] = energy[gy][gx] / cell_px / 255.0
            edges[gy][gx]  = edges[gy][gx]  / cell_px / 255.0

    # normalize energy & edges to [0,1] across the grid (helps visual contrast)
    def _norm(m):
        flat = [v for row in m for v in row]
        lo, hi = min(flat), max(flat)
        if hi - lo < 1e-9:
            return [[0.0] * grid for _ in range(grid)]
        return [[(v - lo) / (hi - lo) for v in row] for row in m]
    energy = _norm(energy)
    edges  = _norm(edges)

    total_h = float(sum(hist_r)) or 1.0
    hist_r = [c / total_h for c in hist_r]
    hist_g = [c / total_h for c in hist_g]
    hist_b = [c / total_h for c in hist_b]

    total_rgb = float(sum_r + sum_g + sum_b) or 1.0
    dom = {
        "r": sum_r / total_rgb,
        "g": sum_g / total_rgb,
        "b": sum_b / total_rgb,
    }

    return {
        "energy_map": energy,
        "edge_map":   edges,
        "hist_r":     hist_r,
        "hist_g":     hist_g,
        "hist_b":     hist_b,
        "dominance":  dom,
        "size":       [w, h],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Job orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _new_job():
    jid = uuid.uuid4().hex[:12]
    job = {
        "id":        jid,
        "proc":      None,
        "log":       [],
        "done":      False,
        "exit_code": None,
        "result":    {},
        "cv":        threading.Condition(),
        "started":   time.time(),
    }
    with JOBS_LOCK:
        JOBS[jid] = job
        # Evict the oldest *finished* job if we're over the cap. Running jobs
        # are kept so streaming clients don't lose their handle mid-flight.
        while len(JOBS) > JOBS_MAX:
            for old_id, old_job in list(JOBS.items()):
                if old_job["done"]:
                    JOBS.pop(old_id, None)
                    break
            else:
                break  # nothing finished yet — nothing to evict
    return job


def _run_job(cmd, cwd=None, stdin_data=None, on_done=None, env=None):
    """Spawn a subprocess and capture its stdout/stderr line-by-line."""
    job = _new_job()

    def reader():
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd or ROOT,
                stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            job["proc"] = proc
            if stdin_data is not None:
                try:
                    proc.stdin.write(stdin_data)
                    proc.stdin.close()
                except Exception:
                    pass
            for line in proc.stdout:
                line = line.rstrip("\n")
                with job["cv"]:
                    job["log"].append(line)
                    job["cv"].notify_all()
            proc.wait()
            job["exit_code"] = proc.returncode
        except Exception as e:
            with job["cv"]:
                job["log"].append("[server] error: %s" % e)
                job["exit_code"] = -1
                job["cv"].notify_all()
        finally:
            try:
                if on_done is not None:
                    on_done(job)
            except Exception as e:
                with job["cv"]:
                    job["log"].append("[server] on_done error: %s" % e)
            with job["cv"]:
                job["done"] = True
                job["cv"].notify_all()

    threading.Thread(target=reader, daemon=True).start()
    return job


# ─────────────────────────────────────────────────────────────────────────────
# Log parsing helpers (gen_image_ce)
# ─────────────────────────────────────────────────────────────────────────────

_RE_MORPH    = re.compile(r"\[gen_image_ce\] morphemes=\d+\s+(.*)$")
_RE_CANVAS   = re.compile(r"routed canvas_id=0x([0-9a-fA-F]+)")
_RE_MEAN     = re.compile(r"mean RGB = \(([\d.\-]+),\s*([\d.\-]+),\s*([\d.\-]+)\)")
_RE_PROMPT   = re.compile(r'prompt="([^"]*)"\s+seed=0x([0-9a-fA-F]+)\s+steps=(\d+)\s+wave_iters=(\d+)')


def parse_gen_log(lines):
    out = {
        "morphemes": [],
        "canvas_id": None,
        "mean_rgb":  None,
        "prompt":    None,
        "seed":      None,
        "steps":     None,
        "wave_iters": None,
    }
    for ln in lines:
        m = _RE_MORPH.search(ln)
        if m:
            tokens = m.group(1).strip().split()
            morphs = []
            for t in tokens:
                if "/" in t:
                    surf, pos = t.split("/", 1)
                else:
                    surf, pos = t, "?"
                morphs.append({"surface": surf, "pos": pos})
            out["morphemes"] = morphs
        m = _RE_CANVAS.search(ln)
        if m:
            out["canvas_id"] = "0x" + m.group(1)
        m = _RE_MEAN.search(ln)
        if m:
            out["mean_rgb"] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
        m = _RE_PROMPT.search(ln)
        if m:
            out["prompt"] = m.group(1)
            out["seed"]   = int(m.group(2), 16)
            out["steps"]  = int(m.group(3))
            out["wave_iters"] = int(m.group(4))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HTTP handler
# ─────────────────────────────────────────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "SSSForgeUI/0.1"

    # Quieter logs.
    def log_message(self, fmt, *args):
        sys.stderr.write("[ui] %s - %s\n" % (self.address_string(), fmt % args))

    # ── helpers ────────────────────────────────────────────────────────────
    def _cors(self):
        # Same-origin only by default. The UI is served from the same origin
        # so it doesn't need CORS at all; emitting `*` here would let any
        # other website drive-by-request our local execution endpoints.
        # If the operator wants cross-origin access they can pass --cors *.
        origin_policy = getattr(self.server, "cors_origin", None)
        if origin_policy:
            self.send_header("Access-Control-Allow-Origin", origin_policy)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self.send_error(404, "not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        """Returns (body_dict, error_response_or_None). Callers should bail if
        an error response was already produced."""
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            return {}, None
        raw = self.rfile.read(n)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send_json({"error": "invalid JSON: %s" % e}, 400)
            return None, True
        if not isinstance(obj, dict):
            self._send_json({"error": "JSON body must be an object"}, 400)
            return None, True
        return obj, None

    # ── routing ────────────────────────────────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        p = url.path
        if p == "/" or p == "/index.html":
            return self._send_file(os.path.join(UI_DIR, "index.html"), "text/html; charset=utf-8")
        if p == "/api/models":
            return self._send_json({"models": list_models()})
        if p == "/api/health":
            return self._send_json({"ok": True, "root": ROOT})
        if p.startswith("/api/job/"):
            jid = p.rsplit("/", 1)[-1]
            return self._send_job(jid)
        if p.startswith("/api/stream/"):
            jid = p.rsplit("/", 1)[-1]
            return self._stream_job(jid)
        return self.send_error(404, "not found")

    def do_POST(self):
        url  = urlparse(self.path)
        p    = url.path
        body, err = self._read_json()
        if err:
            return  # error response already sent
        if p == "/api/generate":
            return self._post_generate(body, with_viz=False)
        if p == "/api/viz-generate":
            return self._post_generate(body, with_viz=True)
        if p == "/api/train":
            return self._post_train(body)
        if p == "/api/train-text":
            return self._post_train_text(body)
        if p == "/api/chat":
            return self._post_chat(body)
        return self.send_error(404, "not found")

    # ── action handlers ────────────────────────────────────────────────────
    def _post_generate(self, body, with_viz):
        model      = body.get("model") or ""
        prompt     = body.get("prompt") or ""
        seed       = int(body.get("seed", 0))
        steps      = int(body.get("steps", 50))
        wave_iters = int(body.get("wave_iters", 200))
        guidance   = float(body.get("guidance", 1.0))
        hybrid     = bool(body.get("hybrid", False))

        model_path = _resolve_model(model)
        if not model_path:
            return self._send_json({"error": "model not found in build/models/: %s" % model}, 400)
        if not prompt:
            return self._send_json({"error": "prompt is required"}, 400)

        out_path = os.path.join(TMP_DIR, "gen_%s.ppm" % uuid.uuid4().hex[:10])
        cmd = [
            os.path.join(BUILD_DIR, "gen_image_ce"),
            model_path,
            prompt,
            out_path,
            str(seed),
            str(steps),
            str(wave_iters),
        ]
        if hybrid:
            cmd.append("--hybrid")
        cmd += ["--guidance", "%.4f" % guidance]

        def on_done(job):
            try:
                if job["exit_code"] == 0 and os.path.isfile(out_path):
                    job["result"]["image"] = ppm_to_data_uri(out_path)
                    job["result"]["ppm_path"] = out_path
                    parsed = parse_gen_log(job["log"])
                    job["result"].update(parsed)
                    if with_viz:
                        job["result"]["analysis"] = analyze_ppm(out_path)
                else:
                    job["result"]["error"] = "gen_image_ce exited %s" % job["exit_code"]
            except Exception as e:
                job["result"]["error"] = "post-process: %s" % e

        job = _run_job(cmd, on_done=on_done)
        return self._send_json({"job_id": job["id"], "cmd": cmd})

    def _post_train(self, body):
        dataset       = body.get("dataset") or "data/demo"
        out_base      = body.get("out_base") or "build/models/demo"
        masked_epochs = int(body.get("masked_epochs", 0))

        # Confine dataset to ROOT (any path under the repo is fine to read).
        ds_abs = _confine(dataset, ROOT)
        if not ds_abs or not os.path.isdir(ds_abs):
            return self._send_json({"error": "dataset dir not found or outside repo: %s" % dataset}, 400)

        # Confine output base to build/models/ so a malicious request cannot
        # write artifacts elsewhere on disk.
        out_abs = _confine(out_base, MODELS_DIR)
        if not out_abs:
            return self._send_json({"error": "out_base must be inside build/models/: %s" % out_base}, 400)
        os.makedirs(os.path.dirname(out_abs) or MODELS_DIR, exist_ok=True)

        cmd = [os.path.join(BUILD_DIR, "train_demo"), ds_abs, out_abs]
        if masked_epochs > 0:
            cmd += ["--masked-epochs", str(masked_epochs)]
        job = _run_job(cmd)
        return self._send_json({"job_id": job["id"], "cmd": cmd})

    def _post_train_text(self, body):
        text_in = body.get("input") or "data/sample_en.txt"
        max_n   = int(body.get("max", 5000))
        out     = body.get("out") or "build/models/stream_auto.spai"

        in_abs = _confine(text_in, ROOT)
        if not in_abs or not os.path.isfile(in_abs):
            return self._send_json({"error": "input file not found or outside repo: %s" % text_in}, 400)

        out_abs = _confine(out, MODELS_DIR)
        if not out_abs:
            return self._send_json({"error": "out must be inside build/models/: %s" % out}, 400)
        os.makedirs(os.path.dirname(out_abs) or MODELS_DIR, exist_ok=True)

        cmd = [
            os.path.join(BUILD_DIR, "stream_train"),
            "--input", in_abs,
            "--max", str(max_n),
            "--save", out_abs,
            "--verify",
        ]
        job = _run_job(cmd)
        return self._send_json({"job_id": job["id"], "cmd": cmd})

    def _post_chat(self, body):
        model   = body.get("model") or ""
        message = body.get("message") or ""
        if not model:
            return self._send_json({"error": "model is required"}, 400)
        if not message:
            return self._send_json({"error": "message is required"}, 400)
        model_abs = _resolve_model(model)
        if not model_abs:
            return self._send_json({"error": "model not found in build/models/: %s" % model}, 400)
        cmd = [os.path.join(BUILD_DIR, "chat"), "--load", model_abs]
        # Single-turn: feed the message + :q on stdin so the REPL exits.
        stdin_data = message.rstrip("\n") + "\n:q\n"

        def on_done(job):
            reply = _extract_chat_reply(job["log"])
            job["result"]["reply"] = reply

        job = _run_job(cmd, stdin_data=stdin_data, on_done=on_done)
        return self._send_json({"job_id": job["id"], "cmd": cmd})

    # ── job introspection ──────────────────────────────────────────────────
    def _send_job(self, jid):
        with JOBS_LOCK:
            job = JOBS.get(jid)
        if not job:
            return self._send_json({"error": "unknown job"}, 404)
        with job["cv"]:
            data = {
                "id":        job["id"],
                "done":      job["done"],
                "exit_code": job["exit_code"],
                "log":       list(job["log"]),
                "result":    dict(job["result"]),
            }
        return self._send_json(data)

    def _stream_job(self, jid):
        with JOBS_LOCK:
            job = JOBS.get(jid)
        if not job:
            return self._send_json({"error": "unknown job"}, 404)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()

        idx = 0
        try:
            while True:
                with job["cv"]:
                    got_event = job["cv"].wait_for(
                        lambda: idx < len(job["log"]) or job["done"],
                        timeout=15.0,
                    )
                    new = job["log"][idx:]
                    idx = len(job["log"])
                    done = job["done"]
                    exit_code = job["exit_code"]
                    result = dict(job["result"]) if done else None
                if not got_event and not new and not done:
                    # SSE comment heartbeat — keeps proxies and browsers from
                    # closing the connection during long quiet stretches.
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                for ln in new:
                    payload = json.dumps({"line": ln})
                    self.wfile.write(("data: %s\n\n" % payload).encode("utf-8"))
                    self.wfile.flush()
                if done:
                    final = json.dumps({
                        "done":      True,
                        "exit_code": exit_code,
                        "result":    result,
                    })
                    self.wfile.write(("data: %s\n\n" % final).encode("utf-8"))
                    self.wfile.flush()
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_models():
    out = []
    if not os.path.isdir(MODELS_DIR):
        return out
    for name in sorted(os.listdir(MODELS_DIR)):
        path = os.path.join(MODELS_DIR, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower().lstrip(".")
        if ext not in ("ces", "spai", "sss"):
            continue
        out.append({
            "name": name,
            "ext":  ext,
            "size": os.path.getsize(path),
            "rel":  os.path.relpath(path, ROOT),
            "mtime": int(os.path.getmtime(path)),
        })
    return out


def _confine(path, base):
    """Resolve `path` against `base` and verify it does not escape it.

    Accepts either a relative path or an absolute path; either way the resolved
    result must live inside `base`. Also strips a leading `build/models/` or
    `data/` segment that matches the base, so callers can pass either a bare
    name or the full repo-relative path that /api/models surfaces.
    Returns the realpath on success, None on rejection.
    """
    if not path:
        return None
    base_real = os.path.realpath(base)
    base_rel  = os.path.relpath(base_real, ROOT)

    if os.path.isabs(path):
        candidate = path
    else:
        norm = path.replace("\\", "/")
        if norm.startswith("./"):
            norm = norm[2:]
        # Strip a redundant repo-relative prefix so passing "build/models/x"
        # while base is build/models/ doesn't produce build/models/build/models/x.
        prefix = base_rel.replace(os.sep, "/") + "/"
        if norm.startswith(prefix):
            norm = norm[len(prefix):]
        candidate = os.path.join(base_real, norm)
    real = os.path.realpath(candidate)
    if real != base_real and not real.startswith(base_real + os.sep):
        return None
    return real


def _resolve_model(s):
    """Resolve a user-supplied model identifier to an absolute path inside
    build/models/. Rejects absolute paths, traversal, and anything outside the
    models directory. Returns None on failure so the caller emits a 4xx."""
    if not s or not isinstance(s, str):
        return None
    # Strip a leading "build/models/" so callers can pass either a bare
    # filename or the rel path returned by /api/models.
    name = s
    rel_prefix = "build/models/"
    if name.startswith(rel_prefix):
        name = name[len(rel_prefix):]
    if os.path.isabs(name) or ".." in name.replace("\\", "/").split("/"):
        return None
    real = _confine(name, MODELS_DIR)
    if real and os.path.isfile(real):
        return real
    return None


def _extract_chat_reply(lines):
    """Pull the AI reply out of the chat REPL transcript.

    When stdin is piped, the REPL's `you> ` prompt is not flushed with a
    newline so it ends up concatenated onto the same line as `bot: …`. We
    therefore search for `bot:` anywhere on the line and take everything after
    it as the reply.
    """
    bot_re = re.compile(r"bot:\s*(.*)$")
    bot_replies = []
    rest = []
    skip_prefixes = ("[chat]", "CANVAS chat", "[server]", "Bye", "bye")
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        m = bot_re.search(s)
        if m:
            bot_replies.append(m.group(1).strip())
            continue
        if any(s.startswith(p) for p in skip_prefixes):
            continue
        if s.startswith("you>") or s.startswith("(sim=") or s.startswith("(refine"):
            continue
        if s in (":q", ":quit"):
            continue
        rest.append(s)
    if bot_replies:
        return "\n".join(bot_replies)
    return "\n".join(rest).strip() or "(no reply emitted; check console)"


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _parse_port(s):
    try:
        p = int(s)
    except (TypeError, ValueError):
        raise SystemExit("[ui] invalid port: %r (must be an integer 1..65535)" % s)
    if not (1 <= p <= 65535):
        raise SystemExit("[ui] invalid port: %d (must be 1..65535)" % p)
    return p


def main():
    parser = argparse.ArgumentParser(prog="sss-forge-ui",
        description="SSS Forge UI bridge server.")
    parser.add_argument("port", nargs="?", default="8080",
        help="TCP port to listen on (default 8080)")
    parser.add_argument("--host", default="127.0.0.1",
        help="interface to bind (default 127.0.0.1; pass 0.0.0.0 to expose to "
             "the LAN — note: the API is unauthenticated and runs local "
             "binaries, so only do this on a trusted network)")
    parser.add_argument("--cors", default=None,
        help="value for Access-Control-Allow-Origin. Off by default since the "
             "UI is same-origin; pass an explicit origin or '*' to enable.")
    args = parser.parse_args()
    port = _parse_port(args.port)
    addr = (args.host, port)
    httpd = ThreadingServer(addr, Handler)
    httpd.cors_origin = args.cors
    if args.host == "0.0.0.0":
        print("[ui] WARNING: binding 0.0.0.0 exposes an unauthenticated API "
              "that can run local binaries. Only do this on a trusted network.")
    print("[ui] sss forge serving on http://%s:%d  (root=%s)" % (addr[0], addr[1], ROOT))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ui] shutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
