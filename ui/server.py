#!/usr/bin/env python3
"""
SSS UI Bridge Server
====================
C 엔진 바이너리를 HTTP API로 노출하는 브릿지 서버.
외부 의존성 없음 (Python 3 stdlib만 사용).

Routes:
  GET  /                  → UI (index.html)
  GET  /api/models        → 사용 가능한 모델 목록
  GET  /api/browse        → 서버 디렉토리 탐색 (ROOT 하위만)
  POST /api/generate      → 이미지/영상 생성 (scripts/sss_gen.py 위임)
  POST /api/viz-generate  → 시각화용 생성 (분석 데이터 포함)
  POST /api/train         → CE 엔진 학습 실행 (train_demo)
  POST /api/train-sss     → 스펙트로그램 학습 (scripts/sss_train.py)
  POST /api/train-text    → 텍스트 학습 (stream_train)
  POST /api/upload        → 학습 데이터 업로드 (multipart)
  POST /api/chat          → 텍스트 대화 (chat 바이너리)
  GET  /api/stream/<id>   → SSE: 실행 중인 작업 로그 스트림
"""

import http.server
import json
import subprocess
import os
import sys
import base64
import struct
import threading
import time
import uuid
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from io import BytesIO

# ─── 경로 설정 ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # sss/
BUILD = ROOT / "build"
MODELS = BUILD / "models"
UI_DIR = Path(__file__).resolve().parent  # sss/ui/
DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"

MODELS.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ─── 작업 저장소 (진행 중인 프로세스 로그) ─────────────────────
jobs = {}  # job_id → { "proc": Popen, "log": [str], "done": bool, "result": dict }


# ─── PPM → PNG 변환 (순수 Python, 외부 의존성 없음) ──────────
def ppm_to_png_bytes(ppm_path):
    """P6 PPM을 읽어 PNG bytes로 변환 (zlib + 최소 PNG 구조)."""
    import zlib

    with open(ppm_path, "rb") as f:
        # Parse header
        magic = f.readline().strip()
        if magic not in (b"P6", b"P3"):
            raise ValueError(f"Not a PPM file: {magic}")

        # Skip comments
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())

        line = f.readline()
        maxval = int(line.strip())

        if magic == b"P6":
            data = f.read()
        else:
            # P3 text format
            tokens = f.read().split()
            data = bytes(int(t) for t in tokens)

    # Build PNG
    def make_chunk(chunk_type, data):
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    # IHDR
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8bit RGB

    # IDAT — add filter byte (0=None) to each row
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter: None
        offset = y * w * 3
        row = data[offset:offset + w * 3]
        if maxval != 255:
            row = bytes(int(b * 255 / maxval) for b in row)
        raw.extend(row)

    compressed = zlib.compress(bytes(raw), 6)

    png = b"\x89PNG\r\n\x1a\n"
    png += make_chunk(b"IHDR", ihdr)
    png += make_chunk(b"IDAT", compressed)
    png += make_chunk(b"IEND", b"")

    return png


def ppm_to_base64(ppm_path):
    """PPM → base64 PNG data URI."""
    try:
        png_bytes = ppm_to_png_bytes(ppm_path)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        return None


def analyze_ppm(ppm_path):
    """PPM 이미지를 분석해서 시각화용 데이터를 추출."""
    import math

    with open(ppm_path, "rb") as f:
        magic = f.readline().strip()
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        f.readline()  # maxval
        data = f.read()

    pixels = []
    for i in range(0, len(data) - 2, 3):
        pixels.append((data[i], data[i + 1], data[i + 2]))

    # Color histogram (16 bins per channel)
    hist_r = [0] * 16
    hist_g = [0] * 16
    hist_b = [0] * 16
    for r, g, b in pixels:
        hist_r[r >> 4] += 1
        hist_g[g >> 4] += 1
        hist_b[b >> 4] += 1

    total = len(pixels) or 1
    hist_r = [round(v / total, 4) for v in hist_r]
    hist_g = [round(v / total, 4) for v in hist_g]
    hist_b = [round(v / total, 4) for v in hist_b]

    # Spatial energy map (16×16 blocks) — average brightness per block
    bw, bh = 16, 16
    block_w = w // bw
    block_h = h // bh
    energy_map = []
    for by in range(bh):
        row = []
        for bx in range(bw):
            total_e = 0
            count = 0
            for dy in range(block_h):
                for dx in range(block_w):
                    px = bx * block_w + dx
                    py = by * block_h + dy
                    if py < h and px < w:
                        idx = py * w + px
                        if idx < len(pixels):
                            r, g, b = pixels[idx]
                            total_e += (r + g + b) / 3.0
                            count += 1
            row.append(round(total_e / max(count, 1), 1))
        energy_map.append(row)

    # Edge energy (simple Sobel-like horizontal gradient per block)
    edge_map = []
    for by in range(bh):
        row = []
        for bx in range(bw):
            grad_sum = 0
            count = 0
            for dy in range(block_h):
                for dx in range(1, block_w):
                    px = bx * block_w + dx
                    py = by * block_h + dy
                    if py < h and px < w:
                        idx1 = py * w + px
                        idx0 = py * w + px - 1
                        if idx1 < len(pixels) and idx0 < len(pixels):
                            r1, g1, b1 = pixels[idx1]
                            r0, g0, b0 = pixels[idx0]
                            grad_sum += abs(r1 - r0) + abs(g1 - g0) + abs(b1 - b0)
                            count += 1
            row.append(round(grad_sum / max(count, 1), 1))
        edge_map.append(row)

    # Color dominance
    sum_r = sum(p[0] for p in pixels)
    sum_g = sum(p[1] for p in pixels)
    sum_b = sum(p[2] for p in pixels)
    total_sum = sum_r + sum_g + sum_b + 1
    dominance = {
        "r": round(sum_r / total_sum, 3),
        "g": round(sum_g / total_sum, 3),
        "b": round(sum_b / total_sum, 3),
    }

    return {
        "width": w, "height": h,
        "hist_r": hist_r, "hist_g": hist_g, "hist_b": hist_b,
        "energy_map": energy_map,
        "edge_map": edge_map,
        "dominance": dominance,
    }


# ─── 경로 유효성 (ROOT 하위만 허용) ────────────────────────
def safe_under_root(p):
    """경로가 ROOT 하위인지 확인. 통과 시 resolved Path, 아니면 None."""
    try:
        target = Path(p).expanduser().resolve()
    except Exception:
        return None
    root = ROOT.resolve()
    try:
        target.relative_to(root)
        return target
    except ValueError:
        return None


# ─── multipart/form-data 파서 (stdlib only) ─────────────────
def parse_multipart(body_bytes, boundary):
    """간단한 multipart 파서.
    Returns: list of dicts with keys: name, filename, content_type, data (bytes)."""
    boundary_bytes = ("--" + boundary).encode("latin-1")
    parts = body_bytes.split(boundary_bytes)
    result = []
    for raw in parts:
        if not raw or raw in (b"--\r\n", b"--", b"\r\n"):
            continue
        if raw.startswith(b"\r\n"):
            raw = raw[2:]
        if raw.endswith(b"\r\n"):
            raw = raw[:-2]
        sep = raw.find(b"\r\n\r\n")
        if sep < 0:
            continue
        header_block = raw[:sep].decode("latin-1", errors="replace")
        data = raw[sep + 4:]
        if data.endswith(b"\r\n"):
            data = data[:-2]

        name = None
        filename = None
        content_type = "application/octet-stream"
        for line in header_block.split("\r\n"):
            low = line.lower()
            if low.startswith("content-disposition:"):
                for tok in line.split(";"):
                    tok = tok.strip()
                    if tok.startswith("name="):
                        name = tok[5:].strip().strip('"')
                    elif tok.startswith("filename="):
                        filename = tok[9:].strip().strip('"')
            elif low.startswith("content-type:"):
                content_type = line.split(":", 1)[1].strip()
        if name is None:
            continue
        result.append({
            "name": name,
            "filename": filename,
            "content_type": content_type,
            "data": data,
        })
    return result


# ─── 모델 스캔 ───────────────────────────────────────────────
def scan_models():
    """build/models/ 내의 .ces, .spai, .sss 파일 목록."""
    models = []
    if not MODELS.exists():
        return models
    for f in sorted(MODELS.iterdir()):
        if f.suffix in (".ces", ".spai", ".sss"):
            models.append({
                "name": f.stem,
                "type": f.suffix[1:],
                "path": str(f),
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return models


# ─── 작업 실행 (백그라운드 스레드) ────────────────────────────
def run_job(job_id, cmd, cwd=None, on_done=None):
    """subprocess를 백그라운드에서 실행하고 stdout을 jobs[job_id]["log"]에 축적."""
    job = jobs[job_id]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=cwd or str(ROOT), text=True, bufsize=1,
        )
        job["proc"] = proc
        for line in proc.stdout:
            job["log"].append(line.rstrip("\n"))
        proc.wait()
        job["exit_code"] = proc.returncode
    except Exception as e:
        job["log"].append(f"[ERROR] {e}")
        job["exit_code"] = -1
    finally:
        job["done"] = True
        if on_done:
            on_done(job)


def start_job(cmd, cwd=None, on_done=None):
    """새 작업 시작, job_id 반환."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"proc": None, "log": [], "done": False, "exit_code": None, "result": {}}
    t = threading.Thread(target=run_job, args=(job_id, cmd, cwd, on_done), daemon=True)
    t.start()
    return job_id


# ─── HTTP 핸들러 ──────────────────────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 깔끔한 로그
        sys.stderr.write(f"[sss-ui] {args[0]}\n")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    # ── GET ──────────────────────────────────────────────────
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Serve UI
        if path == "/" or path == "/index.html":
            html_path = UI_DIR / "index.html"
            if html_path.exists():
                data = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", len(data))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_json({"error": "index.html not found"}, 404)
            return

        # Model list
        if path == "/api/models":
            self.send_json({"models": scan_models()})
            return

        # Filesystem browser (ROOT-restricted)
        if path == "/api/browse":
            params = parse_qs(parsed.query)
            req_dir = params.get("dir", [str(DATA_DIR)])[0]
            target = safe_under_root(req_dir)
            if target is None:
                self.send_json({"error": "access denied"}, 403)
                return
            if not target.exists() or not target.is_dir():
                self.send_json({"error": f"not a directory: {req_dir}"}, 400)
                return
            entries = []
            try:
                items = sorted(target.iterdir(),
                               key=lambda p: (not p.is_dir(), p.name.lower()))
            except PermissionError:
                self.send_json({"error": "permission denied"}, 403)
                return
            for item in items:
                try:
                    is_dir = item.is_dir()
                    size_kb = round(item.stat().st_size / 1024, 1) if not is_dir else 0
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
            self.send_json({
                "path": str(target),
                "rel_path": str(target.relative_to(root)) if target != root else "",
                "parent": parent,
                "root": str(root),
                "entries": entries,
            })
            return

        # Job log stream (SSE)
        if path.startswith("/api/stream/"):
            job_id = path.split("/")[-1]
            if job_id not in jobs:
                self.send_json({"error": "job not found"}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            job = jobs[job_id]
            sent = 0
            while True:
                while sent < len(job["log"]):
                    line = job["log"][sent]
                    self.wfile.write(f"data: {json.dumps({'line': line})}\n\n".encode())
                    self.wfile.flush()
                    sent += 1
                if job["done"]:
                    result = {"done": True, "exit_code": job["exit_code"]}
                    result.update(job.get("result", {}))
                    self.wfile.write(f"data: {json.dumps(result)}\n\n".encode())
                    self.wfile.flush()
                    break
                time.sleep(0.1)
            return

        # Job status
        if path.startswith("/api/job/"):
            job_id = path.split("/")[-1]
            if job_id not in jobs:
                self.send_json({"error": "job not found"}, 404)
                return
            job = jobs[job_id]
            self.send_json({
                "done": job["done"],
                "exit_code": job["exit_code"],
                "log": job["log"][-50:],
                "result": job.get("result", {}),
            })
            return

        self.send_json({"error": "not found"}, 404)

    # ── POST ─────────────────────────────────────────────────
    def do_POST(self):
        path = urlparse(self.path).path

        # ── Image / video generation (Phase 5 unified path) ──
        # /api/generate is now a thin alias on top of the Python
        # unified entry point. The .ces / gen_image_ce branch was
        # removed when Phase 5 retired the legacy generator. The
        # request body keeps the historical schema (model, prompt,
        # seed, detail, steps); new fields (frames, conditions,
        # atmos_reference, out_w/h, atmos) are forwarded straight
        # through to scripts/sss_gen.py.
        if path == "/api/generate":
            body = self.read_body()
            model = body.get("model", str(MODELS / "demo.sss"))
            prompt = body.get("prompt", "red")
            seed = str(body.get("seed", 0))
            frames = int(body.get("frames", 1))
            if frames < 1:
                self.send_json({"error": "frames must be >= 1"}, 400); return
            if not Path(model).exists():
                self.send_json({"error": f"model not found: {model}"}, 400)
                return

            out_name = f"gen_{uuid.uuid4().hex[:6]}"
            out_path = BUILD / (out_name + ".ppm")
            out_dir  = BUILD / out_name if frames > 1 else None

            detail = f"{float(body.get('detail', 1.5)):.3f}"
            steps  = str(int(body.get("steps", 120)))
            cmd = [
                sys.executable, str(ROOT / "scripts" / "sss_gen.py"),
                model, prompt, str(out_path),
                seed, detail, steps,
            ]
            if body.get("atmos"):       cmd.append("--atmos")
            if body.get("out_w"):       cmd += ["--out-w", str(int(body["out_w"]))]
            if body.get("out_h"):       cmd += ["--out-h", str(int(body["out_h"]))]
            if frames > 1:
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd += ["--frames", str(frames), "--out-dir", str(out_dir)]
            if body.get("atmos_reference"):
                cmd += ["--atmos-from", str(body["atmos_reference"])]
            for c in (body.get("conditions") or []):
                lab = c.get("label"); intensity = c.get("intensity", 0.5)
                if not lab:
                    self.send_json({"error":
                        "conditions[].label is required"}, 400); return
                cmd += ["--condition", str(lab), f"{float(intensity):.3f}"]

            def on_done(job, _out=out_path, _out_dir=out_dir, _frames=frames):
                if _frames > 1 and _out_dir is not None and _out_dir.is_dir():
                    frames_out = []
                    for fp in sorted(_out_dir.glob("frame_*.ppm")):
                        try:
                            frames_out.append(ppm_to_base64(str(fp)))
                        except Exception:
                            pass
                    job["result"]["frames"] = frames_out
                    if frames_out:
                        job["result"]["image"] = frames_out[0]
                    try:
                        for fp in _out_dir.glob("*"): fp.unlink()
                        _out_dir.rmdir()
                    except OSError:
                        pass
                    for line in job["log"]:
                        if line.startswith("wrote ") or "frames to" in line:
                            job["result"]["info"] = line
                    return
                if _out.exists():
                    data_uri = ppm_to_base64(str(_out))
                    job["result"]["image"] = data_uri
                    job["result"]["ppm_path"] = str(_out)
                    for line in job["log"]:
                        if line.startswith("wrote "):
                            job["result"]["info"] = line

            job_id = start_job(cmd, on_done=on_done)
            self.send_json({"job_id": job_id})
            return

        # ── Training ─────────────────────────────────────────
        if path == "/api/train":
            body = self.read_body()
            dataset = body.get("dataset", str(ROOT / "data" / "demo"))
            out_base = body.get("out_base", str(MODELS / "demo"))
            masked_epochs = body.get("masked_epochs", 0)

            cmd = [str(BUILD / "train_demo"), dataset, out_base]
            if masked_epochs > 0:
                cmd += ["--masked-epochs", str(masked_epochs)]

            def on_done(job):
                ces = Path(out_base + ".ces")
                spai = Path(out_base + ".spai")
                job["result"]["ces_exists"] = ces.exists()
                job["result"]["spai_exists"] = spai.exists()
                if ces.exists():
                    job["result"]["ces_size_kb"] = round(ces.stat().st_size / 1024, 1)

            job_id = start_job(cmd, on_done=on_done)
            self.send_json({"job_id": job_id})
            return

        # ── Chat (synchronous single-turn) ────────────────────
        if path == "/api/chat":
            body = self.read_body()
            model = body.get("model", "")
            message = body.get("message", "")
            mode = body.get("mode", "gen")  # gen | retr | both

            if not model or not Path(model).exists():
                self.send_json({"error": "model path required"}, 400)
                return

            # Run chat in single-turn mode via stdin pipe
            cmd = [str(BUILD / "chat"), "--load", model]
            try:
                # Send message + :q to exit
                input_text = f"{message}\n:q\n"
                result = subprocess.run(
                    cmd, input=input_text, capture_output=True,
                    text=True, timeout=30, cwd=str(ROOT),
                )
                lines = result.stdout.strip().split("\n")
                # Filter out prompt markers and system lines
                response_lines = []
                capture = False
                for line in lines:
                    if ">" in line and message in line:
                        capture = True
                        continue
                    if capture and not line.startswith("[chat]") and line.strip() != ":q":
                        if line.strip():
                            response_lines.append(line)

                self.send_json({
                    "response": "\n".join(response_lines) if response_lines else "(no response)",
                    "raw": lines[-20:],  # last 20 lines for debug
                })
            except subprocess.TimeoutExpired:
                self.send_json({"error": "timeout"}, 500)
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return

        # ── Text training (stream_train) ──────────────────────
        if path == "/api/train-text":
            body = self.read_body()
            input_file = body.get("input", str(ROOT / "data" / "wiki5k.txt"))
            max_lines = str(body.get("max", 5000))
            out_model = body.get("out", str(MODELS / "text_model.spai"))

            cmd = [
                str(BUILD / "stream_train"),
                "--input", input_file,
                "--max", max_lines,
                "--save", out_model,
                "--verify",
            ]
            job_id = start_job(cmd)
            self.send_json({"job_id": job_id})
            return

        # ── Viz Generate (analysis-enriched, Phase 5 unified) ────
        # Same unified path as /api/generate above; the only
        # difference is that the on_done callback also runs the
        # per-pixel analysis pass and surfaces the result-line
        # parsing the old morpheme / canvas_id parser used to do.
        # The legacy `.ces` branch and morpheme-FFT log parsing
        # are gone with `gen_image_ce`.
        if path == "/api/viz-generate":
            body = self.read_body()
            model = body.get("model", str(MODELS / "demo.sss"))
            prompt = body.get("prompt", "red")
            seed = str(body.get("seed", 0))
            if not Path(model).exists():
                self.send_json({"error": f"model not found: {model}"}, 400)
                return

            out_name = f"viz_{uuid.uuid4().hex[:6]}.ppm"
            out_path = BUILD / out_name
            detail = f"{float(body.get('detail', 1.5)):.3f}"
            steps  = str(int(body.get("steps", 120)))
            cmd = [
                sys.executable, str(ROOT / "scripts" / "sss_gen.py"),
                model, prompt, str(out_path),
                seed, detail, steps,
            ]
            if body.get("atmos"): cmd.append("--atmos")

            def on_done(job):
                if out_path.exists():
                    data_uri = ppm_to_base64(str(out_path))
                    job["result"]["image"] = data_uri
                    analysis = analyze_ppm(str(out_path))
                    job["result"]["analysis"] = analysis
                    job["result"]["steps"] = int(steps)
                    job["result"]["seed"] = int(seed)
                    for line in job["log"]:
                        if line.startswith("wrote "):
                            job["result"]["info"] = line

            job_id = start_job(cmd, on_done=on_done)
            self.send_json({"job_id": job_id})
            return

        # ── Spectrogram (.sss) training ───────────────────────
        if path == "/api/train-sss":
            body = self.read_body()
            labels = body.get("labels")
            root_dir = body.get("root")
            out_model = body.get("out", str(MODELS / "custom.sss"))
            size = str(int(body.get("size", 64)))

            if not labels or not root_dir:
                self.send_json({"error": "labels and root are required"}, 400)
                return
            if not Path(labels).exists():
                self.send_json({"error": f"labels file not found: {labels}"}, 400)
                return
            if not Path(root_dir).is_dir():
                self.send_json({"error": f"root dir not found: {root_dir}"}, 400)
                return
            # Ensure out_model has .sss extension
            if not out_model.endswith(".sss"):
                out_model += ".sss"
            Path(out_model).parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python3", str(ROOT / "scripts" / "sss_train.py"),
                "--labels", labels,
                "--root", root_dir,
                "--out", out_model,
                "--size", size,
            ]

            def on_done(job):
                p = Path(out_model)
                job["result"]["sss_exists"] = p.exists()
                if p.exists():
                    job["result"]["sss_size_kb"] = round(p.stat().st_size / 1024, 1)
                    job["result"]["sss_path"] = str(p)

            job_id = start_job(cmd, on_done=on_done)
            self.send_json({"job_id": job_id})
            return

        # ── File upload (multipart) ───────────────────────────
        if path == "/api/upload":
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                self.send_json({"error": "Content-Type must be multipart/form-data"}, 400)
                return
            boundary = None
            for tok in ctype.split(";"):
                tok = tok.strip()
                if tok.lower().startswith("boundary="):
                    boundary = tok[9:].strip().strip('"')
            if not boundary:
                self.send_json({"error": "boundary missing"}, 400)
                return
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                self.send_json({"error": "empty body"}, 400)
                return
            raw = self.rfile.read(length)
            try:
                parts = parse_multipart(raw, boundary)
            except Exception as e:
                self.send_json({"error": f"multipart parse failed: {e}"}, 400)
                return

            session = "sess_" + uuid.uuid4().hex[:8]
            session_name = None
            for p in parts:
                if p["name"] == "session" and not p["filename"]:
                    session_name = p["data"].decode("utf-8", errors="replace").strip()
            if session_name:
                # sanitize
                safe = "".join(c for c in session_name if c.isalnum() or c in "._-")[:40]
                if safe:
                    session = safe
            target_dir = UPLOADS_DIR / session
            target_dir.mkdir(parents=True, exist_ok=True)

            saved = []
            labels_path = None
            for p in parts:
                fname = p["filename"]
                if not fname:
                    continue
                # strip path components
                fname = Path(fname).name
                if not fname:
                    continue
                dst = target_dir / fname
                with open(dst, "wb") as f:
                    f.write(p["data"])
                saved.append({"name": fname, "size": len(p["data"])})
                if p["name"] == "labels" or fname.endswith(".tsv") or fname == "labels.txt":
                    labels_path = str(dst)

            self.send_json({
                "dir": str(target_dir),
                "rel_dir": str(target_dir.relative_to(ROOT)),
                "labels": labels_path,
                "rel_labels": str(Path(labels_path).relative_to(ROOT)) if labels_path else None,
                "files": saved,
            })
            return

        self.send_json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ─── 서버 시작 ────────────────────────────────────────────────
def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    print(f"""
╔═══════════════════════════════════════╗
║          SSS Engine UI Server         ║
║                                       ║
║   http://localhost:{port}              ║
║                                       ║
║   Models: {MODELS}
║   Engine: {BUILD}
╚═══════════════════════════════════════╝
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[sss-ui] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
