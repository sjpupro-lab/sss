#!/usr/bin/env python3
"""
SSS UI Bridge Server
====================
C 엔진 바이너리를 HTTP API로 노출하는 브릿지 서버.
외부 의존성 없음 (Python 3 stdlib만 사용).

Routes:
  GET  /                  → UI (index.html)
  GET  /api/models        → 사용 가능한 모델 목록
  POST /api/generate      → 이미지 생성 (gen_image_ce)
  POST /api/train         → 학습 실행 (train_demo)
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

MODELS.mkdir(parents=True, exist_ok=True)

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

        # ── Image generation ─────────────────────────────────
        if path == "/api/generate":
            body = self.read_body()
            model = body.get("model", str(MODELS / "demo.ces"))
            prompt = body.get("prompt", "red")
            seed = str(body.get("seed", 0))
            steps = str(body.get("steps", 50))
            wave = str(body.get("wave_iters", 200))
            hybrid = body.get("hybrid", False)
            guidance = str(body.get("guidance", 1.0))

            if not Path(model).exists():
                self.send_json({"error": f"model not found: {model}"}, 400)
                return

            out_name = f"gen_{uuid.uuid4().hex[:6]}.ppm"
            out_path = BUILD / out_name

            cmd = [
                str(BUILD / "gen_image_ce"),
                model, prompt, str(out_path),
                seed, steps, wave,
            ]
            if hybrid:
                cmd += ["--hybrid", "--guidance", guidance]

            def on_done(job):
                if out_path.exists():
                    data_uri = ppm_to_base64(str(out_path))
                    job["result"]["image"] = data_uri
                    job["result"]["ppm_path"] = str(out_path)
                    # Parse mean RGB from log
                    for line in job["log"]:
                        if "mean RGB" in line:
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

        # ── Viz Generate (analysis-enriched) ──────────────
        if path == "/api/viz-generate":
            body = self.read_body()
            model = body.get("model", str(MODELS / "demo.ces"))
            prompt = body.get("prompt", "red")
            seed = str(body.get("seed", 0))
            steps = str(body.get("steps", 50))
            wave = str(body.get("wave_iters", 200))

            if not Path(model).exists():
                self.send_json({"error": f"model not found: {model}"}, 400)
                return

            out_name = f"viz_{uuid.uuid4().hex[:6]}.ppm"
            out_path = BUILD / out_name

            cmd = [
                str(BUILD / "gen_image_ce"),
                model, prompt, str(out_path), seed, steps, wave,
            ]

            def on_done(job):
                if out_path.exists():
                    data_uri = ppm_to_base64(str(out_path))
                    job["result"]["image"] = data_uri
                    analysis = analyze_ppm(str(out_path))
                    job["result"]["analysis"] = analysis

                    # Parse morphemes and routing from log
                    morphemes = []
                    canvas_id = ""
                    mean_rgb = [0, 0, 0]
                    for line in job["log"]:
                        if "morphemes=" in line:
                            # Extract morpheme tokens
                            parts = line.split("morphemes=")[1] if "morphemes=" in line else ""
                            import re
                            morph_matches = re.findall(r'(\S+)/(\S+)', parts)
                            morphemes = [{"token": m[0], "pos": m[1]} for m in morph_matches]
                        if "routed canvas_id=" in line:
                            canvas_id = line.split("canvas_id=")[1].strip()
                        if "mean RGB" in line:
                            import re
                            m = re.search(r'\(([^)]+)\)', line)
                            if m:
                                vals = m.group(1).split(",")
                                mean_rgb = [float(v.strip()) for v in vals[:3]]

                    job["result"]["morphemes"] = morphemes
                    job["result"]["canvas_id"] = canvas_id
                    job["result"]["mean_rgb"] = mean_rgb
                    job["result"]["steps"] = int(steps)
                    job["result"]["wave_iters"] = int(wave)
                    job["result"]["seed"] = int(seed)

            job_id = start_job(cmd, on_done=on_done)
            self.send_json({"job_id": job_id})
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
