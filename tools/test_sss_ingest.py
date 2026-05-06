"""Smoke test for tools/sss_ingest.

Covers:
  - extension routing (image / text / video / unknown)
  - per-type summary shape and counts
  - PNG and PPM image branches (sss_image_io)
  - UTF-8 text + latin-1 fallback
  - memory grows monotonically
  - resize endpoint mapping (corners hit corners)
  - video branch only when ffmpeg is on PATH

Run:
    python3 tools/test_sss_ingest.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SSS_UNIFIED_DIR"] = tmp
        from tools.sss_unified import SSSPipeline
        from tools.sss_ingest import (
            ingest_file, _resize_nn, fingerprint, ingest_csv)
        from tools.sss_image_io import write_png, write_ppm

        # _resize_nn endpoint check: corners of source map to corners
        # of target after the linspace fix.
        src = np.arange(255 * 255, dtype=np.uint8).reshape(255, 255)
        out = _resize_nn(src, 256, 256)
        assert out[0, 0] == src[0, 0], "top-left corner lost"
        assert out[-1, -1] == src[-1, -1], "bottom-right corner lost"
        print("OK  resize    : endpoints preserved (255→256 round-trip)")

        pipe = SSSPipeline(db_path=os.path.join(tmp, "mem"))
        pipe.seed_foundation()
        baseline = pipe.memory.total_cells()
        assert baseline > 0
        print(f"OK  seed      : {baseline} cells")

        # ── fingerprint matches the C contract ───────────
        fp = fingerprint("red flower")
        assert fp.shape == (256,) and fp.dtype == np.float32
        # L2-normalised: |fp| ≈ 1.0 unless empty.
        norm = float(np.sqrt(np.sum(fp * fp)))
        assert abs(norm - 1.0) < 1e-5, norm
        assert np.allclose(fingerprint(""), 0.0)
        print(f"OK  fp        : 256-bin L2-normed ({norm:.6f})")

        # ── PNG image WITHOUT label is refused ───────────
        img = np.zeros((96, 128, 3), np.uint8)
        img[:, :64] = (50, 50, 210)
        img[:, 64:] = (50, 170, 50)
        png_path = os.path.join(tmp, "img.png"); write_png(png_path, img)
        before = pipe.memory.total_cells()
        s = ingest_file(png_path, pipe.memory)
        assert s.get("error") and "라벨" in s["error"], s
        assert pipe.memory.total_cells() == before
        print(f"OK  no-label  : refused with '{s['error']}' (no cells added)")

        # ── PNG image WITH label adds 5 cells with amp/phase ─
        s = ingest_file(png_path, pipe.memory, label="red green stripe")
        assert s["type"] == "image"
        assert s["cells_added"] == 5
        assert s["tags"] == ["red", "green", "stripe"]
        assert set(s["blocks"]) == {"hair", "face", "upper", "lower", "bg"}
        assert pipe.memory.total_cells() == before + 5
        # Spot-check a cell carries amp + phase as a contiguous ndarray.
        sample = pipe.memory.cells["face"][-1]
        assert sample["amp"] is not None and sample["phase"] is not None
        assert sample["amp"].shape == sample["phase"].shape
        assert sample["amp"].ndim == 3 and sample["amp"].shape[1] == 3
        assert sample["amp"].dtype == np.float32
        print(f"OK  png+label : +{s['cells_added']} cells, "
              f"amp shape={sample['amp'].shape} ({sample['amp'].dtype})")

        # ── PPM image with label ─────────────────────────
        ppm_path = os.path.join(tmp, "img.ppm"); write_ppm(ppm_path, img)
        before = pipe.memory.total_cells()
        s = ingest_file(ppm_path, pipe.memory, label="ppm sample")
        assert s["type"] == "image" and s["cells_added"] == 5
        assert pipe.memory.total_cells() == before + 5
        print(f"OK  ppm+label : +{s['cells_added']} cells")

        # ── CSV batch ────────────────────────────────────
        csv_dir = os.path.join(tmp, "csv_dir")
        os.makedirs(csv_dir, exist_ok=True)
        a_path = os.path.join(csv_dir, "a.png"); write_png(a_path, img)
        b_path = os.path.join(csv_dir, "b.png"); write_png(b_path, img)
        csv_path = os.path.join(csv_dir, "labels.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("filepath,label\n")          # header — must be skipped
            f.write("a.png,red apple\n")
            f.write("b.png,blue sky cloud\n")
            f.write("missing.png,gone\n")        # error row
        before = pipe.memory.total_cells()
        s = ingest_csv(csv_path, pipe.memory, base_dir=csv_dir)
        assert s["type"] == "csv"
        assert s["files_ok"] == 2 and s["cells_added"] == 10
        assert len(s["files_err"]) == 1
        assert pipe.memory.total_cells() == before + 10
        print(f"OK  csv       : ok={s['files_ok']}, err={len(s['files_err'])}, "
              f"+{s['cells_added']} cells")

        # ── ingest_file routes .csv → ingest_csv ─────────
        before = pipe.memory.total_cells()
        s = ingest_file(csv_path, pipe.memory, base_dir=csv_dir)
        assert s["type"] == "csv" and s["files_ok"] == 2
        assert pipe.memory.total_cells() == before + 10
        print(f"OK  csv route : ingest_file dispatches to ingest_csv")

        # ── path traversal: rows that escape base_dir are refused ─
        evil_csv = os.path.join(csv_dir, "evil.csv")
        outside = os.path.join(tmp, "outside.png"); write_png(outside, img)
        with open(evil_csv, "w", encoding="utf-8") as f:
            # All four entries should be refused as per-row errors,
            # without reading any of those files.
            f.write("../outside.png,escape via parent\n")
            f.write("../../etc/passwd,absolute escape\n")
            f.write(f"{outside},absolute path\n")
            f.write("a.png,legit relative\n")        # this one IS allowed
        before = pipe.memory.total_cells()
        s = ingest_csv(evil_csv, pipe.memory, base_dir=csv_dir)
        # Only the legit relative row should add cells.
        assert s["files_ok"] == 1, s
        assert s["cells_added"] == 5
        assert len(s["files_err"]) == 3, s
        # The errors should mention the sandbox refusal.
        joined_errs = " ".join(e["error"] for e in s["files_err"])
        assert "absolute path not allowed" in joined_errs
        assert "escapes base_dir" in joined_errs
        assert pipe.memory.total_cells() == before + 5
        print(f"OK  csv sandbox: blocked {len(s['files_err'])}/4, "
              f"allowed {s['files_ok']}/4")

        # ── UTF-8 text ───────────────────────────────────
        txt = os.path.join(tmp, "note.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("first line\n빨간 캐릭터\n" + ("a" * 300) + "\n")
        before = pipe.memory.total_cells()
        s = ingest_file(txt, pipe.memory)
        assert s["type"] == "text"
        assert s["lines"] == 3
        # first line + korean line + long line split into 2 chunks (300 ASCII bytes).
        assert s["cells_added"] == 4, s
        assert pipe.memory.total_cells() == before + 4
        print(f"OK  txt utf-8 : +{s['cells_added']} cells, lines={s['lines']}")

        # ── latin-1 fallback ────────────────────────────
        bad = os.path.join(tmp, "bad.txt")
        with open(bad, "wb") as f:
            f.write(b"plain\n\xe9\xe9\xe9 not utf8 \xff\n")
        s = ingest_file(bad, pipe.memory)
        assert s["type"] == "text"
        print(f"OK  txt latin1: +{s['cells_added']} cells")

        # ── unknown extension raises ─────────────────────
        unk = os.path.join(tmp, "x.bin")
        open(unk, "wb").write(b"\x00")
        try:
            ingest_file(unk, pipe.memory)
        except ValueError as e:
            print(f"OK  unknown   : raised '{e}'")
        else:
            print("FAIL: unknown extension did not raise")
            return 1

        # ── video (only if ffmpeg available) ─────────────
        if shutil.which("ffmpeg"):
            vid = os.path.join(tmp, "test.mp4")
            cp = subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi",
                "-i", "testsrc=duration=1:size=64x64:rate=12",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", vid
            ], capture_output=True)
            if cp.returncode == 0:
                before = pipe.memory.total_cells()
                s = ingest_file(vid, pipe.memory)
                assert s["type"] == "video"
                assert s["frames_processed"] >= 1
                assert s["cells_added"] >= 5
                assert pipe.memory.total_cells() > before
                print(f"OK  video     : frames={s['frames_processed']}, "
                      f"+{s['cells_added']} cells")
            else:
                tail = cp.stderr.decode("utf-8", errors="replace")[-160:]
                print(f"SKIP video    : ffmpeg lavfi failed: {tail.strip()}")
        else:
            print("SKIP video    : ffmpeg not in PATH")

        print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
