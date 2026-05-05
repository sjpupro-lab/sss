"""Smoke test for tools/sss_unified.

Verifies that the unified pipeline's CEMemory.add() routes through the
C bridge (ce_storage_add_typed) and that a single end-to-end prompt
run produces an image, evaluates it, and writes the new cells back
into both the Python sidecar and CEStorage.

Run:
    make pybridge
    python3 tools/test_sss_unified.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SSS_UNIFIED_DIR"] = tmp
        # Import after env override so BASE_DIR resolves there.
        from tools.sss_unified import SSSPipeline, run_sss_pipeline

        pipeline = SSSPipeline(db_path=os.path.join(tmp, "mem"))
        # Foundation seeding must populate both views in lockstep:
        # every memory.add() call goes through ce_storage_add_typed.
        pipeline.seed_foundation()
        py_cells = pipeline.memory.total_cells()
        ce_cells = pipeline.memory.ce_storage_count()
        assert py_cells > 0
        assert ce_cells == py_cells, (
            f"sidecar/CEStorage drift after seed: py={py_cells} ce={ce_cells}")
        print(f"OK  seed     : py={py_cells} ce={ce_cells}")

        # Single end-to-end prompt. Image must come back; if accepted,
        # both counters step in lockstep again. If rejected (low score
        # on a fresh memory), the CEStorage count stays put.
        result, image, frames = pipeline.run("빨간 캐릭터가 웃는 이미지 그려줘")
        assert image is not None and image.shape == (256, 256, 3)
        assert "score" in result and 0.0 <= result["score"]["final"] <= 1.0

        py_after = pipeline.memory.total_cells()
        ce_after = pipeline.memory.ce_storage_count()
        assert ce_after == py_after, (
            f"sidecar/CEStorage drift after run: py={py_after} ce={ce_after}")
        delta = py_after - py_cells
        print(f"OK  run      : py={py_after} ce={ce_after} "
              f"(stored={result.get('stored')}, +{delta} cells)")
        print(f"OK  score    : final={result['score']['final']:.3f} "
              f"prompt_match={result['score']['prompt_match']:.3f} "
              f"strategy={result['search']['strategy']}")

        # Module-level convenience shares one lazy pipeline.
        r2, img2, _ = run_sss_pipeline("파란 캐릭터가 슬픈 이미지 만들어줘")
        assert img2 is not None
        assert "score" in r2
        print(f"OK  module-fn: final={r2['score']['final']:.3f}")

        # Save() round-trip: must produce a .ces and meta.json.
        pipeline.memory.save()
        ces_path = os.path.join(tmp, "mem", "storage.ces")
        meta_path = os.path.join(tmp, "mem", "meta.json")
        assert os.path.exists(ces_path), ces_path
        assert os.path.exists(meta_path), meta_path
        print(f"OK  persist  : .ces={os.path.getsize(ces_path)}B "
              f"meta={os.path.getsize(meta_path)}B")

        print("PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
