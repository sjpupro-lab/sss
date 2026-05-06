"""Smoke test for tools/sss_pose_radar.

Covers:
  - synthesised character-like silhouette runs end-to-end
  - returned dict carries bbox / joints / motions / dirty_zones
  - dirty-zone spans are clipped within the canvas
  - motion entries carry the documented schema fields
  - debug files (mask.png, radar.png, skeleton_overlay.png,
    dirty_rows.png, pose_motion.json) write to disk when out_dir is set
  - ingest_labeled_image populates memory.pose_cells AND keeps the
    legacy 5 row-block cells / total_cells contract intact

Run:
    python3 tools/test_sss_pose_radar.py
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _synth_character(h=256, w=256):
    """Solid-coloured 256×256 figure on a white background.

    The shape is a head circle, neck stem, torso rectangle, two arm
    rectangles spreading outward, and two legs. Far from photorealistic
    but plenty of structure for the silhouette / joint logic to bite
    on.
    """
    img = np.full((h, w, 3), 240, dtype=np.uint8)         # bright bg
    fg  = np.array([60, 80, 200], dtype=np.uint8)         # blue-ish

    # head: circle at (128, 50), r=24
    yy, xx = np.ogrid[:h, :w]
    head = (yy - 50) ** 2 + (xx - 128) ** 2 <= 24 * 24
    img[head] = fg

    # neck: 12-wide rectangle from y=70 to y=84
    img[70:84, 122:134] = fg

    # torso: 56-wide rectangle from y=84 to y=170
    img[84:170, 100:156] = fg

    # arms: angled rectangles. left: from shoulder to (60, 150).
    for r in range(0, 70):
        col = 100 - int(r * 0.5)
        row = 90 + r
        if 0 <= row < h and 0 <= col < w:
            img[row, max(0, col - 4):col + 4] = fg
    # right: from shoulder to (196, 150).
    for r in range(0, 70):
        col = 156 + int(r * 0.5)
        row = 90 + r
        if 0 <= row < h and 0 <= col < w:
            img[row, col - 4:min(w, col + 4)] = fg

    # legs: two rectangles
    img[170:230, 110:128] = fg
    img[170:230, 128:146] = fg
    return img


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SSS_UNIFIED_DIR"] = tmp

        from tools.sss_pose_radar import analyze_pose_radar
        from tools.sss_image_io import write_png
        from tools.sss_unified import SSSPipeline
        from tools.sss_ingest import ingest_labeled_image, ingest_file

        # ── 1. analyze_pose_radar on a numpy ndarray ──────────
        char = _synth_character()
        result = analyze_pose_radar(char, "blue character")
        assert result["type"] == "pose_radar"
        assert result["label"] == "blue character"

        bbox = result["bbox"]
        assert isinstance(bbox, list) and len(bbox) == 4
        assert 0 <= bbox[0] < bbox[2] < 256
        assert 0 <= bbox[1] < bbox[3] < 256
        # Synth figure should occupy a hefty portion of the canvas
        # without taking the whole frame.
        assert 0.04 < result["area_frac"] < 0.7, result
        print(f"OK  analyze   : bbox={bbox} area={result['area_frac']:.3f} "
              f"q={result['quality']:.3f}")

        # ── 2. joint dict has the documented anchors ──────────
        joints = result["joints"]
        for name in ("head", "neck", "shoulder_l", "shoulder_r",
                     "elbow_l", "elbow_r", "hand_l", "hand_r",
                     "waist", "hip_l", "hip_r",
                     "knee_l", "knee_r", "foot_l", "foot_r"):
            assert name in joints, f"missing joint {name!r}"
            x, y = joints[name]
            assert 0 <= x < 256 and 0 <= y < 256, (name, x, y)
        # Head should be above the feet and shoulders should be wider
        # than the head x-position (the synth figure has clear arms).
        assert joints["head"][1] < joints["foot_l"][1]
        assert joints["shoulder_l"][0] <= joints["shoulder_r"][0]
        print(f"OK  joints    : {len(joints)} anchors, "
              f"head→foot Δy={joints['foot_l'][1] - joints['head'][1]}")

        # ── 3. dirty zones clipped, motion schema intact ──────
        dirty = result["dirty_zones"]
        for name in ("hair", "face", "arms", "cloth", "body"):
            assert name in dirty, f"missing zone {name!r}"
            a, b = dirty[name]
            assert 0 <= a <= b < 256, (name, a, b)
        motions = result["motions"]
        assert isinstance(motions, list) and len(motions) >= 4
        for m in motions:
            for k in ("name", "origin", "vector",
                      "amplitude", "frequency", "dirty_rows"):
                assert k in m, (m, k)
            assert isinstance(m["origin"], list) and len(m["origin"]) == 2
            assert isinstance(m["vector"], list) and len(m["vector"]) == 2
            assert m["amplitude"] >= 0.0
            assert m["frequency"] >= 0.0
            assert (isinstance(m["dirty_rows"], list)
                    and len(m["dirty_rows"]) == 2)
        names = {m["name"] for m in motions}
        for must in ("hair_wave", "blink", "arm_sway",
                     "cloth_sway", "body_bounce"):
            assert must in names, (must, names)
        print(f"OK  motion    : {len(motions)} rules → {sorted(names)}")

        # ── 4. _arrays sidecar (mask + radar fields) ──────────
        arrays = result.get("_arrays", {})
        assert "mask" in arrays and arrays["mask"].shape == (256, 256)
        assert "row_density" in arrays and arrays["row_density"].shape == (256,)
        assert "col_density" in arrays and arrays["col_density"].shape == (256,)
        assert "edge_map" in arrays and arrays["edge_map"].shape == (256, 256)
        assert "fft_response" in arrays and arrays["fft_response"].ndim == 2
        print(f"OK  arrays    : mask={arrays['mask'].dtype} "
              f"row_density max={float(arrays['row_density'].max()):.3f}")

        # ── 5. CLI / out_dir branch writes debug PNGs + JSON ──
        out_dir = os.path.join(tmp, "pose_dbg")
        png_path = os.path.join(tmp, "char.png")
        write_png(png_path, char)
        result2 = analyze_pose_radar(png_path, "blue character",
                                     out_dir=out_dir)
        for name in ("mask.png", "radar.png",
                     "skeleton_overlay.png", "dirty_rows.png",
                     "pose_motion.json"):
            p = os.path.join(out_dir, name)
            assert os.path.exists(p), p
            assert os.path.getsize(p) > 0, p
        # JSON sidecar must round-trip.
        with open(os.path.join(out_dir, "pose_motion.json")) as f:
            loaded = json.load(f)
        assert loaded["type"] == "pose_radar"
        assert "_arrays" not in loaded
        print(f"OK  debug io  : 5 files written, json bbox={loaded['bbox']}")

        # ── 6. ingest_labeled_image stores pose without breaking
        #      the legacy 5 row-block cells / total_cells contract ──
        pipe = SSSPipeline(db_path=os.path.join(tmp, "mem"))
        pipe.seed_foundation()
        before_cells = pipe.memory.total_cells()
        before_pose  = pipe.memory.total_pose_cells()

        s = ingest_labeled_image(png_path, pipe.memory,
                                 "blue character",
                                 source="test_pose")
        assert s["type"] == "image"
        # Legacy contract intact.
        assert s["cells_added"] == 5, s
        assert pipe.memory.total_cells() == before_cells + 5
        # New pose sidecar populated.
        assert s["pose_cells_added"] == 1
        assert pipe.memory.total_pose_cells() == before_pose + 1
        pose_summary = s["pose_radar"]
        assert pose_summary.get("error") is None, pose_summary
        assert pose_summary["motion_count"] >= 4
        assert pose_summary["dirty_zone_count"] == 5
        print(f"OK  ingest    : +5 row cells (total={pipe.memory.total_cells()}), "
              f"+1 pose cell, motions={pose_summary['motion_count']}")

        # ── 7. find_pose retrieves by tag ──────────────────────
        hits = pipe.memory.find_pose(["pose", "blue"])
        assert hits, hits
        data = hits[0]["data"]
        assert data["type"] == "pose_radar"
        assert "joints" in data and "motions" in data
        # No matches → empty list, never an error.
        assert pipe.memory.find_pose(["wholly_unknown_tag_xyz"]) == []
        print(f"OK  find_pose : {len(hits)} hit(s), top tags={hits[0]['tags'][:4]}")

        # ── 8. ingest_file routes images through pose_radar too ─
        before_pose = pipe.memory.total_pose_cells()
        s = ingest_file(png_path, pipe.memory, label="blue character v2")
        assert s["type"] == "image" and s["cells_added"] == 5
        assert s.get("pose_cells_added") == 1
        assert pipe.memory.total_pose_cells() == before_pose + 1
        print(f"OK  route     : ingest_file → pose sidecar +1")

        # ── 9. pipeline.run consults pose cells for video motion ─
        # (smoke check: with a matching pose entry we still produce
        # frames; absence of pose data just means the deterministic
        # fallback runs.)
        result, image, frames = pipe.run(
            "파란 캐릭터가 흔드는 영상 만들어줘")
        assert image is not None and image.shape == (256, 256, 3)
        assert frames is not None and len(frames) > 0
        # MotionEngine is wired to memory; no exception means the
        # find_pose lookup at least ran.
        print(f"OK  motion run: frames={len(frames)} pose_cells="
              f"{pipe.memory.total_pose_cells()}")

        print("PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
