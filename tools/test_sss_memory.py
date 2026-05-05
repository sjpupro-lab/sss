"""Smoke test for tools/sss_memory.CEMemory.

Verifies that add_cell routes through ce_storage_add_typed: every Python
add must show up in the underlying CEStorage with matching count.
Re-load via .ces round-trip is also exercised so the storage_io path
(v3) stays compatible with what the bridge writes.

Run:
    make pybridge
    python3 tools/test_sss_memory.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.sss_memory import CEMemory  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        mem = CEMemory(os.path.join(tmp, "mem"))

        rng = np.random.default_rng(0)
        canny = rng.integers(0, 256, size=(64, 256), dtype=np.uint8)
        depth = rng.integers(0, 256, size=(64, 256), dtype=np.uint8)
        color_avg = np.array([180, 90, 60], dtype=np.float32)

        # Add three cells in slot "top", two in "bot".
        ids_top = [mem.add_cell("top", canny, depth, color_avg, 0.5,
                                source=f"smoke_top_{i}") for i in range(3)]
        ids_bot = [mem.add_cell("bot", canny, None, color_avg, 0.7,
                                source=f"smoke_bot_{i}") for i in range(2)]

        assert ids_top == [0, 1, 2], ids_top
        assert ids_bot == [0, 1], ids_bot

        total = mem.ce_storage_count()
        per_top = mem.ce_storage_count_by_slot("top")
        per_bot = mem.ce_storage_count_by_slot("bot")
        assert total == 5, f"expected 5 entries in CEStorage, got {total}"
        assert per_top == 3, per_top
        assert per_bot == 2, per_bot

        # Sidecar metadata should line up with block_idx returned by C.
        for i, c in enumerate(mem.cells["top"]):
            assert c["id"] == i
            assert c["source"] == f"smoke_top_{i}"

        # Keyframe lookup round-trips.
        kf = mem.get_keyframe("top", 0)
        assert kf is not None and len(kf) == 64

        # Save and verify .ces lands on disk.
        mem.save()
        ces_path = os.path.join(tmp, "mem", "storage.ces")
        assert os.path.exists(ces_path), ces_path
        size = os.path.getsize(ces_path)
        assert size > 16, f"ces file too small: {size}"

        # Reload into a fresh CEMemory via the bridge: count must match.
        mem2 = CEMemory(os.path.join(tmp, "mem2"))
        # Direct ctypes call to load — Python sidecar stays empty, but
        # CEStorage count should reflect the loaded entries.
        from tools.sss_memory import _lib
        ok = _lib().sss_memory_load(mem2._handle, ces_path.encode("utf-8"))
        assert ok == 1
        assert mem2.ce_storage_count() == 5

        # After load, a new add_cell continues numbering from the tail.
        new_id = mem2.add_cell("top", canny, depth, color_avg, 0.4)
        assert new_id == 3, f"expected next top idx to be 3, got {new_id}"

        print("OK  ce_storage_count =", mem.ce_storage_count())
        print("OK  per-slot counts  : top=%d bot=%d" % (per_top, per_bot))
        print("OK  keyframe bytes   :", kf[:8].hex(), "...")
        print("OK  reload + append  : next top idx =", new_id)
        print("PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
