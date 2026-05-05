"""sss_memory — CEMemory backed by ce_storage_add_typed via ctypes.

Drop-in replacement for the in-memory `CEMemory` class used by the SSS
self-upgrade loop. Every add_cell() call routes through the C bridge
(`sss_memory_add_typed` -> `ce_storage_add_typed`) so cells land in real
CEStorage as CE_TYPE_IMAGE entries with type-tagged keyframe + delta.

Out-of-band metadata (quality, source, use_count, generation) is kept
Python-side because it does not belong in CEStorage entries; both views
share the same block_idx so they line up 1:1.

Usage
-----
    from tools.sss_memory import CEMemory
    mem = CEMemory("/tmp/sss_mem")     # initialises ctypes lib + sidecar
    cid = mem.add_cell("top", canny, depth, color_avg, quality=0.5)
    mem.save()                          # writes /tmp/sss_mem/storage.ces

The library is built by `make pybridge` (produces
`build/libsss_pybridge.so`). Path resolution:
  1) $SSS_PYBRIDGE_LIB if set
  2) build/libsss_pybridge.so under the repo root
  3) ./libsss_pybridge.so
"""
from __future__ import annotations

import ctypes
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

# Block-name -> CEStorage slot. Matches ROW_BLOCKS order in the
# self-upgrade script.
_BLOCK_SLOTS = {"top": 0, "mid_top": 1, "mid_bot": 2, "bot": 3}

# CEType.CE_TYPE_IMAGE — must stay in sync with ce_core/ce_type.h.
_CE_TYPE_IMAGE = 1

_DEFAULT_CANVAS_ID = 1


def _find_lib() -> str:
    env = os.environ.get("SSS_PYBRIDGE_LIB")
    if env and os.path.exists(env):
        return env
    here = Path(__file__).resolve().parent
    repo = here.parent
    candidates = [
        repo / "build" / "libsss_pybridge.so",
        repo / "libsss_pybridge.so",
        Path.cwd() / "build" / "libsss_pybridge.so",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "libsss_pybridge.so not found. Run `make pybridge` from the repo root, "
        "or set SSS_PYBRIDGE_LIB to the .so path."
    )


def _load_lib():
    lib = ctypes.CDLL(_find_lib())

    lib.sss_memory_create.argtypes = [ctypes.c_uint32]
    lib.sss_memory_create.restype = ctypes.c_void_p

    lib.sss_memory_destroy.argtypes = [ctypes.c_void_p]
    lib.sss_memory_destroy.restype = None

    lib.sss_memory_add_typed.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint16, ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
    ]
    lib.sss_memory_add_typed.restype = ctypes.c_uint32

    lib.sss_memory_count.argtypes = [ctypes.c_void_p]
    lib.sss_memory_count.restype = ctypes.c_uint32

    lib.sss_memory_count_by_slot.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint16,
    ]
    lib.sss_memory_count_by_slot.restype = ctypes.c_uint32

    lib.sss_memory_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.sss_memory_save.restype = ctypes.c_int

    lib.sss_memory_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.sss_memory_load.restype = ctypes.c_int

    lib.sss_memory_get_keyframe.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint16, ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.sss_memory_get_keyframe.restype = ctypes.c_int

    return lib


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = _load_lib()
    return _LIB


def _slot_for(block_name: str) -> int:
    if block_name not in _BLOCK_SLOTS:
        raise KeyError(
            f"unknown block_name {block_name!r}; expected one of {list(_BLOCK_SLOTS)}"
        )
    return _BLOCK_SLOTS[block_name]


def _encode_block_bytes(canny, depth, color_avg) -> bytes:
    """Pack the cell's signal into a contiguous byte stream for ce_feed.

    The C side hashes whatever bytes we hand it into a 64-byte CEUnit, so
    layout only matters for distinguishability — same bytes -> same
    keyframe. We concatenate color_avg (3 bytes), canny edges, then
    depth (if present). Order is fixed so re-ingest is deterministic.
    """
    chunks = []

    if isinstance(color_avg, np.ndarray):
        ca = np.clip(color_avg, 0, 255).astype(np.uint8).tobytes()
    else:
        ca = bytes(int(max(0, min(255, v))) for v in color_avg)
    chunks.append(ca)

    if canny is not None:
        chunks.append(np.ascontiguousarray(canny, dtype=np.uint8).tobytes())

    if depth is not None:
        chunks.append(np.ascontiguousarray(depth, dtype=np.uint8).tobytes())

    return b"".join(chunks)


class CEMemory:
    """CE Cell store backed by ce_storage_add_typed.

    Public surface matches the in-memory class in the SSS self-upgrade
    script: add_cell / get_best_cells / update_cell_quality / get_stats /
    save. Calling code does not need to know the C bridge exists.
    """

    def __init__(self, db_path: str, canvas_id: int = _DEFAULT_CANVAS_ID,
                 initial_capacity: int = 1024):
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.canvas_id = int(canvas_id)
        self._handle = _lib().sss_memory_create(int(initial_capacity))
        if not self._handle:
            raise RuntimeError("sss_memory_create returned NULL")
        # Sidecar metadata, keyed by block_name -> list[cell_dict]. The
        # list index equals block_idx in CEStorage so the two views stay
        # aligned with no extra bookkeeping.
        self.cells = defaultdict(list)
        self.generation = 0
        self.stats = {"adds": 0, "rejects": 0, "updates": 0}

    def __del__(self):
        try:
            if getattr(self, "_handle", None):
                _lib().sss_memory_destroy(self._handle)
                self._handle = None
        except Exception:
            pass

    # ── primary API ────────────────────────────────────────────────

    def add_cell(self, block_name, canny, depth, color_avg, quality,
                 source: str = "") -> int:
        """Append a typed cell. Returns its block_idx (0-based per slot)."""
        slot = _slot_for(block_name)
        payload = _encode_block_bytes(canny, depth, color_avg)
        buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        bidx = _lib().sss_memory_add_typed(
            self._handle, self.canvas_id, slot, _CE_TYPE_IMAGE,
            buf, len(payload),
        )
        if bidx == 0xFFFFFFFF:
            raise RuntimeError("sss_memory_add_typed failed (OOM or bad input)")

        cell = {
            "id": int(bidx),
            "canny": canny,
            "depth": depth,
            "color_avg": color_avg.tolist() if isinstance(color_avg, np.ndarray) else list(color_avg),
            "quality": float(quality),
            "generation": self.generation,
            "source": source,
            "use_count": 0,
        }
        self.cells[block_name].append(cell)
        self.stats["adds"] += 1
        return int(bidx)

    def get_best_cells(self, block_name, n=5):
        cells = self.cells.get(block_name, [])
        return sorted(cells, key=lambda c: c["quality"], reverse=True)[:n]

    def update_cell_quality(self, block_name, cell_id, new_quality):
        cell = self.cells[block_name][cell_id]
        alpha = 0.3
        cell["quality"] = cell["quality"] * (1 - alpha) + new_quality * alpha
        cell["use_count"] += 1
        self.stats["updates"] += 1

    def get_stats(self):
        total = sum(len(v) for v in self.cells.values())
        avg_q = 0.0
        if total > 0:
            all_q = [c["quality"] for cells in self.cells.values() for c in cells]
            avg_q = float(np.mean(all_q))
        return {
            "total_cells": total,
            "avg_quality": round(avg_q, 4),
            "blocks": {k: len(v) for k, v in self.cells.items()},
            "ce_storage_count": int(_lib().sss_memory_count(self._handle)),
            **self.stats,
        }

    def save(self):
        ces_path = os.path.join(self.db_path, "storage.ces")
        ok = _lib().sss_memory_save(self._handle, ces_path.encode("utf-8"))
        meta = {
            "generation": self.generation,
            "stats": self.stats,
            "block_counts": {k: len(v) for k, v in self.cells.items()},
            "ces_path": ces_path,
            "ces_saved": bool(ok),
        }
        with open(os.path.join(self.db_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ── pass-throughs to the bridge for callers that want raw CE access ──

    def ce_storage_count(self) -> int:
        return int(_lib().sss_memory_count(self._handle))

    def ce_storage_count_by_slot(self, block_name: str) -> int:
        return int(_lib().sss_memory_count_by_slot(
            self._handle, self.canvas_id, _slot_for(block_name)))

    def get_keyframe(self, block_name: str, block_idx: int) -> bytes:
        """Return the 64-byte CEUnit keyframe for a given cell, or None."""
        out = (ctypes.c_uint8 * 64)()
        ok = _lib().sss_memory_get_keyframe(
            self._handle, self.canvas_id, _slot_for(block_name),
            int(block_idx), out)
        return bytes(out) if ok else None
