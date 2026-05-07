"""sss_atmos — Python wrapper for the Atmos scene-object engine.

Drives `ce_scene_build_from_rgba` / `ce_scene_tick_with_profiles` /
`ce_scene_render` / `ce_scene_persist_to_storage` through the
sss_pybridge ctypes ABI.

Usage
-----
    from tools.sss_atmos import AtmosScene, MOTION_NAMES

    scene = AtmosScene.from_rgba(rgba, w, h)
    print(scene.objects)             # list[ObjectSummary]
    for tick in range(60):
        frame = scene.render(width=256, height=256)
        scene.tick()
    scene.seek(0)

The opaque C handle is owned by the AtmosScene; it's released either
explicitly (`scene.close()`) or when the object is garbage-collected.
The shared library is the same one tools/sss_memory.py loads —
`build/libsss_pybridge.so`, produced by `make pybridge`.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


# Mirrors CEMotionType in ce_move_profile.h — the row payload returned
# by sss_atmos_scene_dump_objects encodes the type as a single byte.
MOTION_NAMES = {
    0: "STATIC",
    1: "FLOW",
    2: "RIGID",
    3: "ORGANIC",
    4: "PARTICLE",
}


@dataclass
class ObjectSummary:
    """Per-object snapshot returned by AtmosScene.objects.

    `motion_type` is the integer code (see MOTION_NAMES); cx/cy live in
    the same 0..255 normalised space the renderer uses internally.
    """
    id: int
    motion_type: int
    base_cx: int
    base_cy: int
    color_r: int
    color_g: int
    color_b: int

    @property
    def motion_name(self) -> str:
        return MOTION_NAMES.get(self.motion_type, "UNKNOWN")


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
        "libsss_pybridge.so not found. Run `make pybridge` from the repo "
        "root, or set SSS_PYBRIDGE_LIB to the .so path."
    )


_LIB: Optional[ctypes.CDLL] = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ctypes.CDLL(_find_lib())

    lib.sss_atmos_scene_create.argtypes = []
    lib.sss_atmos_scene_create.restype = ctypes.c_void_p

    lib.sss_atmos_scene_destroy.argtypes = [ctypes.c_void_p]
    lib.sss_atmos_scene_destroy.restype = None

    lib.sss_atmos_scene_build_from_rgba.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int, ctypes.c_uint16,
    ]
    lib.sss_atmos_scene_build_from_rgba.restype = ctypes.c_uint16

    lib.sss_atmos_scene_object_count.argtypes = [ctypes.c_void_p]
    lib.sss_atmos_scene_object_count.restype = ctypes.c_uint16

    lib.sss_atmos_scene_dump_objects.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint16,
    ]
    lib.sss_atmos_scene_dump_objects.restype = ctypes.c_uint16

    lib.sss_atmos_scene_tick.argtypes = [ctypes.c_void_p]
    lib.sss_atmos_scene_tick.restype = None

    lib.sss_atmos_scene_seek.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.sss_atmos_scene_seek.restype = None

    lib.sss_atmos_scene_render.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int,
    ]
    lib.sss_atmos_scene_render.restype = None

    lib.sss_atmos_scene_persist.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint16,
    ]
    lib.sss_atmos_scene_persist.restype = ctypes.c_uint16

    _LIB = lib
    return lib


class AtmosScene:
    """Pythonic wrapper around the C `sss_atmos_scene` handle.

    The handle is opened lazily by `AtmosScene.from_rgba` and released
    by `close()` (also wired to `__del__` so accidental leaks during
    REPL use just turn into garbage-collected closes).
    """

    def __init__(self):
        self._lib = _load_lib()
        self._handle = self._lib.sss_atmos_scene_create()
        if not self._handle:
            raise MemoryError("sss_atmos_scene_create returned NULL")

    # ── factories ──────────────────────────────────────────────────
    @classmethod
    def from_rgba(cls, rgba: np.ndarray, width: int, height: int,
                  base_id: int = 0) -> "AtmosScene":
        """Decompose an RGBA image into Atmos scene objects.

        `rgba` must be a row-major H*W*4 uint8 array (or anything
        np.ascontiguousarray turns into one). Returns a fresh
        AtmosScene populated with the discovered objects + profiles.
        """
        scene = cls()
        scene.build_from_rgba(rgba, width, height, base_id=base_id)
        return scene

    # ── instance API ───────────────────────────────────────────────
    def build_from_rgba(self, rgba: np.ndarray, width: int, height: int,
                        base_id: int = 0) -> int:
        """Run ce_scene_build_from_rgba on the supplied RGBA buffer.

        Returns the number of objects found. Subsequent calls overwrite
        the scene (the C side calls ce_scene_init).
        """
        arr = np.ascontiguousarray(rgba, dtype=np.uint8)
        if arr.size != width * height * 4:
            raise ValueError(
                f"rgba size mismatch: got {arr.size} bytes, "
                f"expected {width*height*4} for {width}x{height}"
            )
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        return int(self._lib.sss_atmos_scene_build_from_rgba(
            self._handle, ptr, int(width), int(height), int(base_id)
        ))

    @property
    def object_count(self) -> int:
        return int(self._lib.sss_atmos_scene_object_count(self._handle))

    @property
    def objects(self) -> List[ObjectSummary]:
        n = self.object_count
        if n == 0:
            return []
        buf = (ctypes.c_uint8 * (n * 8))()
        wrote = self._lib.sss_atmos_scene_dump_objects(
            self._handle, buf, ctypes.c_uint16(n)
        )
        out = []
        for i in range(int(wrote)):
            row = buf[i * 8:(i + 1) * 8]
            obj_id = row[0] | (row[1] << 8)
            out.append(ObjectSummary(
                id=obj_id,
                motion_type=row[2],
                base_cx=row[3],
                base_cy=row[4],
                color_r=row[5],
                color_g=row[6],
                color_b=row[7],
            ))
        return out

    def tick(self) -> None:
        self._lib.sss_atmos_scene_tick(self._handle)

    def seek(self, target_tick: int) -> None:
        self._lib.sss_atmos_scene_seek(self._handle, int(target_tick))

    def render(self, width: int = 128, height: int = 128) -> np.ndarray:
        """Render the current scene state into an HxWx3 uint8 array."""
        if width <= 0 or height <= 0:
            raise ValueError("width/height must be positive")
        out = np.empty((height, width, 3), dtype=np.uint8)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self._lib.sss_atmos_scene_render(self._handle, ptr,
                                         int(width), int(height))
        return out

    def persist(self, memory, canvas_id: int, scene_id: int) -> int:
        """Persist scene objects through the sss_memory CEStorage handle.

        `memory` must be a `tools.sss_memory.CEMemory` (or anything that
        exposes the same `_handle` attribute pointing at the C
        sss_memory). Returns the number of CE_TYPE_SCENE entries
        written.
        """
        handle = getattr(memory, "_handle", None)
        if handle is None:
            raise TypeError(
                "memory must expose a `_handle` ctypes pointer "
                "(use tools.sss_memory.CEMemory)"
            )
        return int(self._lib.sss_atmos_scene_persist(
            handle, self._handle,
            ctypes.c_uint32(canvas_id),
            ctypes.c_uint16(scene_id),
        ))

    def close(self) -> None:
        if self._handle:
            self._lib.sss_atmos_scene_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ["AtmosScene", "ObjectSummary", "MOTION_NAMES"]
