"""SSS image I/O — pure Python + numpy + zlib.

Stdlib-only PNG / PPM encoder + decoder so the SSS pipeline and the
unified UI can render images without depending on cv2 or Pillow. Built
to run unmodified on Termux.

Public surface (what the rest of the codebase imports):

    numpy_to_png_bytes(arr)        -> bytes
    numpy_to_png_data_uri(arr)     -> "data:image/png;base64,..."
    write_png(path, arr)
    read_png(path)                 -> ndarray (BGR or grayscale)
    write_ppm(path, arr)
    read_ppm(path)                 -> ndarray (BGR)
    imwrite(path, arr)             -> True   (auto-dispatch by extension)
    imencode(fmt, arr)             -> (True, bytes)

Conventions match the rest of SSS / cv2: input/output 3-channel arrays
are BGR, uint8. Internally we transpose to RGB on write and back on
read so the on-disk PNG/PPM stays standard.
"""
from __future__ import annotations

import base64
import os
import struct
import zlib
from typing import Tuple

import numpy as np

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# ────────────────────────────────────────────────────────────
# PNG encoder
# ────────────────────────────────────────────────────────────
def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


def numpy_to_png_bytes(arr) -> bytes:
    """Encode a numpy array as PNG. Accepts:
        (H, W)         uint8  → grayscale
        (H, W, 3)      uint8  → BGR (transposed to RGB before encoding)
        (H, W, 4)      uint8  → BGRA (transposed to RGBA before encoding)
    Other dtypes are clipped to [0, 255] and cast to uint8.
    Returns the full PNG file contents.
    """
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)

    if a.ndim == 2:
        h, w = a.shape
        color_type = 0  # grayscale
        pixels = a
    elif a.ndim == 3 and a.shape[2] == 3:
        h, w = a.shape[:2]
        color_type = 2  # RGB
        # BGR → RGB so the on-disk PNG renders correctly elsewhere.
        pixels = a[..., ::-1]
    elif a.ndim == 3 and a.shape[2] == 4:
        h, w = a.shape[:2]
        color_type = 6  # RGBA
        pixels = a[..., [2, 1, 0, 3]]
    elif a.ndim == 3 and a.shape[2] == 1:
        h, w = a.shape[:2]
        color_type = 0
        pixels = a[..., 0]
    else:
        raise ValueError(f"unsupported array shape {a.shape}")

    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)

    # Prepend a filter byte (type 0 = None) to every scanline. Vectorised
    # via column-stack rather than per-row append so we never copy in a
    # Python loop.
    flat_w = pixels.shape[1] * (1 if pixels.ndim == 2 else pixels.shape[2])
    rows = pixels.reshape(h, flat_w)
    filter_col = np.zeros((h, 1), dtype=np.uint8)
    raw = np.concatenate([filter_col, rows], axis=1).tobytes()

    bit_depth = 8
    ihdr_data = struct.pack(
        ">IIBBBBB",
        w, h, bit_depth, color_type,
        0,  # compression: deflate
        0,  # filter: PNG default
        0,  # interlace: none
    )

    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr_data)
        + _png_chunk(b"IDAT", zlib.compress(raw, 6))
        + _png_chunk(b"IEND", b"")
    )


def numpy_to_png_data_uri(arr) -> str:
    """Return a `data:image/png;base64,...` URI for direct use as an
    `<img src=>`. Same input contract as numpy_to_png_bytes."""
    png = numpy_to_png_bytes(arr)
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def write_png(path, arr) -> bool:
    with open(path, "wb") as f:
        f.write(numpy_to_png_bytes(arr))
    return True


# ────────────────────────────────────────────────────────────
# PNG decoder (8-bit grayscale / RGB / RGBA, non-interlaced)
# ────────────────────────────────────────────────────────────
def _png_unfilter(row: np.ndarray, prev_row: np.ndarray,
                  filt_type: int, bpp: int) -> np.ndarray:
    n = row.size
    out = row.astype(np.int32, copy=True)
    if filt_type == 0:                    # None
        return out.astype(np.uint8)
    if filt_type == 2:                    # Up — fully vectorisable
        return ((out + prev_row.astype(np.int32)) & 0xFF).astype(np.uint8)

    prev = prev_row.astype(np.int32)
    if filt_type == 1:                    # Sub
        for i in range(bpp, n):
            out[i] = (out[i] + out[i - bpp]) & 0xFF
        return out.astype(np.uint8)
    if filt_type == 3:                    # Average
        for i in range(n):
            left = out[i - bpp] if i >= bpp else 0
            up = prev[i]
            out[i] = (out[i] + (left + up) // 2) & 0xFF
        return out.astype(np.uint8)
    if filt_type == 4:                    # Paeth
        for i in range(n):
            a = out[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            if pa <= pb and pa <= pc:
                pred = a
            elif pb <= pc:
                pred = b
            else:
                pred = c
            out[i] = (out[i] + pred) & 0xFF
        return out.astype(np.uint8)

    raise ValueError(f"unknown PNG filter type {filt_type}")


def _decode_png(data: bytes) -> np.ndarray:
    if data[:8] != _PNG_SIGNATURE:
        raise ValueError("not a PNG file")

    width = height = bit_depth = color_type = None
    idat = bytearray()
    pos = 8
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length  # length(4) + type(4) + data + crc(4)
        if ctype == b"IHDR":
            (width, height, bit_depth, color_type, comp, filt,
             interlace) = struct.unpack(">IIBBBBB", body)
            if comp != 0 or filt != 0:
                raise ValueError("unsupported PNG: nonzero comp/filter")
            if interlace != 0:
                raise ValueError("interlaced PNG not supported")
            if bit_depth != 8:
                raise ValueError(f"only 8-bit PNG supported (got {bit_depth})")
        elif ctype == b"IDAT":
            idat.extend(body)
        elif ctype == b"IEND":
            break

    if width is None:
        raise ValueError("PNG: missing IHDR")
    bpp = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if bpp is None:
        raise ValueError(f"unsupported PNG color_type {color_type}")

    raw = zlib.decompress(bytes(idat))
    stride = width * bpp
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    src = 0
    for y in range(height):
        ftype = raw[src]; src += 1
        row = np.frombuffer(raw[src:src + stride], dtype=np.uint8).copy()
        src += stride
        out[y] = _png_unfilter(row, prev, ftype, bpp)
        prev = out[y]

    if color_type == 0:
        return out.reshape(height, width)
    if color_type == 2:
        rgb = out.reshape(height, width, 3)
        return np.ascontiguousarray(rgb[..., ::-1])           # → BGR
    if color_type == 4:
        return out.reshape(height, width, 2)
    # color_type == 6
    rgba = out.reshape(height, width, 4)
    return np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])      # → BGRA


def read_png(path) -> np.ndarray:
    with open(path, "rb") as f:
        return _decode_png(f.read())


# ────────────────────────────────────────────────────────────
# PPM (P6) — already used by tools/sss_unified.py
# ────────────────────────────────────────────────────────────
def write_ppm(path, arr) -> bool:
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        rgb = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[2] == 3:
        rgb = a[..., ::-1]                # BGR → RGB
    else:
        raise ValueError(f"PPM: unsupported shape {a.shape}")
    h, w = rgb.shape[:2]
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb.tobytes())
    return True


def _ppm_token(f) -> bytes:
    """Read one whitespace-delimited token from a PPM stream, skipping
    `# ...` comments. Returns the token bytes, or b'' at EOF."""
    out = bytearray()
    while True:
        ch = f.read(1)
        if not ch:
            return bytes(out)
        if ch == b"#":
            while ch and ch != b"\n":
                ch = f.read(1)
            continue
        if ch in (b" ", b"\t", b"\n", b"\r"):
            if out:
                return bytes(out)
            continue
        out += ch


def read_ppm(path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = _ppm_token(f)
        if magic != b"P6":
            raise ValueError(f"not a P6 PPM (magic={magic!r})")
        w = int(_ppm_token(f))
        h = int(_ppm_token(f))
        maxval = int(_ppm_token(f))
        if maxval != 255:
            raise ValueError(f"PPM maxval {maxval} not supported (need 255)")
        body = f.read(h * w * 3)
    rgb = np.frombuffer(body, dtype=np.uint8).reshape(h, w, 3)
    return np.ascontiguousarray(rgb[..., ::-1])               # RGB → BGR


# ────────────────────────────────────────────────────────────
# cv2-style dispatchers
# ────────────────────────────────────────────────────────────
def imwrite(path, arr) -> bool:
    """cv2.imwrite-compatible. Routes by extension. Supports
    .png and .ppm. Raises NotImplementedError for .jpg/.jpeg —
    pure-Python JPEG is impractical without a DCT/Huffman codec; use
    cv2 or Pillow if you need JPEG."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".png":
        return write_png(path, arr)
    if ext == ".ppm":
        return write_ppm(path, arr)
    if ext in (".jpg", ".jpeg"):
        raise NotImplementedError(
            "JPEG encoding requires cv2 or Pillow; "
            "tools/sss_image_io supports PNG and PPM only.")
    raise ValueError(f"imwrite: unsupported extension {ext!r}")


def imencode(fmt: str, arr) -> Tuple[bool, bytes]:
    """cv2.imencode-compatible. Returns (ok, bytes). `fmt` may be the
    extension with or without the leading dot, e.g. '.png' or 'png'."""
    f = fmt.lower().lstrip(".")
    if f == "png":
        return True, numpy_to_png_bytes(arr)
    if f == "ppm":
        a = np.asarray(arr)
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        if a.ndim == 2:
            rgb = np.stack([a, a, a], axis=-1)
        elif a.ndim == 3 and a.shape[2] == 3:
            rgb = a[..., ::-1]
        else:
            return False, b""
        h, w = rgb.shape[:2]
        return (True,
                f"P6\n{w} {h}\n255\n".encode("ascii")
                + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
    return False, b""
