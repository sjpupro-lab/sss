#!/usr/bin/env python3
"""PNG -> PPM P6 256x256 converter (stdlib-only).

Decodes 8-bit RGB or RGBA non-interlaced PNGs and downsamples to
GRID_SIZE x GRID_SIZE (256x256) via box-average. PPM has no alpha,
so RGBA is composited on white.

Usage: png_to_ppm256.py <in.png> <out.ppm>
"""
import struct
import sys
import zlib

GRID = 256


def decode_png(path):
    with open(path, "rb") as fp:
        data = fp.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if comp != 0 or filt != 0 or interlace != 0:
                raise ValueError("only standard non-interlaced PNGs supported")
            if bit_depth != 8:
                raise ValueError(f"only 8-bit channels supported (got {bit_depth})")
            if color_type not in (2, 6):
                raise ValueError(f"only RGB(2)/RGBA(6) supported (got color_type {color_type})")
        elif ctype == b"IDAT":
            idat += chunk
        elif ctype == b"IEND":
            break

    bpp = 3 if color_type == 2 else 4
    raw = zlib.decompress(bytes(idat))
    stride = width * bpp
    out = bytearray(width * height * bpp)
    prev = bytearray(stride)
    rp = 0
    op = 0
    for _ in range(height):
        ftype = raw[rp]
        rp += 1
        line = bytearray(raw[rp : rp + stride])
        rp += stride
        if ftype == 0:
            pass
        elif ftype == 1:  # Sub
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa = abs(p - a)
                pb = abs(p - b)
                pc = abs(p - c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        else:
            raise ValueError(f"bad PNG filter {ftype}")
        out[op : op + stride] = line
        op += stride
        prev = line

    if color_type == 6:  # composite RGBA on white
        rgb = bytearray(width * height * 3)
        for i in range(width * height):
            r, g, b, a = out[i * 4 : i * 4 + 4]
            ia = 255 - a
            rgb[i * 3 + 0] = (r * a + 255 * ia + 127) // 255
            rgb[i * 3 + 1] = (g * a + 255 * ia + 127) // 255
            rgb[i * 3 + 2] = (b * a + 255 * ia + 127) // 255
        out = rgb
    return width, height, bytes(out)


def resize_to_grid(width, height, rgb):
    """Box-average downsample / nearest-neighbor upsample to GRID x GRID."""
    out = bytearray(GRID * GRID * 3)
    for oy in range(GRID):
        sy0 = (oy * height) // GRID
        sy1 = max(sy0 + 1, ((oy + 1) * height) // GRID)
        for ox in range(GRID):
            sx0 = (ox * width) // GRID
            sx1 = max(sx0 + 1, ((ox + 1) * width) // GRID)
            r = g = b = n = 0
            for sy in range(sy0, sy1):
                base = sy * width
                for sx in range(sx0, sx1):
                    p = (base + sx) * 3
                    r += rgb[p]
                    g += rgb[p + 1]
                    b += rgb[p + 2]
                    n += 1
            o = (oy * GRID + ox) * 3
            out[o] = r // n
            out[o + 1] = g // n
            out[o + 2] = b // n
    return bytes(out)


def main():
    if len(sys.argv) != 3:
        sys.stderr.write(f"usage: {sys.argv[0]} <in.png> <out.ppm>\n")
        sys.exit(2)
    in_path, out_path = sys.argv[1], sys.argv[2]
    w, h, rgb = decode_png(in_path)
    sys.stderr.write(f"[png_to_ppm256] {in_path}: {w}x{h} -> {GRID}x{GRID}\n")
    pixels = resize_to_grid(w, h, rgb)
    with open(out_path, "wb") as fp:
        fp.write(f"P6\n{GRID} {GRID}\n255\n".encode())
        fp.write(pixels)
    sys.stderr.write(f"[png_to_ppm256] wrote {out_path}\n")


if __name__ == "__main__":
    main()
