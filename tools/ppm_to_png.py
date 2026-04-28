#!/usr/bin/env python3
"""Convert one or more PPM (P6) files to PNG using PIL.

Usage:
    ppm_to_png.py <in1.ppm> [<in2.ppm> ...]   # writes alongside as .png
    ppm_to_png.py --out <dir> <in1.ppm> ...   # writes into <dir>
"""
import os
import sys
from PIL import Image


def main():
    args = sys.argv[1:]
    out_dir = None
    if args and args[0] == "--out":
        if len(args) < 2:
            sys.stderr.write("error: --out requires a directory argument\n")
            print(__doc__)
            return 2
        out_dir = args[1]
        args = args[2:]
        os.makedirs(out_dir, exist_ok=True)
    if not args:
        print(__doc__)
        return 2
    for src in args:
        img = Image.open(src)
        base = os.path.splitext(os.path.basename(src))[0]
        if out_dir:
            dst = os.path.join(out_dir, base + ".png")
        else:
            dst = os.path.splitext(src)[0] + ".png"
        img.save(dst, "PNG", optimize=True)
        print(f"{src} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
