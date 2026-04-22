#!/usr/bin/env python3
"""
animate_training.py — render a twinkling-grid animation from a
stream_train event log.

Input
-----
Binary "CEVT" log produced by `stream_train --log <path>`:

    header (12 B):  "CEVT" + u32 version + u32 reserved
    per record  :   u32 clause_idx
                    u8  decision ( 0=new KF, 1=delta, 2=skip )
                    u16 byte_count
                    u8[byte_count * 2]   (y, x) pairs

Each (y, x) pair is one cell that the clause's bytes write into
on the 256x256 grid. Replaying these gives a live-looking
"data flowing into cells" animation instead of a post-hoc
snapshot.

Output
------
MP4 at 1280x720 with:
    left   — 720x720 grid (256x256 upscaled nearest-neighbor),
             each clause flashes white on its cells, then fades
             to cyan (new KF) or magenta (delta) over ~12 frames
    right  — HUD: clause index, elapsed, KF / delta counters,
             decision rate histograms
    grid   — faint per-16-cell gridlines for the "checkerboard"
             look; fully hidden behind bright cells

Usage
-----
    python tools/animate_training.py build/models/events.log
    python tools/animate_training.py events.log \
        --out training_animation.mp4 \
        --seconds 45 --fps 30

Batch size per frame auto-scales so the whole log fits into the
requested duration; use --clauses-per-frame to override.
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GRID       = 256
GRID_PIX   = 720     # left panel size
FRAME_W    = 1280
FRAME_H    = 720

DECAY      = 0.92    # per-frame brightness / color decay
FLASH_BOOST = 2.0    # how bright the hit frame is before decay

COLOR_NEW_KF  = (0.30, 1.00, 1.00)   # cyan
COLOR_DELTA   = (1.00, 0.30, 1.00)   # magenta
COLOR_SKIP    = (0.70, 0.70, 0.70)   # grey / silver


# ──────────────────────────────────────────────── log reader

def read_events(path):
    """Stream the event log; yield (clause_idx, decision, (ys, xs))."""
    with open(path, 'rb') as f:
        hdr = f.read(12)
        if len(hdr) != 12 or hdr[:4] != b'CEVT':
            raise ValueError(f'not a CEVT log: {path}')
        (version, _res) = struct.unpack('<II', hdr[4:])
        if version != 1:
            raise ValueError(f'unsupported event log version: {version}')
        while True:
            head = f.read(4 + 1 + 2)
            if len(head) < 7:
                return
            clause_idx, decision, n_bytes = struct.unpack('<IBH', head)
            payload = f.read(n_bytes * 2)
            if len(payload) != n_bytes * 2:
                return
            arr = np.frombuffer(payload, dtype=np.uint8).reshape(n_bytes, 2)
            yield clause_idx, decision, arr[:, 0], arr[:, 1]


# ──────────────────────────────────────────────── rendering

def load_font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              'C:/Windows/Fonts/consola.ttf',
              'C:/Windows/Fonts/arial.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render_frame(brightness, color_r, color_g, color_b,
                 hud, ft_title, ft_stat, ft_sm):
    img  = Image.new('RGB', (FRAME_W, FRAME_H), (10, 10, 14))
    draw = ImageDraw.Draw(img)

    # Grid panel: brightness * color, upscaled nearest to 720x720.
    r = np.clip(color_r * brightness * 255.0, 0, 255).astype(np.uint8)
    g = np.clip(color_g * brightness * 255.0, 0, 255).astype(np.uint8)
    b = np.clip(color_b * brightness * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    grid_img = Image.fromarray(rgb, 'RGB').resize((GRID_PIX, GRID_PIX), Image.NEAREST)
    img.paste(grid_img, (0, 0))

    # Faint per-16 gridlines (every 16 cells = 45 px in 720 space).
    step = GRID_PIX // 16
    for i in range(1, 16):
        p = i * step
        draw.line([(p, 0), (p, GRID_PIX)], fill=(30, 30, 40), width=1)
        draw.line([(0, p), (GRID_PIX, p)], fill=(30, 30, 40), width=1)

    # HUD on the right.
    hx = GRID_PIX + 24
    draw.text((hx, 12), "SPATIAL-PATTERN-AI", fill=(240, 240, 240), font=ft_title)
    draw.text((hx, 44), "live training replay",    fill=(120, 180, 255), font=ft_stat)

    y = 84
    lines = [
        f"clause:        {hud['clause']:>6} / {hud['total']}",
        f"time:          {hud['time']:>6.1f} s",
        "",
        f"new keyframes: {hud['kf']:>6}",
        f"deltas:        {hud['dl']:>6}",
        f"skipped:       {hud['sk']:>6}",
        "",
        f"active cells:  {hud['active']:>6}",
        f"delta rate:    {hud['dr']:>6.1f} %",
        "",
        f"legend:",
    ]
    for line in lines:
        draw.text((hx, y), line, fill=(210, 210, 210), font=ft_stat); y += 22

    # Legend colour swatches.
    for (col, label) in [(COLOR_NEW_KF, "new KF (cyan)"),
                         (COLOR_DELTA,  "delta (magenta)"),
                         (COLOR_SKIP,   "skip (grey)")]:
        cr, cg, cb = (int(c * 255) for c in col)
        draw.rectangle([(hx, y + 4), (hx + 16, y + 18)], fill=(cr, cg, cb))
        draw.text((hx + 24, y), label, fill=(200, 200, 200), font=ft_stat)
        y += 22

    # Progress bar.
    bar_y = FRAME_H - 60
    progress = (hud['clause'] + 1) / max(hud['total'], 1)
    draw.rectangle([(24, bar_y), (GRID_PIX - 24, bar_y + 14)], outline=(90, 90, 110))
    draw.rectangle([(24, bar_y),
                    (24 + int((GRID_PIX - 48) * progress), bar_y + 14)],
                   fill=(90, 220, 140))
    draw.text((24, bar_y + 20),
              f"{int(progress * 100)}%   {hud['clause']+1}/{hud['total']} clauses",
              fill=(180, 180, 200), font=ft_sm)

    draw.text((FRAME_W - 320, FRAME_H - 22),
              "github.com/sjpupro-lab/CANVAS",
              fill=(60, 60, 80), font=ft_sm)

    return img


# ──────────────────────────────────────────────── main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="CEVT event log from stream_train --log")
    ap.add_argument("--out", default=None,
                    help="output MP4 (default: same dir as log, .mp4 extension)")
    ap.add_argument("--seconds", type=float, default=45.0,
                    help="target video duration (default 45 s)")
    ap.add_argument("--fps", type=int, default=30,
                    help="frame rate (default 30)")
    ap.add_argument("--clauses-per-frame", type=int, default=0,
                    help="fixed batch size; 0 = auto-scale to --seconds")
    ap.add_argument("--keep-frames", action="store_true",
                    help="keep the temporary PNG frames")
    args = ap.parse_args()

    if args.out is None:
        base = os.path.splitext(args.log)[0]
        args.out = base + '_animation.mp4'

    # First pass: count clauses to plan batch size.
    total = 0
    for _ in read_events(args.log):
        total += 1
    if total == 0:
        print("empty log")
        return 1

    target_frames = max(int(args.seconds * args.fps), 1)
    batch = (args.clauses_per_frame
             if args.clauses_per_frame > 0
             else max(1, (total + target_frames - 1) // target_frames))
    frames = (total + batch - 1) // batch

    print(f"[anim] log: {args.log}")
    print(f"[anim] clauses: {total:,}  batch: {batch}  frames: {frames}  duration: {frames/args.fps:.1f} s")
    print(f"[anim] out:  {args.out}")

    # State.
    bright  = np.zeros((GRID, GRID), dtype=np.float32)
    color_r = np.zeros((GRID, GRID), dtype=np.float32)
    color_g = np.zeros((GRID, GRID), dtype=np.float32)
    color_b = np.zeros((GRID, GRID), dtype=np.float32)
    kf_n = dl_n = sk_n = 0

    ft_title = load_font(26)
    ft_stat  = load_font(17)
    ft_sm    = load_font(13)

    tmpdir = tempfile.mkdtemp(prefix='canvas_anim_')
    frame_idx = 0
    clause_ctr = 0

    def hud_dict():
        total_seen = kf_n + dl_n + sk_n
        return {
            'clause': clause_ctr,
            'total':  total,
            'time':   frame_idx / args.fps,
            'kf':     kf_n,
            'dl':     dl_n,
            'sk':     sk_n,
            'active': int((bright > 0.05).sum()),
            'dr':     (dl_n / total_seen * 100.0) if total_seen else 0.0,
        }

    batch_iter = []

    def flush_batch():
        nonlocal frame_idx, bright, color_r, color_g, color_b, kf_n, dl_n, sk_n
        # Decay first, then write new hits so freshest cells are brightest.
        bright  *= DECAY
        color_r *= DECAY
        color_g *= DECAY
        color_b *= DECAY

        for (idx, decision, ys, xs) in batch_iter:
            if decision == 0:
                kf_n += 1
                cr, cg, cb = COLOR_NEW_KF
            elif decision == 1:
                dl_n += 1
                cr, cg, cb = COLOR_DELTA
            else:
                sk_n += 1
                cr, cg, cb = COLOR_SKIP
            bright[ys, xs]  = np.maximum(bright[ys, xs], FLASH_BOOST)
            color_r[ys, xs] = cr
            color_g[ys, xs] = cg
            color_b[ys, xs] = cb

        img = render_frame(np.minimum(bright, 1.0), color_r, color_g, color_b,
                           hud_dict(), ft_title, ft_stat, ft_sm)
        img.save(os.path.join(tmpdir, f'frame_{frame_idx:06d}.png'))
        frame_idx += 1

    for idx, decision, ys, xs in read_events(args.log):
        batch_iter.append((idx, decision, ys, xs))
        clause_ctr = idx + 1
        if len(batch_iter) >= batch:
            flush_batch()
            batch_iter = []

    if batch_iter:
        flush_batch()

    # 2 seconds of final still so the viewer sees the steady state.
    for _ in range(args.fps * 2):
        bright  *= DECAY
        color_r *= DECAY
        color_g *= DECAY
        color_b *= DECAY
        img = render_frame(np.minimum(bright, 1.0), color_r, color_g, color_b,
                           hud_dict(), ft_title, ft_stat, ft_sm)
        img.save(os.path.join(tmpdir, f'frame_{frame_idx:06d}.png'))
        frame_idx += 1

    # Encode.
    cmd = ['ffmpeg', '-y',
           '-framerate', str(args.fps),
           '-i', os.path.join(tmpdir, 'frame_%06d.png'),
           '-vf', 'format=yuv420p',
           '-c:v', 'libx264',
           '-preset', 'medium',
           '-crf', '20',
           args.out]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"\nffmpeg not installed; PNG frames left in {tmpdir}")
        return 0

    if result.returncode != 0:
        print(f"\nffmpeg failed:\n{result.stderr[-400:]}")
        return 1

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[anim] wrote {args.out} ({size_mb:.1f} MB, {frame_idx} frames)")

    if not args.keep_frames:
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)
    else:
        print(f"[anim] frames kept in {tmpdir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
