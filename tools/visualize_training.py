#!/usr/bin/env python3
"""
CANVAS training visualizer.

Reads every .spai checkpoint in a directory, renders a dashboard-style
PNG for each, then stitches them into an MP4 showing how the engine
evolves across training steps.

Per frame:
  * A-channel log heatmap (byte-frequency pattern)
  * RGB composite scaled by A (semantic / function / extended channel
    contribution, including the B channel that spec v2 Mod D
    activated via POS seeds)
  * KF / Delta / active-cells / per-channel std / adaptive weights
  * Up to 5 sample keyframe labels

File format reference (include/spatial_io.h):

  Header (32B):
    char magic[4] == "SPAI"
    uint32 version           (current = 5)
    uint32 kf_count
    uint32 df_count
    uint32 reserved[3]

  Tagged records:
    0x01  keyframe
    0x02  delta
    0x03  channel weights
    0x04  canvas
    0x05  subtitle track
    0x06  EMA tables (4 × GRID_TOTAL float, v4+)

Usage:
  python3 tools/visualize_training.py build/models
"""
import os
import sys
import struct
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GRID       = 256
GRID_TOTAL = GRID * GRID
FRAME_W    = 1280
FRAME_H    = 720

TAG_KEYFRAME = 0x01
TAG_DELTA    = 0x02
TAG_WEIGHTS  = 0x03
TAG_CANVAS   = 0x04
TAG_SUBTITLE = 0x05
TAG_EMA      = 0x06


# ─────────────────────────────────────────────────────────── .spai reader

def _read_exact(f, n, what):
    b = f.read(n)
    if len(b) != n:
        raise IOError(f"truncated while reading {what}: wanted {n}, got {len(b)}")
    return b


def _read_keyframe(f, version):
    kf_id, = struct.unpack('<I', _read_exact(f, 4, 'kf.id'))
    label  = _read_exact(f, 64, 'kf.label').split(b'\x00', 1)[0].decode('utf-8', errors='replace')
    tbc,   = struct.unpack('<I', _read_exact(f, 4, 'kf.text_byte_count'))
    if version >= 5:
        topic_hash,   = struct.unpack('<I', _read_exact(f, 4, 'kf.topic_hash'))
        seq_in_topic, = struct.unpack('<I', _read_exact(f, 4, 'kf.seq_in_topic'))
    else:
        topic_hash, seq_in_topic = 0, 0
    A = np.frombuffer(_read_exact(f, GRID_TOTAL * 2, 'kf.A'), dtype=np.uint16).reshape(GRID, GRID).copy()
    R = np.frombuffer(_read_exact(f, GRID_TOTAL,     'kf.R'), dtype=np.uint8 ).reshape(GRID, GRID).copy()
    G = np.frombuffer(_read_exact(f, GRID_TOTAL,     'kf.G'), dtype=np.uint8 ).reshape(GRID, GRID).copy()
    B = np.frombuffer(_read_exact(f, GRID_TOTAL,     'kf.B'), dtype=np.uint8 ).reshape(GRID, GRID).copy()
    return {
        'id': kf_id, 'label': label, 'byte_count': tbc,
        'topic_hash': topic_hash, 'seq_in_topic': seq_in_topic,
        'A': A, 'R': R, 'G': G, 'B': B,
    }


def _skip_delta(f, version):
    _ = struct.unpack('<I',   _read_exact(f, 4,  'df.id'))
    _ = struct.unpack('<I',   _read_exact(f, 4,  'df.parent_id'))
    _ = _read_exact(f, 64, 'df.label')
    count, = struct.unpack('<I', _read_exact(f, 4, 'df.count'))
    _ = struct.unpack('<f',   _read_exact(f, 4,  'df.change_ratio'))
    entry_size = 9 if version >= 4 else 8
    if count > 0:
        _ = _read_exact(f, count * entry_size, 'df.entries')
    return count


def _read_weights(f):
    return struct.unpack('<4f', _read_exact(f, 16, 'weights'))


def _read_ema(f):
    buf = _read_exact(f, GRID_TOTAL * 4 * 4, 'ema block')
    ema_R     = np.frombuffer(buf[0                 : GRID_TOTAL*4   ], dtype=np.float32).reshape(GRID, GRID).copy()
    ema_G     = np.frombuffer(buf[GRID_TOTAL*4      : GRID_TOTAL*4*2 ], dtype=np.float32).reshape(GRID, GRID).copy()
    ema_B     = np.frombuffer(buf[GRID_TOTAL*4*2    : GRID_TOTAL*4*3 ], dtype=np.float32).reshape(GRID, GRID).copy()
    ema_count = np.frombuffer(buf[GRID_TOTAL*4*3    : GRID_TOTAL*4*4 ], dtype=np.float32).reshape(GRID, GRID).copy()
    return ema_R, ema_G, ema_B, ema_count


def read_spai(path):
    """Return a dict of aggregated statistics for a .spai file."""
    with open(path, 'rb') as f:
        magic = _read_exact(f, 4, 'magic')
        if magic != b'SPAI':
            raise ValueError(f"not a SPAI file: {path} (magic={magic!r})")
        version, kf_count, df_count = struct.unpack('<3I', _read_exact(f, 12, 'header counts'))
        _ = _read_exact(f, 12, 'reserved')

        weights = (1.0, 1.0, 1.0, 1.0)
        ema_R = ema_G = ema_B = ema_count = np.zeros((GRID, GRID), dtype=np.float32)
        keyframes = []
        delta_entries_total = 0

        # Interleaved KF / Delta records first, then trailing tags.
        kfs_read = 0
        dfs_read = 0
        while kfs_read < kf_count or dfs_read < df_count:
            tag_byte = f.read(1)
            if not tag_byte:
                raise IOError(f"unexpected EOF in record stream ({kfs_read}/{kf_count} KF, {dfs_read}/{df_count} Δ)")
            tag = tag_byte[0]
            if tag == TAG_KEYFRAME:
                keyframes.append(_read_keyframe(f, version))
                kfs_read += 1
            elif tag == TAG_DELTA:
                delta_entries_total += _skip_delta(f, version)
                dfs_read += 1
            else:
                raise IOError(f"unexpected tag 0x{tag:02x} in record stream")

        # Trailing optional blocks. Stop at EOF or unknown tag.
        while True:
            tag_byte = f.read(1)
            if not tag_byte:
                break
            tag = tag_byte[0]
            if tag == TAG_WEIGHTS:
                weights = _read_weights(f)
            elif tag == TAG_EMA:
                ema_R, ema_G, ema_B, ema_count = _read_ema(f)
            elif tag == TAG_CANVAS:
                # Canvas records are large (~2 MB) but we only need their
                # count for reporting. Skip the raw bytes: slot_count(4)
                # + canvas_type(4) + frame_type(4) + parent_canvas_id(4)
                # + changed_ratio(4) + classified(4) + SlotMeta×32 (20B each)
                # + A (CV_TOTAL*2) + R + G + B (CV_TOTAL each).
                # CV_WIDTH = 2048, CV_HEIGHT = 1024 → CV_TOTAL = 2_097_152.
                hdr = struct.unpack('<6I', _read_exact(f, 24, 'canvas header'))
                _ = _read_exact(f, 32 * 20, 'canvas slot meta')
                cv_total = 2048 * 1024
                _ = _read_exact(f, cv_total * 2, 'canvas A')
                _ = _read_exact(f, cv_total,     'canvas R')
                _ = _read_exact(f, cv_total,     'canvas G')
                _ = _read_exact(f, cv_total,     'canvas B')
            elif tag == TAG_SUBTITLE:
                count, = struct.unpack('<I', _read_exact(f, 4, 'subtitle count'))
                if count > 0:
                    _ = _read_exact(f, count * 20, 'subtitle entries')
            else:
                # Unknown tag — stop gracefully (forward-compat).
                break

    # Aggregate channel stats over every keyframe.
    agg_A = np.zeros((GRID, GRID), dtype=np.float64)
    agg_R = np.zeros((GRID, GRID), dtype=np.float64)
    agg_G = np.zeros((GRID, GRID), dtype=np.float64)
    agg_B = np.zeros((GRID, GRID), dtype=np.float64)
    active_cells = np.zeros((GRID, GRID), dtype=np.float64)
    for kf in keyframes:
        mask = kf['A'] > 0
        agg_A         += kf['A'].astype(np.float64)
        agg_R[mask]   += kf['R'][mask].astype(np.float64)
        agg_G[mask]   += kf['G'][mask].astype(np.float64)
        agg_B[mask]   += kf['B'][mask].astype(np.float64)
        active_cells  += mask.astype(np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.where(active_cells > 0, active_cells, 1.0)
        agg_R = agg_R / denom
        agg_G = agg_G / denom
        agg_B = agg_B / denom

    return {
        'path': path,
        'version': version,
        'kf_count': kf_count,
        'df_count': df_count,
        'delta_entries_total': delta_entries_total,
        'weights': weights,
        'keyframes': keyframes,
        'agg_A': agg_A,
        'agg_R': agg_R,
        'agg_G': agg_G,
        'agg_B': agg_B,
        'ema_R': ema_R,
        'ema_G': ema_G,
        'ema_B': ema_B,
        'ema_count': ema_count,
    }


# ─────────────────────────────────────────────────────────── rendering

def _heatmap_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            lut[i] = (0, 0, int(128 * t / 0.25))
        elif t < 0.5:
            lut[i] = (0, int(200 * (t - 0.25) / 0.25), int(128 + 127 * (t - 0.25) / 0.25))
        elif t < 0.75:
            lut[i] = (int(255 * (t - 0.5) / 0.25), int(200 + 55 * (t - 0.5) / 0.25),
                      int(255 - 128 * (t - 0.5) / 0.25))
        else:
            lut[i] = (255, 255, int(127 + 128 * (t - 0.75) / 0.25))
    return lut

_LUT = _heatmap_lut()


def heatmap_image(data, size=(380, 380)):
    d = np.asarray(data, dtype=np.float64)
    if d.max() > 0:
        d = np.log1p(d)
        d = (d / d.max() * 255).astype(np.uint8)
    else:
        d = d.astype(np.uint8)
    return Image.fromarray(_LUT[d], 'RGB').resize(size, Image.NEAREST)


def rgb_composite_image(A, R, G, B, size=(380, 380)):
    amax = float(A.max())
    if amax == 0:
        return Image.new('RGB', size, (0, 0, 0))
    brightness = np.log1p(A) / np.log1p(amax)
    scale = 1.5
    rr = np.clip(R * brightness * scale, 0, 255).astype(np.uint8)
    gg = np.clip(G * brightness * scale, 0, 255).astype(np.uint8)
    bb = np.clip(B * brightness * scale, 0, 255).astype(np.uint8)
    rgb = np.stack([rr, gg, bb], axis=-1)
    return Image.fromarray(rgb, 'RGB').resize(size, Image.NEAREST)


def _load_font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
              'C:/Windows/Fonts/consola.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render_frame(model, idx, total, out_path):
    img  = Image.new('RGB', (FRAME_W, FRAME_H), (15, 15, 25))
    draw = ImageDraw.Draw(img)
    ft_title = _load_font(26)
    ft_stat  = _load_font(17)
    ft_sm    = _load_font(13)

    A = model['agg_A']
    R, G, B = model['agg_R'], model['agg_G'], model['agg_B']
    kf_n, df_n = model['kf_count'], model['df_count']
    name = os.path.basename(model['path']).replace('.spai', '')
    mask = A > 0
    active = int(mask.sum())
    a_max  = float(A.max())
    a_mean = float(A[mask].mean()) if active else 0.0
    r_std  = float(R[mask].std())  if active else 0.0
    g_std  = float(G[mask].std())  if active else 0.0
    b_std  = float(B[mask].std())  if active else 0.0
    ema_active = int((model['ema_count'] > 0).sum())
    w = model['weights']

    draw.text((30, 12), f"SPATIAL-PATTERN-AI   KF={kf_n}  Δ={df_n}  v{model['version']}",
              fill=(255, 255, 255), font=ft_title)
    draw.text((30, 42), name, fill=(120, 180, 255), font=ft_stat)

    img.paste(heatmap_image(A),                      (30,  75))
    img.paste(rgb_composite_image(A, R, G, B),       (440, 75))
    draw.text((30,  460), "A-channel (byte frequency, log-scaled)",
              fill=(160, 160, 160), font=ft_sm)
    draw.text((440, 460), "RGB composite (POS / function / extended)",
              fill=(160, 160, 160), font=ft_sm)

    sx = 860
    lines = [
        f"Keyframes:    {kf_n:>6}",
        f"Deltas:       {df_n:>6}",
        f"Δ entries:    {model['delta_entries_total']:>6}",
        f"Active cells: {active:>6} ({active / GRID_TOTAL * 100:5.1f}%)",
        f"A max:        {a_max:>6.0f}",
        f"A mean:       {a_mean:>6.1f}",
        "",
        f"R std (act):  {r_std:>6.2f}",
        f"G std (act):  {g_std:>6.2f}",
        f"B std (act):  {b_std:>6.2f}",
        "",
        f"Weights  A {w[0]:.2f}  R {w[1]:.2f}  G {w[2]:.2f}  B {w[3]:.2f}",
        "",
        f"EMA active:   {ema_active:>6}",
    ]
    for i, line in enumerate(lines):
        draw.text((sx, 80 + i * 22), line, fill=(200, 200, 200), font=ft_stat)

    progress = (idx + 1) / max(total, 1)
    bar_y = 490
    draw.rectangle([(30, bar_y), (410, bar_y + 18)], outline=(80, 80, 80))
    draw.rectangle([(30, bar_y), (30 + int(380 * progress), bar_y + 18)], fill=(80, 200, 120))
    draw.text((420, bar_y), f"{idx + 1}/{total}", fill=(180, 180, 180), font=ft_stat)

    draw.line([(30, 530), (FRAME_W - 30, 530)], fill=(50, 50, 70))
    draw.text((30, 540), "Sample keyframes:", fill=(255, 220, 100), font=ft_stat)
    for i, kf in enumerate(model['keyframes'][:5]):
        label = kf['label'][:80]
        draw.text((30, 565 + i * 20),
                  f"  KF{kf['id']:>3}  topic {kf['topic_hash']:08x}  seq {kf['seq_in_topic']}  {label}",
                  fill=(180, 180, 180), font=ft_sm)

    draw.text((FRAME_W - 320, FRAME_H - 22),
              "github.com/sjpupro-lab/CANVAS", fill=(60, 60, 80), font=ft_sm)
    img.save(out_path)


# ─────────────────────────────────────────────────────────── main

def main():
    if len(sys.argv) < 2:
        model_dir = 'build/models'
    else:
        model_dir = sys.argv[1]

    out_dir = os.path.join(model_dir, 'viz')
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(model_dir) if f.endswith('.spai'))
    if not files:
        print(f"No .spai files in {model_dir}")
        return 1

    print(f"Found {len(files)} model file(s) in {model_dir}")
    models = []
    for name in files:
        path = os.path.join(model_dir, name)
        try:
            m = read_spai(path)
            print(f"  {name}   KF={m['kf_count']:<5}  Δ={m['df_count']:<6}  v{m['version']}")
            models.append(m)
        except Exception as exc:
            print(f"  {name}   skipped ({exc})")

    if not models:
        print("No readable models")
        return 1

    for i, m in enumerate(models):
        frame_path = os.path.join(out_dir, f'frame_{i:04d}.png')
        render_frame(m, i, len(models), frame_path)
        print(f"  rendered frame {i}: {os.path.basename(m['path'])}")

    video_path = os.path.join(model_dir, 'training_evolution.mp4')
    if len(models) == 1:
        cmd = ['ffmpeg', '-y', '-loop', '1',
               '-i', os.path.join(out_dir, 'frame_0000.png'),
               '-t', '5',
               '-vf', 'fps=30,format=yuv420p',
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
               video_path]
    else:
        cmd = ['ffmpeg', '-y',
               '-framerate', '1/3',
               '-i', os.path.join(out_dir, 'frame_%04d.png'),
               '-vf', 'fps=30,format=yuv420p',
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
               video_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print("\nffmpeg not installed — skipping video. Frames are in:", out_dir)
        return 0

    if result.returncode == 0:
        size_kb = os.path.getsize(video_path) / 1024
        print(f"\nVideo: {video_path} ({size_kb:.0f} KB)")
    else:
        print(f"\nffmpeg failed:\n{result.stderr[:400]}")

    print("\nOutputs:")
    print(f"  models: {model_dir}/*.spai")
    print(f"  frames: {out_dir}/frame_*.png")
    print(f"  video:  {video_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
