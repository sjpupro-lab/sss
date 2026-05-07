#!/usr/bin/env python3
"""sin 주파수로 pose00~pose15를 순환하며 캐릭터 애니메이션 프레임 생성.

기본 모드 (sss_gen pose 프롬프트):
  - 각 캐릭터에 대해 N_CYCLES × 16포즈 = 총 프레임
  - sss_gen을 pose 프롬프트로 호출하여 PPM 생성
  - PNG로 업스케일 저장

Atmos 모드 (`--atmos`):
  - sss_gen으로 한 장의 reference 이미지를 만든 뒤
  - tools.sss_atmos.AtmosScene으로 오브젝트 분해
  - ce_scene_tick_with_profiles + ce_scene_render로
    매 프레임을 직접 렌더 (FFT pose 루프 없이 ~ms/frame)
  - STATIC 배경/FLOW 흐름/ORGANIC 호흡이 자동으로 적용된다.
출력: build/anim_frames/<char>/frame_XXXX.png
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# repo-root on sys.path so `from tools.sss_atmos import …` works whether
# the script is invoked directly or via `python -m`.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from PIL import Image     # noqa: E402  (import after sys.path tweak)


SSS_GEN   = "./build/sss_gen"
MODEL     = "build/models/sanrio_anim.sss"
OUT_BASE  = "build/anim_frames"
N_POSES   = 16
N_CYCLES  = 3    # 3바퀴 반복
SCALE     = 8    # 64 → 512px
SEED      = 7
ATMOS_REF_SIZE = 64    # reference 생성 해상도 (sss_gen)
ATMOS_FRAME_SIZE = 512 # 매 프레임 렌더 해상도

CHARS = {
    "keroppi":  ("green", "frog",   "keroppi"),
    "kitty":    ("white", "cat",    "kitty"),
    "mymelody": ("pink",  "rabbit", "mymelody"),
}


def _run_sss_gen(prompt: str, out_path: str) -> bool:
    cmd = [SSS_GEN, MODEL, prompt, out_path, str(SEED), "1.5"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(out_path):
        sys.stderr.write(
            f"  warning: {prompt} 생성 실패 - {result.stderr.strip()}\n")
        return False
    return True


def _read_ppm_rgba(path: str):
    """Load a PPM-or-PNG (whatever sss_gen produced) into a (H, W, 4)
    uint8 numpy array. Imported here so the default sss_gen path doesn't
    pay for numpy at import time."""
    import numpy as np  # local import — only the atmos path needs it
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    h, w, _ = arr.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = arr
    rgba[..., 3]  = 255
    return rgba, w, h


def make_pose_frames():
    """Default pipeline: pose-prompt sss_gen + nearest-neighbour upscale."""
    os.makedirs(OUT_BASE, exist_ok=True)
    for char, (color, shape, face) in CHARS.items():
        out_dir = os.path.join(OUT_BASE, char)
        os.makedirs(out_dir, exist_ok=True)
        tmp_ppm = f"/tmp/sss_anim_{char}.ppm"

        frame_idx = 0
        total = N_CYCLES * N_POSES
        print(f"\n[{char}] {total}프레임 생성 중...")

        for cycle in range(N_CYCLES):
            for pose_idx in range(N_POSES):
                prompt = f"{color} {shape} {face} pose{pose_idx:02d}"
                if _run_sss_gen(prompt, tmp_ppm):
                    img = Image.open(tmp_ppm).resize(
                        (64 * SCALE, 64 * SCALE), Image.NEAREST)
                else:
                    img = Image.new("RGB",
                                    (64 * SCALE, 64 * SCALE),
                                    (200, 200, 200))
                out_path = os.path.join(out_dir, f"frame_{frame_idx:04d}.png")
                img.save(out_path)
                frame_idx += 1
                if frame_idx % 8 == 0:
                    print(f"  {frame_idx}/{total} 완료 (pose{pose_idx:02d})")
        print(f"  [{char}] 완료: {frame_idx}프레임 → {out_dir}")


def make_atmos_frames(n_frames: int):
    """Atmos pipeline: one reference per character, then per-frame
    ce_scene_tick_with_profiles + ce_scene_render."""
    from tools.sss_atmos import AtmosScene  # local import — needs .so

    os.makedirs(OUT_BASE, exist_ok=True)
    for char, (color, shape, face) in CHARS.items():
        out_dir = os.path.join(OUT_BASE, char)
        os.makedirs(out_dir, exist_ok=True)
        ref_ppm = f"/tmp/sss_atmos_ref_{char}.ppm"
        prompt = f"{color} {shape} {face}"

        print(f"\n[{char}] Atmos reference 생성중 (prompt={prompt!r})…")
        if not _run_sss_gen(prompt, ref_ppm):
            print(f"  [{char}] reference 실패 — skip")
            continue

        rgba, w, h = _read_ppm_rgba(ref_ppm)
        with AtmosScene.from_rgba(rgba, w, h) as scene:
            n_obj = scene.object_count
            objs = scene.objects
            print(f"  [{char}] {n_obj} objects 분해됨")
            for o in objs:
                print(f"    id={o.id} motion={o.motion_name} "
                      f"cx={o.base_cx} cy={o.base_cy} "
                      f"color=({o.color_r},{o.color_g},{o.color_b})")

            for f in range(n_frames):
                rgb = scene.render(width=ATMOS_FRAME_SIZE,
                                   height=ATMOS_FRAME_SIZE)
                Image.fromarray(rgb, mode="RGB").save(
                    os.path.join(out_dir, f"frame_{f:04d}.png"))
                scene.tick()
                if (f + 1) % 16 == 0:
                    print(f"  {f+1}/{n_frames} 프레임")
        print(f"  [{char}] Atmos 완료: {n_frames}프레임 → {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atmos", action="store_true",
                   help="Use the Atmos motion engine "
                        "(ce_scene_tick_with_profiles) instead of "
                        "per-pose sss_gen calls.")
    p.add_argument("--n-frames", type=int, default=N_CYCLES * N_POSES,
                   help="Frame count per character (atmos mode only). "
                        "Default = N_CYCLES * N_POSES.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.atmos:
        make_atmos_frames(args.n_frames)
    else:
        make_pose_frames()

    print("\n=== 전체 완료 ===")
    for char in CHARS:
        d = os.path.join(OUT_BASE, char)
        if not os.path.isdir(d):
            continue
        n = len([f for f in os.listdir(d) if f.endswith('.png')])
        print(f"  {char}: {n}프레임")


if __name__ == "__main__":
    main()
