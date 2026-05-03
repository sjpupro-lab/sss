#!/usr/bin/env python3
"""
sin 주파수로 pose00~pose15를 순환하며 캐릭터 애니메이션 프레임 생성

각 캐릭터에 대해:
  - N_CYCLES 사이클 × 16포즈 = 총 프레임 수
  - sss_gen을 pose 프롬프트로 호출하여 PPM 생성
  - PNG로 업스케일 저장

출력: build/anim_frames/<char>/frame_XXXX.png
"""
import os
import math
import subprocess
from PIL import Image

SSS_GEN   = "./build/sss_gen"
MODEL     = "build/models/sanrio_anim.sss"
OUT_BASE  = "build/anim_frames"
N_POSES   = 16
N_CYCLES  = 3    # 3바퀴 반복
SCALE     = 8    # 64 → 512px
SEED      = 7

CHARS = {
    "keroppi":  ("green", "frog",   "keroppi"),
    "kitty":    ("white", "cat",    "kitty"),
    "mymelody": ("pink",  "rabbit", "mymelody"),
}

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
            # sin 주파수: pose_idx가 0→15→0→15... 순환
            # 프롬프트: "green frog keroppi pose03" 형태
            prompt = f"{color} {shape} {face} pose{pose_idx:02d}"

            cmd = [SSS_GEN, MODEL, prompt, tmp_ppm, str(SEED), "1.0"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if not os.path.exists(tmp_ppm):
                print(f"  경고: {prompt} 생성 실패 - {result.stderr.strip()}")
                # 빈 프레임 대신 이전 프레임 복사 또는 회색
                img = Image.new("RGB", (64 * SCALE, 64 * SCALE), (200, 200, 200))
            else:
                img = Image.open(tmp_ppm).resize(
                    (64 * SCALE, 64 * SCALE), Image.NEAREST
                )

            out_path = os.path.join(out_dir, f"frame_{frame_idx:04d}.png")
            img.save(out_path)
            frame_idx += 1

            if frame_idx % 8 == 0:
                print(f"  {frame_idx}/{total} 완료 (pose{pose_idx:02d})")

    print(f"  [{char}] 완료: {frame_idx}프레임 → {out_dir}")

print("\n=== 전체 완료 ===")
for char in CHARS:
    d = os.path.join(OUT_BASE, char)
    n = len([f for f in os.listdir(d) if f.endswith('.png')])
    print(f"  {char}: {n}프레임")
