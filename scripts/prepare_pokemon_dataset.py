#!/usr/bin/env python3
"""
포켓몬 CSV 데이터를 sss_train.py 파이프라인용으로 변환

사용법:
  python3 scripts/prepare_pokemon_dataset.py \
      --csv <pokemon.csv 경로> \
      --out data/pokemon_sss

출력:
  data/pokemon_sss/
    ├── labels.tsv          (fname\tcolor\tshape\tface)
    └── pokemon_XXXX_name.ppm  (64x64 PPM 이미지 801개)

labels.tsv 컬럼 매핑:
  color = type1  (주 타입 → 색상 특성)
  shape = type2  (부 타입 → 형태 특성, 없으면 'solo')
  face  = generation + legendary 여부
"""
import argparse
import os
import pandas as pd
import numpy as np

TYPE_COLORS = {
    'fire':     (240, 80,  20),
    'water':    (40,  100, 240),
    'grass':    (50,  190, 50),
    'electric': (255, 210, 0),
    'psychic':  (240, 50,  160),
    'ice':      (140, 220, 255),
    'dragon':   (70,  30,  200),
    'dark':     (50,  30,  70),
    'fairy':    (255, 150, 200),
    'normal':   (170, 170, 130),
    'fighting': (190, 50,  30),
    'flying':   (130, 170, 255),
    'poison':   (150, 50,  190),
    'ground':   (200, 150, 70),
    'rock':     (150, 130, 70),
    'bug':      (110, 170, 30),
    'ghost':    (90,  70,  150),
    'steel':    (150, 170, 190),
}

def get_color(t):
    if pd.isna(t) or str(t) not in TYPE_COLORS:
        return (128, 128, 128)
    return TYPE_COLORS[str(t)]

def norm(v, lo, hi):
    if hi == lo: return 0.5
    return float(v - lo) / float(hi - lo)

def make_ppm(row, stats_min, stats_max, path):
    H, W = 64, 64
    img = np.zeros((H, W, 3), dtype=np.float32)
    c1 = get_color(row['type1'])
    c2 = get_color(row['type2'])

    # 배경 그라디언트
    for y in range(H):
        for x in range(W):
            t = x / W
            for ch in range(3):
                img[y, x, ch] = (c1[ch] * (1 - t) + c2[ch] * t) / 255.0

    # 스탯 바
    stats = [('hp',(1,.3,.3)),('attack',(1,.6,.1)),('defense',(.3,.6,1)),
             ('sp_attack',(.8,.3,1)),('sp_defense',(.3,.9,.6)),('speed',(1,.9,.1))]
    for i, (col, color) in enumerate(stats):
        h_frac = norm(row[col], stats_min[col], stats_max[col])
        bar_h = max(2, int(h_frac * 18))
        bx = 2 + i * 10
        for y in range(H - bar_h, H - 1):
            for x in range(bx, bx + 9):
                if 0 <= x < W:
                    img[y, x] = list(color)

    # 전설 테두리
    if int(row['is_legendary']) == 1:
        for x in range(W):
            for y in range(4):
                img[y, x] = [1.0, 0.84, 0.0]

    # 세대 점
    gen = int(row['generation'])
    for g in range(gen):
        px = 2 + g * 6
        if px + 3 < W:
            for dy in range(3):
                for dx in range(3):
                    img[2 + dy, px + dx] = [1.0, 1.0, 1.0]

    # base_total 원
    cx, cy = W // 2, H // 2 - 8
    tot_frac = norm(row['base_total'], stats_min['base_total'], stats_max['base_total'])
    radius = int(tot_frac * 12) + 5
    for y in range(max(0, cy - radius), min(H, cy + radius)):
        for x in range(max(0, cx - radius), min(W, cx + radius)):
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if radius - 2 <= dist <= radius:
                img[y, x] = [1.0, 1.0, 1.0]
            elif dist < radius - 2:
                for ch in range(3):
                    img[y, x, ch] = min(1.0, c1[ch] / 255.0 * 1.4)

    with open(path, 'wb') as f:
        f.write(f"P6\n{W} {H}\n255\n".encode('ascii'))
        f.write((np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='/home/ubuntu/pokemon_dataset/pokemon.csv')
    ap.add_argument('--out', default='data/pokemon_sss')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    stat_cols = ['hp','attack','defense','sp_attack','sp_defense','speed','base_total']
    stats_min = {c: df[c].min() for c in stat_cols}
    stats_max = {c: df[c].max() for c in stat_cols}

    os.makedirs(args.out, exist_ok=True)
    tsv_lines = []
    print(f"포켓몬 {len(df)}마리 변환 중...")

    for idx, row in df.iterrows():
        safe = row['name'].lower().replace(' ','_').replace("'",'').replace('.','').replace(':','').replace('/','_')
        fname = f"pokemon_{row['pokedex_number']:04d}_{safe}.ppm"
        make_ppm(row, stats_min, stats_max, os.path.join(args.out, fname))

        color = str(row['type1']).lower() if not pd.isna(row['type1']) else 'normal'
        shape = str(row['type2']).lower() if not pd.isna(row['type2']) else 'solo'
        face  = f"gen{int(row['generation'])}"
        if int(row['is_legendary']) == 1:
            face += '_legendary'
        tsv_lines.append(f"{fname}\t{color}\t{shape}\t{face}")

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(df)}")

    with open(os.path.join(args.out, 'labels.tsv'), 'w') as f:
        f.write('\n'.join(tsv_lines) + '\n')

    print(f"완료: {args.out}/labels.tsv ({len(df)}개)")

if __name__ == '__main__':
    main()
