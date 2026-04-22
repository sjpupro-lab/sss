#!/usr/bin/env python3
"""
Kaggle GPU trainer for SPATIAL-PATTERN-AI (experimental).

Purpose:
- Use free Kaggle GPU to train up to 50,000 clauses quickly.
- Build aggregated channel tables (A/R/G/B) on CUDA tensors.
- Auto-save checkpoints and final model.

Notes:
- This script is an external GPU training route intended for Kaggle.
- It does not replace the C runtime path; it produces portable .pt checkpoints.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch

GRID = 256
WEIGHT_BASE = 1.0
WEIGHT_WORD = 5.0
WEIGHT_MORPHEME = 3.0

CLAUSE_RE = re.compile(r"[^.!?]+[.!?]")
MORPH_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_]+")

PARTICLE_SUFFIXES = (
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "와",
    "과",
    "도",
    "로",
    "만",
)

ENDING_SUFFIXES = (
    "다",
    "는다",
    "했다",
    "한다",
    "이다",
    "였다",
)


@dataclass
class GpuModel:
    a_sum: torch.Tensor
    r_mean: torch.Tensor
    g_mean: torch.Tensor
    b_mean: torch.Tensor
    row_total_a: torch.Tensor
    trained_clauses: int


def split_clauses(text: str) -> List[str]:
    out = []
    for m in CLAUSE_RE.finditer(text):
        c = m.group(0).strip()
        if len(c) >= 10:
            out.append(c)
    return out


def token_pos(token: str) -> str:
    if any(token.endswith(s) for s in PARTICLE_SUFFIXES):
        return "PARTICLE"
    if any(token.endswith(s) for s in ENDING_SUFFIXES):
        return "ENDING"
    if token.endswith("한") or token.endswith("운") or token.endswith("은"):
        return "ADJ"
    if token.endswith("다"):
        return "VERB"
    return "NOUN"


def pos_seed(pos: str) -> Tuple[float, float, float]:
    if pos == "NOUN":
        return 40.0, 30.0, 100.0
    if pos == "VERB":
        return 120.0, 40.0, 140.0
    if pos == "ADJ":
        return 170.0, 35.0, 180.0
    if pos == "PARTICLE":
        return 8.0, 85.0, 90.0
    if pos == "ENDING":
        return 12.0, 95.0, 110.0
    return 210.0, 20.0, 200.0


def byte_positions(text: str) -> Tuple[List[int], List[int]]:
    bs = text.encode("utf-8", errors="ignore")
    ys = [(i % GRID) for i in range(len(bs))]
    xs = [int(b) for b in bs]
    return ys, xs


def word_positions(text: str) -> Tuple[List[int], List[int]]:
    bs = text.encode("utf-8", errors="ignore")
    ys: List[int] = []
    xs: List[int] = []
    for i, b in enumerate(bs):
        if b in (32, 9, 10, 13):
            continue
        ys.append(i % GRID)
        xs.append(int(b))
    return ys, xs


def morpheme_positions_with_seeds(text: str) -> List[Tuple[int, int, float, float, float]]:
    out: List[Tuple[int, int, float, float, float]] = []
    bs = text.encode("utf-8", errors="ignore")

    # Build byte offset map from character index to byte index.
    char_to_byte = [0]
    acc = 0
    for ch in text:
        acc += len(ch.encode("utf-8", errors="ignore"))
        char_to_byte.append(acc)

    for m in MORPH_TOKEN_RE.finditer(text):
        token = m.group(0)
        if len(token) < 2:
            continue
        pos = token_pos(token)
        r_seed, g_seed, b_seed = pos_seed(pos)

        start_char = m.start()
        end_char = m.end()
        start_b = char_to_byte[start_char]
        end_b = char_to_byte[end_char]
        for bi in range(start_b, min(end_b, len(bs))):
            b = int(bs[bi])
            if b in (32, 9, 10, 13):
                continue
            out.append((bi % GRID, b, r_seed, g_seed, b_seed))
    return out


def scatter_add_2d(dst: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor, values: torch.Tensor) -> None:
    flat = ys * GRID + xs
    dst.view(-1).index_add_(0, flat, values)


def train_gpu(clauses: List[str], checkpoint_every: int, out_dir: str) -> GpuModel:
    device = torch.device("cuda")
    a_sum = torch.zeros((GRID, GRID), dtype=torch.float32, device=device)
    r_sum = torch.zeros((GRID, GRID), dtype=torch.float32, device=device)
    g_sum = torch.zeros((GRID, GRID), dtype=torch.float32, device=device)
    b_sum = torch.zeros((GRID, GRID), dtype=torch.float32, device=device)

    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    for i, clause in enumerate(clauses, start=1):
        by, bx = byte_positions(clause)
        if by:
            ys = torch.tensor(by, dtype=torch.long, device=device)
            xs = torch.tensor(bx, dtype=torch.long, device=device)
            vals = torch.full((len(by),), WEIGHT_BASE, dtype=torch.float32, device=device)
            scatter_add_2d(a_sum, ys, xs, vals)

        wy, wx = word_positions(clause)
        if wy:
            ys = torch.tensor(wy, dtype=torch.long, device=device)
            xs = torch.tensor(wx, dtype=torch.long, device=device)
            vals = torch.full((len(wy),), WEIGHT_WORD, dtype=torch.float32, device=device)
            scatter_add_2d(a_sum, ys, xs, vals)

        morph = morpheme_positions_with_seeds(clause)
        if morph:
            ys = torch.tensor([m[0] for m in morph], dtype=torch.long, device=device)
            xs = torch.tensor([m[1] for m in morph], dtype=torch.long, device=device)
            vals = torch.full((len(morph),), WEIGHT_MORPHEME, dtype=torch.float32, device=device)
            scatter_add_2d(a_sum, ys, xs, vals)

            r_vals = torch.tensor([m[2] for m in morph], dtype=torch.float32, device=device) * WEIGHT_MORPHEME
            g_vals = torch.tensor([m[3] for m in morph], dtype=torch.float32, device=device) * WEIGHT_MORPHEME
            b_vals = torch.tensor([m[4] for m in morph], dtype=torch.float32, device=device) * WEIGHT_MORPHEME
            scatter_add_2d(r_sum, ys, xs, r_vals)
            scatter_add_2d(g_sum, ys, xs, g_vals)
            scatter_add_2d(b_sum, ys, xs, b_vals)

        if checkpoint_every > 0 and (i % checkpoint_every == 0):
            eps = 1e-8
            r_mean = r_sum / torch.clamp(a_sum, min=eps)
            g_mean = g_sum / torch.clamp(a_sum, min=eps)
            b_mean = b_sum / torch.clamp(a_sum, min=eps)
            ckpt = {
                "trained_clauses": i,
                "a_sum": a_sum.detach().cpu(),
                "r_mean": r_mean.detach().cpu(),
                "g_mean": g_mean.detach().cpu(),
                "b_mean": b_mean.detach().cpu(),
                "row_total_a": a_sum.sum(dim=1).detach().cpu(),
                "meta": {
                    "weights": {"base": WEIGHT_BASE, "word": WEIGHT_WORD, "morpheme": WEIGHT_MORPHEME},
                    "device": "cuda",
                },
            }
            p = os.path.join(out_dir, f"gpu_checkpoint_{i:06d}.pt")
            torch.save(ckpt, p)
            print(f"[checkpoint] saved: {p}")

        if i % 1000 == 0 or i == len(clauses):
            dt = time.time() - t0
            print(f"[train] {i}/{len(clauses)} clauses, elapsed={dt:.1f}s")

    eps = 1e-8
    r_mean = r_sum / torch.clamp(a_sum, min=eps)
    g_mean = g_sum / torch.clamp(a_sum, min=eps)
    b_mean = b_sum / torch.clamp(a_sum, min=eps)
    row_total = a_sum.sum(dim=1)

    return GpuModel(
        a_sum=a_sum.detach().cpu(),
        r_mean=r_mean.detach().cpu(),
        g_mean=g_mean.detach().cpu(),
        b_mean=b_mean.detach().cpu(),
        row_total_a=row_total.detach().cpu(),
        trained_clauses=len(clauses),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kaggle GPU trainer (up to 50,000 clauses)")
    p.add_argument("--input", required=True, help="UTF-8 text corpus path")
    p.add_argument("--max-clauses", type=int, default=50000, help="max clauses to train")
    p.add_argument("--checkpoint-every", type=int, default=5000, help="checkpoint interval")
    p.add_argument("--out-dir", default="build/gpu_models", help="output directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU not available. In Kaggle, enable GPU (T4/P100) first.")
        return 2

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    clauses = split_clauses(text)
    if not clauses:
        print("ERROR: no valid clauses found")
        return 3

    clauses = clauses[: args.max_clauses]
    print(f"[setup] device=cuda clauses={len(clauses)} max={args.max_clauses}")

    model = train_gpu(clauses, args.checkpoint_every, args.out_dir)

    final_path = os.path.join(args.out_dir, "gpu_model_final.pt")
    torch.save(
        {
            "trained_clauses": model.trained_clauses,
            "a_sum": model.a_sum,
            "r_mean": model.r_mean,
            "g_mean": model.g_mean,
            "b_mean": model.b_mean,
            "row_total_a": model.row_total_a,
            "meta": {
                "weights": {"base": WEIGHT_BASE, "word": WEIGHT_WORD, "morpheme": WEIGHT_MORPHEME},
                "device": "cuda",
                "script": "tools/kaggle_gpu_train.py",
            },
        },
        final_path,
    )

    active = int((model.a_sum > 0).sum().item())
    avg_a = float(model.a_sum[model.a_sum > 0].mean().item()) if active > 0 else 0.0
    print(f"[done] saved final model: {final_path}")
    print(f"[done] trained_clauses={model.trained_clauses} active_cells={active} avg_A={avg_a:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
