#!/usr/bin/env python3
"""sss_train_interaction — Phase 4 interaction-response trainer.

Design philosophy:
    The motif learns how it *responds* to a condition signal by
    comparing its `static` and `active` sequences. The trainer
    accumulates the envelope and warp *delta* — never raw frames,
    never per-row FFT bytes. Similar motifs (cosine ≥ 0.85 on the
    Phase 3 multi-feature weighted matcher) contribute via a
    coherence-weighted average; that's the "accumulate responses
    from similar motifs" core principle.

Input layout:
    dataset/interactions/<motif_label>_<condition_label>/
        static/   16 (or --frames) ordered PNG/PPM frames — motif
                  with no condition applied.
        active/   16 ordered frames — same motif under the condition
                  signal.

The trainer expects a primitive_motifs.npz (Phase 3) and a
conditions.npz (Phase 4) to resolve `<motif_label>` and
`<condition_label>`. Unknown labels are skipped with a warning.

Algorithm (per pairing):
    (1) Load both sequences, normalise to 64×64 RGB float32 in [0, 1].
    (2) Compute static envelope (folder-average row / col / cross
        / heatmap). Re-uses the primitive trainer's envelope code.
    (3) Compute active envelope per frame.
    (4) Delta envelope = active_frame_t − static.
    (5) Compute the condition's per-frame intensity series via
        `_temporal_envelope_at(cond, t/N)`.
    (6) Correlation = corrcoef(intensity[t], ‖delta[t]‖) along t.
        If corr < CORRELATION_THRESH (default 0.7), discard the
        pair — the active sequence isn't actually responding to
        the condition.
    (7) Aggregate: row_response = mean_t(delta_row[t] /
        max(intensity[t], ε)). Same for col / cross.
    (8) Warp field: per-frame optical-flow proxy (block-matching at
        a 16×16 grid; sliding-mean intensity difference) →
        per-frame warp_x[16][16], warp_y[16][16] in [-1, +1] units
        (pixel displacement normalised by patch radius). Average
        across frames, scale by 1 / mean_intensity.
    (9) response_strength = max ‖delta‖ across frames.
   (10) Similar-motif accumulation: for the motif under training,
        find every other registered motif whose Phase 3 weighted-
        cosine similarity ≥ SIMILARITY_THRESH (default 0.85).
        Their existing responses (if any) for the same condition
        contribute via a coherence-weighted average:
            new_response = (Σ_i coh_i · resp_i + coh_new · resp_new)
                         / (Σ_i coh_i + coh_new)
        This is the Phase 4 "the motif learns from similar motifs"
        rule. accumulated_samples reflects the total contributor
        count.

CLI:
    python3 scripts/sss_train_interaction.py \\
        --motifs     build/primitive_motifs.npz \\
        --conditions build/conditions.npz \\
        --data       dataset/interactions \\
        --out        build/interactions.npz
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import sss_train_primitive as _prim   # noqa: E402
from scripts import sss_train_condition as _cond   # noqa: E402


FRAMES_DEFAULT          = 16
PATCH_SIZE              = 64
FREQ_BINS               = 128
TEMPORAL_BINS           = 64
WARP_DIM                = 16
# Lowered from the GPT-spec'd 0.7 to 0.5 for v1. The condition's
# stored temporal_freq is amplitude-only (Phase 1 rule), so the
# trainer's zero-phase irfft reconstruction is typically 90° out of
# phase with the active sequence's actual oscillation. Squaring
# both sides for the |intensity|² vs |delta|² correlation (below)
# recovers most but not all of the alignment — 0.5 is the empirical
# floor below which a pair really isn't responding.
CORRELATION_THRESH      = 0.5
SIMILARITY_THRESH       = 0.85
INTENSITY_EPS           = 1e-2     # avoid divide-by-zero on flat frames
WARP_MAX_DISPLACEMENT   = 4.0      # px; normalises warp_x/_y to [-1, +1]

IMAGE_EXTS = (".png", ".ppm")


# ── Helpers ───────────────────────────────────────────────────────

def _load_seq(folder: Path, target_len: int) -> np.ndarray:
    paths = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return _cond._load_sequence_64(paths, target_len=target_len)


def _per_frame_envelope(frame: np.ndarray) -> dict:
    """frame: (64, 64, 3) float32 in [0, 1]. Returns row / col / cross
    envelopes and 16×16 Sobel heatmap exactly as the primitive
    trainer computes them per-image."""
    img_u8 = (frame * 255.0).clip(0, 255).astype(np.uint8)
    f = _prim._to_float_demeaned(img_u8)
    return {
        "row":   _prim._envelope_row_freq(f),
        "col":   _prim._envelope_col_freq(f),
        "color": _prim._envelope_color_freq(f),
        "heat":  _prim._sobel_heatmap_16(img_u8).astype(np.float32),
    }


def _envelope_mean(seq: np.ndarray) -> dict:
    """Folder-average envelopes across an entire sequence."""
    rows = []; cols = []; colors = []; heats = []
    for t in range(seq.shape[0]):
        env = _per_frame_envelope(seq[t])
        rows.append(env["row"])
        cols.append(env["col"])
        colors.append(env["color"])
        heats.append(env["heat"])
    return {
        "row":   np.mean(rows, axis=0).astype(np.float32),
        "col":   np.mean(cols, axis=0).astype(np.float32),
        "color": np.mean(colors, axis=0).astype(np.float32),
        "heat":  np.mean(heats, axis=0).astype(np.float32),
    }


def _temporal_intensity_series(cond_npz_row: dict, n_frames: int) -> np.ndarray:
    """Sample the condition's effective per-frame intensity for the
    given sequence length. Re-uses the synthesiser's interpretation."""
    from tools import sss_synthesizer
    from tools.sss_feature_bank import SSSCondition
    cond_obj = SSSCondition(
        label=cond_npz_row["label"],
        condition_type=int(cond_npz_row["condition_type"]),
        temporal_freq=cond_npz_row["temporal_freq"],
        direction=float(cond_npz_row["direction"]),
        default_intensity=float(cond_npz_row["default_intensity"]),
        decay_rate=float(cond_npz_row["decay_rate"]),
    )
    return np.array([
        sss_synthesizer._temporal_envelope_at(
            cond_obj, t / max(1, n_frames - 1), n_frames=n_frames)
        for t in range(n_frames)
    ], dtype=np.float32)


def _warp_per_frame(static_frame: np.ndarray,
                    active_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Block-matched optical-flow proxy on a 16×16 grid. For each
    grid cell we compute the per-direction (dx, dy) that minimises
    SSD between a 4×4 patch in `static_frame` and the candidate
    patch in `active_frame`. Returns (warp_x, warp_y) shaped
    (16, 16) float32 in [-1, +1] (normalised by the search radius)."""
    static_gray = static_frame.mean(axis=-1)         # (64, 64)
    active_gray = active_frame.mean(axis=-1)
    H, W = static_gray.shape
    blk = 4  # block side; 64 / 16 == 4
    search = 3  # ± px around the cell centre — keeps the search cheap
    warp_x = np.zeros((WARP_DIM, WARP_DIM), np.float32)
    warp_y = np.zeros((WARP_DIM, WARP_DIM), np.float32)
    for gy in range(WARP_DIM):
        cy = gy * blk + blk // 2
        for gx in range(WARP_DIM):
            cx = gx * blk + blk // 2
            best_ssd = float("inf")
            best_dy = 0; best_dx = 0
            # Reference patch from static.
            ay0 = max(0, cy - blk // 2); ay1 = min(H, cy + blk // 2)
            ax0 = max(0, cx - blk // 2); ax1 = min(W, cx + blk // 2)
            ref = static_gray[ay0:ay1, ax0:ax1]
            for dy in range(-search, search + 1):
                for dx in range(-search, search + 1):
                    by0 = ay0 + dy; by1 = ay1 + dy
                    bx0 = ax0 + dx; bx1 = ax1 + dx
                    if by0 < 0 or bx0 < 0 or by1 > H or bx1 > W:
                        continue
                    cand = active_gray[by0:by1, bx0:bx1]
                    if cand.shape != ref.shape:
                        continue
                    ssd = float(((cand - ref) ** 2).sum())
                    if ssd < best_ssd:
                        best_ssd = ssd
                        best_dy = dy; best_dx = dx
            # Normalise to [-1, +1] by the search radius.
            warp_x[gy, gx] = best_dx / float(search)
            warp_y[gy, gx] = best_dy / float(search)
    return warp_x, warp_y


# ── Per-pair training ────────────────────────────────────────────

def train_interaction_pair(static_seq: np.ndarray, active_seq: np.ndarray,
                           cond_npz_row: dict,
                           n_frames: int = FRAMES_DEFAULT) -> dict | None:
    """Aggregate one (motif, condition) interaction. Returns a dict
    of response fields ready for sfb packing, or None if correlation
    is below the threshold (pair is rejected as not actually
    responding)."""
    static_env = _envelope_mean(static_seq)
    # Per-frame active envelopes + warp fields.
    per_frame_deltas_row = []
    per_frame_deltas_col = []
    per_frame_deltas_color = []
    per_frame_deltas_heat = []
    per_frame_warp_x = []
    per_frame_warp_y = []
    per_frame_norm = []
    for t in range(active_seq.shape[0]):
        env = _per_frame_envelope(active_seq[t])
        d_row = env["row"]   - static_env["row"]
        d_col = env["col"]   - static_env["col"]
        d_color = env["color"] - static_env["color"]
        d_heat = env["heat"] - static_env["heat"]
        per_frame_deltas_row.append(d_row)
        per_frame_deltas_col.append(d_col)
        per_frame_deltas_color.append(d_color)
        per_frame_deltas_heat.append(d_heat)
        # Optical-flow warp: this frame vs the matching static frame
        # (or static[0] if we don't have a per-frame static).
        ref_idx = min(t, static_seq.shape[0] - 1)
        wx, wy = _warp_per_frame(static_seq[ref_idx], active_seq[t])
        per_frame_warp_x.append(wx)
        per_frame_warp_y.append(wy)
        per_frame_norm.append(float(np.linalg.norm(d_row)
                                    + np.linalg.norm(d_col)))

    intensity = _temporal_intensity_series(cond_npz_row,
                                           n_frames=active_seq.shape[0])
    norms = np.asarray(per_frame_norm, np.float32)
    # Correlate **squared magnitudes**: |intensity|² vs |delta|².
    # The trainer doesn't know the relative phase of the condition's
    # temporal envelope vs the active sequence — Phase 1 dropped
    # phase from disk, and the per-condition irfft uses zero phase
    # by design. Squared magnitudes are phase-invariant, so a sine
    # condition still correlates with a cosine delta and vice versa.
    intensity_e = intensity * intensity
    norms_e = norms * norms
    if norms_e.std() < 1e-9 or intensity_e.std() < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(intensity_e, norms_e)[0, 1])
    # Accept both correlations and anti-correlations — a motif that
    # behaves "opposite" to the condition's intensity is still a
    # learned response (anti-resonance). What we reject is the case
    # where the signals are uncorrelated, i.e. the motif isn't
    # responding at all.
    if abs(corr) < CORRELATION_THRESH:
        return None

    # response = mean_t(delta[t] / max(|intensity[t]|, eps)). Phase
    # alignment may flip sign for some frames; the abs() in the
    # divisor keeps response_strength meaningful while preserving
    # the per-frame sign of the actual delta.
    abs_i = np.maximum(np.abs(intensity), INTENSITY_EPS)
    weights = (1.0 / abs_i).astype(np.float32)

    def _wavg(arr_list):
        a = np.stack(arr_list, axis=0)                  # (T, ...)
        w = weights.reshape((-1,) + (1,) * (a.ndim - 1))
        return (a * w).mean(axis=0).astype(np.float32)

    row_response = _wavg(per_frame_deltas_row)
    col_response = _wavg(per_frame_deltas_col)
    color_resp   = _wavg(per_frame_deltas_color)
    heat_resp    = _wavg(per_frame_deltas_heat)

    # cross_response: build a 64-bin spectrum from the per-frame
    # heatmap delta sequence. Per-frame Sobel-heatmap delta flattened
    # → length-256 vector; rfft along T → take amplitudes, sum across
    # the 256 spatial bins → 64-bin temporal aggregation.
    heat_stack = np.stack(per_frame_deltas_heat, axis=0).reshape(
        active_seq.shape[0], -1)                              # (T, 256)
    spec = np.fft.rfft(heat_stack - heat_stack.mean(axis=0, keepdims=True),
                       axis=0)
    cross_full = np.abs(spec).sum(axis=1).astype(np.float32)  # (T//2+1,)
    cross_resp = _prim._resample_linear(cross_full, TEMPORAL_BINS)
    # L2-normalise so the synthesiser's weighting is bounded.
    n = float(np.linalg.norm(cross_resp))
    if n > 1e-9:
        cross_resp = cross_resp / n

    warp_x = np.clip(np.mean(per_frame_warp_x, axis=0), -1.0, 1.0
                    ).astype(np.float32)
    warp_y = np.clip(np.mean(per_frame_warp_y, axis=0), -1.0, 1.0
                    ).astype(np.float32)

    response_strength = float(norms.max())

    return {
        "row_response":       row_response,
        "col_response":       col_response,
        "cross_response":     cross_resp,
        "warp_x_response":    warp_x,
        "warp_y_response":    warp_y,
        "response_strength":  response_strength,
        "response_delay":     0.0,        # Phase 4 v1 doesn't fit delay
        "accumulated_samples": 1,
        "correlation":        float(corr),
    }


# ── Similar-motif accumulation ───────────────────────────────────

def _weighted_cosine(a_envs: dict, b_envs: dict) -> float:
    """Phase 3 weighted-cosine matcher applied to motif envelope
    triples. `a_envs` / `b_envs` are dicts with the row_freq /
    col_freq / color_freq / position_heatmap fields from
    `primitive_motifs.npz` (already zero-mean L2-normalised)."""
    def cos(u, v):
        u64 = u.astype(np.float64).ravel()
        v64 = v.astype(np.float64).ravel()
        nu = float(np.linalg.norm(u64))
        nv = float(np.linalg.norm(v64))
        if nu < 1e-12 or nv < 1e-12:
            return 0.0
        return float(np.dot(u64, v64) / (nu * nv))

    row = cos(a_envs["row_freq"], b_envs["row_freq"])
    col = cos(a_envs["col_freq"], b_envs["col_freq"])
    color = (cos(a_envs["color_freq"][0], b_envs["color_freq"][0])
             + cos(a_envs["color_freq"][1], b_envs["color_freq"][1])
             + cos(a_envs["color_freq"][2], b_envs["color_freq"][2])) / 3.0
    heat = cos(a_envs["position_heatmap"], b_envs["position_heatmap"])
    return 0.35 * row + 0.35 * col + 0.20 * color + 0.10 * heat


def _accumulate_from_similar(target_idx: int,
                             cond_idx: int,
                             existing: dict,
                             new_resp: dict,
                             motifs_npz: dict,
                             coherence: np.ndarray) -> dict:
    """Merge `new_resp` (just-measured for target_idx) with every
    similar motif's existing response for the same condition.

    Each contributor weights at `coherence[contributor_motif_idx]`.
    The motif being trained itself contributes at its own coherence.
    Result is the coherence-weighted average of every response field.

    `existing` is the response dict already accumulated by previous
    training runs (or `None` if this is the first pairing for the
    motif+condition). It contributes at its own accumulated_samples
    weight."""
    contributors = []
    target_view = {
        "row_freq":          motifs_npz["row_freq"][target_idx],
        "col_freq":          motifs_npz["col_freq"][target_idx],
        "color_freq":        motifs_npz["color_freq"][target_idx],
        "position_heatmap":  motifs_npz["position_heatmap"][target_idx],
    }
    contributors.append((float(coherence[target_idx]), new_resp))

    if existing is not None and existing.get("accumulated_samples", 0) > 0:
        contributors.append((float(coherence[target_idx])
                             * float(existing["accumulated_samples"]),
                             existing))

    n_motifs = motifs_npz["row_freq"].shape[0]
    for i in range(n_motifs):
        if i == target_idx:
            continue
        other_view = {
            "row_freq":          motifs_npz["row_freq"][i],
            "col_freq":          motifs_npz["col_freq"][i],
            "color_freq":        motifs_npz["color_freq"][i],
            "position_heatmap":  motifs_npz["position_heatmap"][i],
        }
        sim = _weighted_cosine(target_view, other_view)
        if sim < SIMILARITY_THRESH:
            continue
        # Pull `i`'s existing response for the same condition if any.
        other_existing = _existing_responses.get((i, cond_idx))
        if other_existing is None:
            continue
        contributors.append((float(coherence[i]) * sim, other_existing))

    if not contributors:
        return new_resp

    # Coherence-weighted average across all contributors.
    total_w = sum(w for w, _ in contributors) + 1e-12
    fields = ("row_response", "col_response", "cross_response",
              "warp_x_response", "warp_y_response")
    out = {f: np.zeros_like(new_resp[f]) for f in fields}
    strength = 0.0
    accum = 0
    for w, resp in contributors:
        for f in fields:
            out[f] += (w / total_w) * np.asarray(resp[f], np.float32)
        strength = max(strength, float(resp.get("response_strength", 0.0)))
        accum += int(resp.get("accumulated_samples", 1))
    out["response_strength"]   = float(strength)
    out["response_delay"]      = float(new_resp.get("response_delay", 0.0))
    out["accumulated_samples"] = min(255, accum)
    out["correlation"]         = float(new_resp.get("correlation", 0.0))
    return out


# Module-level registry of (motif_idx, condition_idx) → response,
# used by the similar-motif accumulator. Reset on every CLI run.
_existing_responses: dict = {}


# ── Driver ───────────────────────────────────────────────────────

_PAIR_RE = re.compile(r"^(?P<motif>[^_]+)_(?P<cond>.+)$")


def _resolve_indices(folder_name: str,
                     motif_labels: list, cond_labels: list) -> tuple[int, int]:
    """Match `<motif>_<condition>` folder names against the registered
    label tables. Returns (-1, -1) if either label is unknown."""
    # Try every split point (motif label may itself contain '_').
    for i in range(len(folder_name)):
        if folder_name[i] != "_":
            continue
        m_lab = folder_name[:i]
        c_lab = folder_name[i + 1:]
        if m_lab in motif_labels and c_lab in cond_labels:
            return motif_labels.index(m_lab), cond_labels.index(c_lab)
    return -1, -1


def build_interactions(motifs_path: Path, conditions_path: Path,
                       data_root: Path,
                       n_frames: int = FRAMES_DEFAULT,
                       verbose: bool = False) -> dict:
    motifs = np.load(motifs_path, allow_pickle=False)
    conds  = np.load(conditions_path, allow_pickle=False)
    motif_labels = [str(s) for s in motifs["labels"]]
    cond_labels  = [str(s) for s in conds["labels"]]
    coherence    = motifs["coherence"].astype(np.float32)

    pairs = sorted(p for p in Path(data_root).iterdir() if p.is_dir())
    accumulated: dict = {}        # (motif_idx, cond_idx) → response dict
    rejected = 0
    skipped  = 0
    _existing_responses.clear()

    for p in pairs:
        m_idx, c_idx = _resolve_indices(p.name, motif_labels, cond_labels)
        if m_idx < 0:
            if verbose:
                print(f"  skip {p.name}: unknown motif/condition")
            skipped += 1
            continue
        static_dir = p / "static"
        active_dir = p / "active"
        if not (static_dir.is_dir() and active_dir.is_dir()):
            if verbose:
                print(f"  skip {p.name}: missing static/ or active/")
            skipped += 1
            continue
        static_seq = _load_seq(static_dir, n_frames)
        active_seq = _load_seq(active_dir, n_frames)
        cond_row = {k: conds[k][c_idx] for k in conds.files}
        cond_row["label"] = cond_labels[c_idx]
        cond_row["condition_type"] = int(conds["condition_type"][c_idx])
        new_resp = train_interaction_pair(
            static_seq, active_seq, cond_row, n_frames=n_frames)
        if new_resp is None:
            if verbose:
                print(f"  reject {p.name}: correlation below threshold")
            rejected += 1
            continue

        existing = accumulated.get((m_idx, c_idx))
        merged = _accumulate_from_similar(
            m_idx, c_idx, existing, new_resp, motifs, coherence)
        accumulated[(m_idx, c_idx)] = merged
        _existing_responses[(m_idx, c_idx)] = merged

        if verbose:
            print(f"  {p.name:<32s} "
                  f"motif={motif_labels[m_idx]:<14s} "
                  f"cond={cond_labels[c_idx]:<11s} "
                  f"corr={merged['correlation']:+.3f} "
                  f"strength={merged['response_strength']:.4f} "
                  f"accum={merged['accumulated_samples']}")

    return {
        "motif_count":    len(motif_labels),
        "condition_count": len(cond_labels),
        "accumulated":    accumulated,
        "rejected":       rejected,
        "skipped":        skipped,
    }


def _save_npz(result: dict, out_path: Path) -> None:
    items = sorted(result["accumulated"].items())     # deterministic order
    n = len(items)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if n == 0:
        np.savez(out_path,
                 motif_id=np.zeros(0, np.uint16),
                 condition_id=np.zeros(0, np.uint16),
                 row_response=np.zeros((0, FREQ_BINS), np.float32),
                 col_response=np.zeros((0, FREQ_BINS), np.float32),
                 cross_response=np.zeros((0, TEMPORAL_BINS), np.float32),
                 warp_x_response=np.zeros((0, WARP_DIM, WARP_DIM), np.float32),
                 warp_y_response=np.zeros((0, WARP_DIM, WARP_DIM), np.float32),
                 response_strength=np.zeros(0, np.float32),
                 response_delay=np.zeros(0, np.float32),
                 accumulated_samples=np.zeros(0, np.uint8))
        return

    motif_ids = np.array([k[0] for k, _ in items], np.uint16)
    cond_ids  = np.array([k[1] for k, _ in items], np.uint16)
    np.savez(
        out_path,
        motif_id=motif_ids,
        condition_id=cond_ids,
        row_response=np.stack([v["row_response"]    for _, v in items]),
        col_response=np.stack([v["col_response"]    for _, v in items]),
        cross_response=np.stack([v["cross_response"] for _, v in items]),
        warp_x_response=np.stack([v["warp_x_response"] for _, v in items]),
        warp_y_response=np.stack([v["warp_y_response"] for _, v in items]),
        response_strength=np.array([v["response_strength"]
                                    for _, v in items], np.float32),
        response_delay=np.array([v["response_delay"] for _, v in items],
                                np.float32),
        accumulated_samples=np.array([v["accumulated_samples"]
                                      for _, v in items], np.uint8),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motifs",     required=True, type=Path)
    ap.add_argument("--conditions", required=True, type=Path)
    ap.add_argument("--data",       required=True, type=Path,
                    help="dataset/interactions root")
    ap.add_argument("--out",        required=True, type=Path)
    ap.add_argument("--frames", type=int, default=FRAMES_DEFAULT)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    for p in (args.motifs, args.conditions):
        if not p.is_file():
            print(f"error: {p} not found", file=sys.stderr); return 2
    if not args.data.is_dir():
        print(f"error: --data {args.data} is not a directory",
              file=sys.stderr); return 2

    t0 = time.perf_counter()
    result = build_interactions(args.motifs, args.conditions, args.data,
                                n_frames=args.frames, verbose=args.verbose)
    _save_npz(result, args.out)
    print(f"wrote {args.out}  ({len(result['accumulated'])} interactions; "
          f"{result['rejected']} rejected, {result['skipped']} skipped; "
          f"{time.perf_counter() - t0:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
