"""sss_synthesizer — Phase 4 resonance-based signal synthesiser.

Design philosophy (Phase 4):

The sss engine treats *images* and *videos* through the same motif
dictionary. An image is the `temporal_length = 0` special case of a
video; a video is what comes out when a condition signal is applied
to the same static motif along a time axis.

Given a (motif, condition, intensity, t, seed) tuple, the
synthesiser produces *envelope* deltas (row / col / cross-resonance
amplitude shifts) and a *warp field* (16×16 → 64×64 displacement
map) that the renderer applies on top of a base waveform. The
synthesiser never reads from a training image directly — only
spectral envelopes and learned response coefficients.

Public surface:

    synth_frame(bank, motif, condition, intensity, t, seed)
        → SynthOutput(envelope_row, envelope_col, cross, warp_x, warp_y)
        Combines the motif's baseline envelopes + warp_self with
        the InteractionResponse keyed by (motif, condition).
        Scales the response by `intensity` and the condition's
        temporal_freq envelope sampled at time `t`.

    apply_warp(image_64x64x3, warp_x_16, warp_y_16, strength)
        → image_64x64x3
        Bilinearly upsamples the 16×16 warp field to 64×64 and
        resamples the image pixels accordingly.

    synthesise_from_envelope(env_row, env_col, cross, seed,
                             size=64) → image_64x64x3
        Naïve renderer that turns row/col/cross envelopes into a
        single-frame image via the Phase 4 cross-axis modulation
        rule:

            for k in 0..63:
                interference[i,j] += cross[k]
                    * cos(2π·(row_freq[k] + col_freq[k])·i)
                    * cos(2π·(row_freq[k] - col_freq[k])·j)

        Plus a per-seed random row+col baseline so per-seed
        diversity is preserved (Phase 1 invariant).

No phase. No raw frames. Every output is regenerated each call
from the same on-disk envelope tables."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.sss_feature_bank import (
    SSSCondition, SSSFeatureBank, SSSInteractionResponse, SSSMotif,
    SFB_FREQ_BINS, SFB_HEATMAP_DIM, SFB_TEMPORAL_BINS, SFB_WARP_DIM,
    CONDITION_CONTINUOUS, CONDITION_IMPULSE, CONDITION_OSCILLATORY,
)


# ── Output container ─────────────────────────────────────────────

@dataclass
class SynthOutput:
    envelope_row: np.ndarray   # (128,)
    envelope_col: np.ndarray   # (128,)
    cross:         np.ndarray  # (64,)
    warp_x:        np.ndarray  # (16, 16) float32 in [-1, 1]
    warp_y:        np.ndarray  # (16, 16) float32 in [-1, 1]


# ── Helpers ──────────────────────────────────────────────────────

def _temporal_envelope_at(cond: SSSCondition, t: float,
                          n_frames: int = 16) -> float:
    """Sample the condition's effective intensity at time t ∈ [0, 1].

    Phase 4 keeps `temporal_freq[64]` as the spectral signature of
    the time axis (no phase). The 64-bin storage is *resampled*
    (not truncated) to the target-length rfft bin count before
    irfft, so frequency content survives the round-trip even when
    the source was originally a much shorter sequence (e.g. 16
    frames → 9 bins → resampled to 64 → must come back to 9 to
    reconstruct).

    For impulse conditions we additionally apply `decay_rate` so
    `effective(t) = irfft_sample · exp(-decay_rate · t · n_frames)`.
    Continuous conditions ignore decay."""
    amp = np.asarray(cond.temporal_freq, dtype=np.float32)
    target_len = max(2, int(n_frames))
    nf = target_len // 2 + 1
    # Linear-resample the stored 64-bin envelope down to the
    # target's nf bins. Truncation would discard the high-bin
    # content where the actual peak typically lives after the
    # trainer's 64-bin upsample.
    if amp.size != nf:
        src_x  = np.linspace(0.0, 1.0, amp.size, endpoint=True)
        target_x = np.linspace(0.0, 1.0, nf, endpoint=True)
        amp_nf = np.interp(target_x, src_x, amp).astype(np.float32)
    else:
        amp_nf = amp
    # Zero phase (Phase 1 amp-only rule).
    spec = amp_nf.astype(np.complex64)
    waveform = np.fft.irfft(spec, n=target_len).astype(np.float32)
    # Sample at fractional t ∈ [0, 1].
    pos = float(np.clip(t, 0.0, 1.0)) * (target_len - 1)
    lo  = int(np.floor(pos)); hi = min(target_len - 1, lo + 1)
    frac = pos - lo
    base = float(waveform[lo] * (1.0 - frac) + waveform[hi] * frac)
    if cond.condition_type == CONDITION_IMPULSE:
        base *= float(np.exp(-cond.decay_rate * pos))
    return base


def _find_response(bank: SSSFeatureBank, motif_id: int,
                   condition_id: int) -> SSSInteractionResponse | None:
    """Linear scan for the (motif, condition) response. Returns None
    when the bank hasn't seen this pairing — the synthesiser then
    falls back to the motif's self-warp + baseline envelope only."""
    for r in bank.responses:
        if r.motif_id == motif_id and r.condition_id == condition_id:
            return r
    return None


# ── synth_frame ──────────────────────────────────────────────────

def synth_frame(bank: SSSFeatureBank, motif: SSSMotif,
                condition: SSSCondition, intensity: float,
                t: float, seed: int,
                n_frames: int = 16) -> SynthOutput:
    """Produce one frame's worth of envelope + warp coefficients.

    Steps:
      (1) Start from the motif's baseline envelopes (row_freq /
          col_freq / cross_resonance) and self-warp.
      (2) Look up the InteractionResponse for (motif, condition);
          fall back to a neutral response (zero deltas) when absent.
      (3) Compute the per-frame condition strength via the
          condition's temporal envelope sampled at `t`. Multiply
          by `intensity` and the response_strength.
      (4) Add the scaled response deltas onto the baseline.

    `bank` is consulted only to walk `bank.responses` — the
    synthesiser does not modify any state."""
    if motif is None or condition is None:
        raise ValueError("synth_frame requires motif and condition")

    # Resolve motif / condition IDs (linear scans — Phase 4 v1
    # banks are small enough that this is cheap; future versions may
    # index).
    motif_id = -1
    for i, m in enumerate(bank.motifs):
        if m is motif or m.label == motif.label:
            motif_id = i; break
    cond_id = -1
    for i, c in enumerate(bank.conditions):
        if c is condition or c.label == condition.label:
            cond_id = i; break

    response = _find_response(bank, motif_id, cond_id) if \
        (motif_id >= 0 and cond_id >= 0) else None

    # Temporal envelope sampled at this frame. `intensity` is the
    # user-supplied scaling (0..1); the per-frame condition envelope
    # comes from the trained spectrum.
    cond_t = _temporal_envelope_at(condition, t, n_frames=n_frames)
    scale = float(intensity) * cond_t

    # Baseline (no response): motif envelopes + self-warp.
    out_row = np.asarray(motif.row_freq,        dtype=np.float32).copy()
    out_col = np.asarray(motif.col_freq,        dtype=np.float32).copy()
    out_x   = np.asarray(motif.cross_resonance, dtype=np.float32).copy()
    out_wx  = np.asarray(motif.warp_self_x,     dtype=np.float32).copy()
    out_wy  = np.asarray(motif.warp_self_y,     dtype=np.float32).copy()

    if response is not None:
        rs = float(response.response_strength)
        s  = rs * scale
        out_row += s * np.asarray(response.row_response,    dtype=np.float32)
        out_col += s * np.asarray(response.col_response,    dtype=np.float32)
        out_x   += s * np.asarray(response.cross_response,  dtype=np.float32)
        out_wx  += s * np.asarray(response.warp_x_response, dtype=np.float32)
        out_wy  += s * np.asarray(response.warp_y_response, dtype=np.float32)

    # Per-seed jitter on the warp field — keeps Phase 1's per-seed
    # diversity rule alive at the motif level. The jitter scale is
    # tiny (0.02) so static synthesis is essentially deterministic
    # but two seeds visibly differ in their warp.
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    jitter_x = (rng.random((SFB_WARP_DIM, SFB_WARP_DIM),
                           dtype=np.float32) - 0.5) * 0.04
    jitter_y = (rng.random((SFB_WARP_DIM, SFB_WARP_DIM),
                           dtype=np.float32) - 0.5) * 0.04
    out_wx = np.clip(out_wx + jitter_x, -1.0, 1.0)
    out_wy = np.clip(out_wy + jitter_y, -1.0, 1.0)

    return SynthOutput(
        envelope_row=out_row, envelope_col=out_col,
        cross=out_x, warp_x=out_wx, warp_y=out_wy,
    )


# ── Warp application ─────────────────────────────────────────────

def _bilinear_upsample(field: np.ndarray, target: int) -> np.ndarray:
    """16×16 → target×target bilinear upsample of a single-channel
    field. Returns float32."""
    src = field.astype(np.float32)
    dim = src.shape[0]
    if dim == target:
        return src.copy()
    yi = np.linspace(0, dim - 1, target).astype(np.float32)
    xi = np.linspace(0, dim - 1, target).astype(np.float32)
    y0 = np.floor(yi).astype(np.int64); y1 = np.minimum(y0 + 1, dim - 1)
    x0 = np.floor(xi).astype(np.int64); x1 = np.minimum(x0 + 1, dim - 1)
    fy = (yi - y0).reshape(-1, 1)
    fx = (xi - x0).reshape(1, -1)
    a = src[y0[:, None], x0[None, :]]
    b = src[y0[:, None], x1[None, :]]
    c = src[y1[:, None], x0[None, :]]
    d = src[y1[:, None], x1[None, :]]
    return ((1 - fy) * ((1 - fx) * a + fx * b) +
                  fy  * ((1 - fx) * c + fx * d))


def apply_warp(image: np.ndarray, warp_x: np.ndarray,
               warp_y: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Resample `image` (H, W, 3) according to a 16×16 warp field
    bilinearly upsampled to (H, W). Displacement units are pixels
    scaled by `strength`. Out-of-range samples use edge clamping.

    Inputs:
        image:   float32 or uint8 (H, W, 3). uint8 input is
                 promoted to float32 internally; output dtype
                 matches input.
        warp_x:  (16, 16) float32 in [-1, +1]. +1 = displace one
                 max-displacement unit (set via `strength` in pixels).
        warp_y:  same shape, vertical component.
        strength: maximum pixel displacement when warp == ±1.
                 Phase 4 default is 8 px (12.5 % of a 64-px frame)
                 which produces visible motion without tearing.
    """
    arr = np.asarray(image)
    orig_dtype = arr.dtype
    if arr.dtype != np.float32:
        arr_f = arr.astype(np.float32)
    else:
        arr_f = arr
    H, W, C = arr_f.shape

    wx = _bilinear_upsample(np.asarray(warp_x, dtype=np.float32), W)
    wy = _bilinear_upsample(np.asarray(warp_y, dtype=np.float32), H)
    dx = wx * float(strength)
    dy = wy * float(strength)

    # Sign convention: warp_y > 0 displaces content downward.
    # output[y, x] = input[y - dy, x] — i.e. the destination samples
    # from above to display "what was up here a moment ago" lower
    # in the frame. This is the standard "displacement field"
    # convention used by optical flow.
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    sy = np.clip(yy - dy, 0.0, H - 1.0)
    sx = np.clip(xx - dx, 0.0, W - 1.0)

    y0 = np.floor(sy).astype(np.int64); y1 = np.minimum(y0 + 1, H - 1)
    x0 = np.floor(sx).astype(np.int64); x1 = np.minimum(x0 + 1, W - 1)
    fy = (sy - y0)[..., None]
    fx = (sx - x0)[..., None]

    a = arr_f[y0, x0]; b = arr_f[y0, x1]
    c = arr_f[y1, x0]; d = arr_f[y1, x1]
    out = ((1 - fy) * ((1 - fx) * a + fx * b) +
                  fy  * ((1 - fx) * c + fx * d))

    if orig_dtype == np.uint8:
        return np.clip(out, 0, 255).astype(np.uint8)
    return out.astype(orig_dtype)


# ── Naïve renderer from envelope ─────────────────────────────────

def synthesise_from_envelope(env_row: np.ndarray, env_col: np.ndarray,
                             cross: np.ndarray, seed: int,
                             size: int = 64) -> np.ndarray:
    """Produce a (size, size, 3) float32 image in [0, 1] from the
    three envelope components. The Phase 4 cross-axis interpretation:

        interference[i,j] = Σ_k cross[k]
                          · cos(2π·(row_freq[k] + col_freq[k])·i / size)
                          · cos(2π·(row_freq[k] - col_freq[k])·j / size)

    plus a row/col baseline reconstructed via per-seed random phase
    (Phase 1 invariant)."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    # Row baseline: irfft of env_row[:NF] with random phase.
    NF_row = size // 2 + 1
    NF_col = size // 2 + 1
    row_amp = np.asarray(env_row, dtype=np.float32)
    col_amp = np.asarray(env_col, dtype=np.float32)
    # Take the first NF bins as the row/column spectrum.
    row_spec = (row_amp[:NF_row]
                * np.exp(1j * rng.uniform(-np.pi, np.pi, NF_row))
                ).astype(np.complex64)
    col_spec = (col_amp[:NF_col]
                * np.exp(1j * rng.uniform(-np.pi, np.pi, NF_col))
                ).astype(np.complex64)
    row_signal = np.fft.irfft(row_spec, n=size).astype(np.float32)
    col_signal = np.fft.irfft(col_spec, n=size).astype(np.float32)
    # Compose a (size, size) gray baseline as the outer sum.
    gray = row_signal[None, :] + col_signal[:, None]
    # Cross-axis interference: per the Phase 4 rule. cross[k]
    # modulates a (row_freq[k] ± col_freq[k]) carrier pair.
    cross_a = np.asarray(cross, dtype=np.float32)
    K = min(cross_a.size, row_amp.size, col_amp.size)
    if K > 0:
        ii = np.arange(size, dtype=np.float32) / size
        jj = np.arange(size, dtype=np.float32) / size
        interference = np.zeros((size, size), dtype=np.float32)
        # Pick the K most-energetic indices to keep the inner loop
        # bounded — 64 bins × 64×64 is fine inline.
        for k in range(K):
            r_k = float(row_amp[k]) if k < row_amp.size else 0.0
            c_k = float(col_amp[k]) if k < col_amp.size else 0.0
            x_k = float(cross_a[k])
            if abs(x_k) < 1e-7:
                continue
            sum_f  = (r_k + c_k)
            diff_f = (r_k - c_k)
            ci = np.cos(2 * np.pi * sum_f  * ii)         # (size,)
            cj = np.cos(2 * np.pi * diff_f * jj)         # (size,)
            interference += x_k * np.outer(cj, ci)
        gray = gray + interference

    # Map to [0, 1] with a soft squash and broadcast to RGB.
    gray = (gray - gray.min())
    span = max(1e-6, float(gray.max() - gray.min()))
    gray = np.clip(gray / span, 0.0, 1.0)
    return np.stack([gray, gray, gray], axis=-1)
