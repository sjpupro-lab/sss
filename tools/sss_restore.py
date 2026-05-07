"""SSS image restoration / enhancement engine.

Built on the same row-and-column FFT primitives as sss_rowvae:
every operation projects the image to its row spectrum (Y axis,
per channel) or column spectrum (X axis, per channel), edits the
band that matters, and inverse-FFTs back. Damaged regions get
*sequential row inference* — the previous row anchors the next,
so a restored region is grown row-by-row instead of being filled
as an opaque blob.

Public surface (all return BGR uint8 ndarrays the same shape as
the input):

    SSSRestore(model_path=None, prompt="")
        .restore(image, mask=None, levels=(32, 64, 128, 256))
        .upscale(image, scale=2)
        .sharpen(image, strength=0.5)
        .denoise(image, strength=0.5)
        .color_correct(image, target_prompt="")
        .enhance(image, prompt="")

    auto_detect_damage(image, threshold=0.0035) -> mask  (H, W) float in [0, 1]
    load_sss_model(path) -> SSSModel                    (lightweight reader)

Pure numpy + tools.sss_image_io. No cv2, no Pillow, no scipy. Reference
cells (when a model is loaded) are looked up via ce_feed + ce_distance
through the existing pybridge so search uses the same key the C engine
uses at generate time. Memory: at most one (H, W, 3) intermediate
during sweeps; pyramid levels are processed one at a time.

The CLI wraps the same path:

    python3 tools/sss_restore.py input.png \\
        --mode enhance --prompt "vivid" --out result.png
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# When this file is launched directly (`python3 tools/sss_restore.py
# ...`), `tools` isn't on sys.path yet. Insert the repo root so the
# package-qualified imports below work in both test and CLI mode.
if __package__ in (None, "") and __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

from tools import sss_image_io


# ─────────────────────────────────────────────────────────────
# .sss model loader (Python port of ce_core/sss_io.c)
# ─────────────────────────────────────────────────────────────
SSS_MAGIC_V9 = 0x53535839  # 'SSX9'
SSS_MAGIC_V8 = 0x53535838  # 'SSX8'
SSS_FP_LEN   = 256

CE_COLOR = 1
CE_SHAPE = 2
CE_FACE  = 3


@dataclass
class SSSCell:
    type: int
    label: str
    ce_key: bytes              # 64 bytes
    fp: np.ndarray             # (256,) float32
    amp: np.ndarray            # variable shape, see _reshape_amp
    phase: np.ndarray          # same shape as amp


@dataclass
class SSSModel:
    height: int
    width: int
    nf: int
    nf_low: int
    cells: list = field(default_factory=list)


def _read_u32(f) -> int:
    b = f.read(4)
    if len(b) != 4:
        raise IOError(".sss: truncated u32")
    return struct.unpack("<I", b)[0]


def _read_floats(f, n: int) -> np.ndarray:
    b = f.read(4 * n)
    if len(b) != 4 * n:
        raise IOError(f".sss: truncated float[{n}] block")
    return np.frombuffer(b, dtype="<f4").astype(np.float32).copy()


def _reshape_amp(cell_type: int, raw: np.ndarray, m: SSSModel) -> np.ndarray:
    """Match the C-side cell layouts in sss_rowvae.c:
       COLOR  amp shape (H, NF_LOW, 3)
       FACE   amp shape (H, NF_HIGH, 3)
       SHAPE  amp shape (W, NF)        — gray, no channel
    Returns the original flat array if the size doesn't match
    (forward-compatible: a future cell type just stays flat)."""
    H, W, NF, NF_LOW = m.height, m.width, m.nf, m.nf_low
    NF_HIGH = NF - NF_LOW
    if cell_type == CE_COLOR and raw.size == H * NF_LOW * 3:
        return raw.reshape(H, NF_LOW, 3)
    if cell_type == CE_FACE and raw.size == H * NF_HIGH * 3:
        return raw.reshape(H, NF_HIGH, 3)
    if cell_type == CE_SHAPE and raw.size == W * NF:
        return raw.reshape(W, NF)
    return raw


def load_sss_model(path) -> SSSModel:
    """Read a .sss v9 (or v8) file and return an SSSModel. Raises
    IOError / ValueError on any malformed header or short read."""
    with open(path, "rb") as f:
        magic = _read_u32(f)
        if magic not in (SSS_MAGIC_V8, SSS_MAGIC_V9):
            raise ValueError(f".sss: bad magic {magic:#010x}")
        version = _read_u32(f)
        is_v8 = (magic == SSS_MAGIC_V8 and version == 8)
        if not is_v8 and (magic != SSS_MAGIC_V9 or version != 9):
            raise ValueError(f".sss: unsupported version {version}")

        m = SSSModel(
            height=_read_u32(f),
            width=_read_u32(f),
            nf=_read_u32(f),
            nf_low=_read_u32(f),
        )
        n_cells = _read_u32(f)
        for _ in range(n_cells):
            ctype = _read_u32(f)
            label_len = _read_u32(f)
            label_bytes = f.read(label_len) if label_len > 0 else b""
            if len(label_bytes) != label_len:
                raise IOError(".sss: truncated label")
            label = label_bytes.decode("utf-8", errors="replace")

            if is_v8:
                # v8 didn't write the ce_key; regenerate via the C
                # bridge to match runtime behaviour.
                from tools.sss_memory import ce_feed_bytes
                ce_key = ce_feed_bytes(label)
            else:
                ce_key = f.read(64)
                if len(ce_key) != 64:
                    raise IOError(".sss: truncated ce_key")

            fp = _read_floats(f, SSS_FP_LEN)
            amp_len = _read_u32(f)
            amp = _read_floats(f, amp_len) if amp_len > 0 else np.zeros(0, np.float32)
            phase_len = _read_u32(f)
            phase = (_read_floats(f, phase_len)
                     if phase_len > 0 else np.zeros(0, np.float32))

            m.cells.append(SSSCell(
                type=ctype, label=label, ce_key=ce_key, fp=fp,
                amp=_reshape_amp(ctype, amp, m),
                phase=_reshape_amp(ctype, phase, m),
            ))
    return m


# ─────────────────────────────────────────────────────────────
# Prompt → reference cell lookup
# ─────────────────────────────────────────────────────────────
def _pick_reference_cells(model: Optional[SSSModel], prompt: str):
    """For each cell type return (cell, distance) of the closest cell
    to any prompt token, or None when no model / no match. Uses the
    same ce_feed + ce_distance pair the C engine uses at generate
    time, so the picked reference is consistent with sss_gen output.

    The match threshold mirrors sss_rowvae.c::find_candidate_cells'
    `MATCH_TIGHT = 100` — anything looser pulls in semantically
    unrelated cells (e.g. "warm" loosely matching a shape cell)."""
    if not model or not prompt:
        return {CE_COLOR: None, CE_SHAPE: None, CE_FACE: None}
    try:
        from tools.sss_memory import ce_feed_bytes, ce_distance
    except (ImportError, RuntimeError):
        return {CE_COLOR: None, CE_SHAPE: None, CE_FACE: None}

    tokens = [t for t in prompt.split() if t]
    if not tokens:
        return {CE_COLOR: None, CE_SHAPE: None, CE_FACE: None}

    keys = [ce_feed_bytes(t) for t in tokens]

    refs = {}
    for ctype in (CE_COLOR, CE_SHAPE, CE_FACE):
        best = None
        best_d = 1_000_000
        for cell in model.cells:
            if cell.type != ctype:
                continue
            d = min(ce_distance(k, cell.ce_key) for k in keys)
            if d < best_d:
                best_d = d
                best = cell
        refs[ctype] = (best, best_d) if (best and best_d <= 100) else None
    return refs


# ─────────────────────────────────────────────────────────────
# FFT helpers (numpy)
# ─────────────────────────────────────────────────────────────
def _row_fft(image_f: np.ndarray):
    """rfft along the X axis. image_f is float (H, W, 3) in [0, 1].
    Returns amp, phase shaped (H, NF, 3)."""
    spec = np.fft.rfft(image_f, axis=1)
    return np.abs(spec), np.angle(spec)


def _row_ifft(amp: np.ndarray, phase: np.ndarray, W: int) -> np.ndarray:
    spec = amp * np.exp(1j * phase)
    return np.fft.irfft(spec, n=W, axis=1).astype(np.float32)


def _col_fft(image_f: np.ndarray):
    """rfft along the Y axis. image_f is float (H, W, 3). Returns
    amp, phase shaped (NF, W, 3)."""
    spec = np.fft.rfft(image_f, axis=0)
    return np.abs(spec), np.angle(spec)


def _col_ifft(amp: np.ndarray, phase: np.ndarray, H: int) -> np.ndarray:
    spec = amp * np.exp(1j * phase)
    return np.fft.irfft(spec, n=H, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Resize (nearest / bilinear, numpy only)
# ─────────────────────────────────────────────────────────────
def _resize_nn(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    H, W = arr.shape[:2]
    if H == h and W == w:
        return arr.copy()
    yi = np.linspace(0, max(0, H - 1), h).round().astype(np.int64)
    xi = np.linspace(0, max(0, W - 1), w).round().astype(np.int64)
    return arr[yi[:, None], xi[None, :]]


def _resize_bilinear(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize for float (H, W, 3). Slow loops are avoided —
    everything is vectorised over (h, w)."""
    H, W = arr.shape[:2]
    if H == h and W == w:
        return arr.copy()
    yf = np.linspace(0, max(0, H - 1), h)
    xf = np.linspace(0, max(0, W - 1), w)
    y0 = np.clip(np.floor(yf).astype(np.int64), 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.clip(np.floor(xf).astype(np.int64), 0, W - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    wy = (yf - y0).reshape(-1, 1, 1)
    wx = (xf - x0).reshape(1, -1, 1)
    a = arr[y0[:, None], x0[None, :]]
    b = arr[y0[:, None], x1[None, :]]
    c = arr[y1[:, None], x0[None, :]]
    d = arr[y1[:, None], x1[None, :]]
    top = a * (1 - wx) + b * wx
    bot = c * (1 - wx) + d * wx
    return (top * (1 - wy) + bot * wy).astype(arr.dtype)


# ─────────────────────────────────────────────────────────────
# Sequential row inference
# ─────────────────────────────────────────────────────────────
def _row_inpaint_pass(image: np.ndarray, mask: np.ndarray, *,
                      reverse: bool = False) -> np.ndarray:
    """One sweep of sequential row inference. For every row that has
    any damaged pixel (mask sum > 0), blend its row FFT toward the
    previous *good* row's FFT amp + phase, then irfft. Damaged
    pixels take the irfft output; clean pixels stay untouched.

    reverse=True walks bottom→top with the next row as anchor.
    """
    H, W, _ = image.shape
    out = image.copy()
    rows = range(H - 1, -1, -1) if reverse else range(H)
    anchor_amp = None
    anchor_phase = None
    for r in rows:
        damaged = mask[r].max() > 0.5
        if anchor_amp is None and not damaged:
            spec = np.fft.rfft(out[r], axis=0)
            anchor_amp = np.abs(spec)
            anchor_phase = np.angle(spec)
            continue
        if not damaged:
            # Clean row → update the anchor with light EMA so the
            # spectrum drifts slowly along the image instead of
            # snapping to a single reference.
            spec = np.fft.rfft(out[r], axis=0)
            cur_amp = np.abs(spec)
            cur_phase = np.angle(spec)
            anchor_amp   = 0.5 * cur_amp   + 0.5 * anchor_amp
            anchor_phase = cur_phase  # latest clean phase is more reliable
            continue
        # Damaged row → reconstruct.
        if anchor_amp is None:
            # Hit a damaged region before any clean row; defer.
            continue
        new_spec = anchor_amp * np.exp(1j * anchor_phase)
        new_row = np.fft.irfft(new_spec, n=W, axis=0).astype(np.float32)
        # Pixel-wise blend: damaged pixels take new, clean pixels keep
        # the original. Mask is (W,) here.
        m = mask[r][:, None]
        out[r] = m * new_row + (1.0 - m) * out[r]
    return out


def _col_inpaint_pass(image: np.ndarray, mask: np.ndarray, *,
                      reverse: bool = False) -> np.ndarray:
    """Column-axis twin of _row_inpaint_pass."""
    # Implemented by transposing → row pass → transpose back.
    img_t = np.transpose(image, (1, 0, 2)).copy()
    mask_t = mask.T.copy()
    out_t = _row_inpaint_pass(img_t, mask_t, reverse=reverse)
    return np.transpose(out_t, (1, 0, 2))


def _restore_one_scale(image_f: np.ndarray, mask_f: np.ndarray) -> np.ndarray:
    """Both directions on both axes, then average. Damaged pixels
    only get the averaged result; clean pixels are untouched."""
    a = _row_inpaint_pass(image_f, mask_f, reverse=False)
    b = _row_inpaint_pass(image_f, mask_f, reverse=True)
    c = _col_inpaint_pass(image_f, mask_f, reverse=False)
    d = _col_inpaint_pass(image_f, mask_f, reverse=True)
    avg = (a + b + c + d) * 0.25
    m = mask_f[..., None]
    return (m * avg + (1.0 - m) * image_f).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Auto damage detection
# ─────────────────────────────────────────────────────────────
def auto_detect_damage(image: np.ndarray, threshold: float = 0.0035) -> np.ndarray:
    """Detect rows/columns whose mean-square distance to BOTH
    neighbours exceeds `threshold` (image is normalised to [0, 1] for
    the comparison). Returns a (H, W) float mask, 1.0 on every pixel
    of a flagged row or column. Threshold defaults to 0.0035 ≈ 15/255
    per channel — anything smaller fires on natural texture noise."""
    img = image.astype(np.float32) / 255.0
    H, W, _ = img.shape
    mask = np.zeros((H, W), np.float32)

    # Rows
    for r in range(1, H - 1):
        du = float(np.mean((img[r] - img[r - 1]) ** 2))
        dd = float(np.mean((img[r] - img[r + 1]) ** 2))
        if du > threshold and dd > threshold:
            mask[r, :] = 1.0
    # Columns
    for c in range(1, W - 1):
        dl = float(np.mean((img[:, c] - img[:, c - 1]) ** 2))
        dr = float(np.mean((img[:, c] - img[:, c + 1]) ** 2))
        if dl > threshold and dr > threshold:
            mask[:, c] = 1.0
    return mask


# ─────────────────────────────────────────────────────────────
# SSSRestore
# ─────────────────────────────────────────────────────────────
class SSSRestore:
    """Stateful entry point. Holds the loaded model + cached prompt
    references so a single instance can run many ops without re-paying
    the model load."""

    def __init__(self, model_path: Optional[str] = None, prompt: str = ""):
        if model_path is None:
            model_path = os.environ.get("SSS_MODEL_PATH", "") or _find_sss_model()
        self.model_path = model_path or ""
        self.model: Optional[SSSModel] = None
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = load_sss_model(self.model_path)
            except (IOError, ValueError):
                # Bad / partial model file — fall back to model-less
                # operation. Restore still works, it just can't borrow
                # high-frequency texture.
                self.model = None
        self._prompt = prompt
        self._refs = _pick_reference_cells(self.model, prompt)

    # ── reference cell access ──────────────────────────────────
    def set_prompt(self, prompt: str):
        self._prompt = prompt
        self._refs = _pick_reference_cells(self.model, prompt)

    def _ref(self, ctype):
        item = self._refs.get(ctype)
        return item[0] if item else None

    # ── 1) restore: pyramid + sequential row/col inference ─────
    def restore(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
                levels=(32, 64, 128, 256)) -> np.ndarray:
        """Mask is 1 on damaged pixels, 0 on clean. None → auto-detect.
        Pyramid: coarse-to-fine. Each level downsamples image+mask,
        runs the row/col sweep, upsamples back, and blends only over
        the damaged area. Smaller levels give the structural fill;
        larger levels add the texture detail."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("restore: expected (H, W, 3) BGR uint8")
        H, W, _ = image.shape

        if mask is None:
            mask = auto_detect_damage(image)
        mask = mask.astype(np.float32)
        if mask.shape != (H, W):
            mask = _resize_nn(mask[..., None], H, W)[..., 0]

        # Mobile-friendly: skip levels larger than the image.
        usable = sorted({int(L) for L in levels if int(L) >= 8 and int(L) <= max(H, W)})
        if not usable:
            usable = [min(H, W)]
        if usable[-1] != max(H, W):
            usable.append(max(H, W))

        img_f = image.astype(np.float32) / 255.0
        out = img_f.copy()

        for lvl in usable:
            lvl_h = min(H, lvl)
            lvl_w = min(W, lvl)
            small_img  = _resize_bilinear(out, lvl_h, lvl_w)
            small_mask = _resize_nn(mask[..., None], lvl_h, lvl_w)[..., 0]
            if small_mask.max() < 0.5:
                continue                                # nothing damaged at this scale
            healed = _restore_one_scale(small_img, small_mask)
            healed_full = _resize_bilinear(healed, H, W)
            # Replace only the damaged pixels at full res.
            m = mask[..., None]
            out = m * healed_full + (1.0 - m) * out

        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return out

    # ── 2) upscale: zero-pad rfft + reference high-freq inject ─
    def upscale(self, image: np.ndarray, scale: int = 2) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("upscale: expected (H, W, 3) BGR uint8")
        scale = int(max(1, scale))
        if scale == 1:
            return image.copy()
        H, W, _ = image.shape
        new_H, new_W = H * scale, W * scale
        new_NF = new_W // 2 + 1
        old_NF = W // 2 + 1
        col_NF = new_H // 2 + 1
        col_old_NF = H // 2 + 1

        img_f = image.astype(np.float32) / 255.0

        # ── Row pass: rfft existing rows, zero-pad to new_NF, irfft.
        spec = np.fft.rfft(img_f, axis=1)             # (H, old_NF, 3)
        amp = np.abs(spec); phase = np.angle(spec)

        # Borrow weak high-frequency texture from the reference COLOR
        # / FACE cell when one matched the prompt. The cell's spectrum
        # is at the model's NF resolution (typically NF=33 for 64-wide
        # training); we re-bin onto [old_NF, new_NF) at 0.3× weight so
        # the upscale inherits the same character of detail the engine
        # produces at gen time without overwriting the source.
        ref_amp = self._ref_row_amp_full()
        if ref_amp is not None:
            row_inject = _stretch_band(ref_amp, H, target_h=H,
                                       target_NF=new_NF, src_lo=0,
                                       fill_lo=old_NF, fill_hi=new_NF) * 0.3
            new_amp = np.zeros((H, new_NF, 3), np.float32)
            new_amp[:, :old_NF, :] = amp
            new_amp[:, old_NF:, :] = row_inject[:, old_NF:, :]
            new_phase = np.zeros((H, new_NF, 3), np.float32)
            # Mirror existing phase; the high-band injection rides on
            # zero phase which still produces edge-rich detail.
            new_phase[:, :old_NF, :] = phase
        else:
            new_amp = np.zeros((H, new_NF, 3), np.float32)
            new_phase = np.zeros((H, new_NF, 3), np.float32)
            new_amp[:, :old_NF, :] = amp
            new_phase[:, :old_NF, :] = phase

        wide = np.fft.irfft(new_amp * np.exp(1j * new_phase),
                            n=new_W, axis=1).astype(np.float32)

        # ── Column pass on the newly wide image: extend H → new_H.
        spec_c = np.fft.rfft(wide, axis=0)            # (col_old_NF, new_W, 3)
        amp_c = np.abs(spec_c); phase_c = np.angle(spec_c)
        new_amp_c = np.zeros((col_NF, new_W, 3), np.float32)
        new_phase_c = np.zeros((col_NF, new_W, 3), np.float32)
        new_amp_c[:col_old_NF, :, :]  = amp_c
        new_phase_c[:col_old_NF, :, :] = phase_c
        # Optional shape-cell injection on the column high band.
        ref_col = self._ref_col_amp_full()
        if ref_col is not None:
            col_inject = _stretch_col_band(ref_col, target_h_NF=col_NF,
                                           target_w=new_W,
                                           fill_lo=col_old_NF,
                                           fill_hi=col_NF) * 0.3
            new_amp_c[col_old_NF:, :, :] = col_inject[col_old_NF:, :, :]

        big = np.fft.irfft(new_amp_c * np.exp(1j * new_phase_c),
                           n=new_H, axis=0).astype(np.float32)

        return np.clip(big * 255.0, 0, 255).astype(np.uint8)

    # ── 3) sharpen: high-freq amp boost ────────────────────────
    def sharpen(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        s = float(np.clip(strength, 0.0, 1.0))
        if s <= 1e-6:
            return image.copy()
        img_f = image.astype(np.float32) / 255.0
        H, W, _ = img_f.shape

        # Row direction.
        spec = np.fft.rfft(img_f, axis=1)
        amp  = np.abs(spec); phase = np.angle(spec)
        NF = amp.shape[1]
        k = np.arange(NF, dtype=np.float32) / max(1, NF - 1)         # 0..1
        # Boost upper half; slight damp on the very top bin (Nyquist
        # is the noise-prone band, see denoise too).
        boost = 1.0 + s * np.where(k >= 0.5, k - 0.5, 0.0) * 2.0       # 1.0 → 1+s
        boost[-1] *= max(0.0, 1.0 - 0.4 * s)
        amp = amp * boost.reshape(1, NF, 1)
        img_f = np.fft.irfft(amp * np.exp(1j * phase), n=W, axis=1).astype(np.float32)

        # Column direction.
        spec = np.fft.rfft(img_f, axis=0)
        amp  = np.abs(spec); phase = np.angle(spec)
        NFc = amp.shape[0]
        kc = np.arange(NFc, dtype=np.float32) / max(1, NFc - 1)
        boost = 1.0 + s * np.where(kc >= 0.5, kc - 0.5, 0.0) * 2.0
        boost[-1] *= max(0.0, 1.0 - 0.4 * s)
        amp = amp * boost.reshape(NFc, 1, 1)
        img_f = np.fft.irfft(amp * np.exp(1j * phase), n=H, axis=0).astype(np.float32)

        return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)

    # ── 4) denoise: clip outlier amps ──────────────────────────
    def denoise(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        s = float(np.clip(strength, 0.0, 1.0))
        if s <= 1e-6:
            return image.copy()
        img_f = image.astype(np.float32) / 255.0
        H, W, _ = img_f.shape

        # Per-bin, per-channel median across rows; rows whose amp at
        # that bin is far above the median get pulled toward median.
        spec = np.fft.rfft(img_f, axis=1)
        amp  = np.abs(spec); phase = np.angle(spec)
        med  = np.median(amp, axis=0, keepdims=True)
        mad  = np.median(np.abs(amp - med), axis=0, keepdims=True) + 1e-9
        z    = (amp - med) / (mad * 1.4826)        # robust z-score
        # Anything > 2σ is treated as anomalous; pull toward median by
        # `s` (full clamp at strength=1).
        outlier = (z > 2.0).astype(np.float32) * s
        amp = amp * (1.0 - outlier) + med * outlier
        img_f = np.fft.irfft(amp * np.exp(1j * phase), n=W, axis=1).astype(np.float32)

        # Same for columns.
        spec = np.fft.rfft(img_f, axis=0)
        amp  = np.abs(spec); phase = np.angle(spec)
        med  = np.median(amp, axis=1, keepdims=True)
        mad  = np.median(np.abs(amp - med), axis=1, keepdims=True) + 1e-9
        z    = (amp - med) / (mad * 1.4826)
        outlier = (z > 2.0).astype(np.float32) * s
        amp = amp * (1.0 - outlier) + med * outlier
        img_f = np.fft.irfft(amp * np.exp(1j * phase), n=H, axis=0).astype(np.float32)

        return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)

    # ── 5) color_correct: low-freq amp shift ───────────────────
    def color_correct(self, image: np.ndarray,
                      target_prompt: str = "") -> np.ndarray:
        img_f = image.astype(np.float32) / 255.0
        H, W, _ = img_f.shape

        # Keyword shortcuts: per-channel low-band gain on row+col DC.
        keyword_gains = {
            "warm":  np.array([0.96, 1.00, 1.06], np.float32),
            "cool":  np.array([1.06, 1.02, 0.96], np.float32),
            "vivid": None,
            "bright": None,
            "dim":   None,
        }
        kw = (target_prompt or "").lower().strip()
        if kw in keyword_gains and keyword_gains[kw] is not None:
            return _apply_channel_gain(img_f, keyword_gains[kw])
        if kw == "vivid":
            # Saturate via slight pull away from the mean luma.
            mean = img_f.mean(axis=2, keepdims=True)
            return _to_uint8(img_f + 0.20 * (img_f - mean))
        if kw == "bright":
            return _to_uint8(img_f * 1.10 + 0.04)
        if kw == "dim":
            return _to_uint8(img_f * 0.92)

        # Otherwise look up a COLOR cell and blend its low-frequency
        # row-amp toward the image's. Phase is left intact so the
        # original structure is preserved — only the colour pull moves.
        if target_prompt:
            self.set_prompt(target_prompt)
        ref = self._ref(CE_COLOR)
        if ref is None or ref.amp.ndim != 3:
            return image.copy()                       # no reference → no-op
        spec = np.fft.rfft(img_f, axis=1)             # (H, NF_img, 3)
        amp  = np.abs(spec); phase = np.angle(spec)
        NF_img = amp.shape[1]
        NF_low_ref = ref.amp.shape[1]
        # Resample ref amp to (H, NF_low_ref, 3)-aligned at the image
        # H / NF_img by simple nearest indexing along H.
        if ref.amp.shape[0] != H:
            ri = np.linspace(0, ref.amp.shape[0] - 1, H).round().astype(np.int64)
            ref_amp = ref.amp[ri]
        else:
            ref_amp = ref.amp
        nf_pull = min(NF_img, NF_low_ref)
        # Blend at 0.4 — strong enough to shift hue, gentle enough to
        # preserve the source character.
        amp[:, :nf_pull, :] = (
            0.6 * amp[:, :nf_pull, :] + 0.4 * ref_amp[:, :nf_pull, :])
        img_f = np.fft.irfft(amp * np.exp(1j * phase), n=W, axis=1).astype(np.float32)
        return _to_uint8(img_f)

    # ── 6) enhance: prompt → chained ops ───────────────────────
    def enhance(self, image: np.ndarray, prompt: str = "") -> np.ndarray:
        """Tiny rule-based parser. Korean and English keywords map to
        the underlying ops; an empty prompt does denoise + light
        sharpen by default."""
        text = (prompt or "").lower()
        out = image
        if not text:
            out = self.denoise(out, 0.4)
            out = self.sharpen(out, 0.3)
            return out

        # Keyword detection. Multiple keywords compose.
        if any(k in text for k in ("선명", "sharp", "clarity", "clear")):
            out = self.sharpen(out, 0.6)
        if any(k in text for k in ("깨끗", "denoise", "노이즈")):
            out = self.denoise(out, 0.6)
        if any(k in text for k in ("밝", "bright")):
            out = self.color_correct(out, "bright")
        if any(k in text for k in ("어둡", "dim")):
            out = self.color_correct(out, "dim")
        if any(k in text for k in ("따뜻", "warm")):
            out = self.color_correct(out, "warm")
        if any(k in text for k in ("차가", "쿨", "cool")):
            out = self.color_correct(out, "cool")
        if any(k in text for k in ("진하", "vivid", "선명한 색")):
            out = self.color_correct(out, "vivid")
        if out is image:
            # No keyword hit — fall back to gentle defaults.
            out = self.denoise(image, 0.3)
            out = self.sharpen(out, 0.3)
        return out

    # ── reference-cell row/col helpers ─────────────────────────
    def _ref_row_amp_full(self):
        """Return the FACE/COLOR reference cells' row-amp in
        (H_ref, NF_ref, 3) form, FACE preferred for high-band detail.
        None if neither reference is loaded."""
        ref = self._ref(CE_FACE) or self._ref(CE_COLOR)
        if ref is None or ref.amp.ndim != 3:
            return None
        return ref.amp

    def _ref_col_amp_full(self):
        """SHAPE cell's column-amp shape (W, NF). Broadcast along the
        channel axis so callers can use it like the row variant."""
        ref = self._ref(CE_SHAPE)
        if ref is None or ref.amp.ndim != 2:
            return None
        return ref.amp


# ─────────────────────────────────────────────────────────────
# Spectrum-band stretching helpers (used by upscale)
# ─────────────────────────────────────────────────────────────
def _stretch_band(ref_amp: np.ndarray, src_h: int, *, target_h: int,
                  target_NF: int, src_lo: int, fill_lo: int, fill_hi: int) -> np.ndarray:
    """Map a (H_ref, NF_ref, 3) reference amp onto (target_h, target_NF, 3)
    so the high band [fill_lo:fill_hi] gets a plausible amp signature
    instead of zero-padding silence. Uses linear axis interpolation —
    cheap and bounded."""
    H_ref, NF_ref = ref_amp.shape[0], ref_amp.shape[1]
    out = np.zeros((target_h, target_NF, 3), np.float32)
    if NF_ref <= 0:
        return out
    # Row index map.
    yi = np.linspace(0, H_ref - 1, target_h).round().astype(np.int64)
    # Frequency index map for [fill_lo, fill_hi).
    bins = np.arange(fill_lo, fill_hi, dtype=np.float32)
    if bins.size == 0:
        return out
    # Stretch the available NF_ref bins across the whole [src_lo, target_NF)
    # band, then sample at `bins`. ki ∈ [0, NF_ref).
    span = max(1, target_NF - src_lo)
    ki = np.clip(((bins - src_lo) / span * (NF_ref - 1)).round().astype(np.int64),
                 0, NF_ref - 1)
    grabbed = ref_amp[yi[:, None], ki[None, :]]   # (target_h, len(bins), 3)
    out[:, fill_lo:fill_hi, :] = grabbed
    return out


def _stretch_col_band(ref_amp: np.ndarray, *, target_h_NF: int, target_w: int,
                      fill_lo: int, fill_hi: int) -> np.ndarray:
    """Same idea, but for SHAPE cells — gray (W_ref, NF_ref) broadcast
    to 3 channels. Output shape (target_h_NF, target_w, 3)."""
    W_ref, NF_ref = ref_amp.shape
    out = np.zeros((target_h_NF, target_w, 3), np.float32)
    if NF_ref <= 0:
        return out
    xi = np.linspace(0, W_ref - 1, target_w).round().astype(np.int64)
    bins = np.arange(fill_lo, fill_hi, dtype=np.float32)
    if bins.size == 0:
        return out
    span = max(1, target_h_NF)
    ki = np.clip((bins / span * (NF_ref - 1)).round().astype(np.int64),
                 0, NF_ref - 1)
    grabbed = ref_amp[xi[None, :], ki[:, None]]    # (len(bins), target_w)
    grabbed = grabbed[..., None].repeat(3, axis=2)
    out[fill_lo:fill_hi, :, :] = grabbed
    return out


# ─────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────
def _apply_channel_gain(img_f: np.ndarray, gain: np.ndarray) -> np.ndarray:
    out = img_f * gain.reshape(1, 1, 3)
    return _to_uint8(out)


def _to_uint8(img_f: np.ndarray) -> np.ndarray:
    return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)


def _find_sss_model() -> str:
    """Guess a sensible default — same lookup the unified server uses
    when SSS_MODEL_PATH isn't set: prefer the most recent .sss in
    build/models/."""
    candidates = []
    for root in ("build/models", "."):
        if os.path.isdir(root):
            for fn in os.listdir(root):
                if fn.endswith(".sss"):
                    candidates.append(os.path.join(root, fn))
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def _read_image(path: str) -> np.ndarray:
    """Same loader sss_ingest uses — PNG/PPM via sss_image_io,
    everything else via ffmpeg."""
    from tools.sss_ingest import _read_image_any
    return _read_image_any(path)


def _cli():
    ap = argparse.ArgumentParser(description="SSS image restore / enhance")
    ap.add_argument("input")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--mode", default="enhance",
                    choices=("restore", "upscale", "sharpen",
                             "denoise", "color", "enhance"))
    ap.add_argument("--model", default=None,
                    help="path to .sss model (defaults to SSS_MODEL_PATH "
                         "or the newest build/models/*.sss)")
    ap.add_argument("--prompt", default="",
                    help="reference prompt (e.g. 'cat face')")
    ap.add_argument("--strength", type=float, default=0.5,
                    help="sharpen / denoise strength in [0, 1]")
    ap.add_argument("--scale", type=int, default=2,
                    help="upscale factor")
    ap.add_argument("--mask", default=None,
                    help="restore mask PNG (white = damaged); "
                         "omitted → auto-detect")
    ap.add_argument("--target", default="",
                    help="color_correct target prompt or keyword "
                         "(warm / cool / vivid / bright / dim)")
    args = ap.parse_args()

    img = _read_image(args.input)
    rest = SSSRestore(model_path=args.model, prompt=args.prompt)

    if args.mode == "restore":
        mask = None
        if args.mask:
            m_img = _read_image(args.mask)
            mask = (m_img.mean(axis=2) > 127).astype(np.float32)
        out = rest.restore(img, mask)
    elif args.mode == "upscale":
        out = rest.upscale(img, scale=args.scale)
    elif args.mode == "sharpen":
        out = rest.sharpen(img, strength=args.strength)
    elif args.mode == "denoise":
        out = rest.denoise(img, strength=args.strength)
    elif args.mode == "color":
        out = rest.color_correct(img, target_prompt=args.target)
    else:                                              # enhance
        out = rest.enhance(img, prompt=args.prompt)

    sss_image_io.write_png(args.out, out)
    print(f"wrote {args.out}  shape={out.shape}  in={img.shape}  mode={args.mode}")


if __name__ == "__main__":
    _cli()
