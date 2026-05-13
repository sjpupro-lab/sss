# sss — resonance-based visual-signal synthesiser

> **시간이 가둬진 신호와 시간이 흐르는 신호를 동일한 motif 사전으로
> 다루는 공명 기반 시각 신호 합성 엔진이며, motif는 자신과 유사한
> motif의 조건 응답을 누적하여 학습한다.**
>
> An image is the `frames = 1` special case of a video. Both run
> through the same motif dictionary, the same synthesiser, and the
> same entry point. The pipeline therefore needs exactly one.

`sss` stores trained visual content as a **feature spectral
dictionary** (`.sfb`) — never pixels, never raw frames, never phase.
Motifs hold amplitude envelopes (row / column / colour) and a 16×16
spatial heatmap. Conditions hold time-axis amplitude envelopes.
Interaction responses hold the per-(motif, condition) delta that
falls out of paired observation. The generator synthesises a new
signal that follows the envelopes — no row or pixel of a training
image is ever directly invoked, and varying the seed varies the
output deterministically.

---

## Quick start

```bash
# 1) Build the C side (statics-only, no third-party deps).
make sss_gen pybridge

# 2) Train a small spectrogram model (Phase 1 path).
python3 scripts/sss_train.py \
    --labels data/sanrio/labels.tsv \
    --root   data/sanrio \
    --out    build/sanrio.sss \
    --size   64

# 3) Generate a single image.
./build/sss_gen build/sanrio.sss "white cat kitty" out.ppm 1 1.5 120

# 4) Or generate an 8-frame video (Phase 4 seed-variation baseline).
./build/sss_gen build/sanrio.sss "white cat kitty" out.ppm \
    --frames 8 --out-dir /tmp/anim

# 5) Same CLI, condition-driven motion (Phase 4 .sfb path).
python3 scripts/sss_gen.py build/feature_bank.sfb "flag" out.ppm \
    --frames 16 --out-dir /tmp/anim --condition wind 0.5
```

The HTTP surface (POST `/api/sss_gen` on `ui/unified_server.py`)
accepts the same parameters in JSON. See
[`docs/migration_phase5.md`](docs/migration_phase5.md) for the
full request body schema and the old → new endpoint mapping.

---

## Storage in one diagram

```
  dataset/primitive/<label>/*.{png,ppm}     dataset/conditions/<label>/seq/*.ppm
  dataset/labeled/*.{png,ppm}                dataset/interactions/<motif>_<cond>/{static,active}/
        │                                          │
        │  sss_train_primitive.py                  │  sss_train_condition.py
        │  sss_train_motif_memory.py               │  sss_train_interaction.py
        ▼                                          ▼
  primitive_motifs.npz + motif_memory.npz    conditions.npz + interactions.npz
                       \                    /
                        \  sss_build_feature_bank.py  --conditions  --interactions
                         ▼
                  build/feature_bank.sfb               ←  v2 (Phase 4)
                         │
                         │  scripts/sss_gen.py   /  build/sss_gen
                         │
                         ▼
                  PPM (frames = 1)  or  PPM sequence (frames > 1)
```

The `.sfb` file is a flat, self-describing container of five record
arrays (motifs, relations, identities, conditions, responses).
**No phase, no per-row FFT bytes, no raw pixel / frame data** is
ever written to disk — the schema simply has no field that could
carry them. See `ce_core/sss_feature_bank.h` for the byte map and
`tools/test_sfb_v2.py` for the v1 ↔ v2 round-trip guarantees.

---

## What lives where

### Engine (C)

| Path | Role |
|---|---|
| `ce_core/sss_rowvae.{c,h}` | Sculpt + Radio image engine (Phase 1). amp-only radio tuning, per-seed random phase init. |
| `ce_core/sss_feature_bank.{c,h}` | `.sfb` v2 format: save / load / validate. |
| `ce_core/sss_pybridge.{c,h}` | ctypes-friendly C ABI for the Python side. Exposes generate, save_v2, load_v2, probe_v2, scene I/O. |
| `ce_core/sss_io.c` | Legacy `.sss` v9 loader (kept for the Phase 1–3 fast path). |
| `tools/sss_gen.c` | Single CLI. `--frames`, `--out-dir`, `--out-w/h`, `--atmos`, Phase-4 flag stubs. |

### Trainers + synthesiser (Python)

| Path | Role |
|---|---|
| `scripts/sss_train.py` | Phase 1 spectrogram trainer (.sss output). |
| `scripts/sss_train_primitive.py` | Phase 3 primitive-motif trainer (1 motif per dataset folder). |
| `scripts/sss_train_motif_memory.py` | Phase 3 motif-memory builder (relations + identities + position heatmap). |
| `scripts/sss_train_condition.py` | Phase 4 condition-signal trainer (continuous / impulse / oscillatory). |
| `scripts/sss_train_interaction.py` | Phase 4 interaction-response trainer (similar-motif coherence-weighted accumulation). |
| `scripts/sss_build_feature_bank.py` | Combines the four .npz intermediates into a v2 `.sfb`. |
| `scripts/sss_gen.py` | Canonical generation entry. Routes `.sss` → C binary, `.sfb` / `--condition` / `--atmos-from` → Python synth. |
| `tools/sss_synthesizer.py` | `synth_frame()` + `apply_warp()` + `synthesise_from_envelope()`. The Phase 4 resonance synthesiser. |
| `tools/sss_feature_bank.py` | Python mirror of the C `.sfb` I/O (byte-identical save / load). |

### UI

| Path | Role |
|---|---|
| `ui/unified_server.py` | Production HTTP server. Exposes `/api/sss_gen` (canonical) and aliases `/api/generate`, `/api/viz-generate`, `/api/atmos`. |
| `ui/server.py` | Compatibility server with the same endpoints; forwards to `scripts/sss_gen.py`. |
| `ui/index.html` | Forge UI. Frames slider replaces the old engine-toggle panel. |
| `ui/start.sh` | One-command launcher. |

### Tests (run with `python3 tools/test_*.py`)

| Test | Phase | What it locks |
|---|---|---|
| `test_sss_ingest.py` | 1 | Label-required image ingest; amp-determinism on re-ingest; no phase on disk. |
| `test_sss_memory.py` | 1 | CEStorage round-trip via the ctypes bridge. |
| `test_sss_unified.py` | 1 | `SSSPipeline` end-to-end smoke. |
| `test_sss_pose_radar.py` | 1 | Pose / radar perception sidecar. |
| `test_sss_cluster.py` | 1 | Morpheme-cluster discovery on a trained `.sss`. |
| `test_sss_rowvae.py` | 1 | sanrio diversity benchmark (≥ 0.05 pairwise MSE; < 25 dB PSNR vs. training). |
| `test_phase_ablation_regression.py` | 1 | FFT-level scenarios A/B/C — random phase produces diverse output. |
| `test_feature_bank.py` | 2 | `.sfb` round-trip + C ↔ Python byte parity + 1 000-motif benchmark. |
| `test_train_primitive.py` | 3 | 4-motif synthetic dataset; coherence > 0.85; off-diag max < 0.6. |
| `test_train_motif_memory.py` | 3 | Compound scenes; ABOVE / BELOW relation extraction. |
| `test_build_feature_bank.py` | 3 | Primitive → memory → `.sfb` end-to-end. |
| `test_sfb_v2.py` | 4 | v1 ↔ v2 compat; new motif + condition + response round-trip. |
| `test_train_condition.py` | 4 | 5-signal classifier (gravity → CONTINUOUS via frame-delta early-out). |
| `test_train_interaction.py` | 4 | Similar-motif coherence-weighted accumulation. |
| `test_synthesizer.py` | 4 | The 4 GPT scenarios + `apply_warp` displacement. |
| `test_sss_gen_video.py` | 5 | C single-image, C `--frames 4`, C rejection of Phase-4 flags, Python condition-intensity sweep. |

`make test` runs the 21-test C-side regression suite (`tests/*.c`).

---

## Phase history (one line each)

| Phase | Headline |
|---|---|
| **1** | Amp-only radio tuning. Per-seed random phase init drives diversity; trained-side phase is removed from `.sss`. |
| **2** | `.sfb` (SSS Feature Bank) — flat container of motifs / relations / identities. C ↔ Python byte-identical. |
| **3** | Primitive motif trainer + memory builder + `.sfb` compositor. DC-removed zero-mean L2 envelopes, weighted-cosine matcher. |
| **4** | Condition-Interaction paradigm + `.sfb` v2. Condition signals, interaction responses, warp fields, cross-axis side-band synthesis. Image is `temporal_length = 0`; video is `temporal_length = N`. |
| **5** | One generator, one CLI (`sss_gen` / `scripts/sss_gen.py`), one HTTP endpoint (`/api/sss_gen`). Legacy `sss_animate` and `gen_image_ce` retired. |

Full per-phase write-ups, byte maps, ablations, and benchmark
tables live in [`PIPELINE.md`](PIPELINE.md). The historical README
covering the pre-Phase-5 codebase is in
[`docs/legacy_readme.md`](docs/legacy_readme.md).

---

## Design rules the schema enforces

These are the invariants the file format itself makes unviolable —
short of a `SFB_VERSION` bump.

  - **No phase information** is ever written to disk. Phase 1
    established the rule for the spectrogram generator; every later
    phase extends it.
  - **No per-row FFT** bytes. Phase 3 collapses the legacy
    `(H × NF × 3)` per-row layout into a single 128-bin row_freq
    envelope per motif.
  - **No original pixel coordinates.** `position_heatmap` is a
    coarse 16×16 probability surface, not a pixel-addressable map.
  - **No raw frames.** Conditions store an amplitude envelope of
    the time axis (`temporal_freq[64]`); responses store the
    envelope + warp delta that pairing produced.

The generator's output diversity comes from a per-seed random
phase at noise-init time. **Generation is deterministic per seed**
— a fixed seed reproduces the same output bit-for-bit. Varying
the seed varies the waveform.

---

## HTTP surface

```
POST /api/sss_gen        (canonical)
POST /api/generate       (alias — `frames` defaults to 1)
POST /api/viz-generate   (alias + per-pixel analysis on the response)
POST /api/atmos          (preserved — Atmos JSON decompose; not generation)
```

Request body (all fields optional except `prompt`):

```json
{
  "model":           "build/feature_bank.sfb",
  "prompt":          "kitty waving paw",
  "seed":            42,
  "steps":           120,
  "detail":          1.5,
  "frames":          16,
  "out_w":           256,
  "out_h":           256,
  "atmos":           false,
  "conditions":      [{"label": "wind", "intensity": 0.5}],
  "atmos_reference": "path/to/reference.png"
}
```

Response:

- `frames = 1` → `result.image` is a single PNG data URI.
- `frames > 1` → `result.frames[]` is a list of PNG data URIs;
  `result.image` is the first frame for back-compat clients.

curl one-liner:

```bash
curl -X POST http://localhost:8090/api/sss_gen \
    -H 'Content-Type: application/json' \
    -d '{"model":"build/feature_bank.sfb","prompt":"flag",
         "frames":1,"conditions":[{"label":"wind","intensity":0.5}]}'
```

---

## Building

```bash
make sss_gen     # the C CLI (Phase 1–3 fast path; supports --frames)
make pybridge    # libsss_pybridge.so (ctypes bridge for the Python side)
make all         # full ce_core build
make test        # 21-test C-side regression
```

Phase 5 deleted `sss_animate` and `gen_image_ce`. The `legacy_demo`
target still builds `train_demo`, `make_demo_dataset`, and
`verify_hybrid` for pre-Phase-1 reproducibility.

Python dependencies: `numpy`. The synthesiser, format I/O, and
trainers are pure-stdlib + NumPy; nothing else is required at
runtime.

---

## Building a feature bank from scratch

```bash
# 1) Primitive motifs — one motif per dataset/primitive/<label>/ folder.
python3 scripts/sss_train_primitive.py \
    --data dataset/primitive --out build/primitive_motifs.npz

# 2) Motif memory — relations + identities from labelled scenes.
python3 scripts/sss_train_motif_memory.py \
    --motifs   build/primitive_motifs.npz \
    --labeled  dataset/labeled \
    --out      build/motif_memory.npz

# 3) Conditions — one condition signal per dataset/conditions/<label>/ folder.
python3 scripts/sss_train_condition.py \
    --data dataset/conditions --out build/conditions.npz

# 4) Interactions — paired static/active sequences under
#    dataset/interactions/<motif>_<condition>/.
python3 scripts/sss_train_interaction.py \
    --motifs     build/primitive_motifs.npz \
    --conditions build/conditions.npz \
    --data       dataset/interactions \
    --out        build/interactions.npz

# 5) Compose everything into a v2 .sfb.
python3 scripts/sss_build_feature_bank.py \
    --motifs       build/primitive_motifs.npz \
    --memory       build/motif_memory.npz \
    --conditions   build/conditions.npz \
    --interactions build/interactions.npz \
    --out          build/feature_bank.sfb
```

The compositor is the only step that touches the on-disk format.
Each upstream trainer can be re-run independently; the rebuild is
incremental as long as the .npz intermediates are present.

---

## Reading further

- [`PIPELINE.md`](PIPELINE.md) — per-phase implementation notes,
  byte maps, ablations, and benchmark tables.
- [`docs/migration_phase5.md`](docs/migration_phase5.md) —
  old → new CLI / HTTP mapping for code that used to call
  `sss_animate` or `gen_image_ce`.
- [`docs/legacy_readme.md`](docs/legacy_readme.md) — the README as
  it stood through Phase 4; preserved for the verified benchmarks
  and the CE-Cell / spatial-pattern documentation that this README
  no longer expands inline.
- [`reports/phase5_audit.md`](reports/phase5_audit.md) — the
  Phase 5 pre-analysis (call graph, integration points, migration
  scenarios) that drove the consolidation work.
