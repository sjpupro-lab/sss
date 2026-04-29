# SSS — Spatial Pattern AI + CE-Cell Image Engine

![Main Hero](main_hero.png)

> A two-engine codebase. **Text** is encoded as brightness patterns on a
> 256×256 grid; **images** decompose into 64-byte `CEUnit` cells with an
> RGBA tick clock and a residual codebook. Both share one storage layer
> and one save file.
>
> Inference is integer-only. Float math (DCT / SVD) appears only at
> training time.

```
   text engine                          image engine
   ───────────────                       ─────────────
   "The cat eats rice."                  256×256 PPM
        │                                    │
   3-layer encode                       16×16 stamp + SLIG decompose
        │                                    │
   256×256 RGBA grid                    CEUnit pyramid (3 scale × 3 chan)
        │                                    │
   keyframe / Δ                         block + SLIG entries (CEStorage)
        │                                    │
   ai_save *.spai                       ce_storage_save *.ces
```

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Text engine — Spatial Pattern](#text-engine--spatial-pattern)
- [Image engine — CE Cell](#image-engine--ce-cell)
  - [16×16 block stamp](#1-16x16-atomic-block-stamp)
  - [SLIG signal cells](#2-slig-signal-cells)
  - [Hybrid VAE — encode + decode](#3-hybrid-vae)
  - [Masked train](#4-masked-train)
  - [RGBA tick clock + residual codebook](#5-rgba-tick-clock--residual-codebook)
  - [Tick-sorted dynamic decode](#6-tick-sorted-dynamic-decode)
- [End-to-end demo](#end-to-end-demo)
- [Verified results](#verified-results)
- [Build & run](#build--run)
- [Save / load formats](#save--load-formats)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why this exists

A traditional LLM bakes language into a fixed weight matrix and a
diffusion model bakes images into the same pattern in latent space. Both
are opaque, both retrain instead of incrementally accept new data.

This engine is the inverse:

- **Unlimited parameters** — every new clause / image becomes a frame
  or set of cells, never a retrained weight tensor.
- **Unlimited context** — bounded only by disk.
- **Inspectable** — text grids are heatmaps you can open; image cells
  carry tagged `(scale, channel)` with a documented 64-byte layout.
- **Incremental** — a new clause is one delta or one new keyframe, a
  new image is 1024 block stamps + 9 SLIG sets + a residual codebook
  pass. Never a full retrain.
- **Embedded-friendly** — runs in Termux / on Windows MSYS.
- **Integer at inference** — float only at SVD / DCT during learning.

The cost: pattern encoding is a bet that byte-level spatial statistics
plus a tick clock carry enough signal to substitute for an attention
matrix. Current benchmarks say "yes, useful as retrieval + recall +
deterministic image generation", not "yes, this replaces a transformer."

---

## Text engine — Spatial Pattern

The historical core. A clause becomes a 256×256 RGBA grid through a
three-layer encoder, then enters a video-codec-style keyframe / delta
store.

### Three-layer summation

| Layer | Target | Weight | Captures |
|---|---|---|---|
| **Base** | every byte | +1 | raw byte positions |
| **Morpheme** | noun/verb/adj byte ranges | +3 | morpheme-level structure |
| **Word** | space-split word bytes | +5 | word-level emphasis |

Overlap tiers in `A`: `1 / 4 / 6 / 9`.
On `"귀여운 고양이가 밥을 먹는다."`: active 40 px, max 9, total 297
(`= 40 + 185 + 72`, conservation holds).

### RGBA channels

| Channel | Type | Role | How it's set |
|---|---|---|---|
| **A** | u16 | brightness / importance | 3-layer sum |
| **R** | u8 | semantic | morpheme POS seed + diagonal diffusion |
| **G** | u8 | functional | morpheme POS seed + vertical diffusion |
| **B** | u8 | extended | morpheme POS seed + horizontal diffusion + EMA |

`update_rgb_directional` propagates each channel along its own axis;
the engine maintains a per-cell EMA across the corpus.

### Keyframe / delta storage

```
  first clause                              → new keyframe
  best cosine-A ≥ 0.3 vs any KF             → delta (sparse — only moved cells)
  best similarity < 0.3                     → new keyframe
```

`apply_delta(base, entries, count, out)` reconstructs the target grid
bit-for-bit. `test_io` verifies `|sim_before − sim_after| < 1e-3`
across 700 clauses.

### Matching cascade

`spatial_match()` is the unified core:

```
  Step 1 (coarse)  bucket index for KF ≥ 100, full overlap_score otherwise.
                   topk_select → top-K (K=8).
  Step 2 (precise) per-mode scorer:
                     PREDICT  → cosine_rgb_weighted
                     SEARCH   → cosine_a_only
                     QA       → rg_score    (0..1)
                     GENERATE → bg_score    (0..1)
```

All scorers normalise to `[0, 1]`, so threshold comparisons are well-
defined everywhere.

### Canvas pool (subtitle routing)

A 2048×1024 canvas tiles 32 × 256² clause slots. `pool_match` jumps
straight to slots of the query's `DataType` (prose / dialog / code /
short) before cascading.

---

## Image engine — CE Cell

The image side reuses the same `CEStorage` (one file = `*.ces`). Every
storage entry is a 140-byte row tagged with one of the modalities:

```c
typedef enum {
    CE_TYPE_TEXT     = 0,
    CE_TYPE_IMAGE    = 1,   /* 16×16 block stamps */
    CE_TYPE_AUDIO    = 2,
    CE_TYPE_SLIG     = 3,   /* SLIG cells per (scale, channel) */
    CE_TYPE_RESIDUAL = 4    /* codebook descriptors only */
} CEType;
```

The image flow is `block stamp → SLIG decompose → codebook → tick-
sorted decode`.

### 1. 16×16 atomic block stamp

`ce_storage_ingest_rgba_16` cuts the 256×256 image into 16 × 16 blocks
(=256 blocks). Each block goes through `ce_feed_image_16` and produces
**4 quadrant CEUnits** (TL/TR/BL/BR). 256 blocks × 4 quadrants =
**1024 CE_TYPE_IMAGE entries** per image, all sharing one
`canvas_id`. `slot = block-row`, `block_idx = (block-col << 2) | quadrant`.

### 2. SLIG signal cells

`slig_decompose_channel` runs per (scale_level, channel) on the image's
YCbCr planes:

- `SLIG_LEVEL_COARSE` (32×32), `MID` (128), `FINE` (256)
- `SLIG_CH_Y / CB / CR`

3 × 3 = **9 sets per image**. Each set holds up to 32 cells; each cell
is a 64-byte `CEUnit` with 16-coefficient DCT signals on `inc.G/B`,
metadata on `inc.R`, energy / audio bins on `inc.A`, and event /
audio-link bytes on the `dec` half.

Persistence (Phase 6+): `ce_storage_persist_slig_set` writes each cell
as a `CE_TYPE_SLIG` row with `slot = scale_level * 3 + channel` and
`block_idx = cell index`. `ce_storage_load_slig_sets` reassembles the
9-grid from any matching `canvas_id`.

### 3. Hybrid VAE

`ce_hybrid_vae.{c,h}` wraps a single API around the two paths:

```c
HybridEncodeResult res;
hybrid_vae_encode(&res, &storage, &codebook, rgba, w, h, canvas_id);
//  → 1024 block stamps (CE_TYPE_IMAGE)
//  → 9 SLIG sets       (CE_TYPE_SLIG, also linked via audio_bins)
//  → 9 codebook indices for the (scale, channel) buckets

uint8_t out_rgb[256*256*3];
hybrid_vae_decode(out_rgb, &storage, canvas_id, res.sets, &cfg);
//  base = block-stamp colour restoration
//  detail = SLIG residual chain (coarse → mid → fine, per channel)
//  blend(base, detail) + wave refine + CFG guidance + YCbCr→RGB
```

The encode step links the two paths: each SLIG cell's `dec.A.audio_bins`
records the originating `canvas_id`, so a later search across cells
can recover which block stamp set produced them.

### 4. Masked train

`ce_masked_train.{c,h}` teaches the latent to fill in missing cells.
Per epoch:

```
  1. slig_decompose_v2 → SligDecomposed
       cells[0..structure_end)         basis
       cells[structure_end..edge_end)  residual edge
       cells[edge_end..texture_end)    residual texture
       cells[texture_end..color_end)   correction (color)
       cells[color_end..num_cells)     correction (event)

  2. mask the correction (or further) cells
  3. ce_denoise_loop predicts the masked positions
  4. ce_compute_loss(predicted, original)
  5. ce_update_params adjusts CEGenConfig
  6. converged cells go back into storage
```

Mask schedule is progressive:

```
  epoch 0..33%   correction-only mask
  epoch 33..67%  residual + correction mask
  epoch 67..100% random 50% mask
```

### 5. RGBA tick clock + residual codebook

The image cells carry an implicit ordering — every `CEStorageEntry`
maps to a `TickRGBA` derived from `(slot, block_idx)` and
`audio_amps`:

| Channel | What it means | Read from |
|---|---|---|
| **R** | local cell index inside its set / block | `block_idx` |
| **G** | scale level (0=COARSE, 1=MID, 2=FINE) | `slot / 3` |
| **B** | render layer / channel (Y / Cb / Cr) | `slot % 3` |
| **A** | amplitude / residual strength | `audio_amps` max + `sigma>>8` |

Carry chain mirrors `slig_tick_math::tick_add(plus, 1)`:

```c
void ce_tick_step(TickRGBA *t) {
    if (++t->r == 0) {        /* R wraps 255→0 */
        if (++t->g == 0) {    /* carry into G  */
            if (++t->b == 0) {/* carry into B  */
                ++t->a;       /* and into A    */
            }
        }
    }
}
```

`ce_tick_compare` ranks `B > G > R > A`, so `ce_tick_sorted_indices`
walks the storage in render order: layer-by-layer, coarse-to-fine,
cells in index order, weakest-first within ties.

The **residual codebook** (`ce_residual_codebook.{c,h}`) is a
256-entry dictionary keyed by L1 distance, bucketed by scale_level so
chroma never collides with luma.

```c
typedef struct {
    CEUnit   unit;            /* the actual correction CEUnit */
    uint8_t  scale_level;     /* SligScaleLevel (or 0xFF wildcard) */
    uint8_t  direction;       /* SligDir */
    uint8_t  strength;        /* 0..255 */
    TickRGBA tick;
    uint32_t used_count;
} CEResidualCode;
```

Storage descriptor (`CE_TYPE_RESIDUAL`) reuses the keyframe bytes:

```
  bytes[0]    = codebook_idx
  bytes[1]    = strength
  bytes[2..5] = TickRGBA  (R, G, B, A)
  bytes[6..7] = x position (u16 LE)
  bytes[8..9] = y position (u16 LE)
  bytes[10..] = reserved (zero)
```

**No raw patch payload in storage.** When `train_demo --residual-codebook`
is on, `masked_train_image` runs each correction cell through
`ce_residual_codebook_add_or_lookup` (threshold-based). On the
`data/demo` 10-image set, **96 correction cells reduce to 6–9 codebook
patterns** — about 10× pattern compression on the encode side.

### 6. Tick-sorted dynamic decode

`hybrid_vae_decode` Step 4b walks `CE_TYPE_RESIDUAL` entries in tick
order (B > G > R > A), looks up each patch, and stamps an 8×8
gaussian-weighted patch onto `blend_y` centred at `(x & 0xFF, y & 0xFF)`.
The amplitude cap is 64 per single patch; `tick.g` parity flips the
sign so different cells can both add and subtract.

```
  cell drawn first  (low B, low G)  → coarse structure
  cell drawn middle (mid G)         → texture detail
  cell drawn last   (high G)        → high-frequency residual
  amplitude (A) tie-breaks          → weak first, strong stamps over it
```

This replaces the older fixed-phase decode with a per-cell ordering
read directly from storage. `cfg.residual_book == NULL` falls back to
the previous behaviour, so existing callers see no difference.

---

## End-to-end demo

```bash
make demo_tools                            # builds make_demo_dataset, train_demo,
                                           # gen_image_ce, verify_hybrid

./build/make_demo_dataset data/demo        # 10 PPM images + labels.tsv

# Joint text+image training
./build/train_demo data/demo build/models/demo
#   IMG=10240   block stamps
#   TEXT=38     per-morpheme bridges
#   HYB_blocks=10240  (= IMG)
#   HYB_cells=648     SLIG cells from hybrid_vae_encode

# With masked-train + residual codebook
./build/train_demo data/demo build/models/demo_cb \
    --masked-epochs 5 --residual-codebook
#   masked-train summary: stored_cells=253 residuals=96
#   codebook=6–9 patterns (≈10× compression on correction cells)

# Generate (canvas-routed default path)
./build/gen_image_ce build/models/demo.ces "red apple" \
    build/red_apple.ppm 0 50 200

# Generate (hybrid VAE path)
./build/gen_image_ce build/models/demo_cb.ces "red apple" \
    build/red_apple_hybrid.ppm 0 50 200 --hybrid --guidance 1.5

# Roundtrip PSNR over a folder
./build/verify_hybrid data/demo/colors/*.ppm data/demo/fruits/*.ppm
#   per-file dB output + average across the batch
```

`gen_image_ce` morpheme-tokenises the prompt, votes a winning
`canvas_id` via `ce_search_by_type(CE_TYPE_TEXT, ...)`, then either:

- (default) `ce_generate_image_canvas_routed` — denoise + decode +
  16×16 atomic wave-refine on entries with the matching canvas_id, **or**
- (`--hybrid`) `hybrid_vae_decode` — block-stamp colour base + SLIG
  detail (loaded from CEStorage via `ce_storage_load_slig_sets`) +
  optional codebook patches + wave refine + guidance.

---

## Verified results

`make test` passes 19/19 upper engine binaries on this branch. The
`ce_core` suite passes 20/21 (`test_slig_signal` is a documented
Windows-MSYS-only baseline issue with file-write permissions, present
before any of this work and unchanged by it).

### Test surface

```
  Upper engine (make test):
    test_grid             6/6
    test_morpheme         5/5
    test_layers           3/3
    test_match            5/5
    test_keyframe         6/6
    test_context          5/5
    test_integration      4/4
    test_io               7/7
    test_cascade          6/6
    test_canvas           8/8
    test_adaptive         8/8
    test_subtitle         8/8
    test_recluster        7/7
    test_refine          13/13
    test_image_roundtrip  3/3
    test_image_gen       68/68   (incl. hybrid VAE + masked-train E2E)
    test_tick_math       28/28
    test_material        47/47

  ce_core (make -C ce_core test):
    test_core / test_storage / test_engine / test_denoise / test_extend /
    test_gen / test_ingest / test_decode_image_block / test_feed_image /
    test_image_wave_refine / test_pipeline / test_stress_10k /
    test_slig_energy / test_storage_ingest16 / test_decode16          PASS

    test_hybrid_vae        16/16   (encode + decode + roundtrip PSNR)
    test_masked_train      14/14   (config / single image / batch / progressive)
    test_slig_persist      19/19   (persist + load + tick-sorted iter)
    test_residual_codebook 27/27   (lookup + add + (x, y) roundtrip)
    test_residual_decode    8/8   (codebook on/off changes blend_y)
    test_slig_signal       30/33   (3 fails: Windows-MSYS file-write only)
```

### Hybrid VAE roundtrip on synthetic images

```
  test_hybrid_vae:
    solid 2×2 image       → block_entries=1024, total_cells=14, PSNR  6.3 dB
    synth gradient image  → PSNR 13.2 dB
    base-only restoration → non-blank
    detail-only render    → non-blank
    wave refine baseline  → PSNR > 5 dB
```

### Demo pipeline (10 images, deterministic seed 0)

| Prompt | Routed canvas | Mean RGB | Verdict |
|---|---|---:|---|
| `"red apple"`       | apple     | (196, 38, 38)  | red ✓ |
| `"yellow banana"`   | banana    | (213, 195, 46) | yellow ✓ |
| `"purple grape"`    | grape     | (118, 42, 146) | purple ✓ |
| `"green lime"`      | lime      | (38, 195, 38)  | green ✓ |
| `"blue blueberry"`  | blueberry | (38, 39, 196)  | blue ✓ |
| `"orange fruit"`    | orange    | (220, 140, 41) | orange ✓ |

6/6 prompts route to the correct dataset row and decode to the
expected colour, including the `"orange fruit"` case where `"fruit"`
is unknown to the dataset (the morpheme vote still resolves to
`"orange"` because `1/(1+distance)` falls off sharply for unrelated
tokens).

### Masked train on the demo set

```
  --masked-epochs 5 --residual-codebook
  ────────────────────────────────────
  per image: epochs_run=5, final_loss≈80, cells_stored=25, residuals≈10
  batch:     stored_cells=253, residuals=96, codebook=6–9 patterns
```

Loss above the convergence target (`8.0`) is expected — 5 epochs ×
10 images is too small to converge the masked predictor. Phase 13's
test_residual_decode confirms the codebook layer is wired in
correctly (`diff_pixels=984` between codebook on/off, max per-pixel
deviation 77).

### Stress test

`test_stress_10k`:

```
  N = 12,000 entries
    ingest:   5 ms       (2.66M blocks/s)
    search:   0.2 ms     (k=8)
    save:     <1 ms
    load:     <1 ms
    generate: 60 ms      (4-step + 30 wave refines)
  N = 100,000 entries
    ingest:   60 ms
    search:   1.5 ms
    save:    100 ms
    load:    130 ms
    generate: 90 ms
```

All time budgets hold. 1M entries (≈1000 images × 1024 blocks) keep
generate under 0.1 s.

---

## Build & run

```bash
make all                  # all object files + linked test/demo binaries
make test                 # 268 upper-engine cases
make -C ce_core all       # ce_core engine + tests
make -C ce_core test      # ce_core regression
make demo_tools           # train_demo / gen_image_ce / verify_hybrid
```

**Requires:** GCC ≥ 9 (C11), Make, POSIX-ish shell. On Windows MSYS2's
`/usr/bin/gcc` works; the MinGW fork hits POSIX-mkdir issues in
`test_ingest`.

### Streaming text trainer

```bash
make stream
./build/stream_train --input data/kaggle_train.txt \
                     --max 25000 \
                     --save build/models/wiki25k.spai \
                     --checkpoint 5000 \
                     --verify
```

Memory stays flat regardless of source-file size. `--checkpoint N`
emits intermediate `.spai` files; `--verify` re-scans the tail.

### Image training

```bash
# Convert PNG / JPEG to 256×256 PPM
python3 tools/png_to_ppm256.py data/samples/IMG_0304.png \
    build/training/IMG_0304.ppm
make image_tools
gcc -Wall -O2 -Iinclude tools/jpeg_to_ppm256.c \
    -o build/jpeg_to_ppm256 -ljpeg
./build/jpeg_to_ppm256 data/samples/IMG_0305.jpeg \
    build/training/IMG_0305.ppm

# CE-cell trainer (replaces the legacy SpatialAI-only one)
make train_images_ce
./build/train_images_ce build/training/img_model.ces \
    build/training/IMG_0304.ppm \
    build/training/IMG_0305.ppm
```

### `verify_hybrid`

PPM → `hybrid_vae_roundtrip` → PSNR per file + batch average.
Useful for tracking whether the encode/decode regresses across
changes.

```bash
make demo_tools                               # also builds verify_hybrid
./build/verify_hybrid data/demo/colors/*.ppm data/demo/fruits/*.ppm
```

---

## Save / load formats

### `.spai` — text engine state

Magic `SPAI`, current version **6**. v3/4/5 load transparently.

```
  Header 32B    magic + version + kf_count + df_count + reserved
  Records*      tagged stream:
    0x01 Keyframe    id, label, A/R/G/B
    0x02 Delta       id, parent, sparse entries (v6: + cell_deltas[])
    0x03 Weights     ChannelWeight (4× float)
    0x04 Canvas      slot_count, type, parent, A/R/G/B
    0x05 Subtitle    type/topic_hash/canvas_id/slot_id table
    0x06 EMA         R/G/B/count per cell
    0x07 SeqMeta     per-canvas sequence id + timestamp
    0x4A Codebook    SligCodebook patterns (v2.3)
    0x4B ImageIdx    per-keyframe (3×3) codebook indices
```

### `.ces` — CE-Cell storage

Format v2 (current). Each entry is **140 bytes**:

```
   4 B  canvas_id          uint32
   2 B  slot               uint16
   2 B  block_idx          uint16
   1 B  type               CE_TYPE_* (0..4 used today)
   3 B  reserved (zero)
  64 B  keyframe           CEUnit
  64 B  delta              CEUnit
```

Adding `CE_TYPE_SLIG=3` and `CE_TYPE_RESIDUAL=4` is forward-compatible
— the type field is a single byte that already supported up to 254
modalities. Older readers that don't recognise the new values still
load the rest.

For `CE_TYPE_RESIDUAL` the 64-byte `keyframe` slot is reused as a
descriptor (no raw patch payload). See
[§5 RGBA tick clock + residual codebook](#5-rgba-tick-clock--residual-codebook)
for the byte layout.

### Save / load API

```c
SpaiStatus ai_save(const SpatialAI* ai, const char* path);
SpatialAI* ai_load(const char* path, SpaiStatus* out_status);
SpaiStatus ai_save_incremental(const SpatialAI* ai, const char* path);
SpaiStatus ai_peek_header(const char* path, ...);

int ce_storage_save(const CEStorage *s, const char *path);
int ce_storage_load(CEStorage *out, const char *path);
```

`test_io` validates `|sim_before − sim_after| < 1e-3` over a 700-clause
roundtrip and proves that `ai_save_incremental` refuses to shrink an
on-disk model.

---

## Project layout

```
├── include/                    public headers (text engine)
│   ├── spatial_grid.h          256×256 RGBA grid
│   ├── spatial_layers.h        3-layer summation
│   ├── spatial_morpheme.h      Korean longest-match analyser
│   ├── spatial_keyframe.h      Keyframe / delta / SpatialAI
│   ├── spatial_match.h         spatial_match() unified core
│   ├── spatial_context.h       LRU frame cache
│   ├── spatial_canvas.h        2048×1024 canvas + 32 slots
│   ├── spatial_subtitle.h      SubtitleTrack + canvas pool
│   ├── spatial_generate.h      next-clause refine
│   ├── spatial_image.h         image_to_grid / grid_to_image +
│   │                           ai_generate_image_v2_guided / animation
│   └── spatial_io.h            .spai binary format
├── src/                        text engine implementations
│   ├── spatial_grid.c / layers.c / morpheme.c / keyframe.c /
│   │   match.c / context.c / canvas.c / subtitle.c / recluster.c /
│   │   generate.c / io.c / image.c
│   └── spatial_image_gen.c     SLIG v2.3 generator
├── ce_core/                    CE Cell engine
│   ├── ce_core.{c,h}           64-byte CEUnit primitives
│   ├── ce_type.h               CEType (TEXT/IMAGE/AUDIO/SLIG/RESIDUAL)
│   ├── ce_storage.{c,h}        CEStorage + 16×16 ingest + SLIG persist +
│   │                           tick-sorted iter
│   ├── ce_storage_io.{c,h}     .ces v2 reader/writer
│   ├── ce_search.{c,h}         top-k retrieval, by-type filter
│   ├── ce_engine.{c,h}         UNet-equivalent ops (Down/Conv/Attn/Up)
│   ├── ce_denoise.{c,h}        50-step denoise loop
│   ├── ce_decode.{c,h}         latent → image / text
│   ├── ce_extend.{c,h}         sampler + inpaint (CEInpaintMask)
│   ├── ce_gen.{c,h}            top-level generation API
│   ├── ce_feed_image.{c,h}     8×8 / 16×16 block encoders
│   ├── ce_image_wave_refine.{c,h}  0.01→3→0.01 wave refine
│   ├── ce_hybrid_vae.{c,h}     hybrid encode + decode + roundtrip
│   ├── ce_masked_train.{c,h}   PROGRESSIVE mask schedule
│   ├── ce_residual_codebook.{c,h}  256-entry correction dictionary
│   ├── ce_tick.h               TickRGBA carry chain (header-only)
│   ├── slig_signal.{c,h}       SligSignal + decompose + canvas
│   ├── slig_codebook.{c,h}     (scale, channel) pattern dictionary
│   ├── slig_pipeline.{c,h}     v3 5-stage standalone pipeline
│   ├── slig_tick_math.{c,h}    32-bit packed tick + sin/cos/gauss tables
│   └── slig_material_harmonic.{c,h}  Mat-1/2/4 auto-extract
├── tests/                      upper-engine tests (19 binaries)
├── ce_core/tests/              ce_core tests (21 binaries)
├── tools/
│   ├── train_demo.c            joint text+image trainer
│   │                            (--masked-epochs N, --residual-codebook)
│   ├── gen_image_ce.c          prompt → 256×256 PPM
│   │                            (default canvas-routed, --hybrid --guidance N)
│   ├── verify_hybrid.c         hybrid_vae_roundtrip PSNR over a folder
│   ├── train_images_ce.c       single-image CE trainer
│   ├── stream_train.c          line-by-line text trainer
│   ├── chat.c                  interactive REPL
│   ├── make_demo_dataset.c     generates data/demo
│   ├── img2grid.c / grid2img.c PPM ↔ SpatialGrid round-trip
│   └── png_to_ppm256.py / jpeg_to_ppm256.c image converters
├── data/
│   ├── samples/                IMG_0304.png, IMG_0305.jpeg, IMG_0306.jpeg
│   └── demo/                   procedurally generated PPMs + labels.tsv
├── PIPELINE.md                 phase-by-phase log of this branch
├── SPEC.md / SPEC-ENGINE.md    historical spec
└── README.md / README_KO.md
```

---

## Roadmap

Implemented (this branch — see `PIPELINE.md` for the 13-phase log):

- [x] Dead-code purge (`ce_memo`, `ce_hint`, `ce_audio`, `ce_upscale`,
      `ce_generate_image` legacy, `ai_generate_image` v1).
- [x] Hybrid VAE (encode + decode + roundtrip), test suite 16/16.
- [x] Masked train (progressive mask schedule), test suite 14/14.
- [x] `train_demo --masked-epochs N`, `gen_image_ce --hybrid --guidance N`,
      `verify_hybrid`.
- [x] `CE_TYPE_SLIG` / `CE_TYPE_RESIDUAL` modalities; `.ces` v2
      forward-compatible.
- [x] SLIG cellset persistence + tick-sorted storage iteration
      (`ce_tick.h`).
- [x] Residual codebook (256 entries, scale-bucketed, used_count
      tracked).
- [x] `train_demo --residual-codebook` reduces correction cells
      ~10× via the codebook.
- [x] hybrid VAE decode applies `CE_TYPE_RESIDUAL` patches in
      tick order, with positional `(x, y)` 8×8 gaussian stamping.

Pending (deferred per user direction until end-to-end is validated on
a 1000+-image corpus):

- Larger dataset experiment (epochs 50–200 × ≥1000 images), residual
  threshold + amplitude cap retuning.
- `CE_TYPE_RESIDUAL` descriptors gain a per-frame `direction` /
  `scale` field once corpus statistics support a non-wildcard scale tag.
- `hybrid_vae_decode` row-stamp → patch-stamp policy review (currently
  fixed at 8×8; sensible to make this `cfg.patch_radius`).

---

## License

See [LICENSE](LICENSE).
