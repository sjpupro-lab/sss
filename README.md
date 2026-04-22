# SPATIAL-PATTERN-AI (CANVAS)

![Main Hero](main_hero.png)

> A spatial pattern–based AI engine that encodes text as brightness patterns on a 256×256 grid.
> Language is treated like video frames, not token vectors.

```
 "The cat eats rice."          256×256 Grid  (1 clause = 1 frame)
         │                    ┌────────────────────────┐
    UTF-8 bytes               │  ·                     │
         │                    │    ·  ·                │
   ┌─────▼──────┐             │  · · ·  ·   ·         │
   │ X = byte    │             │     ·    ·            │
   │ Y = position│  ──────►   │  ·    ·                │
   │ A = 3-layer │             │       ·  ·            │
   │     sum     │             │  · ·       ·  ·       │
   └─────────────┘             └────────────────────────┘
                                A channel: byte-frequency heatmap
```

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [How it works](#how-it-works)
  - [1. Three-layer bitmap summation](#1-three-layer-bitmap-summation)
  - [2. RGBA channels](#2-rgba-channels)
  - [3. Keyframe / Delta storage](#3-keyframe--delta-storage)
  - [4. Matching cascade](#4-matching-cascade)
  - [5. Canvas Pool (subtitle routing)](#5-canvas-pool-subtitle-routing)
- [Current verified results](#current-verified-results)
- [Build & run](#build--run)
- [Save / Load](#save--load)
- [Project layout](#project-layout)
- [Engine optimizations](#engine-optimizations)
- [Morpheme analyzer](#morpheme-analyzer)
- [vs. Traditional LLM](#vs-traditional-llm)
- [License](#license)

---

## Why this exists

A traditional LLM bakes language into a fixed-size weight matrix. Adding new data
means re-training, and the internal state is a black box.

This engine does the opposite:

- **Unlimited parameters** — each new clause is a new frame, frames stack without limit
- **Unlimited context** — bounded only by disk
- **Inspectable** — `A` channel is a heatmap you can open and eyeball
- **Incremental** — new text = one delta or one new keyframe, never a full retrain
- **Tiny** — one frame ≈ 320 KB, engine core is a few MB; runs on Termux / embedded

The cost: pattern encoding is a bet that byte-level spatial statistics carry
enough signal. The current benchmarks say "yes, enough to be useful as a
retrieval + recall substrate," not "yes, it replaces an LLM."

---

## How it works

### 1. Three-layer bitmap summation

Every clause is encoded through three independent layers that are then summed into
the `A` channel. The weights are picked so the overlap tiers become visibly separated.

| Layer | Target | Weight | What it captures |
|---|---|---|---|
| **Base** | every byte | **+1** | raw byte positions |
| **Morpheme** | noun/verb/adj… byte ranges | **+3** | morpheme-level structure |
| **Word** | space-split word bytes | **+5** | word-level emphasis |

Overlap tiers in the combined `A`:

```
  base only               A = 1
  base + morpheme         A = 4   (1 + 3)
  base + word             A = 6   (1 + 5)
  base + word + morpheme  A = 9   (1 + 3 + 5)  ← content morpheme inside a word
```

Current numbers on `"귀여운 고양이가 밥을 먹는다."` (from `test_layers`):

```
  active  = 40 pixels
  max A   = 9
  total A = 297

  base total   = 40
  word total   = 185
  morph total  = 72
  sum check    = 40 + 185 + 72 = 297   ✓  (conservation holds)
```

### 2. RGBA channels

| Channel | Type | Role | How it's set |
|---|---|---|---|
| **A** | `uint16` | Brightness / importance | 3-layer sum |
| **R** | `uint8`  | Semantic (content-word / meaning) | morpheme POS seed + **diagonal** diffusion |
| **G** | `uint8`  | Functional (particle / ending / punct) | morpheme POS seed + **vertical** diffusion |
| **B** | `uint8`  | Extended (context / order / emotion slot) | morpheme POS seed + **horizontal** diffusion, EMA convergence |

R/G/B are **not fixed tables**. After encoding, `update_rgb_directional`
propagates each channel along its own axis inside the clause's own grid
(never across slots, never across clauses). On top of that, the engine
maintains a per-cell EMA of R/G/B that converges across the whole
corpus:

```
  R ← diagonal neighbors  (↗ ↘ ↙ ↖)    α = 0.05   morpheme / semantic
  G ← vertical neighbors  (↑ ↓)        β = 0.08   word substitution
  B ← horizontal neighbors (← →)       γ = 0.03   clause order / context
  EMA (all 3 channels)     α = 0.10   accumulated across every store
```

POS-keyed seeds (before directional diffusion):

```
  POS_NOUN     R=40   G=30   B=100
  POS_VERB     R=120  G=40   B=140
  POS_ADJ      R=170  G=35   B=180
  POS_PARTICLE R=8    G=85   B=90
  POS_ENDING   R=12   G=95   B=110
  POS_PUNCT    R=5    G=120  B=60
  POS_UNKNOWN  R=210  G=20   B=200
```

Implementation notes that matter:

- **Read/write buffers are separated** (`oldR/newR`, `oldG/newG`, `oldB/newB`)
  so the scan direction doesn't bias the update.
- When the average neighbor delta rounds to zero but isn't actually zero, the
  cell still moves by ±1. Small signals don't get silently killed.
- `A` is *only* a mask here. The RGB update never edits `A`.
- B carries a per-POS prior so `bg_score`, `channel_sim_B`, and the
  engine-wide EMA actually have signal to work with. A per-clause
  hash overlay on B was tried earlier and removed — it stamped over
  the bitmap pattern instead of refining it.

### 3. Keyframe / Delta storage

Video-codec style. `ai_store_auto` decides per clause:

```
  first clause                               → new keyframe (I-frame)
  best cosine-A similarity ≥ 0.3 vs. any KF  → delta (P-frame) against that parent
  best similarity < 0.3                      → new keyframe
```

Delta records are **sparse** — only the cells that actually moved:

```c
typedef struct {
    uint32_t index;   // y*256 + x
    int16_t  diff_A;
    int8_t   diff_R;
    int8_t   diff_G;
} DeltaEntry;         // 8 bytes
```

`apply_delta(base, entries, count, out)` reconstructs the target grid bit-for-bit.
The `test_io` roundtrip test verifies that `|sim_before − sim_after| < 0.001`
over 700 clauses across save → destroy → load.

### 4. Matching cascade — unified 2-stage core

All three matching APIs (`ai_predict`, `match_engine`, `match_cascade`)
now delegate to a single `spatial_match()` function:

```
  query clause
      │
      ▼
  ┌─────────────────────────┐
  │ encode (3 layers)        │
  │ update_rgb_directional   │
  │ ema_apply                │
  └───────────┬─────────────┘
              │
        Step 1: coarse          if (bucket_idx && KF ≥ 100) → hash-bucket hop
              │                 else → full overlap_score scan
              │                 topk_select → Top-K (K=8)
              ▼
        Step 2: precise         mode-specific scorer:
              │                 MATCH_PREDICT  → cosine_rgb_weighted
              │                 MATCH_SEARCH   → cosine_a_only
              │                 MATCH_QA       → rg_score   (0..1)
              │                 MATCH_GENERATE → bg_score   (0..1)
              ▼
        best id + similarity + topk[]
```

`MatchContext` carries the optional bucket index and reserved slots
for block-skip and frame-cache integration. `MatchResult` returns
`best_id`, `best_score`, and a sorted `topk[]` of the final K
candidates.

All channel-pair scorers return values in **[0, 1]** — they're means
over co-active cells, not raw sums. Threshold comparisons against
cosine-normalized scores are well defined everywhere now.

### 5. Canvas Pool (subtitle routing)

Above the per-clause grid there's a **2048×1024 Canvas** that tiles 32 × 256×256
clause slots. A `SubtitleTrack` records `(DataType, canvas_id, slot_id)` for
each stored clause so `pool_match` can jump straight to slots of the query's
own type (prose / dialog / code / short), then cascade through the four
channel-pair stages as above.

This gives the "H.264 scene change" behavior:
a canvas can be `KEYFRAME` or `DELTA-of-parent-canvas`, and save/load
preserves parent_canvas_id + changed_ratio + classified flag.

---

## Current verified results

Everything below is reproduced by `make test` on this branch
(`claude/refactor-canvas-spatial-ai-FJt1Y`).

### Test suite

```
  test_grid         6/6
  test_morpheme     5/5
  test_layers       3/3
  test_match        5/5
  test_keyframe     6/6
  test_context      5/5
  test_integration  4/4
  test_io           7/7
  test_cascade      6/6
  test_canvas       6/6
  test_adaptive     8/8
  test_subtitle     8/8
  ───────────────────────
  total            69/69   ALL TESTS PASSED
```

### Brightness distribution

```
  "귀여운 고양이가 밥을 먹는다."    →  active 40, max A = 9, total 297
  conservation:                     297 = 40 + 185 + 72    (base+word+morph)
```

### Matching integrity

```
  block-skip vs full cosine               0.000% difference
  KF0↔KF1 cosine (similar clauses)       73.2%
  KF0↔KF2 cosine (different clauses)      0.0%
  self-query cosine                     100.0%
```

### Pipeline smoke test (`test_wiki data/sample_en.txt`, 50 clauses)

```
  clauses placed        50 / 50
  canvases              2
  self-query avg sim    100.0%
  cascade step 1 hits   50
  fallbacks             0
  Recall@1 / @5 / @10   100% / 100% / 100%
  next-clause top-1     22.0%     (beats best-of-5 random = 20%)
  save size             20.9 MB   (.spai on disk)
  load + append         OK        (4 canvases, 100 slots after append)
```

### `stream_train` smoke test (1000 clauses over a 10k-line corpus)

```
  ingest                clauses=1000  KF=49  Δ=951   elapsed 4.64 s
  .spai file size       17.2 MB
  verify tail (500)     avg sim 0.9867  min 0.34  max 1.000
  hits ≥ 0.90 / 0.50    490 / 490
  R range               0..210
  G range               0..120
  B range               0..200         (POS seed active, was 0..0)
```

### Unified match performance

Since `match_cascade`, `match_engine`, and `ai_predict` all
delegate to `spatial_match()`, and since the bucket index skips
the O(N) scan past `BUCKET_THRESHOLD = 100`, `stream_train` on the
same 10k-line synthetic corpus went from **58.85 s → 22.14 s**
(−62%) across the refactor.

### Cascade / canvas

```
  cascade early-return on exact clause     ≥ CASCADE_STEP1_THRESHOLD (0.5)
  ai_force_keyframe 1-1 mapping            kf_count == N, df_count == 0
  pool_match jumps to same-type slots      step=1 on matching DataType
  pool_match fallback to other types       step=4 when query type empty
```

---

## Build & run

```bash
# Build everything
make all

# Full test suite (69 tests across 12 binaries)
make test

# Clean
make clean

# Wikipedia-style pipeline probe (uses data/sample_en.txt or data/sample_ko.txt)
make wiki
./build/test_wiki data/sample_en.txt
./build/test_wiki data/sample_en.txt --save build/model.spai
./build/test_wiki data/sample_en.txt --load build/model.spai
```

**Requires:** GCC (C11), Make, POSIX (`posix_memalign`) or MinGW on Windows.
For the visualizer: Python 3, `numpy`, `Pillow`, `ffmpeg`.

### Streaming trainer (`stream_train`)

For large corpora where the full text can't fit in RAM:

```bash
make stream
./build/stream_train --input data/kaggle_train.txt \
                     --max 25000 \
                     --save build/models/wiki25k.spai \
                     --checkpoint 5000 \
                     --verify
```

`stream_train` reads one clause per line with `fgets` (4 KB buffer)
and calls `ai_store_auto` per clause. Memory stays flat regardless
of source-file size. `--checkpoint N` emits `foo.ckpt_NNNNNN.spai`
every N clauses; `--verify` re-scans the tail after training and
reports avg/min/max similarity plus hit counts at 0.9 / 0.5 / 0.1.

### End-to-end practical test

One script that builds, trains, verifies, then points at the visualizer:

```bash
./tools/run_practical_test.sh              # default: 1000 clauses
./tools/run_practical_test.sh 5000
./tools/run_practical_test.sh 25000 data/kaggle_train.txt
```

The checkpoint cadence is `min(N / 2, 5000)`. The corpus falls back
to `data/sample_en.txt` / `data/sample_ko.txt` if no path is given.

### Training-evolution visualizer

```bash
pip install numpy Pillow
python3 tools/visualize_training.py build/models
```

For every `.spai` checkpoint in the directory it renders a 1280×720
PNG (A-channel heatmap, RGB composite, KF / Δ stats, adaptive
weights, EMA coverage, sample keyframe labels) and stitches the
frames into `training_evolution.mp4` via `ffmpeg` at 3 s per frame.
The RGB composite shows B actually contributing — a visual
confirmation that the POS-keyed seed reached the stored grid.

### Benchmarks (optional)

```bash
make bench        # builds bench_stsb / bench_perplexity / bench_word_predict / bench_qa

./build/bench_word_predict  data/sample_en.txt  1000
./build/bench_qa            data/qa.tsv
./build/bench_perplexity    data/sample_en.txt  500
./build/bench_stsb          data/stsb.tsv
```

`bench_word_predict` also exposes `--save`, `--load`, `--load-only`.
If `--save` is omitted, a run that actually trains auto-saves to
`build/models/bench_word_predict_auto.spai`.

### Quick corpus bootstrap (3k-line Wikipedia abstracts)

Streams just enough of `enwiki-latest-abstract1.xml.gz` to extract
3000 abstracts (≈ 2500 usable clauses after short-line filtering).
Works on Kaggle / Linux. No full dump download:

```python
import os, gzip, re, urllib.request
os.makedirs("data", exist_ok=True)
URL  = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract1.xml.gz"
OUT  = "data/kaggle_train.txt"
pat, n = re.compile(rb"<abstract>([^<]+)</abstract>"), 0
with urllib.request.urlopen(URL) as resp, gzip.GzipFile(fileobj=resp) as gz, \
     open(OUT, "wb") as out:
    buf = b""
    while n < 3000:
        c = gz.read(1 << 20)
        if not c: break
        buf += c
        last = 0
        for m in pat.finditer(buf):
            s = m.group(1).strip()
            if len(s) >= 20:
                out.write(s + b"\n"); n += 1
                if n >= 3000: break
            last = m.end()
        buf = buf[last:]
print(f"{OUT}: {n} lines, {os.path.getsize(OUT)/1e6:.1f} MB")
```

Then:

```bash
./tools/run_practical_test.sh 2500 data/kaggle_train.txt
python3 tools/visualize_training.py build/models
```

---

## Save / Load

Binary format `.spai` — `SPAI` magic, current version **5**. Versions
3 and 4 load transparently (missing fields default to zero).

```
  [Header 32B]   magic "SPAI" | version | kf_count | df_count | reserved[3]

  [Records]*     tagged stream, KFs + deltas in insertion order
    tag 0x01  Keyframe:  id + label[64] + text_byte_count +
                         (v5: topic_hash + seq_in_topic) +
                         A + R + G + B
    tag 0x02  Delta:     id + parent_id + label[64] + count + change_ratio +
                         entries[]  (v4+: 9 B/entry with diff_B; v3: 8 B)
    tag 0x03  Weights:   global ChannelWeight (4× float)
    tag 0x04  Canvas:    slot_count, canvas_type, frame_type, parent_canvas_id,
                         changed_ratio, classified, SlotMeta[32], A + R + G + B
    tag 0x05  Subtitle:  count + (type, topic_hash, canvas_id, slot_id, byte_length)[]
    tag 0x06  EMA:       R[65536] + G[65536] + B[65536] + count[65536]  (float each, 1 MB)
```

Version bumps:

- **v3 → v4** added `diff_B` to `DeltaEntry` (sparse delta now covers
  the B channel) and the EMA trailing block.
- **v4 → v5** added `topic_hash` + `seq_in_topic` to every keyframe
  record so `ai_generate_next` can walk same-topic threads instead
  of falling back to `id + 1`.

Public API (`include/spatial_io.h`):

```c
SpaiStatus ai_save(const SpatialAI* ai, const char* path);
SpatialAI* ai_load(const char* path, SpaiStatus* out_status);
SpaiStatus ai_save_incremental(const SpatialAI* ai, const char* path);
SpaiStatus ai_peek_header(const char* path,
                          uint32_t* kf_count, uint32_t* df_count, uint32_t* version);
```

Guarantees validated by `test_io`:

- **Roundtrip integrity**: 700 clauses → save → destroy → load → same query
  cosine within `1e-3`.
- **Incremental growth**: `ai_save_incremental` refuses to shrink
  (returns `SPAI_ERR_STATE` if engine has fewer entries than disk) and
  grows the file by the new records only.
- **Forward compat**: unknown trailing tags are tolerated — older readers
  stop cleanly instead of corrupting.
- **Corrupt-file safety**: bad magic → `SPAI_ERR_MAGIC`,
  bad version → `SPAI_ERR_VERSION`, truncated body → `SPAI_ERR_READ`.

---

## Project layout

```
├── include/                  # Public headers
│   ├── spatial_grid.h        # 256×256 grid, 1D aligned channels
│   ├── spatial_layers.h      # 3-layer summation (base / morpheme / word)
│   ├── spatial_morpheme.h    # Korean morpheme analyzer (longest-match)
│   ├── spatial_keyframe.h    # Keyframe / delta / SpatialAI engine
│   ├── spatial_match.h       # Cosine, cascade modes, adaptive weights
│   ├── spatial_context.h     # Context frames + LRU cache
│   ├── spatial_canvas.h      # 2048×1024 canvas with 32 slots
│   ├── spatial_subtitle.h    # SubtitleTrack + SpatialCanvasPool
│   ├── spatial_generate.h    # Next-clause generation
│   └── spatial_io.h          # .spai binary format (v3)
├── src/                      # Implementations (one per header)
├── dict/                     # Korean dictionaries (nouns/verbs/adj/particles/endings)
├── tests/                    # 12 test binaries, 69 tests total
├── data/                     # Sample corpora + download scripts
├── tools/
│   ├── stream_train.c         # Line-by-line streaming trainer (C binary)
│   ├── run_practical_test.sh  # Build + train + verify wrapper
│   ├── visualize_training.py  # .spai → PNG frames + MP4 video
│   └── kaggle_gpu_train.py    # Optional CUDA training helper
├── Makefile
├── SPEC.md                   # Core spec (Page 1)
├── SPEC-ENGINE.md            # Engine optimization spec (Page 2)
└── README.md / README_KO.md
```

---

## Engine optimizations

All of these live in `src/spatial_match.c` + `src/spatial_keyframe.c` and are
exercised by `test_match`, `test_integration`, `test_cascade`, `test_adaptive`.

| Optimization | Where | Result |
|---|---|---|
| 1D aligned channels (32-byte) | `spatial_grid.c` | AVX2-ready, cache-line friendly |
| Block skip (16×16 sums) | `compute_block_sums`, `cosine_block_skip` | 93% blocks skipped on clauses, **0.000%** accuracy loss |
| Adaptive Top-K (hash buckets) | `bucket_index_*`, `grid_hash` | O(N) → O(N/B + K) for KF ≥ 100 |
| Sparse delta | `compute_delta` / `DeltaEntry` | 16-entry delta = 128 B |
| LRU frame cache | `spatial_context.c` | 90% hit on repeat access |
| Adaptive channel weights | `ChannelWeight` + `weight_update` | winner-take-reward per store |
| Directional RGB diffusion | `update_rgb_directional` | read/write split, min ±1 delta |

Targets: 20–100× combined over a naïve per-cell cosine at 1000+ keyframes,
with accuracy preserved.

---

## Morpheme analyzer

Dictionary-based longest-match tokenizer. No external libs.

```
  Input              Output
  ─────────────────────────────────────────────────
  "귀여운"        → [adj: 귀여운]
  "고양이가"      → [noun: 고양이] + [particle: 가]
  "밥을"          → [noun: 밥] + [particle: 을]
  "먹는다."       → [verb: 먹] + [ending: 는다] + [punct: .]
```

```
  Dictionary
  ├── Nouns       88   (animals, food, objects, nature, people, abstract)
  ├── Verbs       39   (먹, 가, 오, 보, 하, 되, …)
  ├── Adjectives  20   (귀여운, 예쁜, 밝은, 아름다운, …)
  ├── Particles   26   (은/는/이/가/을/를/에서/으로, …)
  └── Endings     20   (는다/었다/았다/겠다, …)
```

POS tags also seed the R / G channels before diffusion:

```
  POS_NOUN     R=40  G=30
  POS_VERB     R=120 G=40
  POS_ADJ      R=170 G=35
  POS_PARTICLE R=8   G=85
  POS_ENDING   R=12  G=95
  POS_PUNCT    R=5   G=120
  POS_UNKNOWN  R=210 G=20
```

---

## vs. Traditional LLM

```
                      Traditional LLM            SPATIAL-PATTERN-AI
  ───────────────────────────────────────────────────────────────────
  Encoding            token → vector → matrix    byte → pixel → 256×256
  Parameters          fixed-size matrix          unlimited frame stack
  Context             bounded window (32K-1M)    unlimited (disk-bound)
  Interpretability    opaque weights             visible heatmap
  Learning            full retrain / SFT         incremental delta
  Per-frame cost      —                          ~320 KB on disk
  Retrieval           attention / embed search   overlap → cosine → cascade
```

Strengths: retrieval, incremental memory, rewindable learning, interpretability,
embedded footprint.
Trade-off: not a generative LLM replacement — it's a substrate for
retrieval-heavy and memory-heavy tasks where pattern persistence matters.

---

## License

See [LICENSE](LICENSE).
