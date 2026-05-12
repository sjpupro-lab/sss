# Phase 5 migration guide — single generation entry

Phase 5 retired the two legacy image generators
(`tools/sss_animate.c` and `legacy_deprecated/gen_image_ce.c`) and
folded their functionality into a single CLI and one HTTP endpoint:

  * **C binary**: `build/sss_gen` — Phase 1–3 spectrogram path (`.sss`
    models) with new `--frames N` and `--out-dir DIR` flags.
  * **Python entry**: `scripts/sss_gen.py` — drives the C binary
    for `.sss` + frames = 1 / no Phase 4 features, and the Phase 4
    Python synthesiser (`tools/sss_synthesizer.py`) for everything
    else.
  * **HTTP**: `/api/sss_gen` on `ui/unified_server.py`. The legacy
    `/api/generate` and `/api/atmos` are kept as thin aliases.

The design philosophy is the Phase 4 invariant carried into the CLI
layer: an image is `temporal_length = 0` and a video is
`temporal_length > 0`, both running through the same synthesiser.
The pipeline therefore needs exactly one entry point.

## CLI mapping

### `sss_animate` → `sss_gen`

| Old (`sss_animate`)                  | New (`sss_gen` / `scripts/sss_gen.py`)              |
|--------------------------------------|----------------------------------------------------|
| `INPUT.ppm`                          | `--atmos-from INPUT.ppm` (Python entry)            |
| `OUT_DIR`                            | `--out-dir OUT_DIR`                                |
| `N_FRAMES`                           | `--frames N`                                       |
| `WIDTH` / `HEIGHT`                   | `--out-w W` / `--out-h H` (already existed)        |
| (implicit auto-inferred motion)      | `--condition LABEL INTENSITY` (Phase 4)            |

The new path takes a `.sfb` feature bank instead of a raw PPM; the
optional `--atmos-from` flag warm-starts from a reference image
while keeping motion driven by the bank's condition signals.

### `gen_image_ce` → `sss_gen`

| Old (`gen_image_ce`)          | New (`sss_gen` / `scripts/sss_gen.py`) |
|-------------------------------|----------------------------------------|
| `model.ces`                   | `model.sss` or `model.sfb`             |
| `prompt`                      | `prompt`                               |
| `out.ppm`                     | `out.ppm`                              |
| `seed`                        | `seed`                                 |
| `steps`                       | `steps`                                |
| `wave_iters`                  | **removed** (Phase 1 radio loop subsumed it) |
| `--hybrid`                    | **removed** (no .sfb equivalent)       |
| `--guidance N.N`              | **removed**                            |

`.ces` is no longer an input format. Phase 3 trainers write `.sfb`
directly; Phase 1 trainers still emit `.sss`. Run Phase 3 to
regenerate any `.ces` you depended on.

## HTTP endpoints

The unified endpoint accepts both image (`frames = 1`) and video
(`frames > 1`) requests; clients just bump `frames` to switch
modes.

```
POST /api/sss_gen        (canonical)
POST /api/generate       (alias — frames defaults to 1)
POST /api/viz-generate   (alias — adds per-pixel analysis to the
                          response; frames typically = 1)
POST /api/atmos          (preserved; the Atmos JSON path stays as
                          is for callers that decompose a PPM into
                          scene-objects without going through the
                          generator)
```

Request body:

```json
{
  "model":          "build/feature_bank.sfb",
  "prompt":         "kitty waving paw",
  "seed":           42,
  "steps":          120,
  "detail":         1.5,
  "frames":         16,
  "out_w":          256,
  "out_h":          256,
  "atmos":          false,
  "conditions": [
    {"label": "wind",  "intensity": 0.5},
    {"label": "blink", "intensity": 1.0}
  ],
  "atmos_reference": "path/to/reference.png"
}
```

Response:

* `frames == 1`: `result.image` is a single PNG data URI.
* `frames  > 1`: `result.frames[]` is a list of PNG data URIs (one
  per frame); `result.image` is the first frame for backward-
  compatible clients that only render the leading image.

## Helper scripts

`scripts/make_anim_frames.py` already drove `sss_gen` — no migration
needed.

## What got deleted

| Path                                  | Phase 5 disposition           |
|---------------------------------------|-------------------------------|
| `tools/sss_animate.c`                 | deleted                       |
| `legacy_deprecated/gen_image_ce.c`    | deleted                       |
| `Makefile` `sss_animate` rule / alias | deleted                       |
| `Makefile` `gen_image_ce` rule / alias| deleted                       |
| `ui/server.py` `.ces` / `gen_image_ce` branch | rewritten to alias `/api/sss_gen` |
| `ui/index.html` engine-toggle UI      | replaced with a single Frames slider |
| `legacy_deprecated/train_demo.c`      | **kept** (still buildable via `make legacy_demo`) |
| `legacy_deprecated/train_images_ce.c` | **kept** (same)               |
| `ce_core/ce_hybrid_vae.{c,h}` and friends | **kept** (still consumed by ce_core tests + `verify_hybrid`) |

## Quick recipes

Image (Phase 1–3 spectrogram path, fastest):

```bash
make sss_gen
./build/sss_gen build/models/demo.sss "red circle" out.ppm 1 1.5 120
```

Image (Phase 4 .sfb, condition-aware):

```bash
python3 scripts/sss_gen.py build/feature_bank.sfb "flag" out.ppm \
    --condition wind 0.5
```

Video (16-frame, condition-driven):

```bash
python3 scripts/sss_gen.py build/feature_bank.sfb "flag" out.ppm \
    --frames 16 --out-dir /tmp/anim --condition wind 0.5
```

Video (Phase 1–3, seed-variation only — no .sfb):

```bash
./build/sss_gen build/models/demo.sss "red apple" out.ppm \
    --frames 8 --out-dir /tmp/anim
```

HTTP image request:

```bash
curl -X POST http://localhost:8090/api/sss_gen \
    -H 'Content-Type: application/json' \
    -d '{"model":"build/feature_bank.sfb","prompt":"flag",
         "frames":1,"conditions":[{"label":"wind","intensity":0.5}]}'
```
