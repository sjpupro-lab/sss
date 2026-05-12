# Phase 5 audit — legacy generator consolidation

Read-only analysis; no engine code modified.

Generators surveyed: `sss_gen`, `sss_animate`, `gen_image_ce`.

## 1. Call graph — ce_core function dependency

Function calls (by name) found in each generator's translation unit. Names are matched against the ce_core/*.h headers to attribute each call to a module.

### `sss_gen` (tools/sss_gen.c)

**Headers included:** `sss_rowvae.h`, `ce_scene_object.h`, `ce_scene_bridge.h`, `ce_move_profile.h`

| function                 | ce_core module  | calls |
|--------------------------|-----------------|-------|
| ce_scene_build_from_rgba | ce_scene_bridge | 1     |
| ce_scene_render          | ce_scene_object | 1     |
| sss_generate             | sss_rowvae      | 1     |
| sss_image_free           | sss_rowvae      | 4     |
| sss_model_free           | sss_rowvae      | 5     |
| sss_model_load           | sss_rowvae      | 1     |

### `sss_animate` (tools/sss_animate.c)

**Headers included:** `ce_scene_object.h`, `ce_scene_bridge.h`, `ce_move_profile.h`

| function                    | ce_core module  | calls |
|-----------------------------|-----------------|-------|
| ce_scene_build_from_rgba    | ce_scene_bridge | 1     |
| ce_scene_render             | ce_scene_object | 1     |
| ce_scene_tick_with_profiles | ce_scene_bridge | 1     |

### `gen_image_ce` (legacy_deprecated/gen_image_ce.c)

**Headers included:** `spatial_morpheme.h`, `ce_storage.h`, `ce_storage_io.h`, `ce_search.h`, `ce_gen.h`, `ce_decode.h`, `ce_denoise.h`, `ce_type.h`, `ce_hybrid_vae.h`, `ce_residual_codebook.h`

| function                        | ce_core module | calls |
|---------------------------------|----------------|-------|
| ce_feed                         | ce_core        | 1     |
| ce_gen_config_default           | ce_denoise     | 1     |
| ce_gen_config_hq                | ce_denoise     | 1     |
| ce_generate_image_canvas_routed | ?              | 1     |
| ce_generate_image_typed         | ?              | 2     |
| ce_init                         | ce_core        | 1     |
| ce_search_by_type               | ce_search      | 1     |
| ce_storage_free                 | ce_storage     | 3     |
| ce_storage_load_slig_sets       | ce_storage     | 1     |
| ce_storage_load_with_codebook   | ce_storage_io  | 1     |
| hybrid_decode_blended           | ?              | 2     |
| hybrid_vae_config_default       | ce_hybrid_vae  | 1     |
| hybrid_vae_decode               | ce_hybrid_vae  | 1     |
| morpheme_init                   | ?              | 1     |
| morpheme_tokenize_clause        | ?              | 1     |

### Overlap matrix (functions called by ≥ 2 generators)

| function                 | callers              | ce_core module  |
|--------------------------|----------------------|-----------------|
| ce_scene_build_from_rgba | sss_gen, sss_animate | ce_scene_bridge |
| ce_scene_render          | sss_gen, sss_animate | ce_scene_object |

### Functions unique to `sss_gen` (kept verbatim post-Phase 5)

`sss_generate`, `sss_image_free`, `sss_model_free`, `sss_model_load`

### Functions unique to legacy generators (drop or move)

**sss_animate:** `ce_scene_tick_with_profiles`

**gen_image_ce:** `ce_feed`, `ce_gen_config_default`, `ce_gen_config_hq`, `ce_generate_image_canvas_routed`, `ce_generate_image_typed`, `ce_init`, `ce_search_by_type`, `ce_storage_free`, `ce_storage_load_slig_sets`, `ce_storage_load_with_codebook`, `hybrid_decode_blended`, `hybrid_vae_config_default`, `hybrid_vae_decode`, `morpheme_init`, `morpheme_tokenize_clause`

## 2. Integration cross-reference

Lines in `ui/`, `scripts/`, `tools/test_*.py`, `docs/`, `README.md`, `PIPELINE.md`, and `Makefile` that mention each generator.

### ui

**`sss_gen` — 17 hit(s):**
  - `ui/index.html:528`: `<!-- .sss-only parameters (sss_gen) -->`
  - `ui/index.html:928`: `$('gen-engine-label').textContent = sss ? '(.sss → sss_gen)' : '(.ces → gen_image_ce)';`
  - `ui/index.html:948`: `log('gen-console', `$ sss_gen ${body.model} "${prompt}" out.ppm ${body.seed} ${body.detail.toFixed(3)} ${body.steps}`, '`
  - `ui/server.py:12`: `POST /api/generate      → 이미지 생성 (.ces→gen_image_ce, .sss→sss_gen)`
  - `ui/server.py:483`: `bin_path = BUILD / "sss_gen"`
  - `ui/server.py:485`: `self.send_json({"error": "sss_gen binary not found — run `make sss_gen`"}, 400)`
  - `ui/server.py:485`: `self.send_json({"error": "sss_gen binary not found — run `make sss_gen`"}, 400)`
  - `ui/server.py:517`: `# sss_gen uses "wrote ... (WxH, prompt=..., seed=..., detail=..., steps=...)")`
  - `ui/server.py:624`: `bin_path = BUILD / "sss_gen"`
  - `ui/server.py:626`: `self.send_json({"error": "sss_gen binary not found — run `make sss_gen`"}, 400)`
  - `ui/server.py:626`: `self.send_json({"error": "sss_gen binary not found — run `make sss_gen`"}, 400)`
  - `ui/server.py:630`: `wave = "0"  # sss_gen has no wave param; report 0 to UI`
  - `ui/unified_server.py:371`: `# bridge for the C engine binaries (gen_image_ce, sss_gen, train_demo,`
  - `ui/unified_server.py:1233`: `# ── Forge: image generation (sss_gen / gen_image_ce) ──`
  - `ui/unified_server.py:1248`: `bin_path = BUILD_DIR / "sss_gen"`
  - `ui/unified_server.py:1250`: `_json(self, {"error": "sss_gen binary not found — "`
  - `ui/unified_server.py:1251`: `"run `make sss_gen`"}, 400)`

**`sss_animate` — 1 hit(s):**
  - `ui/unified_server.py:692`: `# tools.sss_atmos.AtmosScene without re-shelling sss_animate. Returns`

**`gen_image_ce` — 16 hit(s):**
  - `ui/start.sh:5`: `# Old binary-backed server (gen_image_ce / train_demo / chat) lives in`
  - `ui/index.html:502`: `<!-- .ces-only parameters (gen_image_ce) -->`
  - `ui/index.html:928`: `$('gen-engine-label').textContent = sss ? '(.sss → sss_gen)' : '(.ces → gen_image_ce)';`
  - `ui/index.html:954`: `log('gen-console', `$ gen_image_ce ${body.model} "${prompt}" out.ppm ${body.seed} ${body.steps} ${body.wave_iters}` + (b`
  - `ui/server.py:12`: `POST /api/generate      → 이미지 생성 (.ces→gen_image_ce, .sss→sss_gen)`
  - `ui/server.py:495`: `bin_path = BUILD / "gen_image_ce"`
  - `ui/server.py:497`: `self.send_json({"error": "gen_image_ce binary not found — run `make gen_image_ce`"}, 400)`
  - `ui/server.py:497`: `self.send_json({"error": "gen_image_ce binary not found — run `make gen_image_ce`"}, 400)`
  - `ui/server.py:516`: `# Parse summary line from log (gen_image_ce uses "mean RGB",`
  - `ui/server.py:636`: `bin_path = BUILD / "gen_image_ce"`
  - `ui/server.py:638`: `self.send_json({"error": "gen_image_ce binary not found — run `make gen_image_ce`"}, 400)`
  - `ui/server.py:638`: `self.send_json({"error": "gen_image_ce binary not found — run `make gen_image_ce`"}, 400)`
  - `ui/unified_server.py:371`: `# bridge for the C engine binaries (gen_image_ce, sss_gen, train_demo,`
  - `ui/unified_server.py:1233`: `# ── Forge: image generation (sss_gen / gen_image_ce) ──`
  - `ui/unified_server.py:1264`: `bin_path = BUILD_DIR / "gen_image_ce"`
  - `ui/unified_server.py:1266`: `_json(self, {"error": "gen_image_ce binary not "`

### scripts

**`sss_gen` — 11 hit(s):**
  - `scripts/make_anim_frames.py:4`: `기본 모드 (sss_gen pose 프롬프트):`
  - `scripts/make_anim_frames.py:34`: `SSS_GEN   = "./build/sss_gen"`
  - `scripts/make_anim_frames.py:41`: `ATMOS_REF_SIZE = 64    # reference 생성 해상도 (sss_gen)`
  - `scripts/make_anim_frames.py:62`: `"""Load a PPM-or-PNG (whatever sss_gen produced) into a (H, W, 4)`
  - `scripts/make_anim_frames.py:63`: `uint8 numpy array. Imported here so the default sss_gen path doesn't`
  - `scripts/make_anim_frames.py:76`: `"""Default pipeline: pose-prompt sss_gen + nearest-neighbour upscale."""`
  - `scripts/make_anim_frames.py:148`: `"per-pose sss_gen calls.")`
  - `scripts/train_pokemon.sh:23`: `# 1. sss_gen 빌드`
  - `scripts/train_pokemon.sh:24`: `echo "[1/3] sss_gen 빌드..."`
  - `scripts/train_pokemon.sh:25`: `make sss_gen 2>&1 | tail -2`
  - `scripts/train_pokemon.sh:55`: `./build/sss_gen build/models/pokemon.sss "$prompt" "build/sss_pokemon/$fname" 1 1.0`

### docs

**`sss_gen` — 15 hit(s):**
  - `README.md:630`: `make build/sss_gen`
  - `README.md:631`: `./build/sss_gen build/models/demo1k.sss "red circle smile draw" out.ppm 1 1.0 24`
  - `README.md:895`: `Numbers come from `python3 scripts/sss_train.py` + `./build/sss_gen``
  - `README.md:920`: `./build/sss_gen build/models/sanrio.sss "kitty white cat"     out.ppm 1 1.0 24`
  - `README.md:921`: `./build/sss_gen build/models/sanrio.sss "mymelody pink rabbit" out.ppm 1 1.0 24`
  - `README.md:922`: `./build/sss_gen build/models/sanrio.sss "keroppi green frog"   out.ppm 1 1.0 24`
  - `README.md:960`: `![sanrio originals + sss_gen seeds](docs/bench/sanrio_grid.png)`
  - `README.md:963`: `upscaled NN-2× for display); columns 2–4 are `./build/sss_gen` with`
  - `README.md:1135`: `| `sss_gen` on sanrio model, **24-step** sculpt+radio | 0.470 s / image (preview)               |`
  - `README.md:1136`: `| `sss_gen` on sanrio model, **60-step** sculpt+radio | 1.083 s / image (high quality)          |`
  - `README.md:1137`: `| `sss_gen` on sanrio model, **120-step** sculpt+radio | 2.165 s / image (max quality)          |`
  - `README.md:1138`: `| `sss_gen` on demo_1k model, 24-step              | 0.471 s / image                            |`
  - `README.md:1212`: `make sss_gen                                  # build/sss_gen`
  - `README.md:1212`: `make sss_gen                                  # build/sss_gen`
  - `README.md:1365`: `│   ├── sss_gen.c               MAIN PATH — sss_rowvae spectrogram generator`

**`gen_image_ce` — 18 hit(s):**
  - `README.md:407`: `# gen_image_ce, verify_hybrid`
  - `README.md:425`: `./build/gen_image_ce build/models/demo.ces "red apple" \`
  - `README.md:429`: `./build/gen_image_ce build/models/demo_cb.ces "red apple" \`
  - `README.md:437`: ``gen_image_ce` morpheme-tokenises the prompt, votes a winning`
  - `README.md:1157`: `make demo_tools           # train_demo / gen_image_ce / verify_hybrid`
  - `README.md:1364`: `│   ├── gen_image_ce.c          legacy prompt → 256×256 PPM`
  - `README.md:1410`: `- [x] `train_demo --masked-epochs N`, `gen_image_ce --hybrid --guidance N`,`
  - `PIPELINE.md:37`: `| 형태소→이미지 검색 | morpheme_tokenize_clause + ce_search_by_type(CE_TYPE_TEXT) → canvas_id 투표 → ce_generate_image_canvas_rout`
  - `PIPELINE.md:38`: `| Hybrid 생성 (보조) | hybrid_vae_decode + base-only 복원 | ce_hybrid_vae.c, gen_image_ce.c | OK gen_image_ce --hybrid 경로 |`
  - `PIPELINE.md:38`: `| Hybrid 생성 (보조) | hybrid_vae_decode + base-only 복원 | ce_hybrid_vae.c, gen_image_ce.c | OK gen_image_ce --hybrid 경로 |`
  - `PIPELINE.md:110`: `- [x] `gen_image_ce`: morpheme_tokenize_clause + 가중 투표 → canvas_routed`
  - `PIPELINE.md:116`: `- [x] **Phase 6 — CE_TYPE 확장 + SLIG sets 영속화:** `ce_type.h`에 `CE_TYPE_SLIG=3`, `CE_TYPE_RESIDUAL=4` 추가 (기존 0/1/2 값 보존, ``
  - `PIPELINE.md:129`: `- [x] **Phase 4 — 학습 도구 통합:** `train_demo --masked-epochs N` 추가, `gen_image_ce --hybrid --guidance N.N` 추가, `tools/verif`
  - `PIPELINE.md:129`: `- [x] **Phase 4 — 학습 도구 통합:** `train_demo --masked-epochs N` 추가, `gen_image_ce --hybrid --guidance N.N` 추가, `tools/verif`
  - `docs/validation/2026-04-29_ai_framework.md:20`: ``./build/gen_image_ce build/models/demo_cb.ces "red apple" out.ppm 0 50 200 --hybrid``
  - `docs/validation/2026-04-29_ai_framework.md:31`: `- **Root cause**: `gen_image_ce.c` never sets `hcfg.residual_book`, and `ce_storage_save/load` does not persist `CEResid`
  - `docs/validation/2026-04-29_ai_framework.md:50`: `1. ~~**Persist residual codebook**~~ → **DONE.** `.ces` file format bumped to v3 with a trailing `RCBK` section. `ce_sto`
  - `docs/validation/2026-04-29_ai_framework.md:53`: `4. ~~**Compositional voting**~~ → **PARTIAL.** Added `--blend K` to `gen_image_ce`; `vote_canvas_ids_topk` returns the t`

### make

**`sss_gen` — 9 hit(s):**
  - `Makefile:66`: `# it linkable inside the default OBJS list. sss_rowvae (sss_gen +`
  - `Makefile:77`: `sss_animate sss_gen verify_hybrid wave_debug`
  - `Makefile:202`: `$(BUILD_DIR)/sss_gen: tools/sss_gen.c $(OBJS) | $(BUILD_DIR)`
  - `Makefile:202`: `$(BUILD_DIR)/sss_gen: tools/sss_gen.c $(OBJS) | $(BUILD_DIR)`
  - `Makefile:205`: `sss_gen: $(BUILD_DIR)/sss_gen`
  - `Makefile:205`: `sss_gen: $(BUILD_DIR)/sss_gen`
  - `Makefile:206`: `@echo "Built sss_gen. Pipeline:"`
  - `Makefile:209`: `@echo "  ./build/sss_gen build/models/demo.sss \"red circle draw\" out.ppm 1 1.0"`
  - `Makefile:266`: `@echo "Built gen_image_ce (legacy — sss_gen is the main path now)."`

**`sss_animate` — 8 hit(s):**
  - `Makefile:77`: `sss_animate sss_gen verify_hybrid wave_debug`
  - `Makefile:215`: `$(BUILD_DIR)/sss_animate: tools/sss_animate.c $(OBJS) | $(BUILD_DIR)`
  - `Makefile:215`: `$(BUILD_DIR)/sss_animate: tools/sss_animate.c $(OBJS) | $(BUILD_DIR)`
  - `Makefile:218`: `sss_animate: $(BUILD_DIR)/sss_animate`
  - `Makefile:218`: `sss_animate: $(BUILD_DIR)/sss_animate`
  - `Makefile:219`: `@echo "Built sss_animate. Examples:"`
  - `Makefile:220`: `@echo "  ./build/sss_animate input.ppm  build/atmos_anim 60"`
  - `Makefile:221`: `@echo "  ./build/sss_animate input.ppm  build/atmos_anim 120 256 256"`

**`gen_image_ce` — 11 hit(s):**
  - `Makefile:4`: `# CLI tools (gen_image_ce, train_demo, train_images_ce) live in`
  - `Makefile:76`: `legacy_demo demo_tools train_images_ce gen_image_ce \`
  - `Makefile:95`: `# kept reachable for gen_image_ce and test_gen_routed but tagged so`
  - `Makefile:198`: `$(BUILD_DIR)/gen_image_ce: $(LEGACY_DEPRECATED_DIR)/gen_image_ce.c $(OBJS) | $(BUILD_DIR)`
  - `Makefile:198`: `$(BUILD_DIR)/gen_image_ce: $(LEGACY_DEPRECATED_DIR)/gen_image_ce.c $(OBJS) | $(BUILD_DIR)`
  - `Makefile:251`: `legacy_demo: $(BUILD_DIR)/make_demo_dataset $(BUILD_DIR)/train_demo $(BUILD_DIR)/gen_image_ce $(BUILD_DIR)/verify_hybrid`
  - `Makefile:255`: `@echo "  ./build/gen_image_ce build/models/demo.ces \"red apple\" out.ppm 0 50 200"`
  - `Makefile:265`: `gen_image_ce: $(BUILD_DIR)/gen_image_ce`
  - `Makefile:265`: `gen_image_ce: $(BUILD_DIR)/gen_image_ce`
  - `Makefile:266`: `@echo "Built gen_image_ce (legacy — sss_gen is the main path now)."`
  - `Makefile:267`: `@echo "  ./build/gen_image_ce build/synth/demo.ces \"red apple\" out.ppm 0 50 200"`

### Makefile target lines (raw)

**`sss_gen`:**
  - Makefile:66: # it linkable inside the default OBJS list. sss_rowvae (sss_gen +
  - Makefile:77: sss_animate sss_gen verify_hybrid wave_debug
  - Makefile:202: $(BUILD_DIR)/sss_gen: tools/sss_gen.c $(OBJS) | $(BUILD_DIR)
  - Makefile:205: sss_gen: $(BUILD_DIR)/sss_gen
  - Makefile:206: @echo "Built sss_gen. Pipeline:"
  - Makefile:209: @echo "  ./build/sss_gen build/models/demo.sss \"red circle draw\" out.ppm 1 1.0"
  - Makefile:266: @echo "Built gen_image_ce (legacy — sss_gen is the main path now)."

**`sss_animate`:**
  - Makefile:77: sss_animate sss_gen verify_hybrid wave_debug
  - Makefile:215: $(BUILD_DIR)/sss_animate: tools/sss_animate.c $(OBJS) | $(BUILD_DIR)
  - Makefile:218: sss_animate: $(BUILD_DIR)/sss_animate
  - Makefile:219: @echo "Built sss_animate. Examples:"
  - Makefile:220: @echo "  ./build/sss_animate input.ppm  build/atmos_anim 60"
  - Makefile:221: @echo "  ./build/sss_animate input.ppm  build/atmos_anim 120 256 256"

**`gen_image_ce`:**
  - Makefile:4: # CLI tools (gen_image_ce, train_demo, train_images_ce) live in
  - Makefile:76: legacy_demo demo_tools train_images_ce gen_image_ce \
  - Makefile:95: # kept reachable for gen_image_ce and test_gen_routed but tagged so
  - Makefile:198: $(BUILD_DIR)/gen_image_ce: $(LEGACY_DEPRECATED_DIR)/gen_image_ce.c $(OBJS) | $(BUILD_DIR)
  - Makefile:251: legacy_demo: $(BUILD_DIR)/make_demo_dataset $(BUILD_DIR)/train_demo $(BUILD_DIR)/gen_image_ce $(BUILD_DIR)/verify_hybrid
  - Makefile:255: @echo "  ./build/gen_image_ce build/models/demo.ces \"red apple\" out.ppm 0 50 200"
  - Makefile:265: gen_image_ce: $(BUILD_DIR)/gen_image_ce
  - Makefile:266: @echo "Built gen_image_ce (legacy — sss_gen is the main path now)."
  - Makefile:267: @echo "  ./build/gen_image_ce build/synth/demo.ces \"red apple\" out.ppm 0 50 200"

## 3. CLI option mapping

### `sss_animate` → `sss_gen`

| legacy flag                     | purpose                                                          | sss_gen equivalent                                                                                                                      |
|---------------------------------|------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| INPUT.ppm                       | Source frame for Atmos decomposition                             | sss_gen --reference IN.ppm (proposed Phase 5 flag) — or implicit via motif retrieval from the prompt                                    |
| OUT_DIR                         | Output directory for frame_NNNN.ppm                              | sss_gen --out-dir DIR (proposed; today sss_gen writes a single .ppm, Phase 5 must accept a dir when --frames > 1)                       |
| N_FRAMES                        | Number of motion frames to emit                                  | sss_gen --frames N (matches Phase 4 temporal_length)                                                                                    |
| WIDTH / HEIGHT                  | Output resolution                                                | sss_gen --out-w / --out-h (already present)                                                                                             |
| (implicit) Atmos motion profile | Auto-inferred motion type per object via ce_infer_motion_profile | sss_gen --condition <label> <intensity> (Phase 4 InteractionResponse drives motion); auto-infer maps to sfb identity → condition lookup |

### `gen_image_ce` → `sss_gen`

| legacy flag    | purpose                                    | sss_gen equivalent                                                          |
|----------------|--------------------------------------------|-----------------------------------------------------------------------------|
| model.ces      | CEStorage container (pre-.sss legacy)      | sss_gen <model.sfb>  (Phase 4 v2 .sfb supersedes .ces)                      |
| prompt         | Token-tokenised string                     | sss_gen <prompt>  (identical)                                               |
| out.ppm        | Single PPM output                          | sss_gen <out.ppm>  (identical)                                              |
| seed           | Deterministic PRNG seed                    | sss_gen … seed   (identical)                                                |
| steps          | Denoise iterations (pre-Phase-1 wave loop) | sss_gen … steps  (Phase 1 radio steps replaces wave loop)                   |
| wave_iters     | Atomic wave-refine pass count              | MISSING — Phase 1 radio loop subsumed this; no flag needed for new pipeline |
| --hybrid       | Hybrid VAE block-stamp path                | MISSING — pre-Phase-1 colour-base+SLIG decode; not reachable from .sfb      |
| --guidance N.N | Hybrid VAE blend weight                    | MISSING (paired with --hybrid)                                              |

### Identified gaps and proposed treatment

| gap                                                               | why                                                                                                                                         | proposed treatment                                                                                                                                                     |
|-------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| sss_animate: writing N PPMs to a directory                        | sss_gen writes a single .ppm today.                                                                                                         | Phase 5 adds --out-dir + --frames N. Trivial CLI plumbing on the C side; the Python sss_synthesizer already produces per-frame envelopes.                              |
| sss_animate: Atmos motion-profile auto-inference from RGBA pixels | ce_scene_build_from_rgba/_with_profiles infer motion class (STATIC/FLOW/RIGID/ORGANIC/PARTICLE) from pixel statistics, not from prompt+sfb. | Phase 5 may keep Atmos as an optional `--atmos-from PPM` input — converts a reference image to a quick-and-dirty condition set without going through the .sfb trainer. |
| gen_image_ce: .ces ingestion                                      | The .ces container predates .sfb v2. There are no current training pipelines that produce .ces files (Phase 3 wrote directly to .sfb).      | Drop. Existing .ces files in build/models/ are demo artefacts; they can be regenerated via Phase 3 pipeline if needed.                                                 |
| gen_image_ce: Hybrid VAE + guidance                               | Pre-Phase-1 colour-base + SLIG residual decoder. No callers outside legacy demos.                                                           | Drop. The Phase 4 synthesiser's envelope + cross_resonance + warp pipeline subsumes it.                                                                                |

## 4. Build delta

Binaries currently emitted by `Makefile` (via `$(BUILD_DIR)/...`): 40

```
test_grid
test_morpheme
test_layers
test_match
test_keyframe
test_context
test_integration
test_io
test_cascade
test_canvas
test_adaptive
test_subtitle
test_recluster
test_refine
bench_context
bench_refine
test_image_roundtrip
test_image_gen
test_tick_math
test_material
test_gen_routed
test_scene_atmos
bench_v3
img2grid
grid2img
train_images_ce
gen_image_ce
sss_gen
sss_animate
make_demo_dataset
train_demo
verify_hybrid
wave_debug
test_wiki
bench_stsb
bench_perplexity
bench_word_predict
bench_qa
stream_train
chat
```

After Phase 5 removal: 38 binaries (drops `sss_animate` and `gen_image_ce`).

**Object files potentially droppable:** `legacy/ce_gen.o (currently linked via $(LEGACY_OBJS))`.

## 5. Migration scenarios

### A — Immediate removal

**Plan:** Delete tools/sss_animate.c, legacy_deprecated/gen_image_ce.c, the matching Makefile targets, the ui/server.py and ui/unified_server.py call sites, and scripts/make_anim_frames in the same PR. Migrate every caller atomically.

**Pros:**
- Single PR; no transitional code lingers.
- Codebase ends Phase 5 with one generator entry point.

**Cons:**
- Atomic migration is fragile — any missed caller breaks build or runtime.
- Forces the merger of trainer / UI / docs work into one very large PR.

**Recommend when:** Few external callers (none here outside the two UI servers and one script) AND we are willing to land the Phase 5 trainer + UI + doc updates as one PR.

### B — Shim transition

**Plan:** Reduce sss_animate to a thin C wrapper that translates its CLI to `sss_gen --frames N --temporal-mode video` and execvp()s it. gen_image_ce becomes a similar wrapper that prints a deprecation warning and forwards to sss_gen with a default .sfb path. Remove the shims one phase later (Phase 6).

**Pros:**
- Existing scripts and UI keep working unmodified.
- Migration risk is amortised over two PRs.

**Cons:**
- Two-phase delete is overhead for a small surface.
- Shims need their own tests + docs to stay correct.

**Recommend when:** Many external integrations or downstream users depend on the legacy binary names (not the case here — UI is the only meaningful integrator).

### C — Deprecation warning, delete next phase

**Plan:** Keep the binaries buildable; add a stderr warning at the top of main() in both legacy generators ("DEPRECATED: use sss_gen … instead"), update README/PIPELINE to mark them as deprecated. Phase 6 deletes them and the Makefile targets.

**Pros:**
- Zero behavioural change in the PR — fastest to land.
- Gives downstream consumers a published heads-up.

**Cons:**
- Two phases of work for what is a small one-time delete.
- Dead code lingers in the tree for one release cycle.

**Recommend when:** External users / vendored copies of the repo exist and need a published heads-up — not the case here.

### Recommendation

Observed integration footprint outside Makefile: 17 UI line(s), 0 script line(s), 0 test line(s). All callers are in-repo and reachable in a single PR.

**Scenario A (immediate removal) is the recommended path.** Repo-wide grep shows no external integrators, and the test suite already exercises `sss_gen` end-to-end via `tools/test_synthesizer.py` + the Phase 4 trainers. Migrating the two UI handlers (`ui/server.py`, `ui/unified_server.py`) and the one helper script (`scripts/make_anim_frames.py`) atomically with the Makefile cleanup is the smallest total diff.

**Fallback to Scenario C (deprecation warning)** is acceptable if Phase 5 wants to split the work into a small "warn and doc" PR followed by a removal PR — useful if reviewers want to land the UI rewrite and the C cleanup separately.

## 6. Unified UI endpoint design

```
Proposed single endpoint /api/sss_gen replaces /api/generate and
the internal-only /api/atmos. Request body:

    {
      "model":        "build/feature_bank.sfb",
      "prompt":       "kitty waving paw",
      "seed":         42,
      "steps":        120,
      "detail":       1.5,
      "frames":       1                  // 1 = image, N > 1 = video
      "size":         [256, 256],
      "conditions": [                    // optional; Phase 4 v2
        { "label": "wind",       "intensity": 0.5 },
        { "label": "blink",      "intensity": 1.0 }
      ],
      "atmos_reference": "path/to/ref.png"  // optional Atmos warm-start
    }

Response:
  frames == 1 → image/png (single frame, same as today's /api/generate)
  frames  > 1 → application/zip (frame_NNNN.png inside) OR
                application/octet-stream (mp4) when ffmpeg available
                — selected by `Accept` header.

Migration: keep /api/generate as a thin alias that translates to
sss_gen with frames = 1. Keep /api/atmos as a thin alias that
translates to sss_gen with frames > 1 and atmos_reference set.

Alternative considered: split image and video endpoints
(/api/image, /api/video). Rejected because the Phase 4 design
philosophy treats them as the *same* operation — frames=1 is a
special case of frames=N. Two endpoints would re-create the split
this phase is supposed to remove.
```

## 7. Risk roll-up

### build

- sss_animate: 6 Makefile mention(s); 9 integration line(s) outside Makefile.
- gen_image_ce: 9 Makefile mention(s); 45 integration line(s) outside Makefile.

### runtime

- sss_animate: 1 UI handler line(s) need rewiring to sss_gen.
- gen_image_ce: 16 UI handler line(s) need rewiring to sss_gen.

### compatibility

(none observed)


### regression

(none observed)


### docs

- gen_image_ce: 18 doc line(s) mention this — README / PIPELINE need a migration note.

## 8. Recommended Phase 5 work order

1. **C engine:** extend `tools/sss_gen.c` with `--frames N`, `--out-dir DIR`, `--condition LBL INT` (repeatable), `--atmos-from PPM` (replaces the Atmos pixel-decompose path). Existing `--out-w/--out-h` and `--atmos` flags stay.

2. **Python wrappers:** ensure `tools/sss_synthesizer.py` is reachable via a thin Python entry point (`scripts/sss_gen.py`?) so the UI can drive it without shelling out for image-only generation.

3. **UI rewrite:** introduce `/api/sss_gen` per §6. Make `/api/generate` and `/api/atmos` thin aliases.

4. **Script migration:** rewrite `scripts/make_anim_frames.py` to call the new `sss_gen --frames` path.

5. **Makefile + delete:** remove `build/sss_animate` and `build/gen_image_ce` targets + delete the C sources (and `legacy/ce_gen.{c,h}` if no other consumers remain).

6. **Docs:** README + PIPELINE migration notes; `docs/migration_phase5.md` (one page) summarising the CLI mapping.

7. **Regression:** verify Phase 1+2+3+4 tests still PASS; add `tools/test_sss_gen_video.py` exercising `--frames N` end-to-end against a Phase 4 .sfb.


---
Generated by `tools/audit_legacy_generators.py` (Phase 5 pre-analysis, read-only).
