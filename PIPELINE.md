# SSS Image Generation Pipeline

## Diffusion → SSS 1:1 대응

| Diffusion 단계 | SSS 구현 | 파일 | 상태 |
|---|---|---|---|
| Text Tokenizer | morpheme_tokenize_clause | spatial_morpheme.c | OK |
| Text Embedding | ce_feed(token) → CEUnit | ce_core.c | OK |
| Text→Image Cross Attn | ce_cross_attention | ce_engine.c | TODO CEStorage 미연결 |
| VAE Encode (이미지→latent) | ce_feed_image_16 16×16→4 CEUnit (atomic) | ce_feed_image.c | OK production via ce_storage_ingest_rgba_16 |
| VAE Decode (latent→이미지) | ce_decode_image_block | ce_decode.c | OK |
| Latent μ/σ 공간 | dec.A audio_amps[0..3] | slig_signal.c | TODO 렌더러 미사용 |
| Diffusion Noise | ce_noise_init(seed) | ce_core.c | OK |
| Denoising Loop (50스텝) | ce_denoise_loop | ce_denoise.c | TODO 8스텝, CEStorage 비어있음 |
| UNet Down | ce_down | ce_engine.c | TODO CEStorage 미연결 |
| UNet Conv | ce_conv | ce_engine.c | TODO CEStorage 미연결 |
| UNet Self Attn | ce_self_attention | ce_engine.c | TODO CEStorage 미연결 |
| UNet Up | ce_up | ce_engine.c | TODO CEStorage 미연결 |
| Skip Connection | ce_skip_connect | ce_engine.c | TODO CEStorage 미연결 |
| CFG (guidance scale) | apply_guidance / cfg_scale | spatial_image_gen.c | OK |
| LoRA | ~~ce_memo_apply~~ | ~~ce_extend.c~~ | DELETED Phase 1 |
| ControlNet | ~~ce_hint_apply (EDGE/DEPTH/POSE/COLOR)~~ | ~~ce_extend.c~~ | DELETED Phase 1 |
| Inpainting | ce_inpaint + CEInpaintMask | ce_extend.c | OK (masked_train에서 사용) |
| Upscaler | ~~ce_upscale~~ / slig_upscale_harmonic/sbr/wiener/cs | ~~ce_extend.c~~ / slig_pipeline.c | DELETED ce_upscale (Phase 1) / WARN slig v3에서만 |
| Hybrid VAE Encode | hybrid_vae_encode (블록 도장 + SLIG 분해 + 오디오 링크) | ce_hybrid_vae.c | OK Phase 2 |
| Hybrid VAE Decode | hybrid_vae_decode (base+detail 블렌딩 + wave refine) | ce_hybrid_vae.c | OK Phase 2 (SLIG sets 영속화 미구현) |
| Masked Train | masked_train_image (correction 마스킹→denoise→loss→재저장) | ce_masked_train.c | OK Phase 3 |
| Freq 분해 | slig_decompose_v2 (구조/엣지/텍스처/색상/이벤트) | slig_pipeline.c | OK |
| Spatial Attn Mask | slig_apply_masked + mask_left/right/top/bottom | slig_signal.c | TODO 미연결 |
| Atmos 오브젝트 배치 | slig_apply_masked per 형태소 | slig_signal.c | TODO 미연결 |
| Wave Refine | ce_image_wave_refine (0.01→3→0.01 사이클) | ce_image_wave_refine.c | OK ce_generate_image_typed에 연결됨 |
| Tick 정수엔진 | TICK_SIN/COS/GAUSS_TABLE + WAVE_STEPS[7] + RGBA carry | slig_tick_math.c | WARN 부분사용 (hybrid_vae cfg는 placeholder) |
| Material 텍스처 | slig_material_harmonic Mat-1/2/3/4 | slig_material_harmonic.c | WARN spai v6에서만 |
| Audio Sync | ~~ce_audio_load + ce_audio_energy_at~~ → dec.A audio_bins canvas_id 링크 | ~~ce_extend.c~~ / ce_hybrid_vae.c | REPLACED Phase 1 (CEAudioTrack 삭제, hybrid_vae가 audio_bins로 cross-link) |
| Keyframe/Delta 저장 | ai_store_auto_with_image | spatial_keyframe.c | OK |
| CEStorage 훈련 | ce_storage_add_typed | ce_storage.c | OK ce_storage_ingest_rgba / train_images_ce |
| 형태소→이미지 검색 | morpheme_tokenize_clause + ce_search_by_type(CE_TYPE_TEXT) → canvas_id 투표 → ce_generate_image_canvas_routed | ce_search.c, ce_gen.c | OK production (gen_image_ce 기본 경로) |
| Hybrid 생성 (보조) | hybrid_vae_decode + base-only 복원 | ce_hybrid_vae.c, gen_image_ce.c | OK gen_image_ce --hybrid 경로 |

## CE Block 크기 문제

```
현재: 8×8 = 64픽셀 → CEUnit 64바이트
픽셀당 1바이트 → 색상 평균값만 carry chain 누적
공간 디테일 손실 큼

개선안:
  16×16 = 256픽셀 → 4개 CEUnit으로 분산 (채널별)
  또는: 8×8 × 계층 3단계 (coarse/mid/fine) = 3 CEUnit per block
  → 현재 SLIG pyramid (COARSE/MID/FINE) 구조 그대로 CE에 적용
```

## 올바른 훈련 파이프라인

```
이미지 + 레이블 입력
    ↓
1. slig_decompose_v2 → SligDecomposed (구조/엣지/텍스처/색상/이벤트)
2. slig_upscale (harmonic) → 고주파 복원
3. ce_feed(label) → text_CEUnit
4. ce_feed_image(16×16블록) → img_CEUnit   (TODO: 16×16으로 확장)
5. ce_delta(text, img) → delta
6. ce_storage_add_typed(CE_TYPE_IMAGE, img_CEUnit, delta)   (현재 누락)
7. ai_store_auto_with_image → SpatialAI keyframe
8. Mat-1: slig_material_harmonic → dec.A energy 저장
```

## 올바른 생성 파이프라인

```
텍스트 입력
    ↓
1. morpheme_tokenize_clause → 형태소 배열
2. ce_feed(token) → morph_CE per 형태소
3. text_grid(256 RGBA) → cfg_scale 추출 (품사분포 기반)
4. ce_search_by_type(CE_TYPE_IMAGE, morph_CE) → top-K 이미지블록
5. ce_latent_init(noise, kf, delta, prompt, audio=NULL)
6. ce_denoise_loop(50~100스텝) → latent 수렴
7. ce_decode_image_block → coarse 이미지 (블록별)
8. Atmos 배치: 방향키워드 → slig_apply_masked per 형태소
9. ce_image_wave_refine(canvas, top3, 200회) → 파동 수렴
10. slig_apply_material_overlay → 텍스처 복원
11. slig_postprocess → 최종 출력
```

## TODO

### 완료

**1a4c38a — CE 파이프라인 통합**
- [x] ce_feed_image 16×16 블록 버전 (4 quadrant CEUnits)
- [x] ce_denoise 스텝 파라미터화 — `ce_gen_config_hq()` 50-step 프리셋
- [x] CEStorage ↔ SpatialAI 브릿지 — `train_images_ce` (.spai + .ces 동시 저장)
- [x] ce_image_wave_refine 생성 파이프라인 연결 — `ce_generate_image_typed`
- [x] ce_search_by_type 생성 진입점 — `ce_generate_image_typed`
- [x] 10000+ 학습 스케일 검증 — `test_stress_10k` (12K/100K 모두 통과)

**3bd277a — SLIG 측 에너지/방향**
- [x] `slig_decompose_structure` HORIZONTAL → SVD u/v 분산비 기반 H/V/Diag 분류
- [x] `cond_threshold` 누적 에너지 커버리지로 채움 (분해 시)
- [x] `dec.A audio_amps[2]` 컴포넌트별 에너지 비율로 채움 (분해 시)
- [x] `apply_global` 렌더 시 amp[2]로 진폭 변조 (하위호환 sentinel)
- [x] 회귀 테스트 `test_slig_energy` 9/9 통과 + 시각 회귀 0건

**16×16 atomic + per-morpheme CE refactor**
- [x] `ce_storage_ingest_rgba_16`: 16×16 블록 → `ce_feed_image_16` → 4 quadrant CEUnits/block, 1024 entries per 256² 이미지
- [x] `ce_decode_image_block_16`: 4 quadrant → 16×16 RGBA round-trip
- [x] `ce_generate_image_canvas_routed`: caller가 결정한 canvas_id로 wave-refine 타겟 필터, 16×16 atomic 패치 단위로 디코드+타일
- [x] `train_demo`: 16×16 image 인제스트 + per-morpheme TEXT 브릿지 (canvas_id 공유)
- [x] `gen_image_ce`: morpheme_tokenize_clause + 가중 투표 → canvas_routed
- [x] byte-level `ce_generate_image_label_routed` 삭제
- [x] 시각 검증: 6/6 prompts → 정확히 매칭된 색상 (이전 5/6에서 향상)
- [x] 단위 테스트: `test_storage_ingest16`, `test_decode16`, `test_gen_routed`

**RGBA tick 시계 + SLIG/Residual 영속 (Phase 6-9)**
- [x] **Phase 6 — CE_TYPE 확장 + SLIG sets 영속화:** `ce_type.h`에 `CE_TYPE_SLIG=3`, `CE_TYPE_RESIDUAL=4` 추가 (기존 0/1/2 값 보존, `.ces` v2 포맷 호환). `ce_storage_persist_slig_set` / `ce_storage_load_slig_sets` helper로 SligCellSet 9 grid를 (slot=scale*3+channel, block_idx=cell index) 레이아웃으로 영속/재구성. `hybrid_vae_encode`가 codebook 등록 직후 SLIG cells을 storage에 추가. `gen_image_ce --hybrid`가 zero-set 대신 storage에서 sets 로드. `slig_signal.h`의 `SligCellSet`에 struct tag 추가 (forward declaration용).
- [x] **Phase 7 — Tick 시계 채널 분리 매핑:** `ce_core/ce_tick.h` (헤더 only) 신규. `TickRGBA` 구조 + `ce_tick_step` (carry chain `++R; if(0)++G; if(0)++B; if(0)++A;`) + `ce_tick_compare` (B>G>R>A 우선순위). 사용자 명시 채널 분리: R=cell 내부 순서, G=scale_level, B=render layer/channel, A=amplitude. `ce_tick_from_slig` / `ce_tick_from_image`가 storage entry → TickRGBA. `ce_tick_sorted_indices`로 (canvas_id, type mask) 필터 후 tick 순으로 qsort된 entry index 배열 반환. CEUnit byte 레이아웃 미변경 — 매핑은 derived (slot/block_idx + audio_amps에서 추출).
- [x] **Phase 8 — Residual codebook:** `ce_core/ce_residual_codebook.{h,c}` 신규. `CEResidualCode {unit, scale_level, direction, strength, TickRGBA tick, used_count}` + `lookup/add_or_lookup/get` (slig_codebook과 동일한 lookup→reuse / novel→add / full→nearest 정책, 256 entries). scale_level 미스매치는 UINT32_MAX (cross-scale 충돌 차단), 0xFF는 wildcard. `CE_TYPE_RESIDUAL` storage 브릿지 — keyframe 64B를 descriptor로 재해석 (idx 1B + strength 1B + TickRGBA 4B + reserved 2B + zero); `ce_residual_storage_add/unpack` API. **원본 patch는 storage에 저장 안 함**, codebook 레퍼런스만 (사용자 설계 의도 반영).
- [x] **Phase 9 — 회귀 + 신규 테스트:** `tests/test_slig_persist.c` (19 PASS — direct persist/load API + hybrid_vae E2E + tick-sorted iteration의 monotonic property 검증), `tests/test_residual_codebook.c` (24 PASS — init/lookup/add/distance/get + storage descriptor add/unpack roundtrip). ce_core 19/20 통과 (test_slig_signal만 베이스라인 환경 의존 fail 동일). 상위 19/19 통과.
- [x] **Phase 10 — correction → residual codebook 환원:** `MaskedTrainConfig`에 `residual_book` + `residual_threshold` 필드 추가. 수렴 시 basis+edge+texture는 raw `CE_TYPE_SLIG`로, correction(color+event)은 `ce_residual_codebook_add_or_lookup` → `CE_TYPE_RESIDUAL` descriptor (idx + strength + tick만). `train_demo --residual-codebook` 플래그로 활성. data/demo 10장 검증: 96 correction cells → 6~9 codebook patterns로 환원 (~10× 패턴 압축). raw 저장 모드는 backwards-compat로 유지.
- [x] **Phase 11 — hybrid_vae_decode tick 정렬 dynamic 호출:** `HybridVAEConfig`에 `residual_book` 포인터. decode Step 4b 추가: `ce_tick_sorted_indices(.., 1u<<CE_TYPE_RESIDUAL, ..)` → 정렬된 descriptors 순회 → `ce_residual_codebook_get` 패턴 lookup → blend_y에 행 단위 누적. 정렬 우선순위 B>G>R>A는 사용자 명시 의도(layer가 가장 슬로우, amp가 mix strength). cap = 단일 패턴 64. `residual_book == NULL`이면 step 4b 건너뜀 → 기존 caller 영향 없음.
- [x] **Phase 12 — E2E 통합:** `tests/test_residual_decode.c` (8 PASS — encode → codebook 채움 → 같은 storage에 descriptor 추가 → decode 두 번(off/on) 비교, 출력 픽셀 차이 4867개, max_dev 78). 전체 회귀: ce_core 20/21 (test_slig_signal 동일), 상위 19/19. PIPELINE.md 갱신.
- [x] **Phase 13 — Residual descriptor (x, y) positional 필드:** keyframe[6..7]=x_lo/x_hi, [8..9]=y_lo/y_hi (uint16 little-endian). `ce_residual_storage_add(.., uint16_t x, uint16_t y)`, `ce_residual_storage_unpack(.., uint16_t *x, uint16_t *y)`. masked_train의 `store_cells_with_codebook`가 cell의 SligSignal `origin_x/origin_y`를 읽어 stamp. hybrid_vae_decode Step 4b를 행 단위 누적 → 8×8 가우시안 weight 패치 적용 (중심 (px&0xFF, py&0xFF) 기준 -3..+4 윈도우, edge clamp). test_residual_codebook 27/27 (positional roundtrip + high-byte 16-bit 검증), test_residual_decode 8/8 (diff_pixels 984, max_dev 77). 회귀 ce_core 20/21, 상위 19/19.

**SSS 이미지 생성 엔진 리팩토링 (5-phase)**
- [x] **Phase 1 — 데드코드 정리:** `ce_memo_apply/remove`, `ce_hint_apply` (+`CEHintLayer`, `CEHintType`, `CE_HINT_*`), `ce_audio_load/free/energy_at` (+`CEAudioTrack`), `ce_upscale` (+`CEHiresGrid`), `ce_generate_image` (구 API), `ce_generate_upscale`, `ai_generate_image` (v1) 모두 삭제. `ce_denoise_loop` 시그니처에서 audio 파라미터 제거. 데드 테스트(`tests/bench_full.c`, `tests/probe_task_a.c`, `tools/train_images.c`, `requirements-gpu.txt`, `sss prompt v4 split.zip`) 삭제. 샘플 이미지 → `data/samples/`. `ce_inpaint`, `CEInpaintMask`, `ce_sample`, `image_to_grid`, `grid_to_image` 유지.
- [x] **Phase 2 — Hybrid VAE 통합:** `ce_core/ce_hybrid_vae.c/h` 추가. `hybrid_vae_encode`로 1024 블록 도장 + 9개 SLIG 셋 + audio_bins canvas_id 링크. `hybrid_vae_decode`로 색 base × 0.4 + 형태 detail × 0.6 블렌딩 + wave refine. `test_hybrid_vae` 16/16 통과 (PSNR solid 6.3 dB, synth 13.2 dB). 한계: SLIG sets는 .ces에 영속 안 됨(audio_bins 링크만). hybrid_vae cfg ticks 필드는 placeholder (`(void)cfg;`).
- [x] **Phase 3 — 마스크 학습 루프:** `ce_core/ce_masked_train.c/h` 추가. PROGRESSIVE 전략(0~33% correction-only / 33~67% residual+correction / 67~100% random 50%). `test_masked_train` 14/14 통과.
- [x] **Phase 4 — 학습 도구 통합:** `train_demo --masked-epochs N` 추가, `gen_image_ce --hybrid --guidance N.N` 추가, `tools/verify_hybrid.c` 신규 (PPM 배치 PSNR). 데모 데이터셋 10장 검증: train_demo(masked-epochs=3) → demo_masked.ces, gen_image_ce 양쪽 경로 모두 PPM 출력. verify_hybrid 평균 9.80 dB.
- [x] **Phase 5 — 통합 테스트 + 회귀:** `tests/test_image_gen.c`에 hybrid_vae roundtrip + masked_train E2E 추가 (총 68 PASS / 0 FAIL). 회귀: ce_core 17/18 (test_slig_signal은 베이스라인 환경 의존 fail로 Phase 1 이전부터 동일), 상위 19/19, test_stress_10k 12K 엔트리 PASS.

**Phase 5 끝나는 시점에서 통합 흐름 다이어그램**

```
학습:                                                     생성:
PPM 256×256                                                형태소(prompt)
    ↓                                                         ↓
image_to_grid → SpatialGrid                              morpheme_tokenize_clause
    ↓                                                         ↓
  RGBA buffer                                              ce_search_by_type(TEXT)
    ↓                                                       → canvas_id 투표
[A] ce_storage_ingest_rgba_16 ─→ 1024 IMG entries (16×16)    ↓
[B] hybrid_vae_encode          ─→ +블록도장 + SLIG 9셋     ┌────────────────────┐
       └─ codebook_add_or_lookup                            │ default (canvas_routed) │
       └─ dec.A.audio_bins ← canvas_id (cross-link)          │ ce_generate_image_      │
[C] (--masked-epochs N) masked_train_image                   │  canvas_routed:         │
       └─ slig_decompose_v2 → basis/residual/correction       │  denoise + wave refine  │
       └─ ce_denoise_loop → predict masked cells             │  on 16×16 atomic targets │
       └─ ce_compute_loss → ce_update_params                 └────────────────────┘
       └─ store_cells_masked                                 ┌────────────────────┐
[D] morpheme_tokenize → TEXT entries (canvas_id 공유)        │ --hybrid            │
                                                              │ hybrid_vae_decode:  │
                                                              │  base(블록 도장 색) │
                                                              │  + detail(SLIG, 현재  │
                                                              │  zero set) + wave   │
                                                              │  refine + guidance  │
                                                              └────────────────────┘
                                                                       ↓
                                                                256×256 PPM
```

검증 도구 흐름:
```
verify_hybrid <img1.ppm> [<img2.ppm> ...]
  → 각 이미지를 hybrid_vae_roundtrip 통과
  → 평균 PSNR 출력
```

### 후속 작업으로 분리

| 항목 | 분리 사유 |
|---|---|
| **masked-train epoch 50~200 × 1000+장 데이터셋 학습 검증** | Phase 12까지 인프라 완료. 다음은 데이터셋 준비 → 대규모 학습 검증. 사용자 지시: 전체 작업 완료 후 진행. |
| **CE_TYPE_RESIDUAL descriptor positional 정보 추가** | 현재 descriptor는 (idx, strength, tick)만 carry. 행 단위 누적 적용은 heuristic. positional 적용을 위해 descriptor에 (x, y) 필드 또는 RGBA tick 외 추가 메타 필요. |
| **hybrid_vae_decode 행 단위 → 패치 단위 적용** | 1000+장 학습 후 amplitude cap (현재 64), 부호 분기 정책 재조정 검토. |
| `ControlNet edge/depth/pose/color` | 사용자 지시: depth/edge map을 audio_bins 인프라로 전송하는 통합 설계로 검토 (별도 PR) |
| `Audio track: 음악 → 이미지` | 동영상 생성 경로용 (별도 PR). WAV 로더 + 스펙트럼 + 영상 페어 데이터셋 필요 |
| `cfg_scale 자동 도출` (POS 분포) | 현재 hardcoded preset 사용. 형태소 배열의 ADJ/NOUN 비율 → cfg_scale 매핑은 별도 작업 |
| `.ces 델타 압축 (varint/zlib)` | `train_loop_bench` 결과: spatial-prev 델타가 21× 작음 → 압축 시 disk 70%↓ |
| `ce_storage_ingest_rgba(..., dedup_eps)` 옵션 | 메모리 제약 시나리오 (eps=50 → 278× 압축, recall 1/3 비용) |

### 훈련 루프 비교 (`tools/train_loop_bench`)

100 이미지 × 1024 블록 = 102,400 엔트리 합성 데이터셋:

| 변형 | entries | mean\|delta\| | recall | disk |
|---|---:|---:|---:|---:|
| **A spatial-prev** (현재 production) | 102,400 | 3.46 | 99% | 14.3 MB |
| B zero-anchor (delta = fresh) | 102,400 | 74.40 | 99% | 14.3 MB |
| C dedup eps=50 + spatial-prev | 368 | 15.13 | 33% | 51.5 KB |
| C dedup eps=200 + spatial-prev | 16 | 43.53 | 7% | 2.3 KB |

**결론:**
- **A 유지가 정답** (델타가 B의 21× 작음). 현재 .ces가 raw 140B/entry라 disk는 같지만, 델타 압축을 추가하면 A가 압승.
- **C dedup**은 메모리 제약 시나리오에서 옵션 제공 가치: eps=50으로 278× 압축, 단 recall 1/3.
- B는 채택 가치 없음.

후속 작업 후보:
1. `.ces` 델타 압축 (varint 또는 zlib) → A 변형 disk 사이즈 70%↓ 가능
2. `ce_storage_ingest_rgba_v2(..., uint32_t dedup_eps)` 옵션 인자 추가 (eps=0이면 현재 동작)

### 학습 스케일

`test_stress_10k` 측정 (CE_TYPE_IMAGE 엔트리, single thread, -O2):

| N | ingest | search (k=8) | save | load | generate (4-step + wave 30) |
|---:|---:|---:|---:|---:|---:|
| 12,000  | 5 ms (2.66M b/s) | 0.2 ms | <1 ms | <1 ms | 60 ms |
| 100,000 | 60 ms (1.69M b/s) | 1.5 ms | 100 ms | 130 ms | 90 ms |

→ 10K 학습 최소요건은 압도적으로 만족. 1M 엔트리(∼1000 이미지 × 1024 블록) 까지도 generate 0.1s 수준.

---

## `.sfb` Feature Bank (Phase 2 산출물)

`ce_core/sss_feature_bank.{h,c}` + `tools/sss_feature_bank.py` 추가로
**`.sfb`** (SSS Feature Bank) 포맷을 도입. Phase 3에서 학습 산출물을
이 포맷으로 통합하기 위한 사전 작업이다.

| Record | Size (B) | Description |
|--------|---------:|-------------|
| Motif    | 2 864 | label[32] + row/col freq (128 bins f32) + RGB color_freq (3×128 f32) + 16×16 position heatmap (uint8 quantised, 0..255) + coherence/confidence/activation/variation_cluster_id |
| Relation |    20 | src/dst motif_id + (dx, dy) + relation_type (above/below/left/right/near/around/inside) + weight |
| Identity |   104 | label[32] + up to 32 motif_ids + motif_count + confidence |

핵심 설계 결정:
- **pack(1) + little-endian**: 호스트 LE 정적 assert로 검증. byte-swap 코드 0줄.
- **Header에 record_size 포함**: v2 motion 필드 확장 시 version 유지하면서
  record를 키울 수 있도록 `motif_record_size / relation_record_size /
  identity_record_size`를 헤더에 저장 → loader가 stride 단위로 읽음.
- **C ↔ Python byte-identical**: `tools/sss_feature_bank.py`와
  `ce_core/sss_feature_bank.c`는 동일 데이터에 대해 바이트 단위 동일한
  파일을 생성 (test_feature_bank.py가 lock).
- **position_heatmap uint8 양자화**: float[0,1] → uint8 [0,255] (save 시
  `*255` round, load 시 `/255`). 모티프당 1024B → 256B (75% ↓).
  라운드트립 오차 ≤ 1/255.

검증 (`tools/test_feature_bank.py`):
- 100×200×5 라운드트립 + 두 번째 save 바이트 결정성
- 한글 32B label 코드포인트 경계 잘림
- empty bank
- invalid relation src/dst, invalid relation_type, identity motif_count > 32
- bad magic / bad version 거부 (`SFB_ERR_MAGIC` / `SFB_ERR_VERSION`)
- Python save ↔ C save 바이트 동일성
- Python save → C load / C save → Python load 양방향 라운드트립
- **1000 motif × 5000 relation × 50 identity 벤치마크**:
  save 85 ms, load 34 ms, 파일 크기 ~2.83 MB (목표 < 100 ms 통과)

v2 마이그레이션 정책 (사전 정의):
- record_size 변경은 항상 SFB_VERSION 증가와 함께
- v1 reader는 알 수 없는 후행 record-내 바이트를 안전하게 스킵
- 새 필드는 항상 record 끝에 append (기존 offset 보존)
