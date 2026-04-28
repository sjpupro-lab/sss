# SSS Image Generation Pipeline

## Diffusion → SSS 1:1 대응

| Diffusion 단계 | SSS 구현 | 파일 | 상태 |
|---|---|---|---|
| Text Tokenizer | morpheme_tokenize_clause | spatial_morpheme.c | OK |
| Text Embedding | ce_feed(token) → CEUnit | ce_core.c | OK |
| Text→Image Cross Attn | ce_cross_attention | ce_engine.c | TODO CEStorage 미연결 |
| VAE Encode (이미지→latent) | ce_feed_image 8×8→CEUnit | ce_feed_image.c | WARN 블록 작음 |
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
| LoRA | ce_memo_apply | ce_extend.c | TODO 데드코드 |
| ControlNet | ce_hint_apply (EDGE/DEPTH/POSE/COLOR) | ce_extend.c | TODO 데드코드 |
| Inpainting | ce_inpaint + CEInpaintMask | ce_extend.c | TODO 데드코드 |
| Upscaler | slig_upscale_harmonic/sbr/wiener/cs | slig_pipeline.c | WARN v3에서만 |
| Freq 분해 | slig_decompose_v2 (구조/엣지/텍스처/색상/이벤트) | slig_pipeline.c | OK |
| Spatial Attn Mask | slig_apply_masked + mask_left/right/top/bottom | slig_signal.c | TODO 미연결 |
| Atmos 오브젝트 배치 | slig_apply_masked per 형태소 | slig_signal.c | TODO 미연결 |
| Wave Refine | ce_image_wave_refine (0.01→3→0.01 사이클) | ce_image_wave_refine.c | OK ce_generate_image_typed에 연결됨 |
| Tick 정수엔진 | TICK_SIN/COS/GAUSS_TABLE + WAVE_STEPS[7] | slig_tick_math.c | WARN 부분사용 |
| Material 텍스처 | slig_material_harmonic Mat-1/2/3/4 | slig_material_harmonic.c | WARN spai v6에서만 |
| Audio Sync | ce_audio_load + ce_audio_energy_at | ce_extend.c | TODO 데드코드 |
| Keyframe/Delta 저장 | ai_store_auto_with_image | spatial_keyframe.c | OK |
| CEStorage 훈련 | ce_storage_add_typed | ce_storage.c | OK ce_storage_ingest_rgba / train_images_ce |
| 형태소→이미지 검색 | ce_search_by_type(CE_TYPE_IMAGE) | ce_search.c | OK ce_generate_image_typed에 연결됨 |

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

### 후속 작업으로 분리

| 항목 | 분리 사유 |
|---|---|
| `LoRA (ce_memo) 스타일 파인튜닝` | 사용자 지시: 신규 자세/이미지 학습 시 형태소 관계·품사·RGBA 중요도에 따라 CE cell이 자동 추가됨 — 별도 API 불필요. 데드코드 `ce_memo` 정리 검토만 |
| `ControlNet (ce_hint) edge/depth/pose/color` | 사용자 지시: depth/edge map을 audio track 인프라로 전송하는 통합 설계로 검토 (별도 PR) |
| `Audio track: 음악 → 이미지` | 동영상 생성 경로용 (별도 PR). WAV 로더 + 스펙트럼 + 영상 페어 데이터셋 필요 |
| **Cross-modal 텍스트↔이미지 정렬** | E2E에서 발견 — "blue" 프롬프트가 blue KF로 안 옴. CLIP-style joint training 또는 라벨 키 인덱스 필요 |
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
