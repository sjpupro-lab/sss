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
| Wave Refine | ce_image_wave_refine (0.01→3→0.01 사이클) | ce_image_wave_refine.c | TODO 완전 미연결 |
| Tick 정수엔진 | TICK_SIN/COS/GAUSS_TABLE + WAVE_STEPS[7] | slig_tick_math.c | WARN 부분사용 |
| Material 텍스처 | slig_material_harmonic Mat-1/2/3/4 | slig_material_harmonic.c | WARN spai v6에서만 |
| Audio Sync | ce_audio_load + ce_audio_energy_at | ce_extend.c | TODO 데드코드 |
| Keyframe/Delta 저장 | ai_store_auto_with_image | spatial_keyframe.c | OK |
| CEStorage 훈련 | ce_storage_add_typed | ce_storage.c | TODO 훈련 시 미호출 |
| 형태소→이미지 검색 | ce_search_by_type(CE_TYPE_IMAGE) | ce_search.c | TODO 미연결 |

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

## TODO (부족한 부분)

- [ ] ce_feed_image 16×16 블록 버전 추가
- [ ] ce_denoise 스텝 파라미터화 (기본 50)
- [ ] dec.A[2](energy_ratio) → apply_global amp 변조 연결
- [ ] cond_threshold 분해 시 실제 에너지비율로 설정
- [ ] slig_decompose_structure 방향 HORIZONTAL→SVD 방향 계산
- [ ] CEStorage ↔ SpatialAI 브릿지 (.spai + .ces 동시 저장)
- [ ] ce_image_wave_refine 생성 파이프라인에 연결
- [ ] LoRA (ce_memo): 스타일 파인튜닝 인터페이스
- [ ] ControlNet (ce_hint): 엣지/깊이 조건 입력
- [ ] Audio track: 음악→이미지 생성 경로
