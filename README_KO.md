# SSS — Spatial Pattern AI + CE-Cell Image Engine

![Main Hero](main_hero.png)

> 두 엔진이 한 코드베이스에서 같이 굴러갑니다.
> **텍스트**는 256×256 격자에 밝기 패턴으로 인코딩, **이미지**는
> 64-byte `CEUnit`으로 분해하고 RGBA 틱 시계 + 잔차 코드북으로 복원합니다.
> 두 엔진은 같은 저장소(`*.spai` 텍스트, `*.ces` 셀)와 같은 포맷 규약을
> 공유합니다.
>
> 모든 inference는 정수 연산입니다. float는 학습 시 DCT/SVD에만 사용됩니다.

```
   텍스트 엔진                            이미지 엔진
   ───────────                            ───────────
   "고양이가 밥을 먹는다."                 256×256 PPM
        │                                    │
   3-레이어 인코딩                          16×16 블록 도장 + SLIG 분해
        │                                    │
   256×256 RGBA 격자                        CEUnit 피라미드 (3 scale × 3 channel)
        │                                    │
   keyframe / Δ                             블록 + SLIG 엔트리 (CEStorage)
        │                                    │
   ai_save *.spai                           ce_storage_save *.ces
```

---

## 목차

- [왜 만드는가](#왜-만드는가)
- [텍스트 엔진 — Spatial Pattern](#텍스트-엔진--spatial-pattern)
- [이미지 엔진 — CE Cell](#이미지-엔진--ce-cell)
  - [16×16 블록 도장](#1-1616-원자-블록-도장)
  - [SLIG 신호 셀](#2-slig-신호-셀)
  - [Hybrid VAE](#3-hybrid-vae--encode--decode)
  - [마스크 학습](#4-마스크-학습)
  - [RGBA 틱 시계 + 잔차 코드북](#5-rgba-틱-시계--잔차-코드북)
  - [Tick 정렬 동적 디코드](#6-tick-정렬-동적-디코드)
- [E2E 데모](#e2e-데모)
- [검증 결과](#검증-결과)
- [빌드 & 실행](#빌드--실행)
- [저장 포맷](#저장-포맷)
- [프로젝트 구조](#프로젝트-구조)
- [로드맵](#로드맵)
- [라이선스](#라이선스)

---

## 왜 만드는가

전통 LLM은 언어를 고정 가중치 행렬에 굽고, 디퓨전 모델은 이미지를
같은 패턴을 latent 공간에 굽습니다. 둘 다 불투명하고, 새 데이터를
추가하려면 재학습이 필요합니다.

이 엔진은 그 반대입니다.

- **무한 파라미터** — 새 절/이미지가 들어오면 프레임 또는 셀로 누적,
  재학습 없이 디스크가 허용하는 만큼 확장됩니다.
- **무한 컨텍스트** — 디스크 한계가 곧 컨텍스트 한계.
- **검사 가능** — 텍스트 격자는 히트맵으로 열어볼 수 있고, 이미지
  셀은 64-byte 레이아웃이 문서화돼 있습니다.
- **증분 학습** — 새 절은 델타/키프레임 1건, 새 이미지는 1024개
  블록 도장 + 9개 SLIG 셋 + 잔차 코드북 패스 1번. 전부 추가만 합니다.
- **임베디드 친화** — Termux / Windows MSYS에서 그대로 동작합니다.
- **inference 정수 연산** — float는 학습 시 SVD/DCT에서만 등장.

대가: 어텐션 행렬 대신 바이트 단위 공간 통계 + 틱 시계로 신호가
충분한가에 베팅합니다. 현재 벤치마크 기준 "리트리벌 + 회상 + 결정론적
이미지 생성에는 유용", "트랜스포머 대체용은 아님" 정도입니다.

---

## 텍스트 엔진 — Spatial Pattern

원래의 코어. 절 하나가 256×256 RGBA 격자가 되고, 비디오 코덱 식 키프레임/델타
저장소를 거칩니다.

### 3-레이어 합산

| 레이어 | 대상 | 가중치 | 잡는 것 |
|---|---|---|---|
| **Base** | 모든 바이트 | +1 | 바이트 위치 |
| **Morpheme** | 명사/동사/형용사… 바이트 구간 | +3 | 형태소 구조 |
| **Word** | 공백 분리 단어 바이트 | +5 | 단어 단위 강조 |

`A` 채널 겹침 단계: `1 / 4 / 6 / 9`.
`"귀여운 고양이가 밥을 먹는다."` 검증: active 40 px, max 9, total 297
(`= 40 + 185 + 72`, 보존 법칙 OK).

### RGBA 채널

| 채널 | 타입 | 역할 | 계산 |
|---|---|---|---|
| **A** | u16 | 밝기 / 중요도 | 3-layer 합산 |
| **R** | u8 | 의미 | 형태소 POS 시드 + 대각선 확산 |
| **G** | u8 | 기능 | 형태소 POS 시드 + 수직 확산 |
| **B** | u8 | 확장 | 형태소 POS 시드 + 수평 확산 + EMA |

`update_rgb_directional`이 채널마다 자기 축으로 확산시키고, 엔진이
코퍼스 전체에 걸쳐 셀별 EMA를 유지합니다.

### Keyframe / Delta 저장

```
  첫 절                                   → 새 keyframe
  cosine-A 유사도 ≥ 0.3 (어떤 KF 대비)    → 델타 (그 부모 기준)
  유사도 < 0.3                            → 새 keyframe
```

`apply_delta`로 비트 단위 복원. `test_io`가 700절 라운드트립
`|sim_before − sim_after| < 1e-3` 검증.

### 매칭 캐스케이드

`spatial_match()` 통합 코어:

```
  Step 1 (조대 필터)  KF≥100이면 bucket index, 아니면 full overlap_score.
                      topk_select → top-K (K=8).
  Step 2 (정밀 매칭)  모드별 scorer:
                        PREDICT  → cosine_rgb_weighted
                        SEARCH   → cosine_a_only
                        QA       → rg_score    (0..1)
                        GENERATE → bg_score    (0..1)
```

모든 scorer가 `[0, 1]` 범위로 정규화돼 임계값 비교가 안전.

### 캔버스 풀 (자막 라우팅)

2048×1024 캔버스가 32 × 256² 절 슬롯을 타일링. `pool_match`가
질의 `DataType` (prose / dialog / code / short) 슬롯으로 먼저 점프.

---

## 이미지 엔진 — CE Cell

이미지 측은 같은 `CEStorage`를 재사용합니다 (한 파일 = `*.ces`).
모든 storage entry가 140 바이트 row이고 modality 태그가 붙습니다:

```c
typedef enum {
    CE_TYPE_TEXT     = 0,
    CE_TYPE_IMAGE    = 1,   /* 16×16 블록 도장 */
    CE_TYPE_AUDIO    = 2,
    CE_TYPE_SLIG     = 3,   /* (scale, channel) 별 SLIG cells */
    CE_TYPE_RESIDUAL = 4    /* codebook descriptor만 */
} CEType;
```

이미지 흐름은 `블록 도장 → SLIG 분해 → 코드북 → 틱 정렬 디코드`.

### 1. 16×16 원자 블록 도장

`ce_storage_ingest_rgba_16`이 256×256 이미지를 16 × 16 블록(=256개)으로
자르고, 각 블록이 `ce_feed_image_16`을 거쳐 **4 quadrant CEUnit**
(TL/TR/BL/BR)을 생성. 256 × 4 = 이미지 1장당 **1024개 `CE_TYPE_IMAGE`
entry**. 모두 같은 `canvas_id` 공유. `slot = block-row`,
`block_idx = (block-col << 2) | quadrant`.

### 2. SLIG 신호 셀

`slig_decompose_channel`이 이미지의 YCbCr 평면에 대해 (scale_level, channel)별로 분해:

- `SLIG_LEVEL_COARSE` (32×32), `MID` (128), `FINE` (256)
- `SLIG_CH_Y / CB / CR`

3 × 3 = **이미지 1장당 9 셋**. 각 셋 최대 32 cells, 셀 1개 = 64-byte
`CEUnit` (inc.G/B에 16-coefficient DCT, inc.R 메타, inc.A 에너지/오디오 빈,
dec 절반에 이벤트/오디오 링크).

영속화 (Phase 6+): `ce_storage_persist_slig_set`이 각 셀을
`CE_TYPE_SLIG` row로 (`slot = scale_level*3 + channel`,
`block_idx = cell index`) 기록. `ce_storage_load_slig_sets`가 같은
`canvas_id`에서 9-그리드 재구성.

### 3. Hybrid VAE — encode + decode

`ce_hybrid_vae.{c,h}`가 두 경로를 한 API로 묶음:

```c
HybridEncodeResult res;
hybrid_vae_encode(&res, &storage, &codebook, rgba, w, h, canvas_id);
//  → 1024 블록 도장 (CE_TYPE_IMAGE)
//  → 9 SLIG 셋   (CE_TYPE_SLIG, audio_bins로 cross-link)
//  → 9 codebook 인덱스

uint8_t out_rgb[256*256*3];
hybrid_vae_decode(out_rgb, &storage, canvas_id, res.sets, &cfg);
//  base   = 블록 도장 색 복원
//  detail = SLIG 잔차 체인 (coarse → mid → fine, 채널별)
//  blend(base, detail) + wave refine + CFG guidance + YCbCr→RGB
```

encode가 두 경로를 연결합니다 — 각 SLIG 셀의 `dec.A.audio_bins`에
`canvas_id`가 박혀 있어, 셀 검색 시 어떤 블록 도장 셋이 그 셀을
만들었는지 역추적 가능.

### 4. 마스크 학습

`ce_masked_train.{c,h}`가 latent에게 "빠진 셀을 채우는 법"을 가르칩니다:

```
  1. slig_decompose_v2 → SligDecomposed
       cells[0..structure_end)         basis
       cells[structure_end..edge_end)  residual edge
       cells[edge_end..texture_end)    residual texture
       cells[texture_end..color_end)   correction (color)
       cells[color_end..num_cells)     correction (event)

  2. correction 영역 (또는 추가) 마스킹
  3. ce_denoise_loop가 마스크된 자리 예측
  4. ce_compute_loss(predicted, original)
  5. ce_update_params로 CEGenConfig 조정
  6. 수렴 셀을 storage에 다시 저장
```

PROGRESSIVE 마스크 일정:

```
  epoch 0..33%   correction-only 마스크 (쉬움)
  epoch 33..67%  residual + correction 마스크 (중간)
  epoch 67..100% 랜덤 50% 마스크 (어려움)
```

### 5. RGBA 틱 시계 + 잔차 코드북

이미지 셀은 암묵적인 순서를 carry합니다 — 모든 `CEStorageEntry`가
`(slot, block_idx)`와 `audio_amps`에서 `TickRGBA`로 매핑됩니다:

| 채널 | 의미 | 출처 |
|---|---|---|
| **R** | 셋/블록 안 cell 인덱스 | `block_idx` |
| **G** | scale 레벨 (0=COARSE, 1=MID, 2=FINE) | `slot / 3` |
| **B** | render 레이어 / 채널 (Y / Cb / Cr) | `slot % 3` |
| **A** | amplitude / 잔차 강도 | `audio_amps` max + `sigma>>8` |

Carry chain은 `slig_tick_math::tick_add(plus, 1)`과 동일:

```c
void ce_tick_step(TickRGBA *t) {
    if (++t->r == 0) {        /* R 255→0 wrap */
        if (++t->g == 0) {    /* G로 carry    */
            if (++t->b == 0) {/* B로 carry    */
                ++t->a;       /* A로 carry    */
            }
        }
    }
}
```

`ce_tick_compare`는 `B > G > R > A` 우선순위로 정렬하므로,
`ce_tick_sorted_indices`는 storage를 렌더 순서로 walk합니다 —
레이어별, coarse→fine, 셀 인덱스 순, 동률이면 약한 amplitude 먼저.

**잔차 코드북** (`ce_residual_codebook.{c,h}`)은 256-entry 사전,
scale_level로 분리 (chroma와 luma가 충돌하지 않게):

```c
typedef struct {
    CEUnit   unit;            /* correction CEUnit */
    uint8_t  scale_level;     /* SligScaleLevel (0xFF = wildcard) */
    uint8_t  direction;       /* SligDir */
    uint8_t  strength;        /* 0..255 */
    TickRGBA tick;
    uint32_t used_count;
} CEResidualCode;
```

Storage descriptor (`CE_TYPE_RESIDUAL`)는 keyframe 64B를 재해석:

```
  bytes[0]    = codebook_idx
  bytes[1]    = strength
  bytes[2..5] = TickRGBA (R, G, B, A)
  bytes[6..7] = x 위치 (u16 LE)
  bytes[8..9] = y 위치 (u16 LE)
  bytes[10..] = reserved (zero)
```

**원본 patch는 storage에 저장 안 함**. `train_demo --residual-codebook`
켜면 `masked_train_image`가 각 correction 셀을 `ce_residual_codebook_add_or_lookup`
(threshold 기반)으로 환원. data/demo 10장 데이터셋에서 **96 correction
cells → 6~9 codebook patterns** (encode 측 패턴 ~10× 압축).

### 6. Tick 정렬 동적 디코드

`hybrid_vae_decode` Step 4b가 `CE_TYPE_RESIDUAL` entries를 tick 순서
(B > G > R > A)로 walk → 패턴 lookup → `(x & 0xFF, y & 0xFF)` 중심으로
**8×8 가우시안 weight 패치**를 `blend_y`에 stamp. 단일 패턴 amplitude
cap = 64. `tick.g` 패리티로 부호 분기 (양/음 보정).

```
  먼저 그려진 cell  (낮은 B, 낮은 G)  → coarse 구조
  중간에 그려진 cell (중간 G)         → 텍스처 디테일
  마지막 cell      (높은 G)          → 고주파 잔차
  amplitude (A) 동률 깰 때           → 약한 것 먼저, 강한 stamp가 위에 덮어씀
```

이건 기존 fixed-phase 디코드를 storage에서 직접 읽은 per-cell ordering으로
대체합니다. `cfg.residual_book == NULL`이면 기존 동작 그대로 (호환성 보존).

---

## E2E 데모

```bash
make demo_tools                            # make_demo_dataset, train_demo,
                                           # gen_image_ce, verify_hybrid 빌드

./build/make_demo_dataset data/demo        # 10 PPM 이미지 + labels.tsv

# joint text+image 학습
./build/train_demo data/demo build/models/demo
#   IMG=10240        블록 도장
#   TEXT=38          per-morpheme 브릿지
#   HYB_blocks=10240 (= IMG)
#   HYB_cells=648    hybrid_vae_encode가 추가한 SLIG 셀

# 마스크 학습 + 잔차 코드북 활성
./build/train_demo data/demo build/models/demo_cb \
    --masked-epochs 5 --residual-codebook
#   masked-train summary: stored_cells=253 residuals=96
#   codebook=6~9 patterns (correction 셀 ~10× 패턴 압축)

# 생성 (canvas-routed 기본 경로)
./build/gen_image_ce build/models/demo.ces "red apple" \
    build/red_apple.ppm 0 50 200

# 생성 (hybrid VAE 경로)
./build/gen_image_ce build/models/demo_cb.ces "red apple" \
    build/red_apple_hybrid.ppm 0 50 200 --hybrid --guidance 1.5

# PSNR 검증 (PPM 폴더)
./build/verify_hybrid data/demo/colors/*.ppm data/demo/fruits/*.ppm
#   파일별 dB + 배치 평균 출력
```

`gen_image_ce` 흐름:
1. 프롬프트 morpheme 토큰화
2. `ce_search_by_type(CE_TYPE_TEXT, …)`로 winning `canvas_id` 투표
3. (기본) `ce_generate_image_canvas_routed` — denoise + decode + 16×16
   atomic wave-refine, **또는**
   (`--hybrid`) `hybrid_vae_decode` — 블록 도장 base + SLIG detail
   (`ce_storage_load_slig_sets`로 storage에서 로드) + 옵션 codebook 패치 +
   wave refine + guidance.

---

## 검증 결과

`make test` 상위 19/19 통과. `ce_core` 20/21 통과 (`test_slig_signal`은
Phase 1 시작 전부터 동일하게 fail하던 Windows MSYS file-write 환경 의존
문제로, 이번 작업과 무관).

### 테스트 surface

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
    test_image_gen       68/68   (hybrid VAE + masked-train E2E 포함)
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
    test_residual_decode    8/8   (codebook on/off → blend_y 변화)
    test_slig_signal       30/33   (3 fail: Windows MSYS file-write only)
```

### Hybrid VAE roundtrip (합성 이미지)

```
  test_hybrid_vae:
    solid 2×2     → block_entries=1024, total_cells=14, PSNR  6.3 dB
    synth 그래디언트 → PSNR 13.2 dB
    base-only     → 비-blank
    detail-only   → 비-blank
    wave refine   → PSNR > 5 dB
```

### 데모 파이프라인 (10장, seed=0)

| 프롬프트 | 라우팅 캔버스 | 평균 RGB | 결과 |
|---|---|---:|---|
| `"red apple"`       | apple     | (196, 38, 38)  | red ✓ |
| `"yellow banana"`   | banana    | (213, 195, 46) | yellow ✓ |
| `"purple grape"`    | grape     | (118, 42, 146) | purple ✓ |
| `"green lime"`      | lime      | (38, 195, 38)  | green ✓ |
| `"blue blueberry"`  | blueberry | (38, 39, 196)  | blue ✓ |
| `"orange fruit"`    | orange    | (220, 140, 41) | orange ✓ |

6/6 모두 정확. `"orange fruit"`처럼 dataset에 없는 단어가 섞여도
morpheme 투표가 sharp falloff (`1/(1+distance)`)으로 정확히 라우팅.

### 마스크 학습 (데모셋)

```
  --masked-epochs 5 --residual-codebook
  ────────────────────────────────────
  per image: epochs_run=5, final_loss≈80, cells_stored=25, residuals≈10
  batch:     stored_cells=253, residuals=96, codebook=6~9 patterns
```

수렴 임계 (`8.0`)을 넘는 loss는 예상대로 — 5 epoch × 10장은 학습 양 부족.
test_residual_decode가 codebook 레이어 동작 확인 (`diff_pixels=984`,
max_dev=77).

### 스트레스

`test_stress_10k`:

```
  N = 12,000
    ingest:   5 ms       (2.66M blocks/s)
    search:   0.2 ms     (k=8)
    save:     <1 ms
    load:     <1 ms
    generate: 60 ms      (4-step + 30 wave refines)
  N = 100,000
    ingest:   60 ms
    search:   1.5 ms
    save:    100 ms
    load:    130 ms
    generate: 90 ms
```

---

## 빌드 & 실행

```bash
make all                  # 모든 .o + 테스트/데모 바이너리
make test                 # 상위 엔진 268 케이스
make -C ce_core all       # ce_core 엔진 + 테스트
make -C ce_core test      # ce_core 회귀
make demo_tools           # train_demo / gen_image_ce / verify_hybrid
```

**필요 환경:** GCC ≥ 9 (C11), Make, POSIX-ish shell. Windows에선
MSYS2 `/usr/bin/gcc` 권장 (MinGW fork는 `test_ingest`에서 POSIX-mkdir
이슈 있음).

### 스트리밍 텍스트 학습

```bash
make stream
./build/stream_train --input data/kaggle_train.txt \
                     --max 25000 \
                     --save build/models/wiki25k.spai \
                     --checkpoint 5000 \
                     --verify
```

### 이미지 학습

```bash
# PNG/JPEG → 256×256 PPM
python3 tools/png_to_ppm256.py data/samples/IMG_0304.png \
    build/training/IMG_0304.ppm
make image_tools
gcc -Wall -O2 -Iinclude tools/jpeg_to_ppm256.c \
    -o build/jpeg_to_ppm256 -ljpeg
./build/jpeg_to_ppm256 data/samples/IMG_0305.jpeg \
    build/training/IMG_0305.ppm

# CE-cell 학습 (구 SpatialAI-only 학습 도구 대체)
make train_images_ce
./build/train_images_ce build/training/img_model.ces \
    build/training/IMG_0304.ppm \
    build/training/IMG_0305.ppm
```

### `verify_hybrid`

PPM → `hybrid_vae_roundtrip` → 파일별 PSNR + 배치 평균. encode/decode
회귀 추적용.

```bash
make demo_tools
./build/verify_hybrid data/demo/colors/*.ppm data/demo/fruits/*.ppm
```

---

## 저장 포맷

### `.spai` — 텍스트 엔진 상태

매직 `SPAI`, 현재 버전 **6**. v3/4/5는 자동 호환 (필드 누락 시 0).

```
  헤더 32B    magic + version + kf_count + df_count + reserved
  레코드*     태그 스트림:
    0x01 Keyframe    id, label, A/R/G/B
    0x02 Delta       id, parent, sparse entries (v6: + cell_deltas[])
    0x03 Weights     ChannelWeight (4× float)
    0x04 Canvas      slot_count, type, parent, A/R/G/B
    0x05 Subtitle    type/topic_hash/canvas_id/slot_id 표
    0x06 EMA         R/G/B/count per cell
    0x07 SeqMeta     per-canvas sequence id + timestamp
    0x4A Codebook    SligCodebook patterns (v2.3)
    0x4B ImageIdx    per-keyframe (3×3) codebook 인덱스
```

### `.ces` — CE-Cell 저장소

포맷 v2 (현재). entry당 **140 바이트**:

```
   4 B  canvas_id          uint32
   2 B  slot               uint16
   2 B  block_idx          uint16
   1 B  type               CE_TYPE_* (0..4 사용 중)
   3 B  reserved (zero)
  64 B  keyframe           CEUnit
  64 B  delta              CEUnit
```

`CE_TYPE_SLIG=3`, `CE_TYPE_RESIDUAL=4` 추가는 forward-compatible —
type 필드가 1 바이트라 254개 modality까지 지원 가능. 옛 reader가
새 값을 모르면 그 entry만 unknown으로 두고 나머지를 정상 로드.

`CE_TYPE_RESIDUAL`는 64-byte `keyframe`을 descriptor로 재해석 (raw
patch 없음). 바이트 레이아웃은 [§5](#5-rgba-틱-시계--잔차-코드북) 참조.

### Save / load API

```c
SpaiStatus ai_save(const SpatialAI* ai, const char* path);
SpatialAI* ai_load(const char* path, SpaiStatus* out_status);
SpaiStatus ai_save_incremental(const SpatialAI* ai, const char* path);
SpaiStatus ai_peek_header(const char* path, ...);

int ce_storage_save(const CEStorage *s, const char *path);
int ce_storage_load(CEStorage *out, const char *path);
```

`test_io`가 700절 라운드트립 `|sim_before − sim_after| < 1e-3` 검증.
`ai_save_incremental`은 모델이 디스크보다 작아지면 거부 (안전).

---

## 프로젝트 구조

```
├── include/                    공개 헤더 (텍스트 엔진)
│   ├── spatial_grid.h          256×256 RGBA 격자
│   ├── spatial_layers.h        3-레이어 합산
│   ├── spatial_morpheme.h      한국어 longest-match 분석기
│   ├── spatial_keyframe.h      Keyframe / 델타 / SpatialAI
│   ├── spatial_match.h         spatial_match() 통합 코어
│   ├── spatial_context.h       LRU 프레임 캐시
│   ├── spatial_canvas.h        2048×1024 캔버스 + 32 슬롯
│   ├── spatial_subtitle.h      SubtitleTrack + 캔버스 풀
│   ├── spatial_generate.h      next-clause refine
│   ├── spatial_image.h         image_to_grid / grid_to_image +
│   │                           ai_generate_image_v2_guided / animation
│   └── spatial_io.h            .spai 바이너리 포맷
├── src/                        텍스트 엔진 구현
│   ├── spatial_grid.c / layers.c / morpheme.c / keyframe.c /
│   │   match.c / context.c / canvas.c / subtitle.c / recluster.c /
│   │   generate.c / io.c / image.c
│   └── spatial_image_gen.c     SLIG v2.3 생성기
├── ce_core/                    CE Cell 엔진
│   ├── ce_core.{c,h}           64-byte CEUnit primitives
│   ├── ce_type.h               CEType (TEXT/IMAGE/AUDIO/SLIG/RESIDUAL)
│   ├── ce_storage.{c,h}        CEStorage + 16×16 ingest + SLIG 영속 +
│   │                           tick-sorted iter
│   ├── ce_storage_io.{c,h}     .ces v2 reader/writer
│   ├── ce_search.{c,h}         top-k 검색, by-type 필터
│   ├── ce_engine.{c,h}         UNet 등가 ops (Down/Conv/Attn/Up)
│   ├── ce_denoise.{c,h}        50-step denoise loop
│   ├── ce_decode.{c,h}         latent → image / text
│   ├── ce_extend.{c,h}         sampler + inpaint (CEInpaintMask)
│   ├── ce_gen.{c,h}            top-level 생성 API
│   ├── ce_feed_image.{c,h}     8×8 / 16×16 블록 인코더
│   ├── ce_image_wave_refine.{c,h}  0.01→3→0.01 wave refine
│   ├── ce_hybrid_vae.{c,h}     hybrid encode + decode + roundtrip
│   ├── ce_masked_train.{c,h}   PROGRESSIVE 마스크 일정
│   ├── ce_residual_codebook.{c,h}  256-entry correction 사전
│   ├── ce_tick.h               TickRGBA carry chain (header-only)
│   ├── slig_signal.{c,h}       SligSignal + decompose + canvas
│   ├── slig_codebook.{c,h}     (scale, channel) 패턴 사전
│   ├── slig_pipeline.{c,h}     v3 5-stage 독립 파이프라인
│   ├── slig_tick_math.{c,h}    32-bit packed tick + sin/cos/gauss 표
│   └── slig_material_harmonic.{c,h}  Mat-1/2/4 자동 추출
├── tests/                      상위 엔진 테스트 (19 binaries)
├── ce_core/tests/              ce_core 테스트 (21 binaries)
├── tools/
│   ├── train_demo.c            joint text+image 학습기
│   │                           (--masked-epochs N, --residual-codebook)
│   ├── gen_image_ce.c          프롬프트 → 256×256 PPM
│   │                           (기본 canvas-routed, --hybrid --guidance N)
│   ├── verify_hybrid.c         hybrid_vae_roundtrip PSNR 배치
│   ├── train_images_ce.c       단일 이미지 CE 학습기
│   ├── stream_train.c          line-by-line 텍스트 학습기
│   ├── chat.c                  대화형 REPL
│   ├── make_demo_dataset.c     data/demo 생성
│   ├── img2grid.c / grid2img.c PPM ↔ SpatialGrid roundtrip
│   └── png_to_ppm256.py / jpeg_to_ppm256.c 이미지 변환기
├── data/
│   ├── samples/                IMG_0304.png, IMG_0305.jpeg, IMG_0306.jpeg
│   └── demo/                   procedurally 생성된 PPM + labels.tsv
├── PIPELINE.md                 이번 branch의 phase 로그
├── SPEC.md / SPEC-ENGINE.md    역사적 spec
└── README.md / README_KO.md
```

---

## 로드맵

이번 branch에서 완료 (`PIPELINE.md`에 13-phase 로그):

- [x] 데드코드 정리 (`ce_memo`, `ce_hint`, `ce_audio`, `ce_upscale`,
      구 `ce_generate_image`, `ai_generate_image` v1).
- [x] Hybrid VAE (encode + decode + roundtrip), 16/16 통과.
- [x] 마스크 학습 (PROGRESSIVE 마스크 일정), 14/14 통과.
- [x] `train_demo --masked-epochs N`, `gen_image_ce --hybrid --guidance N`,
      `verify_hybrid`.
- [x] `CE_TYPE_SLIG` / `CE_TYPE_RESIDUAL` modality 추가, `.ces` v2
      forward-compatible.
- [x] SLIG cellset 영속 + tick-sorted storage iter (`ce_tick.h`).
- [x] 잔차 코드북 (256-entry, scale-bucketed, used_count tracked).
- [x] `train_demo --residual-codebook` correction 셀 ~10× 압축.
- [x] hybrid VAE decode가 `CE_TYPE_RESIDUAL`을 tick 순으로 적용 +
      positional `(x, y)` 8×8 가우시안 stamp.

대기 (사용자 지시: 1000+장 corpus 검증 후 진행):

- 대규모 데이터셋 실험 (epochs 50–200 × 1000+ 이미지), residual
  threshold + amplitude cap 재조정.
- `CE_TYPE_RESIDUAL` descriptor에 frame별 `direction` / `scale` 필드 추가.
- `hybrid_vae_decode` patch 정책 검토 (현재 8×8 고정 → `cfg.patch_radius`).

---

## 라이선스

[LICENSE](LICENSE) 참조.
