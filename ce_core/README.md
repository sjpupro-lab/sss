# ce_core — CE Cell 결정론적 생성 엔진

64바이트 신호 셀(`CEUnit`) 기반 CPU-only 생성 엔진. 학습된 keyframe 풀에서
modality 일치 항목을 retrieval 한 뒤, **CE Whole-Block Carry Alignment** (전체
grid 동시 carry-tick 정렬)로 seed를 target에 점진적으로 끌어당기면서도 seed
고유 패턴을 보존한다. 같은 (prompt, seed) → 같은 출력. 입력이 바뀌면 출력이
완전히 달라진다. 평균/뭉개기 없음, diffusion 아님.

```
  modality-tagged
  CE Storage                    32x32 latent grid              256x256 이미지
  ─────────────                ─────────────────              ──────────────
  TEXT  keyframes  ─┐            ┌───────────┐                 ┌──────────┐
                    │ retrieval  │ . . . . . │  ce_decode_     │          │
  IMAGE keyframes ──┼──► CEUnit ►│ . . z . . │  image_block    │  RGBA    │
                    │ (by type)  │ . . . . . │ ──────────────► │  256x256 │
  AUDIO keyframes ──┘            └───────────┘  per-block       └──────────┘
                                     ↑↑↑
                                 carry-align (target ↔ seed)
                                  + 주변 cell 반영 + cross-attn
                                  + residual/skip
                                  → 반복 (whole-block carry alignment)
```

이미지 인코딩(`ce_feed_image`)은 RGBA 채널을 분리하여 inc.plus 캐리 체인에
누적하고, 4사분면 평균 차이로 4방향(LR/UD/D1/D2) gradient·edge를
dec.{plus,minus} 에 별도로 저장한다. 텍스트 인코딩(`ce_feed`)은 그대로 유지.

---

## 빌드 & 테스트 (한 줄씩)

```bash
# ce_core 디렉터리로 진입
cd ce_core

# 모든 .o 와 build/ 정리
make clean

# 6개 테스트 바이너리 빌드 + 실행 (PASS/FAIL 출력)
make test
```

개별 테스트만 실행하고 싶을 때:

```bash
# Phase 0: CEUnit 기본 동작 (init/feed/distance/delta/apply/read)
./build/test_core

# Phase 1: storage + 검색 + 노이즈 + 쿼리 그리드
./build/test_storage

# Phase 2: 핵심 엔진 (down/conv/attn/cross/residual/skip/up — carry-align 등가)
./build/test_engine

# Phase 3: carry-align loop + image/text decode + loss
./build/test_denoise

# Phase 4: sampler/memo/hint/inpaint/upscale/audio
./build/test_extend

# 통합: prompt -> image / text 전체 파이프라인
./build/test_gen

# 이미지 modality 인코딩 + 4방향 gradient + type-filtered 검색
./build/test_feed_image

# 단일 CEUnit -> 8x8 RGBA 블록 디코딩
./build/test_decode_image_block
```

---

## 파일 구조

```
ce_core/
├── ce_core.{h,c}        기본 64바이트 셀 + 핵심 6개 API
├── ce_type.h            CEType enum (TEXT/IMAGE/AUDIO modality 태그)
├── ce_feed_image.{h,c}  이미지 전용 인코딩 (RGBA carry tick + 4방향 gradient)
├── ce_storage.{h,c}     type-tagged keyframe + delta 저장소 (auto-grow)
├── ce_storage_io.{h,c}  바이너리 저장/로드 (.ces v2, v1 호환)
├── ce_search.{h,c}      결정론적 노이즈 / 쿼리 그리드 / top-k + by-type 검색
├── ce_engine.{h,c}      carry-align 단계별 연산 (down/conv/attn/cross/residual/skip/up)
├── ce_denoise.{h,c}     whole-block carry alignment 반복 + loss + sampler 모드
├── ce_decode.{h,c}      256x256 이미지 / ASCII 텍스트 출력 + per-block decode
├── ce_extend.{h,c}      sampler / memo / hint / inpaint / upscale / audio
├── ce_ingest.{h,c}      디렉터리 -> CEStorage (PNG/JPG/BMP/TGA/PSD/GIF/PNM, type=IMAGE)
├── ce_gen.{h,c}         최상위 API (generate_image / text / inpaint / upscale)
├── third_party/
│   └── stb_image.h      vendored, public domain (PNG/JPG 디코더)
├── tools/
│   └── ce_ingest.c      CLI: 폴더 -> .ces 변환기
├── Makefile
└── tests/               9개 테스트 바이너리
```

---

## 최소 사용 예제

### 1. 텍스트 코퍼스 학습 → 이미지 생성

```c
#include "ce_gen.h"

CEStorage S;
ce_storage_init(&S, 16);                          // capacity 힌트만; 자동 확장

CEUnit prev; ce_init(&prev);                       // 첫 keyframe 직전 anchor

// 1번 캔버스 / 0번 슬롯에 6개 블록 ingest
const char *blocks[] = {
    "the orange cat eats fish",
    "the gray cat plays with yarn",
    "rain falls on the city street",
    "snow covers the mountain top",
    "the moon rises over the sea",
    "the sun sets behind the trees"
};
for (int i = 0; i < 6; ++i) {
    ce_storage_ingest(&S, /*canvas*/1, /*slot*/0, /*block*/i,
                      (i == 0) ? NULL : &prev,
                      (const uint8_t *)blocks[i],
                      (uint32_t)strlen(blocks[i]));
    prev = S.entries[S.count - 1].keyframe;        // 다음 anchor 로 전달
}

CEGenConfig cfg = ce_gen_config_default();         // total_steps=8, cfg=1.0 등
cfg.total_steps = 50;                               // 50 step (실측 0.49s)

CEImage img;
ce_generate_image(&img, &S,
                  "orange cat eating fish",         // prompt
                  /*seed=*/0xC0FFEE,
                  &cfg,
                  /*memo=*/NULL,
                  /*hint=*/NULL,
                  /*audio=*/NULL);

// img.pixels 는 RGBA 256x256 — PPM 으로 덤프하거나 즉시 화면 표시
ce_storage_free(&S);
```

### 2. 결정론 vs 다양성 확인

```c
CEImage a, b, c;

// 같은 (prompt, seed) → 바이트 단위까지 동일
ce_generate_image(&a, &S, "cat", 42, &cfg, NULL, NULL, NULL);
ce_generate_image(&b, &S, "cat", 42, &cfg, NULL, NULL, NULL);
assert(memcmp(&a, &b, sizeof(CEImage)) == 0);

// seed 만 바꿔도 25% 이상의 픽셀이 달라진다 (다양성)
ce_generate_image(&c, &S, "cat", 99, &cfg, NULL, NULL, NULL);
// memcmp(&a, &c, ...) != 0
```

### 3. Inpaint (anchor 보존 부분 재생성)

```c
CEInpaintMask mask;
for (int i = 0; i < CE_GRID_N; ++i) {
    mask.mask[i] = (i < 256) ? 1 : 0;              // 앞 256셀만 재생성
}

CEImage out;
ce_generate_inpaint(&out, /*original=*/&a, &mask, &S,
                    "snow on mountain", /*seed=*/42, &cfg);
// mask=0 셀의 8x8 픽셀 블록은 원본 그대로 유지
```

### 4. Upscale (32x32 latent → 128x128)

```c
CELatentGrid lo;                                   // 32x32 = 1024 셀
// ... lo 채우기 (예: ce_latent_init / ce_denoise_loop)

CEHiresGrid hi;                                    // 128x128 = 16384 셀
ce_generate_upscale(&hi, &lo, &S, &cfg);
```

### 5. 메모 레이어 (LoRA 등가) 적용

```c
CEUnit adjustments[2];
ce_init(&adjustments[0]); ce_feed(&adjustments[0], (const uint8_t*)"수묵화", 9);
ce_init(&adjustments[1]); ce_feed(&adjustments[1], (const uint8_t*)"먹빛",   6);

CEMemoLayer memo = { .name = "ink_painting",
                     .adjustments = adjustments,
                     .count = 2 };

ce_generate_image(&img, &S, "cat", 42, &cfg,
                  &memo,                            // 스타일 적용
                  NULL, NULL);
```

### 6. 힌트 레이어 (ControlNet 등가) 적용

```c
CEHintLayer hint;
hint.type = CE_HINT_EDGE;
hint.strength = 0.7f;
for (int i = 0; i < CE_GRID_N; ++i) {
    ce_init(&hint.cells[i]);
    // 외곽선 / 깊이 / 포즈 / 컬러 가이드 셀 채우기
}

ce_generate_image(&img, &S, "cat", 42, &cfg,
                  NULL,
                  &hint,                            // 가이드 적용
                  NULL);
```

### 7. 오디오 싱크 (시간축 강도 조절)

```c
uint8_t audio_bytes[44100 * 5];                    // 5초 오디오 raw
// ... audio_bytes 채우기

CEAudioTrack track;
ce_audio_load(&track, audio_bytes, sizeof(audio_bytes), /*seg/step=*/4);

ce_generate_image(&img, &S, "wave", 42, &cfg,
                  NULL, NULL,
                  &track);                          // 매 step 갱신 강도 조절

ce_audio_free(&track);
```

---

## CE Whole-Block Carry Alignment 단계 매핑

이 엔진은 diffusion이 아니다. 단계 이름은 호환성 때문에 유지하지만 의미는
"전체 grid를 동시에 target 방향으로 carry-tick 정렬한다"이며, 매 step 마다
seed 고유 패턴을 일정 비율 보존한다(완전 수렴 금지).

| 단계 | 외부 등가 표현    | CE 함수                        | carry-align 의미              |
|------|-------------------|--------------------------------|-------------------------------|
| 0    | 사전 학습         | `ce_storage_ingest` /          | type-tagged keyframe 적재    |
|      |                   | `ce_ingest_*` (IMAGE)          |                              |
| 1    | x_T 노이즈        | `ce_noise_init(seed)`          | seed 고유 시작 상태          |
| 2    | 노이즈 latent     | `CEQueryGrid`                  | 32x32 동시 정렬 대상         |
| 3    | retrieval         | `ce_search_topk` /             | type-filtered top-k          |
|      |                   | `ce_search_by_type`            | (modality 누수 차단)          |
| 4    | feature map       | `ce_extract_context`           | 주변 cell 카탈로그           |
| 5    | anchor + var.     | `ce_select_kd_pair`            | keyframe + delta 후보        |
| 6    | latent init       | `ce_latent_init` (5-source)    | seed_origin 보존 시작         |
| 7    | UNet encoder      | `ce_down`                      | 의미 압축 방향 carry-tick    |
| 8    | conv block        | `ce_conv`                      | 주변 cell 반영               |
| 9    | self-attn         | `ce_self_attention`            | grid 내 edge continuity      |
| 10   | cross-attn        | `ce_cross_attention` (cfg)     | 프롬프트 조건 alignment      |
| 11   | residual          | `ce_residual`                  | 직전 step 신호 보존          |
| 12   | skip              | `ce_skip_connect`              | down 단계 정보 → up 단계      |
| 13   | UNet decoder      | `ce_up`                        | 디테일 복원 방향 carry-tick   |
| 14   | (반복 step)       | `ce_denoise_loop`              | whole-block carry alignment  |
|      |                   |                                | 반복 (target ↔ seed 균형)    |
| 15   | VAE decoder       | `ce_decode_image` /            | latent → 256x256 RGBA /      |
|      |                   | `ce_decode_image_block` /      | 단일 cell → 8x8 block /      |
|      |                   | `ce_decode_text`               | latent → ASCII text          |
| ext  | LoRA              | `CEMemoLayer`                  | 스타일 delta 추가            |
| ext  | ControlNet        | `CEHintLayer`                  | 가이드 cell 강제             |
| ext  | inpaint           | `ce_inpaint`                   | mask 외 anchor 보존           |
| ext  | upscaler          | `ce_upscale`                   | sub-cell 분기                |
| ext  | (캔버스 고유)     | `CEAudioTrack`                 | 시간축 강도 변조             |

`ce_denoise_loop` 라는 이름은 호환성 위해 유지(파일명·심볼명 변경 없음). 실제
동작은 noise 제거가 아니라 carry-align step 반복이다.

---

## 핵심 6개 API (`ce_core.h`)

```c
void     ce_init(CEUnit *u);                                      // 셀 초기화
void     ce_feed(CEUnit *u, const uint8_t *data, uint32_t len);   // 바이트 → 신호
uint32_t ce_distance(const CEUnit *a, const CEUnit *b);           // 64B SAD
void     ce_delta(CEUnit *out, const CEUnit *anchor,              // 차이
                  const CEUnit *current);
void     ce_apply(CEUnit *out, const CEUnit *anchor,              // 복원
                  const CEUnit *delta);
int64_t  ce_read(const CEUnit *u, int channel);                   // R/G/B/A 읽기
```

가역성:

```
ce_apply(anchor, ce_delta(anchor, x)) == x   (byte-exact, mod 256)
```

모든 상위 연산은 위 6개의 가중 합산. 평균/뭉개기 없음.

---

## 성능 (현 CPU 실측)

```
  256x256 이미지 (4 step) :  0.04 s
  256x256 이미지 (50 step):  0.49 s   (목표 2-3 s)
  ce_distance 처리량      :  225 M cmp/s
  런타임 메모리           :  ~1 MB
  storage 엔트리당        :  140 B (canvas/slot/block/type + keyframe + delta)
```

---

## 제약 조건 (지킨 것)

- `rand()` 금지 → SplitMix64 기반 결정론적 난수만 사용.
- 평균(mean / average) 금지 → 가중 `ce_apply` 체인 + `ce_delta_scale` 만 사용.
- 외부 라이브러리 금지 → `<stdlib.h>`, `<string.h>`, `<stdio.h>`, `<math.h>` 만 사용.
- GPU 불필요 → 50 step 256x256 생성도 0.49 s.
- 각 단계마다 `tests/test_*.c` 가 PASS / FAIL 을 출력.

---

## 이미지 폴더 → 스토리지 (.ces)

PNG/JPG/BMP/TGA 이미지를 8×8 블록으로 분할하여 keyframe + delta 로 저장하는 도구.

### CLI 사용법 (한 줄씩)

```bash
# 빌드 (테스트 + 도구 함께)
cd ce_core && make all

# 폴더 안의 모든 이미지를 ingest 해서 .ces 파일로 저장
./build/ce_ingest /path/to/images -o my_dataset.ces

# 하위 디렉터리까지 재귀적으로 ingest
./build/ce_ingest /path/to/images -o my_dataset.ces --recursive

# 저장된 .ces 파일 정보 보기 (이미지 수 / 블록 수 / 크기 추정)
./build/ce_ingest --info my_dataset.ces
```

출력 예시:

```
ingested      : 3 images from /tmp/ce_real
  seen        : 3
  decoded     : 3
  blocks      : 13564
  errors      : 0
saved         : my_dataset.ces (13564 entries)
```

### C API 사용법

```c
#include "ce_ingest.h"
#include "ce_storage_io.h"

CEStorage S;
ce_storage_init(&S, 1024);

// 1) 단일 파일 ingest
CEIngestStats stats = {0};
ce_ingest_file(&S, "cat.png", &stats);

// 2) 디렉터리 전체 ingest (PNG/JPG/JPEG/BMP/TGA 자동 감지)
ce_ingest_directory(&S, "./images", &stats);

// 3) 재귀 ingest
ce_ingest_directory_recursive(&S, "./dataset", &stats);

// 4) 바이너리로 저장
ce_storage_save(&S, "dataset.ces");

// 5) 다른 프로세스/세션에서 로드
CEStorage L;
ce_storage_load(&L, "dataset.ces");

// 6) ce_generate_image 등에 그대로 전달
ce_generate_image(&img, &L, "orange cat", 42, &cfg, NULL, NULL, NULL);

ce_storage_free(&S);
ce_storage_free(&L);
```

### 인덱싱 규칙

각 이미지는 다음 규칙으로 CEStorage 에 저장된다:

```
canvas_id = FNV-1a(파일경로)        // 파일별 고유 ID
slot      = 블록의 row 좌표 (block_y) // 8픽셀 단위
block_idx = 블록의 col 좌표 (block_x) // 8픽셀 단위

block 입력 = 8 × 8 × 4 = 256 bytes (RGBA)
        ↓ ce_feed_image  (채널 분리 + 4방향 gradient)
keyframe = CEUnit (64 bytes, type=CE_TYPE_IMAGE)
delta    = ce_delta(prev_block, keyframe)  // 이미지 내 체이닝
```

가장자리는 0으로 패딩 (이미지 크기가 8의 배수가 아닐 때). 이미지 ingest 경로는
text용 `ce_feed`를 호출하지 않으며, modality는 entry에 `CE_TYPE_IMAGE`로 태깅된다.

### .ces 바이너리 포맷 (v2, 현재)

```
offset  size  field
   0      4   magic    "CES1" (0x31534543, sanity check)
   4      4   version  uint32 = 2
   8      4   count    uint32 (entry 개수)
  12      4   reserved 0
  16+         entry[count] :
              + uint32  canvas_id
              + uint16  slot
              + uint16  block_idx
              + uint8   type           (CEType: TEXT=0, IMAGE=1, AUDIO=2)
              + uint8   reserved[3]    (zero)
              + 64 byte keyframe
              + 64 byte delta
              = 140 bytes / entry
```

리틀엔디언 고정. 파일 크기 ≈ 16 + 140 × count. v1 (type 없음, 136 B/entry)
파일도 그대로 로드되며, 모든 entry는 `CE_TYPE_TEXT`로 태깅된다.

---

## 자주 쓰는 명령 모음

```bash
# 깨끗한 상태에서 전체 테스트
cd ce_core && make clean && make test

# 빌드만
cd ce_core && make all

# 통합 테스트만 (가장 흥미로운 출력)
cd ce_core && make build/test_gen && ./build/test_gen

# 빠른 sanity check (가장 짧음)
cd ce_core && make build/test_core && ./build/test_core

# 폴더 -> .ces 변환
cd ce_core && make build/ce_ingest && ./build/ce_ingest /path/to/imgs -o out.ces

# .ces 정보 확인
cd ce_core && ./build/ce_ingest --info out.ces
```
