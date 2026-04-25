/* ce_image_wave_refine.h — wave-refine image carving over a 256x256 canvas.
 *
 * 핵심 아이디어:
 *
 *   prompt 입력
 *      ↓
 *   관련된 64x64 후보 공간 top3 선택
 *      ↓
 *   256x256 픽셀을 파도처럼 수정 (move_toward, channel-wise clamp)
 *      ↓
 *   step magnitude wave: 0.01 → 0.1 → 1 → 3 → 1 → 0.1 → 0.01 (cycles per iter)
 *      ↓
 *   이미지 완성
 *
 * 이 모듈은 CEUnit / ce_feed_image / dec-half gradient 와는 독립이다.
 * 픽셀 평면(float RGBA)에서 직접 동작하므로 retrieval 결과를 PixelF target
 * 으로 미리 변환해 넘기면 된다.
 *
 * 본 PR은 단계 1~5(파도 step / top3 weighted target / move_toward / iterate)
 * 까지만 포함한다. 다음에 들어올 예정인 단계는 헤더 주석에만 명시한다:
 *
 *   - 주변 픽셀 가중 평균(혼자 튀는 픽셀 완화)
 *   - seed 흔적 보존 (target 80%, seed 20% 등)
 *   - top3 주기적 재계산 (10 iter마다)
 */
#ifndef CE_IMAGE_WAVE_REFINE_H
#define CE_IMAGE_WAVE_REFINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CE_WAVE_IMG_W   256
#define CE_WAVE_IMG_H   256
#define CE_WAVE_LINK_W   64
#define CE_WAVE_LINK_H   64
#define CE_WAVE_TOPK      3

typedef struct {
    float r, g, b, a; /* per-channel float in [0, 255] (clamped after each step) */
} CEPixelF;

typedef struct {
    /* 64x64 작은 의미 공간. prompt / label / morpheme / word / sentence
     * retrieval 결과와 연결되는 후보. */
    float    score;
    float    semantic[CE_WAVE_LINK_W * CE_WAVE_LINK_H];
    /* 이 후보가 바라보는 256x256 목표 이미지 ("이 후보가 원하는 색 방향"). */
    CEPixelF target[CE_WAVE_IMG_W * CE_WAVE_IMG_H];
} CEImageLink64;

typedef struct {
    /* 256x256 실제 이미지 공간 (float RGBA, [0..255] 범위). */
    CEPixelF px[CE_WAVE_IMG_W * CE_WAVE_IMG_H];
} CEImageCanvas;

/* 파도 step magnitude. iter % 7 로 7-주기 사이클을 돈다.
 * 0.01 → 0.1 → 1 → 3 → 1 → 0.1 → 0.01 */
float ce_wave_step_value(uint32_t iter);

/* top3 후보를 score 가중 평균으로 합쳐 target 픽셀 평면을 만든다.
 * NULL 슬롯은 무시한다. */
void ce_wave_build_target_from_top3(CEPixelF *out_target,
                                    CEImageLink64 *const top3[CE_WAVE_TOPK]);

/* 각 픽셀을 target 방향으로 max_step 만큼만 이동한 뒤 [0, 255] 로 clamp. */
void ce_wave_refine_once(CEImageCanvas *canvas,
                         const CEPixelF *target,
                         float max_step);

/* 전체 파이프라인: target 구성 → iterations 회 파도 refine. */
void ce_image_wave_refine(CEImageCanvas *canvas,
                          CEImageLink64 *const top3[CE_WAVE_TOPK],
                          uint32_t iterations);

#ifdef __cplusplus
}
#endif
#endif /* CE_IMAGE_WAVE_REFINE_H */
