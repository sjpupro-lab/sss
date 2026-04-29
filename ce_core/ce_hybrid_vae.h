/* ce_hybrid_vae.h — 통합 VAE: 블록 도장 + SLIG 파형 + 주파수 복원
 *
 * 디퓨전의 VAE encoder/decoder를 SSS 방식으로 대체하는 통합 파이프라인.
 *
 * ■ Encode (학습)
 *   이미지
 *     → 방법1: ce_storage_ingest_rgba_16 → 1024 CEUnit (색 앵커)
 *     → 방법2: slig_decompose_channel × [3 scale × 3 channel] → SLIG CellSet (형태)
 *     → 링크: SLIG Cell의 dec.A audio_bins에 canvas_id 기록
 *     → codebook에 패턴 등록, Keyframe.image_idx에 인덱스 저장
 *
 * ■ Decode (생성)
 *   CE Cell들
 *     → Step 1: 블록 도장 복원 → 256×256 색 기반 (coarse color)
 *     → Step 2: 주파수 업스케일 (harmonic + SBR) → 고주파 복원
 *     → Step 3: SLIG 틱 렌더 → 형태/디테일 오버레이
 *     → Step 4: base × α + detail × β 블렌딩
 *     → Step 5: wave refine (0.01→3→0.01 사이클)
 *     → 최종 256×256 RGB
 *
 * CE Cell 레이아웃 (SLIG 경로):
 *   inc.R (8B): 메타 (type, scale, sigma, freq, origin, flags)
 *   inc.G (8B): u DCT[0..7]   (세로 주파수)
 *   inc.B (8B): v DCT[0..7]   (가로 주파수)
 *   inc.A (8B): 에너지 프로파일
 *   dec.R (8B): u DCT[8..15]  (고주파 확장)
 *   dec.G (8B): v DCT[8..15]  (고주파 확장)
 *   dec.B (8B): 시간/트리거
 *   dec.A (8B): 오디오트랙 링크 (canvas_id → 블록 도장 참조)
 */
#ifndef CE_HYBRID_VAE_H
#define CE_HYBRID_VAE_H

#include "ce_core.h"
#include "ce_storage.h"
#include "ce_decode.h"
#include "ce_feed_image.h"
#include "ce_image_wave_refine.h"
#include "ce_residual_codebook.h"
#include "slig_signal.h"
#include "slig_pipeline.h"
#include "slig_codebook.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ════════════════════════════════════════════════════════
 *  설정 구조체
 * ════════════════════════════════════════════════════════ */

typedef struct {
    /* 블렌딩 가중치 (base + detail = 1.0) */
    float base_weight;         /* 블록 도장(색) 비중 (기본 0.4) */
    float detail_weight;       /* SLIG(형태) 비중 (기본 0.6) */

    /* 주파수 업스케일 */
    uint8_t  enable_harmonic;  /* 하모닉 확장 (기본 1) */
    uint8_t  enable_sbr;       /* 대역폭 복제 (기본 1) */
    int      target_coeffs;    /* 목표 DCT 계수 수 (기본 32) */

    /* 틱 엔진 */
    uint8_t  structure_ticks;  /* 구조 셀 틱 범위 (기본 5) */
    uint8_t  edge_ticks;       /* 엣지 셀 틱 범위 (기본 10) */
    uint8_t  detail_ticks;     /* 텍스처 셀 틱 범위 (기본 15) */

    /* Wave refine */
    uint32_t wave_iterations;  /* 파동 정제 횟수 (기본 100) */

    /* 후처리 */
    float    guidance_scale;   /* CFG 강도 (기본 1.5) */

    /* Residual codebook (선택). NULL이면 디코드는 base+detail+wave
     * 흐름만 거친다. 비-NULL이면 storage의 CE_TYPE_RESIDUAL entries를
     * tick 순서로 순회해 codebook 패턴을 detail_y에 누적 적용. */
    const CEResidualCodebook *residual_book;
} HybridVAEConfig;

void hybrid_vae_config_default(HybridVAEConfig *cfg);

/* ════════════════════════════════════════════════════════
 *  Encode 결과 (학습 시 반환)
 * ════════════════════════════════════════════════════════ */

typedef struct {
    /* 방법1: 블록 도장 */
    uint32_t block_entries;    /* CEStorage에 추가된 블록 도장 수 */
    uint32_t canvas_id;        /* 이 이미지에 할당된 canvas_id */

    /* 방법2: SLIG 분해 */
    SligCellSet sets[SLIG_NUM_LEVELS][SLIG_NUM_CHANNELS]; /* 3×3 = 9개 셋 */
    uint8_t     codebook_idx[SLIG_NUM_LEVELS][SLIG_NUM_CHANNELS]; /* codebook 인덱스 */

    /* 메타 */
    float    psnr;             /* 복원 품질 (dB) */
    uint32_t total_cells;      /* SLIG 총 셀 수 */
} HybridEncodeResult;

/* ════════════════════════════════════════════════════════
 *  1. ENCODE — 이미지 → CE LATENT
 *
 *  이미지 한 장을 두 가지 방법으로 동시에 압축하고,
 *  오디오트랙(dec.A)으로 두 결과를 링크합니다.
 *
 *  입력:
 *    rgba        — 256×256 RGBA 인터리브드 (256*256*4 = 262144 바이트)
 *    canvas_id   — 이 이미지의 논리 ID (CEStorage 라우팅용)
 *    storage     — 블록 도장이 저장될 CEStorage
 *    codebook    — SLIG 패턴이 등록될 codebook
 *
 *  흐름:
 *    1. RGB → YCbCr 분리
 *    2. ce_storage_ingest_rgba_16 → 블록 도장 1024개
 *    3. slig_decompose_channel × 9 (3 scale × 3 channel) → SLIG 셋
 *    4. slig_upscale_harmonic → 고주파 확장
 *    5. 각 SLIG Cell의 dec.A.audio_bins에 canvas_id 기록 (링크)
 *    6. codebook에 패턴 등록
 * ════════════════════════════════════════════════════════ */

void hybrid_vae_encode(HybridEncodeResult *result,
                       CEStorage          *storage,
                       SligCodebook       *codebook,
                       const uint8_t      *rgba,
                       int                 width,
                       int                 height,
                       uint32_t            canvas_id);

/* ════════════════════════════════════════════════════════
 *  2. DECODE — CE LATENT → 이미지
 *
 *  블록 도장(색 기반) + SLIG 셀(형태) + 주파수 복원을
 *  조합하여 최종 이미지를 생성합니다.
 *
 *  입력:
 *    storage     — 블록 도장이 들어있는 CEStorage
 *    canvas_id   — 복원할 이미지의 canvas_id
 *    sets        — SLIG CellSet 배열 [3][3]
 *    cfg         — 블렌딩/업스케일/wave 설정
 *
 *  흐름:
 *    Step 1: 블록 도장 → 256×256 색 기반
 *    Step 2: SLIG 셀에 harmonic/SBR 업스케일
 *    Step 3: 틱 엔진으로 SLIG 셀 순차 누적
 *    Step 4: base(색) × α + detail(형태) × β
 *    Step 5: wave refine (0.01→3→0.01)
 *    Step 6: YCbCr → RGB 변환
 * ════════════════════════════════════════════════════════ */

void hybrid_vae_decode(uint8_t                *out_rgb,     /* 256×256×3 */
                       const CEStorage        *storage,
                       uint32_t                canvas_id,
                       const SligCellSet       sets[SLIG_NUM_LEVELS][SLIG_NUM_CHANNELS],
                       const HybridVAEConfig  *cfg);

/* ════════════════════════════════════════════════════════
 *  3. 원샷 API — 학습과 즉시 검증
 * ════════════════════════════════════════════════════════ */

/* Encode + 즉시 Decode → PSNR 계산 (학습 품질 확인용) */
float hybrid_vae_roundtrip(uint8_t            *out_rgb,
                           CEStorage          *storage,
                           SligCodebook       *codebook,
                           const uint8_t      *rgba,
                           int                 width,
                           int                 height,
                           uint32_t            canvas_id,
                           const HybridVAEConfig *cfg);

/* ════════════════════════════════════════════════════════
 *  4. 유틸리티
 * ════════════════════════════════════════════════════════ */

/* 블록 도장에서 coarse 색 기반만 복원 (디버그용) */
void hybrid_decode_base_only(uint8_t         *out_rgba,  /* 256×256×4 */
                             const CEStorage *storage,
                             uint32_t         canvas_id);

/* SLIG 셀에서 형태만 복원 (디버그용) */
void hybrid_decode_detail_only(uint8_t             *out_gray,  /* 256×256 */
                               const SligCellSet    sets[SLIG_NUM_LEVELS][SLIG_NUM_CHANNELS],
                               const HybridVAEConfig *cfg);

/* PSNR 계산 */
float hybrid_psnr(const uint8_t *original_rgb,
                  const uint8_t *decoded_rgb,
                  int width, int height);

#ifdef __cplusplus
}
#endif
#endif /* CE_HYBRID_VAE_H */
