/* ce_masked_train.h — 마스크 학습: CE latent에게 "빠진 부분을 추론하는 법" 가르치기
 *
 * 디퓨전이 noise를 추가→제거하는 법을 학습하듯,
 * SSS는 이미지의 일부를 가리고→복원하는 법을 학습한다.
 *
 * ■ 학습 흐름 (이미지 1장당)
 *
 *   1. 이미지 → slig_decompose_v2 → 3단계 분리
 *        basis      = structure cells (큰 윤곽)
 *        residual   = edge + texture cells (디테일)
 *        correction = color + event cells (미세 보정)
 *
 *   2. correction을 마스크로 가림
 *        → "basis + residual만 보고 correction을 예측해봐"
 *
 *   3. ce_denoise_loop로 예측 수행
 *
 *   4. 예측 vs 실제 correction → ce_compute_loss
 *
 *   5. loss가 줄어들도록 파라미터 조정
 *        → 충분히 반복하면 CE latent가 패턴을 학습
 *
 *   6. 수렴한 CE Cell 세트를 CEStorage에 저장
 *        → 이제 새 프롬프트에서도 correction을 "생성"할 수 있음
 *
 * ■ 마스크 전략 (단계별)
 *
 *   Phase 1: correction만 가림 (쉬움 → 색/이벤트 추론 학습)
 *   Phase 2: residual + correction 가림 (중간 → 디테일 추론 학습)
 *   Phase 3: 랜덤 50% 가림 (어려움 → 일반 생성 능력)
 */
#ifndef CE_MASKED_TRAIN_H
#define CE_MASKED_TRAIN_H

#include "ce_core.h"
#include "ce_storage.h"
#include "ce_denoise.h"
#include "ce_residual_codebook.h"
#include "slig_pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ════════════════════════════════════════════════════════
 *  마스크 전략
 * ════════════════════════════════════════════════════════ */

typedef enum {
    MASK_CORRECTION_ONLY = 0,  /* correction cells만 가림 (Phase 1) */
    MASK_RESIDUAL_UP     = 1,  /* residual + correction 가림 (Phase 2) */
    MASK_RANDOM_HALF     = 2,  /* 랜덤 50% 가림 (Phase 3) */
    MASK_PROGRESSIVE     = 3   /* epoch에 따라 자동 전환 */
} MaskStrategy;

/* ════════════════════════════════════════════════════════
 *  학습 설정
 * ════════════════════════════════════════════════════════ */

typedef struct {
    /* 반복 */
    int      epochs;           /* 전체 반복 횟수 (기본 10) */
    int      denoise_steps;    /* 예측 시 denoise 스텝 수 (기본 8) */

    /* 수렴 */
    float    target_loss;      /* 이 이하면 조기 종료 (기본 8.0) */
    float    loss_patience;    /* N epoch 동안 개선 없으면 종료 (기본 3) */

    /* 마스크 */
    MaskStrategy strategy;     /* 마스크 전략 (기본 PROGRESSIVE) */
    uint64_t     mask_seed;    /* 랜덤 마스크 시드 (기본 42) */

    /* 저장 */
    int      store_on_converge; /* 수렴 시 CEStorage에 자동 저장 (기본 1) */

    /* Residual codebook (선택). NULL이면 모든 cells을 raw CE_TYPE_SLIG로
     * 저장 (기존 동작). 비-NULL이면 correction 영역(cells[texture_end..
     * num_cells))은 codebook으로 환원되어 CE_TYPE_RESIDUAL descriptor로
     * 저장됨. basis + edge + texture는 그대로 CE_TYPE_SLIG. 호출자가
     * codebook을 소유하며, 동일 codebook을 batch 호출에 재사용해 패턴
     * 공유를 누적할 수 있다. */
    CEResidualCodebook *residual_book;
    uint32_t            residual_threshold; /* 0이면 CE_RESIDUAL_DEFAULT_THRESHOLD */

    /* Segment loss weights — total = w_struct * L_struct
     *                             + w_texture * L_texture
     *                             + w_color   * L_color
     * Segments map to SligDecomposed buckets: structure = basis cells,
     * texture = edge + texture cells, color = color + event cells.
     * If all three weights are 0 the legacy mean loss is used (so
     * masked_train_config_default keeps the pre-Step-5 behaviour). */
    float    loss_w_structure;   /* default 0 (= legacy mean loss) */
    float    loss_w_texture;     /* default 0 */
    float    loss_w_color;       /* default 0 */

    /* 콜백 (NULL이면 무시) */
    void (*on_epoch)(int epoch, float loss, void *ctx);
    void *cb_ctx;
} MaskedTrainConfig;

void masked_train_config_default(MaskedTrainConfig *cfg);

/* ════════════════════════════════════════════════════════
 *  학습 결과
 * ════════════════════════════════════════════════════════ */

typedef struct {
    float    final_loss;       /* 최종 loss (weighted total when seg weights set) */
    int      epochs_run;       /* 실제 실행된 epoch 수 */
    int      converged;        /* 1이면 target_loss 이하에 도달 */
    uint32_t cells_stored;     /* CEStorage에 저장된 셀 수 (raw + descriptor) */
    uint32_t residual_cells;   /* 그 중 CE_TYPE_RESIDUAL descriptor 개수 */

    /* Per-segment mean loss at the best epoch (zero when the segment
     * had no masked cells). Always populated, regardless of whether
     * weighted aggregation was active — useful for diagnosing which
     * segment dominates total loss. */
    float    loss_structure;
    float    loss_texture;
    float    loss_color;

    /* 수렴된 CE Cell 세트 (correction 추론 결과 포함) */
    SligDecomposed learned;
} MaskedTrainResult;

/* ════════════════════════════════════════════════════════
 *  1. 단일 이미지 학습
 *
 *  한 장의 이미지에 대해 마스크 학습을 수행하고,
 *  수렴된 CE Cell을 CEStorage에 저장합니다.
 * ════════════════════════════════════════════════════════ */

void masked_train_image(MaskedTrainResult      *result,
                        CEStorage              *storage,
                        const uint8_t          *rgba,
                        int                     width,
                        int                     height,
                        uint32_t                canvas_id,
                        const MaskedTrainConfig *cfg);

/* ════════════════════════════════════════════════════════
 *  2. 배치 학습 (여러 이미지)
 *
 *  이미지 배열에 대해 순차적으로 마스크 학습을 수행합니다.
 *  각 이미지의 학습 결과가 CEStorage에 누적되어,
 *  이후 이미지의 추론이 점점 나아집니다.
 * ════════════════════════════════════════════════════════ */

typedef struct {
    const uint8_t *rgba;
    int            width;
    int            height;
    uint32_t       canvas_id;
    const char    *label;       /* 디버그용 이름 (NULL 가능) */
} TrainImageEntry;

typedef struct {
    float    avg_loss;         /* 전체 평균 loss */
    float    best_loss;        /* 가장 낮은 loss */
    float    worst_loss;       /* 가장 높은 loss */
    int      total_converged;  /* 수렴한 이미지 수 */
    int      total_images;     /* 전체 이미지 수 */
} BatchTrainResult;

void masked_train_batch(BatchTrainResult        *result,
                        CEStorage               *storage,
                        const TrainImageEntry   *images,
                        int                      num_images,
                        const MaskedTrainConfig  *cfg);

#ifdef __cplusplus
}
#endif
#endif /* CE_MASKED_TRAIN_H */
