/* slig_signal.h — Signal-Latent Image Generation v2.
 *
 * CE Cell 안에 이미지 주파수 정보를 직접 저장.
 * SSS 파이프라인과 완전 통합:
 *   텍스트 → 256격자 → 형태소별 매칭 → CE Cell 꺼냄 → 신호 조합 → 이미지
 *
 * CE Cell 레이아웃 (64 bytes):
 *
 *   inc half (32B) = 신호 정체성
 *     inc.R (8B): 메타데이터 (type, scale, sigma, freq, origin, flags)
 *     inc.G (8B): u 신호 압축 (세로축 DCT 계수 8개)
 *     inc.B (8B): v 신호 압축 (가로축 DCT 계수 8개)
 *     inc.A (8B): 에너지 프로파일 (스케일별 + 방향별)
 *
 *   dec half (32B) = 관계/디테일
 *     dec.R (8B): u 신호 확장 (DCT 계수 8개 추가, 총 16개)
 *     dec.G (8B): v 신호 확장 (DCT 계수 8개 추가, 총 16개)
 *     dec.B (8B): 시간/트리거 (tick, decay, speed, beam, cond)
 *     dec.A (8B): 오디오트랙 (주파수 빈 4 + 진폭 4)
 */
#ifndef SLIG_SIGNAL_H
#define SLIG_SIGNAL_H

#include "ce_core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── 신호 방향 ── */
enum SligDir {
    SLIG_DIR_HORIZONTAL = 0,   /* 가로×세로 외적        */
    SLIG_DIR_VERTICAL   = 1,   /* 세로×가로 (축 반전)    */
    SLIG_DIR_DIAG_DOWN  = 2,   /* 대각선 ╲              */
    SLIG_DIR_DIAG_UP    = 3,   /* 역대각선 ╱            */
    SLIG_DIR_ZIGZAG     = 4,   /* 지그재그              */
    SLIG_DIR_REVERSE    = 5,   /* 역방향               */
    SLIG_DIR_RIPPLE     = 6,   /* 좌표 시작 원형 물결    */
    SLIG_DIR_BEAM       = 7,   /* 좌표 시작 방향 빔      */
    SLIG_DIR_CROSS      = 8,   /* 좌표 시작 십자 퍼짐    */
    SLIG_DIR_CASCADE    = 9,   /* 연쇄 반응             */
    SLIG_DIR_COUNT      = 10
};

/* ── 멀티스케일 레벨 ── */
enum SligScale {
    SLIG_SCALE_COARSE = 0,     /* 진동폭 32칸 (큰 윤곽)  */
    SLIG_SCALE_MID    = 1,     /* 진동폭 8칸 (중간 구조)  */
    SLIG_SCALE_FINE   = 2,     /* 진동폭 2칸 (세부)      */
    SLIG_SCALE_PIXEL  = 3      /* 진동폭 1칸 (텍스처)    */
};

/* ── CE Cell 메타데이터 플래그 ── */
#define SLIG_FLAG_NEGATIVE   0x01  /* 음수 부호 */
#define SLIG_FLAG_HAS_TRIGGER 0x02 /* 조건부 트리거 있음 */
#define SLIG_FLAG_AUDIO_LINK 0x04  /* 오디오트랙 연결 */

/* ── 런타임 신호 (CE Cell에서 풀어낸 것) ── */
#define SLIG_SIG_LEN    256
#define SLIG_DCT_COEFF   16    /* CE Cell에 저장되는 DCT 계수 수 */

typedef struct {
    /* 메타 */
    uint8_t  dir;              /* SligDir */
    uint8_t  scale;            /* SligScale */
    uint16_t sigma;            /* 진폭 (0~65535) */
    uint8_t  frequency;        /* 진동폭 / 주기 */
    uint8_t  origin_x;         /* 좌표 시작점 X */
    uint8_t  origin_y;         /* 좌표 시작점 Y */
    uint8_t  flags;

    /* 신호 (DCT에서 복원된 전체 길이) */
    int16_t  u[SLIG_SIG_LEN]; /* 세로축 신호 (정수) */
    int16_t  v[SLIG_SIG_LEN]; /* 가로축 신호 (정수) */

    /* 시간/트리거 */
    uint8_t  start_tick;
    uint8_t  decay;            /* 0~255 → 0.80~1.00 감쇠 */
    uint8_t  speed;
    uint8_t  beam_dir;         /* 빔 방향 0~7 */
    uint8_t  beam_width;
    uint8_t  cond_threshold;
    uint8_t  cond_type;        /* 0=없음, 1=밝기, 2=연쇄 */
    uint8_t  phase;

    /* 오디오트랙 */
    uint8_t  audio_bins[4];    /* 주요 주파수 빈 */
    uint8_t  audio_amps[4];    /* 빈별 진폭 */
} SligSignal;

/* ── 캔버스 (정수 틱 누적) ── */
#define SLIG_CANVAS_DIM 256

typedef struct {
    int32_t pixels[SLIG_CANVAS_DIM][SLIG_CANVAS_DIM];
} SligCanvas;

/* ── 이미지 분해 결과 (학습 시) ── */
#define SLIG_MAX_CELLS 32

/* SLIG v2.1: a CellSet now carries channel + scale tags so a single
 * Keyframe can hold a [scale][channel] pyramid (3×3 = 9 sets) without
 * losing track of which slot each set populates. The original
 * single-channel grayscale path stays valid: decompose initialises
 * channel = SLIG_CH_Y and scale_level = SLIG_LEVEL_FINE, which is
 * what a caller using slig_decompose() naïvely sees. */

enum SligChannel {
    SLIG_CH_Y  = 0,    /* luminance — carries shape (≈20 cells)  */
    SLIG_CH_CB = 1,    /* blue chroma — color only (≈6 cells)    */
    SLIG_CH_CR = 2     /* red chroma — color only  (≈6 cells)    */
};
#define SLIG_NUM_CHANNELS 3

enum SligScaleLevel {
    SLIG_LEVEL_COARSE = 0,    /*  32×32 base layer (gross structure) */
    SLIG_LEVEL_MID    = 1,    /* 128×128 mid residual                */
    SLIG_LEVEL_FINE   = 2     /* 256×256 fine residual               */
};
#define SLIG_NUM_LEVELS 3

#define SLIG_CELL_BUDGET_Y   20   /* ≈8 horizontal + ≈11 diag/ripple */
#define SLIG_CELL_BUDGET_CB   6
#define SLIG_CELL_BUDGET_CR   6

typedef struct SligCellSet {
    CEUnit   cells[SLIG_MAX_CELLS];
    uint32_t num_cells;
    uint8_t  channel;       /* SligChannel — tagged when produced */
    uint8_t  scale_level;   /* SligScaleLevel — tagged when produced */
    uint8_t  reserved[2];
} SligCellSet;

/* ══════════════════════════════════════════
 *  1. CE Cell ↔ SligSignal 변환
 * ══════════════════════════════════════════ */

/* CE Cell → 런타임 신호로 풀기 */
void slig_unpack(SligSignal *sig, const CEUnit *cell);

/* 런타임 신호 → CE Cell에 싸기 */
void slig_pack(CEUnit *cell, const SligSignal *sig);

/* ══════════════════════════════════════════
 *  2. 이미지 → CE Cell 세트 (학습 시 사용)
 *     주파수 분해 + 다방향 + 멀티스케일
 * ══════════════════════════════════════════ */

/* 이미지를 분해하여 CE Cell 세트 생성.
 * image: 그레이스케일 uint8, row-major, width×height.
 * 결과: 다방향(가로,세로,대각) × 멀티스케일(32,8,2,1칸) CE Cell들.
 *
 * Backward-compatible wrapper for slig_decompose_channel with
 * channel=SLIG_CH_Y, scale_level=SLIG_LEVEL_FINE,
 * max_cells=SLIG_MAX_CELLS. */
void slig_decompose(SligCellSet *out,
                    const uint8_t *image, int width, int height);

/* Tagged decompose with explicit channel, scale, and cell budget.
 * Output's channel + scale_level fields are populated to match the
 * arguments. max_cells is clamped to SLIG_MAX_CELLS. */
void slig_decompose_channel(SligCellSet *out,
                            const uint8_t *image, int width, int height,
                            uint8_t  channel,
                            uint8_t  scale_level,
                            uint32_t max_cells);

/* ── RGB ↔ YCbCr (BT.601, 8-bit) ────────────────────────────
 * Three planar uint8 buffers, length = width*height each. Cb and
 * Cr are stored centred at 128 (so 128 = no chroma offset). */
void slig_rgb_to_ycbcr_planes(uint8_t *out_y, uint8_t *out_cb, uint8_t *out_cr,
                              const uint8_t *rgb_interleaved,
                              int width, int height);

void slig_ycbcr_planes_to_rgb(uint8_t *out_rgb_interleaved,
                              const uint8_t *y, const uint8_t *cb, const uint8_t *cr,
                              int width, int height);

/* ── Image pyramid helpers (uint8 single-channel, square) ───
 *
 *  slig_image_downsample : box-average — averages a (src_dim/dst_dim)²
 *      block of src into one dst pixel. Both dims must be square; src_dim
 *      must be a positive multiple of dst_dim ≥ dst_dim. No interpolation.
 *
 *  slig_image_upsample   : nearest-neighbour expansion. Each src pixel
 *      maps to a (dst_dim/src_dim)² tile of dst. Constant-block output —
 *      cheap and predictable; sufficient for the multi-scale render
 *      pipeline (the SVD basis at the next level absorbs the blocky
 *      interpolation artifacts naturally). */
void slig_image_downsample(uint8_t *dst, int dst_dim,
                           const uint8_t *src, int src_dim);
void slig_image_upsample  (uint8_t *dst, int dst_dim,
                           const uint8_t *src, int src_dim);

/* ══════════════════════════════════════════
 *  3. 캔버스 조작
 * ══════════════════════════════════════════ */

void slig_canvas_clear(SligCanvas *c);

/* 신호 하나를 캔버스에 적용 (방향/이벤트 자동 판별). */
void slig_apply(SligCanvas *c, const SligSignal *sig);

/* 시간 틱 기반 적용 (이벤트 신호용). */
void slig_apply_at_tick(SligCanvas *c, const SligSignal *sig, uint8_t tick);

/* CE Cell을 직접 캔버스에 적용 (unpack → apply). */
void slig_apply_cell(SligCanvas *c, const CEUnit *cell);

/* CE Cell을 틱 시점에 적용. */
void slig_apply_cell_at_tick(SligCanvas *c, const CEUnit *cell, uint8_t tick);

/* ══════════════════════════════════════════
 *  4. CE Cell 세트 → 이미지 생성
 * ══════════════════════════════════════════ */

/* 여러 CE Cell을 가중 합산하여 이미지 생성.
 * weights: 각 cell의 가중치 (NULL이면 전부 1.0).
 * out_image: uint8 그레이스케일 SLIG_CANVAS_DIM × SLIG_CANVAS_DIM. */
void slig_render(uint8_t *out_image,
                 const CEUnit *cells, int num_cells,
                 const uint8_t *weights);

/* 프로그레시브 렌더: cell 0~up_to까지만 적용. */
void slig_render_progressive(uint8_t *out_image,
                             const CEUnit *cells, int num_cells,
                             int up_to);

/* 시간축 렌더: 틱 0~max_tick까지 이벤트 누적. */
void slig_render_timed(uint8_t *out_image,
                       const CEUnit *cells, int num_cells,
                       uint8_t max_tick);

/* ══════════════════════════════════════════
 *  5. 오디오트랙 브릿지
 * ══════════════════════════════════════════ */

/* 256 격자의 에너지 분포를 오디오 스펙트럼으로 추출.
 * grid_A: 256×256 격자의 A채널 (밝기).
 * out_spectrum: [256] 주파수별 에너지. */
void slig_grid_to_spectrum(uint32_t *out_spectrum,
                           const uint16_t grid_A[256][256]);

/* 스펙트럼으로 CE Cell 가중치 계산.
 * 텍스트 격자의 주파수 분포가 어떤 CE Cell을 강조할지 결정. */
void slig_spectrum_weight(uint8_t *out_weights, int num_cells,
                          const CEUnit *cells,
                          const uint32_t *spectrum);

/* ══════════════════════════════════════════
 *  6. 보간
 * ══════════════════════════════════════════ */

/* 두 CE Cell 사이 보간. t: 0=a, 255=b. */
void slig_interpolate_cell(CEUnit *out,
                           const CEUnit *a, const CEUnit *b,
                           uint8_t t);

/* ══════════════════════════════════════════
 *  6.5  SLIG v2 — sequential AR / NeRF rendering
 *
 *  slig_render_sequential drives the v2 image-generation pipeline:
 *  given an array of SligRenderItems (each carrying a CE-Cell set
 *  pulled from a per-morpheme keyframe match), it lays them down on
 *  the canvas in three stages —
 *
 *    GLOBAL  : NOUN-class items → full-canvas signal application
 *              (the gross structure / shape).
 *    LOCAL   : ADJ-class items  → modulate only the already-drawn
 *              area (color/intensity attributes).
 *    EVENT   : VERB-class items → tick-based ripple/beam/cross
 *              accumulation (motion / state changes).
 *
 *  Items are stable-sorted by render_order ascending before the
 *  three stages so ties within a mode keep input ordering. The
 *  optional grid_A panel (256² A-channel from the SSS encoder)
 *  is forwarded so a future iteration can drive the audio-track
 *  spectrum from inside the renderer; the current build leaves
 *  the spectrum weighting upstream and ignores grid_A.
 *
 *  slig_place_by_grid_coords overrides a signal's origin so the
 *  caller can map a 256-grid coordinate ("byte position of the
 *  morpheme") onto a canvas position ("where on the image"); the
 *  current implementation is a 1:1 pass-through since the engine
 *  uses the same dimension for both.
 * ══════════════════════════════════════════ */

enum SligRenderMode {
    SLIG_RENDER_GLOBAL = 0,   /* nouns: full-canvas signal */
    SLIG_RENDER_LOCAL  = 1,   /* adjectives: modulate drawn region */
    SLIG_RENDER_EVENT  = 2    /* verbs: tick-based events */
};

typedef struct {
    const CEUnit *cells;        /* borrowed; not freed by the renderer */
    uint32_t      cell_count;
    uint8_t       render_mode;   /* SligRenderMode */
    uint8_t       render_order;  /* secondary sort key within a mode */
    uint8_t       render_weight; /* 0..255 strength multiplier */
    uint8_t       grid_x;        /* NeRF anchor — morpheme centroid */
    uint8_t       grid_y;
    /* Optional 256×256 spatial mask. NULL ⇒ full canvas. Per-pixel
     * mask byte multiplies the cell's contribution (0=skip, 255=full).
     * Used by direction-keyword routing in spatial_image_gen — a
     * "왼쪽 산" tag points at slig_make_mask_left so the noun's cells
     * only paint the left half of the canvas. */
    const uint8_t *mask;
    /* Event-stage tick upper bound. Events propagate cumulatively
     * over ticks 0..event_max_tick. Value 0 ⇒ the renderer's default
     * (15 ticks) — used by single-image generation. ai_generate_animation
     * sets this to the current frame index so events evolve over time. */
    uint8_t        event_max_tick;
} SligRenderItem;

void slig_render_sequential(
    uint8_t              *out_image,
    const SligRenderItem *items, uint32_t num_items,
    const uint16_t        grid_A[256][256]);

/* Same pipeline as slig_render_sequential but writes the int32
 * SligCanvas instead of normalising to uint8. Lets a caller
 * normalise over a sub-region (slig_canvas_to_u8_sub) instead of
 * the full 256² — important when small-dim cells leave most of the
 * canvas at zero, which would otherwise drag the global min-max. */
void slig_render_sequential_canvas(
    SligCanvas           *out_canvas,
    const SligRenderItem *items, uint32_t num_items,
    const uint16_t        grid_A[256][256]);

/* Min-max normalise a dim×dim square at (x0, y0) of `c` into out[].
 * Compute min and max only within that sub-region so the active
 * pixels span the full 0..255 output range regardless of zeros
 * elsewhere on the canvas. */
void slig_canvas_to_u8_sub(uint8_t          *out_dim_dim, int dim,
                           const SligCanvas *c,
                           int x0, int y0);

/* One-shot reconstruction: render a single SligCellSet at its
 * native scale into a dim×dim uint8 panel. Used by the residual
 * pyramid (encode + decode) where we need decompose⁻¹ at the same
 * dim that decompose produced the cells from. Empty sets render as
 * mid-gray (128) so they act as "zero residual" in the chain. */
void slig_reconstruct_at_dim(uint8_t           *out_dim_dim, int dim,
                             const SligCellSet *set);

void slig_place_by_grid_coords(
    SligCanvas       *canvas,
    const SligSignal *sig,
    uint8_t grid_x,    uint8_t grid_y,
    uint8_t canvas_cx, uint8_t canvas_cy);

/* ── Spatial masks for layout-driven rendering ────────────
 *
 *  slig_make_mask_left/right/top/bottom build a one-axis half-plane
 *  mask with a soft `feather`-pixel gradient at the boundary. They
 *  power the direction-keyword path in ai_generate_image_v2_guided
 *  ("왼쪽 산", "오른쪽 바다", …).
 *
 *  Buffer layout: row-major uint8 array, length = SLIG_CANVAS_DIM². */
void slig_make_mask_left  (uint8_t *out_256x256, int boundary, int feather);
void slig_make_mask_right (uint8_t *out_256x256, int boundary, int feather);
void slig_make_mask_top   (uint8_t *out_256x256, int boundary, int feather);
void slig_make_mask_bottom(uint8_t *out_256x256, int boundary, int feather);

/* ══════════════════════════════════════════
 *  6.6  SLIG v2 — canvas-target wave refine
 *
 *  Pixel-domain version of ce_image_wave_refine: the same
 *  0.01 → 0.1 → 1 → 3 → 1 → 0.1 → 0.01 amplitude cycle is run
 *  against an int32 SligCanvas, smoothing the rendered image
 *  toward `target` (or toward an internal Gaussian smoothing of
 *  itself when target == NULL). Iterations × cycle_len total
 *  passes are performed.
 * ══════════════════════════════════════════ */

void slig_wave_refine(SligCanvas       *canvas,
                      const SligCanvas *target,
                      int               iterations);

/* ══════════════════════════════════════════
 *  7. 유틸리티
 * ══════════════════════════════════════════ */

/* 캔버스 → 8-bit 이미지 (min-max 정규화). */
void slig_canvas_to_u8(uint8_t out[SLIG_CANVAS_DIM][SLIG_CANVAS_DIM],
                       const SligCanvas *c);

/* PPM 파일 저장 (디버그용). */
int slig_save_ppm(const char *path,
                  const uint8_t img[SLIG_CANVAS_DIM][SLIG_CANVAS_DIM]);

#ifdef __cplusplus
}
#endif
#endif /* SLIG_SIGNAL_H */
