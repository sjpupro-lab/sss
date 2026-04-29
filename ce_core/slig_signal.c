/* slig_signal.c — Signal-Latent Image Generation v2 implementation.
 *
 * CE Cell 안에 이미지 주파수 정보를 직접 담는 통합 엔진.
 * 모든 생성 경로 정수 연산. float는 DCT/SVD (학습 시)에만 사용.
 */

#include "slig_signal.h"
#include "slig_material_harmonic.h"   /* SligMaterialTick + auto-analyzer */
#include "slig_tick_math.h"            /* TICK_COS_TABLE — float-free cos LUT */
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── 내부 헬퍼 ── */
static inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}
static inline uint8_t clamp_u8(int32_t v) {
    return (uint8_t)clamp_i32(v, 0, 255);
}

/* cos(radians) via TICK_COS_TABLE. 256 entries ≡ one full 2π cycle.
 * Drop-in for cosf() in the wave-event paths so the periodic kernel is
 * float-free; surrounding float math (expf, sqrtf, atan2f) is left to
 * later cleanup steps. */
static inline float tick_cosf(float radians) {
    /* Map radians → table index: idx = round(radians × 256 / 2π) mod 256. */
    int idx = (int)(radians * (float)(256.0 / (2.0 * M_PI)));
    idx = ((idx % 256) + 256) % 256;
    return ((int)TICK_COS_TABLE[idx] - 128) * (1.0f / 128.0f);
}

/* ══════════════════════════════════════════════════
 *  DCT 압축/복원 (16 계수 ↔ 256 신호)
 *
 *  신호를 16개 코사인 주파수로 요약.
 *  CE Cell의 inc.G/B + dec.R/G 총 32바이트에 저장.
 * ══════════════════════════════════════════════════ */

/* 256 길이 신호 → 16개 DCT 계수 (float 내부 사용, 학습 시만) */
static void dct_forward_16(uint8_t coeff[SLIG_DCT_COEFF],
                           const float *signal, int len,
                           float *out_max_abs) {
    float raw[SLIG_DCT_COEFF];
    float ma = 0;
    for (int k = 0; k < SLIG_DCT_COEFF; k++) {
        float sum = 0;
        for (int n = 0; n < len; n++)
            sum += signal[n] * cosf((float)M_PI * k * (2*n+1) / (2*len));
        raw[k] = sum;
        float a = fabsf(sum);
        if (a > ma) ma = a;
    }
    *out_max_abs = (ma < 1e-8f) ? 1.0f : ma;
    for (int k = 0; k < SLIG_DCT_COEFF; k++)
        coeff[k] = clamp_u8((int32_t)((raw[k] / *out_max_abs) * 127.0f + 128.0f));
}

/* 16개 DCT 계수 → 256 길이 정수 신호 (정수 연산으로 복원) */
static void dct_inverse_16(int16_t *out, int len,
                           const uint8_t coeff[SLIG_DCT_COEFF]) {
    /* 코사인 테이블 (정수 256 스케일) */
    for (int n = 0; n < len; n++) {
        int32_t sum = 0;
        for (int k = 0; k < SLIG_DCT_COEFF; k++) {
            int32_t c = (int32_t)coeff[k] - 128;  /* [-128, 127] */
            /* cos 근사: 정수 테이블 사용 가능하지만 여기서는 간략 구현 */
            float angle = (float)M_PI * k * (2*n+1) / (2*len);
            int32_t cos_val = (int32_t)(cosf(angle) * 256.0f);
            sum += c * cos_val;
        }
        out[n] = (int16_t)clamp_i32(sum >> 8, -32000, 32000);
    }
}

/* ══════════════════════════════════════════════════
 *  1. CE Cell ↔ SligSignal 변환
 * ══════════════════════════════════════════════════ */

void slig_pack(CEUnit *cell, const SligSignal *sig) {
    ce_init(cell);

    /* inc.R = 메타데이터 (8 bytes) */
    /* CETier R: minus[4] + plus[4] */
    cell->inc.R.plus[0]  = sig->dir;
    cell->inc.R.plus[1]  = sig->scale;
    cell->inc.R.plus[2]  = (sig->sigma >> 8) & 0xFF;
    cell->inc.R.plus[3]  = sig->sigma & 0xFF;
    cell->inc.R.minus[0] = sig->frequency;
    cell->inc.R.minus[1] = sig->origin_x;
    cell->inc.R.minus[2] = sig->origin_y;
    cell->inc.R.minus[3] = sig->flags;

    /* inc.G = u DCT 계수 0~7 */
    /* u 신호를 DCT 계수로: 런타임 신호에서 직접 DCT */
    float u_f[SLIG_SIG_LEN], v_f[SLIG_SIG_LEN];
    for (int i = 0; i < SLIG_SIG_LEN; i++) {
        u_f[i] = (float)sig->u[i];
        v_f[i] = (float)sig->v[i];
    }

    uint8_t u_coeff[SLIG_DCT_COEFF], v_coeff[SLIG_DCT_COEFF];
    float u_max, v_max;
    dct_forward_16(u_coeff, u_f, SLIG_SIG_LEN, &u_max);
    dct_forward_16(v_coeff, v_f, SLIG_SIG_LEN, &v_max);

    cell->inc.G.plus[0]  = u_coeff[0];
    cell->inc.G.plus[1]  = u_coeff[1];
    cell->inc.G.plus[2]  = u_coeff[2];
    cell->inc.G.plus[3]  = u_coeff[3];
    cell->inc.G.minus[0] = u_coeff[4];
    cell->inc.G.minus[1] = u_coeff[5];
    cell->inc.G.minus[2] = u_coeff[6];
    cell->inc.G.minus[3] = u_coeff[7];

    /* inc.B = v DCT 계수 0~7 */
    cell->inc.B.plus[0]  = v_coeff[0];
    cell->inc.B.plus[1]  = v_coeff[1];
    cell->inc.B.plus[2]  = v_coeff[2];
    cell->inc.B.plus[3]  = v_coeff[3];
    cell->inc.B.minus[0] = v_coeff[4];
    cell->inc.B.minus[1] = v_coeff[5];
    cell->inc.B.minus[2] = v_coeff[6];
    cell->inc.B.minus[3] = v_coeff[7];

    /* inc.A = 에너지 프로파일 */
    /* 스케일별: 4칸 평균, 16칸 평균, 64칸 평균, 전체 평균 */
    for (int s = 0; s < 4; s++) {
        int32_t e = 0;
        int block = 1 << (s + 2); /* 4, 8, 16, 32 */
        for (int i = 0; i < SLIG_SIG_LEN && i < block; i++)
            e += abs(sig->u[i]);
        cell->inc.A.plus[s] = clamp_u8(e / block);
    }
    /* 방향별 에너지는 분해 시 채워짐 */
    cell->inc.A.minus[0] = (sig->dir == SLIG_DIR_HORIZONTAL) ? 255 : 0;
    cell->inc.A.minus[1] = (sig->dir == SLIG_DIR_VERTICAL)   ? 255 : 0;
    cell->inc.A.minus[2] = (sig->dir == SLIG_DIR_DIAG_DOWN || sig->dir == SLIG_DIR_DIAG_UP) ? 255 : 0;
    cell->inc.A.minus[3] = (sig->dir >= SLIG_DIR_RIPPLE)     ? 255 : 0;

    /* dec.R = u DCT 계수 8~15 */
    cell->dec.R.plus[0]  = u_coeff[8];
    cell->dec.R.plus[1]  = u_coeff[9];
    cell->dec.R.plus[2]  = u_coeff[10];
    cell->dec.R.plus[3]  = u_coeff[11];
    cell->dec.R.minus[0] = u_coeff[12];
    cell->dec.R.minus[1] = u_coeff[13];
    cell->dec.R.minus[2] = u_coeff[14];
    cell->dec.R.minus[3] = u_coeff[15];

    /* dec.G = v DCT 계수 8~15 */
    cell->dec.G.plus[0]  = v_coeff[8];
    cell->dec.G.plus[1]  = v_coeff[9];
    cell->dec.G.plus[2]  = v_coeff[10];
    cell->dec.G.plus[3]  = v_coeff[11];
    cell->dec.G.minus[0] = v_coeff[12];
    cell->dec.G.minus[1] = v_coeff[13];
    cell->dec.G.minus[2] = v_coeff[14];
    cell->dec.G.minus[3] = v_coeff[15];

    /* dec.B = 시간/트리거 */
    cell->dec.B.plus[0]  = sig->start_tick;
    cell->dec.B.plus[1]  = sig->decay;
    cell->dec.B.plus[2]  = sig->speed;
    cell->dec.B.plus[3]  = sig->beam_dir;
    cell->dec.B.minus[0] = sig->beam_width;
    cell->dec.B.minus[1] = sig->cond_threshold;
    cell->dec.B.minus[2] = sig->cond_type;
    cell->dec.B.minus[3] = sig->phase;

    /* dec.A = 오디오트랙 */
    for (int i = 0; i < 4; i++) {
        cell->dec.A.plus[i]  = sig->audio_bins[i];
        cell->dec.A.minus[i] = sig->audio_amps[i];
    }
}

void slig_unpack(SligSignal *sig, const CEUnit *cell) {
    memset(sig, 0, sizeof(*sig));

    /* 메타데이터 */
    sig->dir       = cell->inc.R.plus[0];
    sig->scale     = cell->inc.R.plus[1];
    sig->sigma     = ((uint16_t)cell->inc.R.plus[2] << 8) | cell->inc.R.plus[3];
    sig->frequency = cell->inc.R.minus[0];
    sig->origin_x  = cell->inc.R.minus[1];
    sig->origin_y  = cell->inc.R.minus[2];
    sig->flags     = cell->inc.R.minus[3];

    /* DCT 계수 추출 */
    uint8_t u_coeff[SLIG_DCT_COEFF], v_coeff[SLIG_DCT_COEFF];

    u_coeff[0]  = cell->inc.G.plus[0];  u_coeff[1]  = cell->inc.G.plus[1];
    u_coeff[2]  = cell->inc.G.plus[2];  u_coeff[3]  = cell->inc.G.plus[3];
    u_coeff[4]  = cell->inc.G.minus[0]; u_coeff[5]  = cell->inc.G.minus[1];
    u_coeff[6]  = cell->inc.G.minus[2]; u_coeff[7]  = cell->inc.G.minus[3];
    u_coeff[8]  = cell->dec.R.plus[0];  u_coeff[9]  = cell->dec.R.plus[1];
    u_coeff[10] = cell->dec.R.plus[2];  u_coeff[11] = cell->dec.R.plus[3];
    u_coeff[12] = cell->dec.R.minus[0]; u_coeff[13] = cell->dec.R.minus[1];
    u_coeff[14] = cell->dec.R.minus[2]; u_coeff[15] = cell->dec.R.minus[3];

    v_coeff[0]  = cell->inc.B.plus[0];  v_coeff[1]  = cell->inc.B.plus[1];
    v_coeff[2]  = cell->inc.B.plus[2];  v_coeff[3]  = cell->inc.B.plus[3];
    v_coeff[4]  = cell->inc.B.minus[0]; v_coeff[5]  = cell->inc.B.minus[1];
    v_coeff[6]  = cell->inc.B.minus[2]; v_coeff[7]  = cell->inc.B.minus[3];
    v_coeff[8]  = cell->dec.G.plus[0];  v_coeff[9]  = cell->dec.G.plus[1];
    v_coeff[10] = cell->dec.G.plus[2];  v_coeff[11] = cell->dec.G.plus[3];
    v_coeff[12] = cell->dec.G.minus[0]; v_coeff[13] = cell->dec.G.minus[1];
    v_coeff[14] = cell->dec.G.minus[2]; v_coeff[15] = cell->dec.G.minus[3];

    /* DCT → 신호 복원 */
    dct_inverse_16(sig->u, SLIG_SIG_LEN, u_coeff);
    dct_inverse_16(sig->v, SLIG_SIG_LEN, v_coeff);

    /* 시간/트리거 */
    sig->start_tick     = cell->dec.B.plus[0];
    sig->decay          = cell->dec.B.plus[1];
    sig->speed          = cell->dec.B.plus[2];
    sig->beam_dir       = cell->dec.B.plus[3];
    sig->beam_width     = cell->dec.B.minus[0];
    sig->cond_threshold = cell->dec.B.minus[1];
    sig->cond_type      = cell->dec.B.minus[2];
    sig->phase          = cell->dec.B.minus[3];

    /* 오디오트랙 */
    for (int i = 0; i < 4; i++) {
        sig->audio_bins[i] = cell->dec.A.plus[i];
        sig->audio_amps[i] = cell->dec.A.minus[i];
    }
}

/* ══════════════════════════════════════════════════
 *  2. 이미지 → CE Cell 세트 (학습)
 * ══════════════════════════════════════════════════ */

/* 간이 power iteration: 한 개 성분 추출 */
static void svd_one(const float *matrix, int dim,
                    float *out_u, float *out_v, float *out_sigma) {
    float *v = (float*)malloc(dim * sizeof(float));
    float *u = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++)
        v[i] = sinf(i * 2.3f) + cosf(i * 1.7f);
    float norm;
    for (int iter = 0; iter < 25; iter++) {
        for (int i = 0; i < dim; i++) {
            u[i] = 0;
            for (int j = 0; j < dim; j++) u[i] += matrix[i*dim+j] * v[j];
        }
        norm = 0;
        for (int i = 0; i < dim; i++) norm += u[i]*u[i];
        norm = sqrtf(norm); if (norm < 1e-10f) norm = 1.0f;
        for (int i = 0; i < dim; i++) u[i] /= norm;
        norm = 0;
        for (int j = 0; j < dim; j++) {
            v[j] = 0;
            for (int i = 0; i < dim; i++) v[j] += matrix[i*dim+j] * u[i];
        }
        for (int j = 0; j < dim; j++) norm += v[j]*v[j];
        *out_sigma = sqrtf(norm);
        if (*out_sigma < 1e-10f) *out_sigma = 1e-10f;
        for (int j = 0; j < dim; j++) v[j] /= *out_sigma;
    }
    memcpy(out_u, u, dim * sizeof(float));
    memcpy(out_v, v, dim * sizeof(float));
    free(u); free(v);
}

void slig_decompose_channel(SligCellSet *out,
                            const uint8_t *image, int width, int height,
                            uint8_t  channel,
                            uint8_t  scale_level,
                            uint32_t max_cells) {
    memset(out, 0, sizeof(*out));
    if (max_cells > SLIG_MAX_CELLS) max_cells = SLIG_MAX_CELLS;
    out->channel     = channel;
    out->scale_level = scale_level;

    int dim = (width < height) ? width : height;
    if (dim > SLIG_CANVAS_DIM) dim = SLIG_CANVAS_DIM;

    /* P2: encode source_dim in sig.phase so slig_render_adaptive can
     * upsample u/v back to canvas dim before rendering — fixes the
     * "small-dim cell signal concentrates in top-left" problem.
     * 0 = default canvas dim (no resample); else literal source_dim. */
    uint8_t source_dim_tag = (uint8_t)((dim < SLIG_CANVAS_DIM) ? dim : 0);

    /* Mat-2: auto-extract material parameters from the centre 8×8 block
     * of the input image once. Each Y-channel cell receives a copy via
     * slig_mat_to_cell (after slig_pack), so downstream renderers can
     * read texture descriptors without re-analysing the image. Skipped
     * on Cb/Cr — chroma cells need dec.A intact for marker dispatch. */
    SligMaterialTick mat_tick;
    int have_mat = 0;
    if (channel == SLIG_CH_Y && dim >= 8) {
        int bx = dim / 2 - 4;
        int by = dim / 2 - 4;
        if (bx < 0)            bx = 0;
        if (by < 0)            by = 0;
        if (bx + 8 > dim)      bx = dim - 8;
        if (by + 8 > dim)      by = dim - 8;
        slig_mat_analyze_block(&mat_tick, &image[by * dim + bx], dim);
        have_mat = 1;
    }

    /* float 변환 + 중앙값 빼기 */
    float *mat = (float*)malloc(dim * dim * sizeof(float));
    float mean = 0;
    for (int i = 0; i < dim * dim; i++) {
        mat[i] = image[i] / 255.0f;
        mean += mat[i];
    }
    mean /= (dim * dim);
    for (int i = 0; i < dim * dim; i++) mat[i] -= mean;

    float *residual = (float*)malloc(dim * dim * sizeof(float));
    memcpy(residual, mat, dim * dim * sizeof(float));

    float *u_f = (float*)malloc(dim * sizeof(float));
    float *v_f = (float*)malloc(dim * sizeof(float));

    /* ── 가로×세로 분해 (horizontal) ── */
    int layers_h = 12;  /* was 8: extended so Y can reach the 20-cell budget */
    for (int k = 0; k < layers_h && out->num_cells < max_cells; k++) {
        float sigma;
        svd_one(residual, dim, u_f, v_f, &sigma);
        /* was 0.01f — lowered so smaller singular values still pack into
         * cells. Below this floor the SVD is mostly numerical noise. */
        if (sigma < 0.003f) break;

        /* 잔차 빼기 */
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                residual[i*dim+j] -= sigma * u_f[i] * v_f[j];

        /* SligSignal 구성 */
        SligSignal sig;
        memset(&sig, 0, sizeof(sig));
        sig.phase = source_dim_tag;
        sig.dir = SLIG_DIR_HORIZONTAL;
        sig.scale = (k < 2) ? SLIG_SCALE_COARSE : (k < 5) ? SLIG_SCALE_MID : SLIG_SCALE_FINE;
        sig.sigma = clamp_i32((int32_t)(sigma * 100), 0, 65535);
        sig.frequency = (1 << (sig.scale + 2));
        sig.origin_x = 128; sig.origin_y = 128;
        for (int i = 0; i < dim; i++) {
            sig.u[i] = (int16_t)(u_f[i] * 10000.0f);
            sig.v[i] = (int16_t)(v_f[i] * 10000.0f);
        }

        /* CE Cell에 싸기 */
        slig_pack(&out->cells[out->num_cells], &sig);
        if (have_mat) slig_mat_to_cell(&out->cells[out->num_cells], &mat_tick);
        out->num_cells++;
    }

    /* ── 대각선 분해 (diag_down): 이미지를 45도 회전 후 SVD ── */
    memcpy(residual, mat, dim * dim * sizeof(float));
    /* 대각 좌표 변환 */
    float *diag_mat = (float*)calloc(dim * dim, sizeof(float));
    for (int y = 0; y < dim; y++)
        for (int x = 0; x < dim; x++) {
            int d = (x + y) % dim;
            int a = abs(x - y) % dim;
            diag_mat[d * dim + a] += residual[y * dim + x];
        }

    for (int k = 0; k < 4 && out->num_cells < max_cells; k++) {
        float sigma;
        svd_one(diag_mat, dim, u_f, v_f, &sigma);
        if (sigma < 0.05f) break;

        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                diag_mat[i*dim+j] -= sigma * u_f[i] * v_f[j];

        SligSignal sig;
        memset(&sig, 0, sizeof(sig));
        sig.phase = source_dim_tag;
        sig.dir = SLIG_DIR_DIAG_DOWN;
        sig.scale = (k < 1) ? SLIG_SCALE_COARSE : SLIG_SCALE_MID;
        sig.sigma = clamp_i32((int32_t)(sigma * 100), 0, 65535);
        sig.frequency = 8;
        sig.origin_x = 128; sig.origin_y = 128;
        for (int i = 0; i < dim; i++) {
            sig.u[i] = (int16_t)(u_f[i] * 10000.0f);
            sig.v[i] = (int16_t)(v_f[i] * 10000.0f);
        }
        slig_pack(&out->cells[out->num_cells], &sig);
        if (have_mat) slig_mat_to_cell(&out->cells[out->num_cells], &mat_tick);
        out->num_cells++;
    }

    /* ── 반대 대각 분해 (diag_up): 직교 대각 좌표 변환 후 SVD ──
     *
     * DIAG_DOWN 의 변환은 (d=(x+y), a=|x-y|) 축에서 패턴을 추출했다.
     * DIAG_UP 은 그 직교 변환 — (d=(x-y+dim)%dim, a=(x+y)%dim) 축. 같은
     * 이미지에서 반대 기울기의 결을 잡아낸다 (예: ╱ vs ╲). 렌더러의
     * SLIG_DIR_DIAG_UP 케이스가 이 두 축으로 u/v 를 읽으므로 cell 의
     * u, v 는 전치 없이 그대로 저장한다. */
    memcpy(residual, mat, dim * dim * sizeof(float));
    float *up_mat = (float*)calloc(dim * dim, sizeof(float));
    for (int y = 0; y < dim; y++)
        for (int x = 0; x < dim; x++) {
            int d = ((x - y) + dim) % dim;
            int a = (x + y) % dim;
            up_mat[d * dim + a] += residual[y * dim + x];
        }
    for (int k = 0; k < 4 && out->num_cells < max_cells; k++) {
        float sigma;
        svd_one(up_mat, dim, u_f, v_f, &sigma);
        if (sigma < 0.05f) break;

        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                up_mat[i*dim+j] -= sigma * u_f[i] * v_f[j];

        SligSignal sig;
        memset(&sig, 0, sizeof(sig));
        sig.phase = source_dim_tag;
        sig.dir = SLIG_DIR_DIAG_UP;
        sig.scale = (k < 1) ? SLIG_SCALE_COARSE : SLIG_SCALE_MID;
        sig.sigma = clamp_i32((int32_t)(sigma * 100), 0, 65535);
        sig.frequency = 8;
        sig.origin_x = 128; sig.origin_y = 128;
        for (int i = 0; i < dim; i++) {
            sig.u[i] = (int16_t)(u_f[i] * 10000.0f);
            sig.v[i] = (int16_t)(v_f[i] * 10000.0f);
        }
        slig_pack(&out->cells[out->num_cells], &sig);
        if (have_mat) slig_mat_to_cell(&out->cells[out->num_cells], &mat_tick);
        out->num_cells++;
    }
    free(up_mat);

    /* ── 지그재그 분해 (zigzag): snake-permutation matrix → SVD ──
     *
     * The renderer's SLIG_DIR_ZIGZAG case reads u[y] but flips x within
     * 4-row groups, so a faithful decomposition needs the SAME
     * permutation applied before SVD. We re-permute mat into zig_mat,
     * then SVD a few rank-1 components; the resulting cells render
     * out as snake-pattern texture overlays distinct from horizontal
     * and diagonal. */
    memcpy(residual, mat, dim * dim * sizeof(float));
    float *zig_mat = (float*)malloc(dim * dim * sizeof(float));
    for (int y = 0; y < dim; y++) {
        int reversed = (y / 4) % 2;
        for (int x = 0; x < dim; x++) {
            int sx = reversed ? (dim - 1 - x) : x;
            zig_mat[y * dim + x] = residual[y * dim + sx];
        }
    }
    for (int k = 0; k < 4 && out->num_cells < max_cells; k++) {
        float sigma;
        svd_one(zig_mat, dim, u_f, v_f, &sigma);
        if (sigma < 0.05f) break;

        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                zig_mat[i*dim+j] -= sigma * u_f[i] * v_f[j];

        SligSignal sig;
        memset(&sig, 0, sizeof(sig));
        sig.phase = source_dim_tag;
        sig.dir = SLIG_DIR_ZIGZAG;
        sig.scale = (k < 1) ? SLIG_SCALE_COARSE : SLIG_SCALE_MID;
        sig.sigma = clamp_i32((int32_t)(sigma * 100), 0, 65535);
        sig.frequency = 6;
        sig.origin_x = 128; sig.origin_y = 128;
        for (int i = 0; i < dim; i++) {
            sig.u[i] = (int16_t)(u_f[i] * 10000.0f);
            sig.v[i] = (int16_t)(v_f[i] * 10000.0f);
        }
        slig_pack(&out->cells[out->num_cells], &sig);
        if (have_mat) slig_mat_to_cell(&out->cells[out->num_cells], &mat_tick);
        out->num_cells++;
    }
    free(zig_mat);

    /* ── 이벤트 신호: 밝기 중심에서 물결 ── */
    if (out->num_cells < max_cells) {
        int32_t cx = 0, cy = 0, total = 0;
        for (int y = 0; y < dim; y++)
            for (int x = 0; x < dim; x++) {
                int32_t v = image[y * dim + x];
                cx += x * v; cy += y * v; total += v;
            }
        if (total > 0) { cx /= total; cy /= total; }
        else { cx = dim/2; cy = dim/2; }

        SligSignal sig;
        memset(&sig, 0, sizeof(sig));
        sig.phase = source_dim_tag;
        sig.dir = SLIG_DIR_RIPPLE;
        sig.sigma = clamp_i32((int32_t)(mean * 30000), 0, 65535);
        sig.frequency = dim / 4;
        sig.origin_x = (uint8_t)clamp_i32(cx, 0, 255);
        sig.origin_y = (uint8_t)clamp_i32(cy, 0, 255);
        sig.speed = 3;
        sig.decay = 230;  /* 0.90 */
        /* u/v는 방사형이라 DCT 불필요, 메타데이터로 충분 */
        sig.u[0] = (int16_t)(mean * 10000);
        slig_pack(&out->cells[out->num_cells], &sig);
        if (have_mat) slig_mat_to_cell(&out->cells[out->num_cells], &mat_tick);
        out->num_cells++;
    }

    free(mat); free(residual); free(diag_mat); free(u_f); free(v_f);
}

void slig_decompose(SligCellSet *out,
                    const uint8_t *image, int width, int height) {
    /* Backward-compatible: Y channel, finest scale, full cell budget. */
    slig_decompose_channel(out, image, width, height,
                           SLIG_CH_Y, SLIG_LEVEL_FINE, SLIG_MAX_CELLS);
}

/* ══════════════════════════════════════════════════
 *  RGB ↔ YCbCr (BT.601) — 8-bit planar
 *
 *  Forward (R, G, B in [0,255]):
 *    Y  = 0.299 R + 0.587 G + 0.114 B
 *    Cb = 128 + (-0.169 R − 0.331 G + 0.500 B)
 *    Cr = 128 + ( 0.500 R − 0.419 G − 0.081 B)
 *
 *  Inverse:
 *    R = Y + 1.402 (Cr − 128)
 *    G = Y − 0.344 (Cb − 128) − 0.714 (Cr − 128)
 *    B = Y + 1.772 (Cb − 128)
 *
 *  Integer math via Q15 fixed point keeps everything deterministic.
 * ══════════════════════════════════════════════════ */

void slig_rgb_to_ycbcr_planes(uint8_t *out_y, uint8_t *out_cb, uint8_t *out_cr,
                              const uint8_t *rgb_interleaved,
                              int width, int height) {
    if (!out_y || !out_cb || !out_cr || !rgb_interleaved) return;
    int n = width * height;
    for (int i = 0; i < n; i++) {
        int32_t R = rgb_interleaved[i*3 + 0];
        int32_t G = rgb_interleaved[i*3 + 1];
        int32_t B = rgb_interleaved[i*3 + 2];
        int32_t Y  = ( 9798 * R + 19235 * G +  3735 * B + 16384) >> 15;
        int32_t Cb = (-5537 * R - 10846 * G + 16384 * B + 16384) >> 15;
        int32_t Cr = (16384 * R - 13720 * G -  2664 * B + 16384) >> 15;
        Cb += 128; Cr += 128;
        out_y [i] = (uint8_t)clamp_i32(Y,  0, 255);
        out_cb[i] = (uint8_t)clamp_i32(Cb, 0, 255);
        out_cr[i] = (uint8_t)clamp_i32(Cr, 0, 255);
    }
}

void slig_ycbcr_planes_to_rgb(uint8_t *out_rgb_interleaved,
                              const uint8_t *y, const uint8_t *cb, const uint8_t *cr,
                              int width, int height) {
    if (!out_rgb_interleaved || !y || !cb || !cr) return;
    int n = width * height;
    for (int i = 0; i < n; i++) {
        int32_t Y  = y[i];
        int32_t Cb = (int32_t)cb[i] - 128;
        int32_t Cr = (int32_t)cr[i] - 128;
        int32_t R = Y + ((45941 * Cr + 16384) >> 15);
        int32_t G = Y - ((11277 * Cb + 23401 * Cr + 16384) >> 15);
        int32_t B = Y + ((58065 * Cb + 16384) >> 15);
        out_rgb_interleaved[i*3 + 0] = (uint8_t)clamp_i32(R, 0, 255);
        out_rgb_interleaved[i*3 + 1] = (uint8_t)clamp_i32(G, 0, 255);
        out_rgb_interleaved[i*3 + 2] = (uint8_t)clamp_i32(B, 0, 255);
    }
}

/* ══════════════════════════════════════════════════
 *  3. 캔버스 조작 + 신호 적용
 * ══════════════════════════════════════════════════ */

void slig_canvas_clear(SligCanvas *c) {
    memset(c, 0, sizeof(*c));
}

/* ── 전역 신호 적용 (방향별) ── */
static void apply_global(SligCanvas *c, const SligSignal *sig) {
    const int N = SLIG_CANVAS_DIM;
    int32_t amp = (int32_t)sig->sigma;
    if (sig->flags & SLIG_FLAG_NEGATIVE) amp = -amp;

    /* dec.A audio_amps[2] carries the per-component energy ratio that
     * slig_decompose_structure measured at training time:
     *   amp[2]/255  ==  sigma^2 / total_energy
     * Scale the rendered magnitude by that ratio so low-energy SVD
     * components contribute proportionally less, matching what the
     * decomposer actually saw.
     *
     * Backward compat: a cell is treated as "unmeasured" only when
     * BOTH audio_amps[2] == 0 AND cond_threshold == 0. The decomposer
     * always sets these two together (cumulative coverage and per-
     * component ratio), so a measured component with a tiny ratio that
     * quantises to amp[2]==0 will still have a non-zero cond_threshold
     * (cumulative coverage ≥ 1) and correctly scale amp to 0. Pre-
     * change .spai files have both fields zeroed and keep the legacy
     * sigma-only amplitude. */
    int measured = (sig->audio_amps[2] != 0) || (sig->cond_threshold != 0);
    if (measured) {
        amp = (int32_t)(((int64_t)amp * (int32_t)sig->audio_amps[2]) / 255);
    }

    /* Conditional gating (cond_type=1, brightness): only contribute
     * where the canvas already has |value| ≥ cond_threshold scaled
     * into canvas units. Threshold byte 0..255 maps to canvas
     * magnitude 0..65280 (×256), which is the rough range for
     * accumulated post-stage-1 values. */
    int      cond_active    = (sig->cond_type == 1);
    int32_t  cond_threshold = (int32_t)sig->cond_threshold * 256;

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (cond_active) {
                int32_t cur = c->pixels[y][x];
                if (cur < 0) cur = -cur;
                if (cur < cond_threshold) continue;
            }
            int32_t u_val, v_val;

            switch (sig->dir) {
            case SLIG_DIR_HORIZONTAL:
                u_val = sig->u[y]; v_val = sig->v[x]; break;
            case SLIG_DIR_VERTICAL:
                u_val = sig->u[x]; v_val = sig->v[y]; break;
            case SLIG_DIR_DIAG_DOWN:
                u_val = sig->u[(x+y) % N]; v_val = sig->v[abs(x-y) % N]; break;
            case SLIG_DIR_DIAG_UP:
                u_val = sig->u[((x-y)+N) % N]; v_val = sig->v[(x+y) % N]; break;
            case SLIG_DIR_ZIGZAG: {
                int reversed = (y / 4) % 2;
                int rx = reversed ? (N - 1 - x) : x;
                u_val = sig->u[y]; v_val = sig->v[rx];
            } break;
            case SLIG_DIR_REVERSE:
                u_val = sig->u[N-1-y]; v_val = sig->v[N-1-x]; break;
            default:
                u_val = sig->u[y]; v_val = sig->v[x]; break;
            }

            /* 정수 곱셈: amp × u × v, 스케일 조정 */
            int64_t product = (int64_t)amp * u_val * v_val;
            c->pixels[y][x] += (int32_t)(product >> 20);
        }
    }
}

/* ── 이벤트 신호 적용 (좌표 시작, 시간 기반) ── */
static void apply_event(SligCanvas *c, const SligSignal *sig, uint8_t tick) {
    if (tick < sig->start_tick) return;
    const int N = SLIG_CANVAS_DIM;
    int elapsed = tick - sig->start_tick;
    /* cond_type / cond_threshold are honoured only for global signals
     * — events have their own origin/decay semantics and gating them
     * by canvas brightness conflicts with the time-evolution model. */

    /* 감쇠: decay 0~255 → 0.80~1.00 */
    float decay_f = 0.80f + (sig->decay / 255.0f) * 0.20f;
    float amp_f = (float)sig->sigma * powf(decay_f, (float)elapsed);
    if (fabsf(amp_f) < 1.0f) return;
    int32_t amp = (int32_t)amp_f;
    if (sig->flags & SLIG_FLAG_NEGATIVE) amp = -amp;

    int32_t radius = elapsed * (int32_t)(sig->speed > 0 ? sig->speed : 1);
    int cx = sig->origin_x, cy = sig->origin_y;

    if (sig->dir == SLIG_DIR_RIPPLE) {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                int32_t dx = x - cx, dy = y - cy;
                int32_t dist_sq = dx*dx + dy*dy;
                int32_t dist = (int32_t)sqrtf((float)dist_sq);
                if (dist > radius + sig->speed * 2) continue;
                int32_t diff = dist - radius;
                if (abs(diff) > sig->speed * 3) continue;
                /* 물결 파형 */
                int32_t freq = sig->frequency > 0 ? sig->frequency : 8;
                float wave = tick_cosf((float)diff * 2.0f * (float)M_PI / freq);
                float falloff = expf(-(float)(diff*diff) / (float)(sig->speed*sig->speed*4));
                c->pixels[y][x] += (int32_t)(amp * wave * falloff) >> 8;
            }
        }
    } else if (sig->dir == SLIG_DIR_BEAM) {
        float angles[8] = {0, M_PI, M_PI/2, -M_PI/2,
                           M_PI/4, -M_PI/4, 3*M_PI/4, -3*M_PI/4};
        float angle = angles[sig->beam_dir % 8];
        float wr = (sig->beam_width / 180.0f) * (float)M_PI;
        if (wr < 0.05f) wr = 0.15f;

        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                float dx = (float)(x-cx), dy = (float)(y-cy);
                float dist = sqrtf(dx*dx+dy*dy);
                if (dist < 1 || dist > radius) continue;
                float da = fabsf(atan2f(dy,dx) - angle);
                if (da > (float)M_PI) da = 2*(float)M_PI - da;
                if (da < wr) {
                    float along = tick_cosf(dist * 2*(float)M_PI / sig->frequency);
                    float across = tick_cosf(da / wr * (float)M_PI / 2);
                    c->pixels[y][x] += (int32_t)(amp * along * across * expf(-dist*0.01f)) >> 8;
                }
            }
        }
    } else if (sig->dir == SLIG_DIR_CROSS) {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                int32_t dx = abs(x-cx), dy = abs(y-cy);
                int32_t minD = dx < dy ? dx : dy;
                int32_t maxD = dx > dy ? dx : dy;
                if (minD < sig->speed * 2 && maxD < radius) {
                    float wave = tick_cosf((float)(maxD-radius)*2*(float)M_PI/sig->frequency);
                    float narrow = expf(-(float)(minD*minD) / (float)(sig->speed*sig->speed));
                    c->pixels[y][x] += (int32_t)(amp * wave * narrow) >> 8;
                }
            }
        }
    } else if (sig->dir == SLIG_DIR_CASCADE) {
        /* 1차 물결 */
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                int32_t dx = x-cx, dy = y-cy;
                int32_t dist = (int32_t)sqrtf((float)(dx*dx+dy*dy));
                if (dist < radius + sig->speed) {
                    int32_t diff = dist - radius;
                    float wave = tick_cosf((float)diff * 2*(float)M_PI / sig->frequency);
                    float fo = expf(-(float)(diff*diff) / (float)(sig->speed*sig->speed*4));
                    c->pixels[y][x] += (int32_t)(amp * wave * fo) >> 8;
                }
                /* 2차 물결 (연쇄) */
                int32_t bounceR = sig->frequency * 3;
                if (radius > bounceR) {
                    int32_t d2 = abs(dist - bounceR);
                    int32_t r2 = radius - bounceR;
                    if (d2 < r2 + sig->speed) {
                        float w2 = tick_cosf((float)(d2-r2) * 2*(float)M_PI / (sig->frequency * 0.7f));
                        float f2 = expf(-(float)((d2-r2)*(d2-r2)) / (float)(sig->speed*sig->speed*3));
                        c->pixels[y][x] += (int32_t)(amp * 0.4f * w2 * f2) >> 8;
                    }
                }
            }
        }
    }
}

void slig_apply(SligCanvas *c, const SligSignal *sig) {
    if (sig->dir < SLIG_DIR_RIPPLE)
        apply_global(c, sig);
    else
        apply_event(c, sig, sig->start_tick);  /* 즉시 적용 */
}

void slig_apply_at_tick(SligCanvas *c, const SligSignal *sig, uint8_t tick) {
    if (sig->dir < SLIG_DIR_RIPPLE)
        apply_global(c, sig);
    else
        apply_event(c, sig, tick);
}

void slig_apply_cell(SligCanvas *c, const CEUnit *cell) {
    SligSignal sig;
    slig_unpack(&sig, cell);
    slig_apply(c, &sig);
}

void slig_apply_cell_at_tick(SligCanvas *c, const CEUnit *cell, uint8_t tick) {
    SligSignal sig;
    slig_unpack(&sig, cell);
    slig_apply_at_tick(c, &sig, tick);
}

/* ══════════════════════════════════════════════════
 *  4. 렌더링
 * ══════════════════════════════════════════════════ */

void slig_canvas_to_u8(uint8_t out[SLIG_CANVAS_DIM][SLIG_CANVAS_DIM],
                       const SligCanvas *c) {
    int32_t mn = INT32_MAX, mx = INT32_MIN;
    const int N = SLIG_CANVAS_DIM;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            if (c->pixels[y][x] < mn) mn = c->pixels[y][x];
            if (c->pixels[y][x] > mx) mx = c->pixels[y][x];
        }
    int32_t range = mx - mn;
    if (range == 0) range = 1;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
            out[y][x] = (uint8_t)(((int64_t)(c->pixels[y][x]-mn)*255) / range);
}

void slig_render(uint8_t *out_image,
                 const CEUnit *cells, int num_cells,
                 const uint8_t *weights) {
    SligCanvas canvas;
    slig_canvas_clear(&canvas);
    for (int i = 0; i < num_cells; i++) {
        SligSignal sig;
        slig_unpack(&sig, &cells[i]);
        /* 가중치 적용 */
        if (weights) {
            uint32_t w = weights[i];
            sig.sigma = (uint16_t)((uint32_t)sig.sigma * w / 255);
        }
        slig_apply(&canvas, &sig);
    }
    slig_canvas_to_u8((uint8_t(*)[SLIG_CANVAS_DIM])out_image, &canvas);
}

void slig_render_progressive(uint8_t *out_image,
                             const CEUnit *cells, int num_cells,
                             int up_to) {
    int n = (up_to > 0 && up_to < num_cells) ? up_to : num_cells;
    slig_render(out_image, cells, n, NULL);
}

void slig_render_timed(uint8_t *out_image,
                       const CEUnit *cells, int num_cells,
                       uint8_t max_tick) {
    SligCanvas canvas;
    slig_canvas_clear(&canvas);
    for (uint8_t t = 0; t <= max_tick; t++) {
        for (int i = 0; i < num_cells; i++)
            slig_apply_cell_at_tick(&canvas, &cells[i], t);
    }
    slig_canvas_to_u8((uint8_t(*)[SLIG_CANVAS_DIM])out_image, &canvas);
}

/* ══════════════════════════════════════════════════
 *  5. 오디오트랙 브릿지
 * ══════════════════════════════════════════════════ */

void slig_grid_to_spectrum(uint32_t *out_spectrum,
                           const uint16_t grid_A[256][256]) {
    /* 256 격자의 각 X열(=바이트값=주파수빈)의 에너지 합산 */
    for (int x = 0; x < 256; x++) {
        uint32_t energy = 0;
        for (int y = 0; y < 256; y++)
            energy += grid_A[y][x];
        out_spectrum[x] = energy;
    }
}

void slig_spectrum_weight(uint8_t *out_weights, int num_cells,
                          const CEUnit *cells,
                          const uint32_t *spectrum) {
    /* 각 CE Cell의 오디오트랙 빈과 스펙트럼의 겹침으로 가중치 계산 */
    uint32_t max_overlap = 0;
    uint32_t overlaps[SLIG_MAX_CELLS];

    for (int i = 0; i < num_cells; i++) {
        SligSignal sig;
        slig_unpack(&sig, &cells[i]);

        uint32_t overlap = 0;
        /* Mat-3: cells whose dec.A holds an auto-extracted material
         * descriptor (HAS_MATERIAL flag) keep their audio_bins/amps
         * fields unmaintained — interpreting those bytes as audio
         * indices would produce arbitrary spectrum lookups. Skip the
         * audio overlap and let the dir-based fallback supply weight. */
        int has_mat = (sig.flags & SLIG_FLAG_HAS_MATERIAL) ? 1 : 0;
        if (!has_mat) {
            for (int b = 0; b < 4; b++) {
                uint8_t bin = sig.audio_bins[b];
                uint8_t amp = sig.audio_amps[b];
                if (amp > 0)
                    overlap += spectrum[bin] * (uint32_t)amp;
            }
        }
        /* 전역 신호는 기본 가중치 */
        if (sig.dir < SLIG_DIR_RIPPLE && overlap == 0)
            overlap = 128;

        overlaps[i] = overlap;
        if (overlap > max_overlap) max_overlap = overlap;
    }

    if (max_overlap == 0) max_overlap = 1;
    for (int i = 0; i < num_cells; i++)
        out_weights[i] = (uint8_t)((uint64_t)overlaps[i] * 255 / max_overlap);
}

/* ══════════════════════════════════════════════════
 *  6. 보간
 * ══════════════════════════════════════════════════ */

void slig_interpolate_cell(CEUnit *out,
                           const CEUnit *a, const CEUnit *b,
                           uint8_t t) {
    const uint8_t *ra = ce_cbytes(a);
    const uint8_t *rb = ce_cbytes(b);
    uint8_t *ro = ce_bytes(out);
    uint32_t ta = 255 - (uint32_t)t;
    uint32_t tb = (uint32_t)t;
    for (int i = 0; i < 64; i++)
        ro[i] = (uint8_t)((ta * ra[i] + tb * rb[i]) / 255);
}

/* ══════════════════════════════════════════════════
 *  7. 유틸리티
 * ══════════════════════════════════════════════════ */

int slig_save_ppm(const char *path,
                  const uint8_t img[SLIG_CANVAS_DIM][SLIG_CANVAS_DIM]) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P6\n%d %d\n255\n", SLIG_CANVAS_DIM, SLIG_CANVAS_DIM);
    for (int y = 0; y < SLIG_CANVAS_DIM; y++)
        for (int x = 0; x < SLIG_CANVAS_DIM; x++) {
            uint8_t v = img[y][x];
            fputc(v, f); fputc(v, f); fputc(v, f);
        }
    fclose(f);
    return 1;
}

/* ══════════════════════════════════════════════════
 *  SLIG v2 — sequential AR/NeRF render & place-by-grid
 * ══════════════════════════════════════════════════ */

/* Stable insertion sort on render_order (small N, ≤64). */
static void sort_items_by_order(SligRenderItem *items, uint32_t n) {
    for (uint32_t i = 1; i < n; i++) {
        SligRenderItem t = items[i];
        int32_t j = (int32_t)i - 1;
        while (j >= 0 && items[j].render_order > t.render_order) {
            items[j + 1] = items[j];
            j--;
        }
        items[j + 1] = t;
    }
}

void slig_place_by_grid_coords(SligCanvas       *canvas,
                               const SligSignal *sig,
                               uint8_t grid_x,    uint8_t grid_y,
                               uint8_t canvas_cx, uint8_t canvas_cy) {
    if (!canvas || !sig) return;
    SligSignal s = *sig;
    /* The engine uses 256² for both grid and canvas, so the linear map
     * is identity. We honour the caller's anchor preference via
     * (canvas_cx, canvas_cy); (grid_x, grid_y) currently kept for
     * future affine extension. */
    s.origin_x = canvas_cx;
    s.origin_y = canvas_cy;
    (void)grid_x; (void)grid_y;
    slig_apply(canvas, &s);
}

void slig_render_sequential_canvas(SligCanvas           *out_canvas,
                                   const SligRenderItem *items, uint32_t num_items,
                                   const uint16_t        grid_A[256][256]) {
    slig_canvas_clear(out_canvas);
    if (num_items == 0 || !items) return;
    (void)grid_A; /* spectrum weighting is applied upstream via slig_spectrum_weight */
    /* Local alias so the existing 3-stage body keeps reading `canvas`. */
    SligCanvas *canvas_ptr = out_canvas;
#define canvas (*canvas_ptr)

    /* Take a working copy so we can sort without mutating caller data. */
    SligRenderItem sorted[64];
    uint32_t n = num_items < 64 ? num_items : 64;
    for (uint32_t i = 0; i < n; i++) sorted[i] = items[i];
    sort_items_by_order(sorted, n);

    /* ── Stage 1: GLOBAL (NOUN) — gross structure ── */
    for (uint32_t i = 0; i < n; i++) {
        const SligRenderItem *it = &sorted[i];
        if (it->render_mode != SLIG_RENDER_GLOBAL) continue;
        if (it->render_weight == 0 || it->cell_count == 0 || !it->cells) continue;
        if (it->mask) {
            /* Spatial-masked path: render to scratch then blend by mask. */
            SligCanvas tmp;
            slig_canvas_clear(&tmp);
            for (uint32_t c = 0; c < it->cell_count; c++) {
                SligSignal sig;
                slig_unpack(&sig, &it->cells[c]);
                sig.sigma = (uint16_t)((uint32_t)sig.sigma * it->render_weight / 255);
                slig_place_by_grid_coords(&tmp, &sig,
                                          it->grid_x, it->grid_y,
                                          it->grid_x, it->grid_y);
            }
            for (int y = 0; y < SLIG_CANVAS_DIM; y++) {
                for (int x = 0; x < SLIG_CANVAS_DIM; x++) {
                    int m = it->mask[y * SLIG_CANVAS_DIM + x];
                    canvas.pixels[y][x] += (tmp.pixels[y][x] * m) / 255;
                }
            }
        } else {
            for (uint32_t c = 0; c < it->cell_count; c++) {
                SligSignal sig;
                slig_unpack(&sig, &it->cells[c]);
                sig.sigma = (uint16_t)((uint32_t)sig.sigma * it->render_weight / 255);
                slig_place_by_grid_coords(&canvas, &sig,
                                          it->grid_x, it->grid_y,
                                          it->grid_x, it->grid_y);
            }
        }
    }

    /* Capture mask of "already drawn" area for stage 2. We threshold at
     * 1/8 of the post-stage-1 max magnitude so weakly-illuminated cells
     * are excluded from local modulation. */
    uint8_t mask[SLIG_CANVAS_DIM * SLIG_CANVAS_DIM];
    int32_t max_abs = 0;
    for (int y = 0; y < SLIG_CANVAS_DIM; y++) {
        for (int x = 0; x < SLIG_CANVAS_DIM; x++) {
            int32_t v = canvas.pixels[y][x];
            if (v < 0) v = -v;
            if (v > max_abs) max_abs = v;
        }
    }
    int32_t threshold = max_abs >> 3;
    for (int y = 0; y < SLIG_CANVAS_DIM; y++) {
        for (int x = 0; x < SLIG_CANVAS_DIM; x++) {
            int32_t v = canvas.pixels[y][x];
            if (v < 0) v = -v;
            mask[y * SLIG_CANVAS_DIM + x] = (v >= threshold) ? 1 : 0;
        }
    }

    /* ── Stage 2: LOCAL (ADJ) — modulate only the drawn region ── */
    for (uint32_t i = 0; i < n; i++) {
        const SligRenderItem *it = &sorted[i];
        if (it->render_mode != SLIG_RENDER_LOCAL) continue;
        if (it->render_weight == 0 || it->cell_count == 0 || !it->cells) continue;
        SligCanvas adj;
        slig_canvas_clear(&adj);
        for (uint32_t c = 0; c < it->cell_count; c++) {
            SligSignal sig;
            slig_unpack(&sig, &it->cells[c]);
            sig.sigma = (uint16_t)((uint32_t)sig.sigma * it->render_weight / 255);
            slig_apply(&adj, &sig);
        }
        for (int y = 0; y < SLIG_CANVAS_DIM; y++) {
            for (int x = 0; x < SLIG_CANVAS_DIM; x++) {
                int idx = y * SLIG_CANVAS_DIM + x;
                if (!mask[idx]) continue;
                int contrib = adj.pixels[y][x] >> 1;
                if (it->mask) contrib = (contrib * it->mask[idx]) / 255;
                canvas.pixels[y][x] += contrib;
            }
        }
    }

    /* ── Stage 3: EVENT (VERB) — tick-based ripple/beam/cross ── */
    for (uint32_t i = 0; i < n; i++) {
        const SligRenderItem *it = &sorted[i];
        if (it->render_mode != SLIG_RENDER_EVENT) continue;
        if (it->render_weight == 0 || it->cell_count == 0 || !it->cells) continue;
        /* Resolve the upper tick bound: 0 (caller default) means "render
         * the full 0..15 evolution" — the still-image case. Animation
         * passes the current frame index so events build up over time. */
        uint8_t max_t = (it->event_max_tick > 0) ? it->event_max_tick : 15;
        if (it->mask) {
            SligCanvas tmp;
            slig_canvas_clear(&tmp);
            for (uint32_t c = 0; c < it->cell_count; c++) {
                SligSignal sig;
                slig_unpack(&sig, &it->cells[c]);
                sig.sigma = (uint16_t)((uint32_t)sig.sigma * it->render_weight / 255);
                sig.origin_x = it->grid_x;
                sig.origin_y = it->grid_y;
                for (uint8_t t = 0; t <= max_t; t++) {
                    slig_apply_at_tick(&tmp, &sig, t);
                }
            }
            for (int y = 0; y < SLIG_CANVAS_DIM; y++) {
                for (int x = 0; x < SLIG_CANVAS_DIM; x++) {
                    int m = it->mask[y * SLIG_CANVAS_DIM + x];
                    canvas.pixels[y][x] += (tmp.pixels[y][x] * m) / 255;
                }
            }
        } else {
            for (uint32_t c = 0; c < it->cell_count; c++) {
                SligSignal sig;
                slig_unpack(&sig, &it->cells[c]);
                sig.sigma = (uint16_t)((uint32_t)sig.sigma * it->render_weight / 255);
                /* Anchor verb events at the morpheme's grid coord. */
                sig.origin_x = it->grid_x;
                sig.origin_y = it->grid_y;
                for (uint8_t t = 0; t <= max_t; t++) {
                    slig_apply_at_tick(&canvas, &sig, t);
                }
            }
        }
    }

    /* ── Stage 4: wave-refine convergence ── */
    slig_wave_refine(&canvas, NULL, 1);
#undef canvas
}

void slig_render_sequential(uint8_t              *out_image,
                            const SligRenderItem *items, uint32_t num_items,
                            const uint16_t        grid_A[256][256]) {
    SligCanvas canvas;
    slig_render_sequential_canvas(&canvas, items, num_items, grid_A);
    slig_canvas_to_u8((uint8_t(*)[SLIG_CANVAS_DIM])out_image, &canvas);
}

void slig_canvas_to_u8_sub(uint8_t          *out_dim_dim, int dim,
                           const SligCanvas *c,
                           int x0, int y0) {
    if (!out_dim_dim || !c || dim <= 0) return;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 + dim > SLIG_CANVAS_DIM) dim = SLIG_CANVAS_DIM - x0;
    if (y0 + dim > SLIG_CANVAS_DIM) dim = SLIG_CANVAS_DIM - y0;

    int32_t mn = INT32_MAX, mx = INT32_MIN;
    for (int y = 0; y < dim; y++) {
        for (int x = 0; x < dim; x++) {
            int32_t v = c->pixels[y0 + y][x0 + x];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
    }
    int32_t range = mx - mn;
    if (range == 0) range = 1;
    for (int y = 0; y < dim; y++) {
        for (int x = 0; x < dim; x++) {
            out_dim_dim[y * dim + x] =
                (uint8_t)(((int64_t)(c->pixels[y0 + y][x0 + x] - mn) * 255) / range);
        }
    }
}

/* ══════════════════════════════════════════════════
 *  SLIG v2 — canvas-target wave refine
 *
 *  Mirrors ce_image_wave_refine's 7-step amplitude cycle
 *  (0.01 → 0.1 → 1 → 3 → 1 → 0.1 → 0.01) but operates on
 *  the int32 SligCanvas. Per-step amplitude is computed as
 *  `(cycle_num × max_abs) / 100`, so the cycle is anchored
 *  to the canvas's own dynamic range.
 *
 *  When `target` is NULL we synthesise one by 3×3 box-blurring
 *  the canvas — this gives self-smoothing convergence without
 *  pulling toward a separate goal image.
 * ══════════════════════════════════════════════════ */

void slig_wave_refine(SligCanvas       *canvas,
                      const SligCanvas *target,
                      int               iterations) {
    if (!canvas || iterations <= 0) return;

    static const int wave_num[7] = { 1, 10, 100, 300, 100, 10, 1 };  /* ×0.01 */

    int32_t max_abs = 0;
    for (int y = 0; y < SLIG_CANVAS_DIM; y++) {
        for (int x = 0; x < SLIG_CANVAS_DIM; x++) {
            int32_t v = canvas->pixels[y][x];
            if (v < 0) v = -v;
            if (v > max_abs) max_abs = v;
        }
    }
    if (max_abs == 0) return;

    int32_t scale = max_abs / 100;
    if (scale < 1) scale = 1;

    /* Synthesise an auto-target via 3×3 box blur if caller didn't pass one. */
    SligCanvas *auto_target = NULL;
    const SligCanvas *tgt = target;
    if (!tgt) {
        auto_target = (SligCanvas *)malloc(sizeof(SligCanvas));
        if (!auto_target) return;
        for (int y = 0; y < SLIG_CANVAS_DIM; y++) {
            for (int x = 0; x < SLIG_CANVAS_DIM; x++) {
                int64_t sum = 0;
                int cnt = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    int yy = y + dy;
                    if (yy < 0 || yy >= SLIG_CANVAS_DIM) continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        int xx = x + dx;
                        if (xx < 0 || xx >= SLIG_CANVAS_DIM) continue;
                        sum += canvas->pixels[yy][xx];
                        cnt++;
                    }
                }
                auto_target->pixels[y][x] = (int32_t)(sum / cnt);
            }
        }
        tgt = auto_target;
    }

    int total_steps = iterations * 7;
    for (int it = 0; it < total_steps; it++) {
        int32_t step = (int32_t)wave_num[it % 7] * scale;
        for (int y = 0; y < SLIG_CANVAS_DIM; y++) {
            for (int x = 0; x < SLIG_CANVAS_DIM; x++) {
                int32_t cur  = canvas->pixels[y][x];
                int32_t goal = tgt->pixels[y][x];
                int32_t diff = goal - cur;
                if      (diff >  step) diff =  step;
                else if (diff < -step) diff = -step;
                canvas->pixels[y][x] = cur + diff;
            }
        }
    }

    if (auto_target) free(auto_target);
}

/* ══════════════════════════════════════════════════
 *  Image pyramid helpers (uint8 single-channel, square)
 * ══════════════════════════════════════════════════ */

void slig_image_downsample(uint8_t *dst, int dst_dim,
                           const uint8_t *src, int src_dim) {
    if (!dst || !src || dst_dim <= 0 || src_dim < dst_dim) return;
    int block = src_dim / dst_dim;
    if (block < 1) block = 1;
    int n = block * block;
    for (int dy = 0; dy < dst_dim; dy++) {
        for (int dx = 0; dx < dst_dim; dx++) {
            int sy0 = dy * block, sx0 = dx * block;
            uint32_t sum = 0;
            for (int yy = 0; yy < block; yy++) {
                int sy = sy0 + yy;
                if (sy >= src_dim) sy = src_dim - 1;
                for (int xx = 0; xx < block; xx++) {
                    int sx = sx0 + xx;
                    if (sx >= src_dim) sx = src_dim - 1;
                    sum += src[sy * src_dim + sx];
                }
            }
            dst[dy * dst_dim + dx] = (uint8_t)(sum / n);
        }
    }
}

void slig_image_upsample(uint8_t *dst, int dst_dim,
                         const uint8_t *src, int src_dim) {
    if (!dst || !src || src_dim <= 0 || dst_dim < src_dim) return;
    /* Nearest-neighbour: dst[y][x] = src[y*src_dim/dst_dim][x*src_dim/dst_dim] */
    for (int dy = 0; dy < dst_dim; dy++) {
        int sy = (dy * src_dim) / dst_dim;
        if (sy >= src_dim) sy = src_dim - 1;
        for (int dx = 0; dx < dst_dim; dx++) {
            int sx = (dx * src_dim) / dst_dim;
            if (sx >= src_dim) sx = src_dim - 1;
            dst[dy * dst_dim + dx] = src[sy * src_dim + sx];
        }
    }
}

/* ══════════════════════════════════════════════════
 *  Classifier-Free Guidance
 *
 *  out[i] = clamp(128 + scale × (cond[i] − 128), 0, 255)
 *
 *  scale is converted to Q8 fixed-point so the per-pixel loop is
 *  integer-only (subtract, multiply, shift). Common scale values:
 *    1.0  → identity (just the conditional render)
 *    1.5  → moderate emphasis
 *    2.0  → strong emphasis (described as "두 배 강조" upstream)
 *    3.0  → caricatured / oversaturated
 * ══════════════════════════════════════════════════ */

void slig_render_guided(uint8_t              *out_image,
                        const SligRenderItem *items, uint32_t num_items,
                        float                 guidance_scale,
                        const uint16_t        grid_A[256][256]) {
    int n = SLIG_CANVAS_DIM * SLIG_CANVAS_DIM;
    slig_render_sequential(out_image, items, num_items, grid_A);
    if (guidance_scale == 1.0f) return;

    int s_q8 = (int)(guidance_scale * 256.0f);
    if (s_q8 < 0) s_q8 = 0;
    for (int i = 0; i < n; i++) {
        int diff = (int)out_image[i] - 128;
        int v    = 128 + (s_q8 * diff) / 256;
        if (v < 0)   v = 0;
        if (v > 255) v = 255;
        out_image[i] = (uint8_t)v;
    }
}

/* ══════════════════════════════════════════════════
 *  Spatial masks for layout-driven rendering
 * ══════════════════════════════════════════════════ */

void slig_apply_masked(SligCanvas       *c,
                       const SligSignal *sig,
                       const uint8_t    *mask) {
    if (!c || !sig || !mask) return;
    SligCanvas tmp;
    slig_canvas_clear(&tmp);
    slig_apply(&tmp, sig);
    for (int y = 0; y < SLIG_CANVAS_DIM; y++) {
        for (int x = 0; x < SLIG_CANVAS_DIM; x++) {
            int m = mask[y * SLIG_CANVAS_DIM + x];
            c->pixels[y][x] += (tmp.pixels[y][x] * m) / 255;
        }
    }
}

/* Half-plane mask along an axis. Half == 0 ("above the boundary")
 * means the lower coordinate range is opaque (255), upper is
 * transparent (0); flip via `invert`. */
static void make_axis_mask(uint8_t *out,
                           int boundary, int feather,
                           int axis_is_x, int invert) {
    if (!out) return;
    int half = feather / 2;
    int start = boundary - half;
    int end   = boundary + half;
    if (start < 0) start = 0;
    if (end > SLIG_CANVAS_DIM - 1) end = SLIG_CANVAS_DIM - 1;
    int span = end - start;
    if (span < 1) span = 1;
    for (int y = 0; y < SLIG_CANVAS_DIM; y++) {
        for (int x = 0; x < SLIG_CANVAS_DIM; x++) {
            int u = axis_is_x ? x : y;
            int v;
            if (u <= start)      v = 255;
            else if (u >= end)   v = 0;
            else                 v = 255 - ((u - start) * 255) / span;
            if (invert) v = 255 - v;
            out[y * SLIG_CANVAS_DIM + x] = (uint8_t)v;
        }
    }
}

void slig_make_mask_left  (uint8_t *out, int boundary, int feather) {
    make_axis_mask(out, boundary, feather, /*axis_is_x=*/1, /*invert=*/0);
}
void slig_make_mask_right (uint8_t *out, int boundary, int feather) {
    make_axis_mask(out, boundary, feather, /*axis_is_x=*/1, /*invert=*/1);
}
void slig_make_mask_top   (uint8_t *out, int boundary, int feather) {
    make_axis_mask(out, boundary, feather, /*axis_is_x=*/0, /*invert=*/0);
}
void slig_make_mask_bottom(uint8_t *out, int boundary, int feather) {
    make_axis_mask(out, boundary, feather, /*axis_is_x=*/0, /*invert=*/1);
}

void slig_reconstruct_at_dim(uint8_t           *out_dim_dim, int dim,
                             const SligCellSet *set) {
    if (!out_dim_dim || dim <= 0) return;
    if (!set || set->num_cells == 0) {
        memset(out_dim_dim, 128, (size_t)dim * dim);
        return;
    }
    SligRenderItem item;
    memset(&item, 0, sizeof(item));
    item.cells         = set->cells;
    item.cell_count    = set->num_cells;
    item.render_mode   = SLIG_RENDER_GLOBAL;
    item.render_weight = 255;
    item.grid_x        = 128;
    item.grid_y        = 128;
    item.mask          = NULL;

    SligCanvas canvas;
    slig_render_sequential_canvas(&canvas, &item, 1, NULL);
    slig_canvas_to_u8_sub(out_dim_dim, dim, &canvas, 0, 0);
}
