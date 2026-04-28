/* slig_pipeline.c — SLIG v2 파이프라인 구현.
 *
 * 수정 사항 (proposal 대비):
 *   1. CEUnit 직접 접근 → slig_pack/slig_unpack 사용
 *   2. 가우시안 splat → slig_apply (방향별 외적 + 이벤트)
 *   3. 잔차 체인 구현 (각 단계가 이전 잔차를 입력)
 *   4. 업스케일: 하모닉 OR SBR (택1) → 위너 → CS
 *   5. 컬러: 블록별 Cb/Cr (평균만이 아닌 공간 분포)
 *   6. 프로그레시브: 5단계 콜백
 *   7. 후처리: 언샤프 마스크 실제 구현
 *   8. 전처리: SVD fallback 추가
 */

#include "slig_pipeline.h"
#include "slig_signal.h"   /* slig_pack, slig_unpack, slig_apply, SligSignal */
#include "slig_tick_math.h" /* tick helpers + TICK_*_TABLE / WAVE_STEPS */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ════════════════════════════════════════════════════
 *  내부 헬퍼
 * ════════════════════════════════════════════════════ */

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* 이미지 픽셀 접근 (경계 클램핑) */
static int px_idx(int w, int h, int ch, int x, int y, int c) {
    if (x < 0) x = 0; if (x >= w) x = w - 1;
    if (y < 0) y = 0; if (y >= h) y = h - 1;
    if (c < 0) c = 0; if (c >= ch) c = ch - 1;
    return (y * w + x) * ch + c;
}

static float img_get(const SligImage *im, int x, int y, int c) {
    return im->data[px_idx(im->width, im->height, im->channels, x, y, c)];
}

static void img_set(SligImage *im, int x, int y, int c, float v) {
    im->data[px_idx(im->width, im->height, im->channels, x, y, c)] = v;
}

static float img_luma(const SligImage *im, int x, int y) {
    if (im->channels == 1) return img_get(im, x, y, 0);
    return 0.299f * img_get(im, x, y, 0)
         + 0.587f * img_get(im, x, y, 1)
         + 0.114f * img_get(im, x, y, 2);
}

/* SligSignal 초기화 + CE Cell에 넣기 */
static int push_signal(SligDecomposed *d, const SligSignal *sig) {
    if (d->num_cells >= 64) return 0;
    slig_pack(&d->cells[d->num_cells], sig);
    d->num_cells++;
    return 1;
}

/* SligCanvas → SligImage 변환 (렌더 결과 추출) */
static void canvas_to_image(SligImage *out, const SligCanvas *c) {
    int32_t mn = INT32_MAX, mx = INT32_MIN;
    int N = SLIG_CANVAS_DIM;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            if (c->pixels[y][x] < mn) mn = c->pixels[y][x];
            if (c->pixels[y][x] > mx) mx = c->pixels[y][x];
        }
    int32_t range = mx - mn;
    if (range == 0) range = 1;

    for (int y = 0; y < out->height && y < N; y++)
        for (int x = 0; x < out->width && x < N; x++) {
            float v = (float)(c->pixels[y][x] - mn) / (float)range;
            img_set(out, x, y, 0, v);
            if (out->channels >= 3) {
                img_set(out, x, y, 1, v);
                img_set(out, x, y, 2, v);
            }
        }
}

/* 간이 power iteration SVD: 한 개 성분 추출 */
static void svd_one(const float *matrix, int dim,
                    float *out_u, float *out_v, float *out_sigma) {
    float *v = (float *)calloc(dim, sizeof(float));
    float *u = (float *)calloc(dim, sizeof(float));
    for (int i = 0; i < dim; i++)
        v[i] = sinf(i * 2.3f) + cosf(i * 1.7f);

    for (int iter = 0; iter < 25; iter++) {
        float norm = 0;
        for (int i = 0; i < dim; i++) {
            u[i] = 0;
            for (int j = 0; j < dim; j++)
                u[i] += matrix[i * dim + j] * v[j];
        }
        for (int i = 0; i < dim; i++) norm += u[i] * u[i];
        norm = sqrtf(norm);
        if (norm < 1e-10f) norm = 1.0f;
        for (int i = 0; i < dim; i++) u[i] /= norm;

        norm = 0;
        for (int j = 0; j < dim; j++) {
            v[j] = 0;
            for (int i = 0; i < dim; i++)
                v[j] += matrix[i * dim + j] * u[i];
        }
        for (int j = 0; j < dim; j++) norm += v[j] * v[j];
        *out_sigma = sqrtf(norm);
        if (*out_sigma < 1e-10f) *out_sigma = 1e-10f;
        for (int j = 0; j < dim; j++) v[j] /= *out_sigma;
    }
    memcpy(out_u, u, dim * sizeof(float));
    memcpy(out_v, v, dim * sizeof(float));
    free(u); free(v);
}

/* ════════════════════════════════════════════════════
 *  이미지 할당/해제
 * ════════════════════════════════════════════════════ */

SligImage *slig_image_alloc(int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0) return NULL;
    SligImage *im = (SligImage *)calloc(1, sizeof(SligImage));
    if (!im) return NULL;
    im->width = width; im->height = height; im->channels = channels;
    im->data = (float *)calloc((size_t)width * height * channels, sizeof(float));
    if (!im->data) { free(im); return NULL; }
    return im;
}

void slig_image_free(SligImage *img) {
    if (img) { free(img->data); free(img); }
}

/* ════════════════════════════════════════════════════
 *  기본 설정
 * ════════════════════════════════════════════════════ */

void slig_preprocess_config_default(SligPreprocessConfig *cfg) {
    if (!cfg) return;
    cfg->noise_threshold = 0.02f;
    cfg->edge_sigma = 1.5f;
    cfg->detect_symmetry = 1;
    cfg->detect_repeat = 1;
}

void slig_decompose_config_default(SligDecomposeConfig *cfg) {
    if (!cfg) return;
    /* Cell budget bumped for fidelity (was 8/8/8/6/4 = 34, → 12/16/16/8/4 = 56).
     * Each additional cell adds 64B; total raw payload 56 × 64 = 3.5 KB
     * (well under SligDecomposed's 64-cell cap × 64B = 4 KB). The biggest
     * PSNR gain comes from edge / texture which capture mid/high-frequency
     * detail; structure rank ~12 is enough for big-shape coverage. */
    cfg->max_structure_cells = 12;
    cfg->max_edge_cells = 16;
    cfg->max_texture_cells = 16;
    cfg->max_color_cells = 8;
    cfg->max_event_cells = 4;
    cfg->energy_target = 0.97f;
}

void slig_upscale_config_default(SligUpscaleConfig *cfg) {
    if (!cfg) return;
    cfg->enable_harmonic = 1;
    cfg->enable_sbr = 0;        /* 하모닉 OR SBR 택1 — 기본은 하모닉 */
    cfg->enable_wiener = 1;
    cfg->enable_cs = 0;
    cfg->target_coeffs = 32;
    cfg->cs_iterations = 50;
}

void slig_render_config_default(SligRenderConfig *cfg) {
    if (!cfg) return;
    cfg->target_width = 256;
    cfg->target_height = 256;
    cfg->guidance_scale = 1.0f;
    cfg->sigma_scale = 1.0f;
    cfg->depth_enable = 0;
    cfg->progressive = 0;
    cfg->on_progress = NULL;
    cfg->progress_ctx = NULL;
}

void slig_postprocess_config_default(SligPostprocessConfig *cfg) {
    if (!cfg) return;
    cfg->enable_sharpen = 1;
    cfg->sharpen_amount = 0.5f;
    cfg->sharpen_radius = 1.0f;
    cfg->enable_debanding = 1;
    cfg->enable_color_correct = 1;
    cfg->saturation = 1.0f;
    cfg->brightness = 0.0f;
    cfg->contrast = 1.0f;
    cfg->enable_histogram_match = 0;
}

/* ════════════════════════════════════════════════════
 *  1. 전처리
 *
 *  히스토그램 정규화 → 바이래터럴 필터 → 대칭 검출
 * ════════════════════════════════════════════════════ */

void slig_preprocess(SligImage *out, SligDecomposed *meta,
                     const SligImage *in, const SligPreprocessConfig *cfg0) {
    SligPreprocessConfig def;
    if (!cfg0) { slig_preprocess_config_default(&def); cfg0 = &def; }
    if (!out || !in || !out->data || !in->data) return;

    size_t n = (size_t)in->width * in->height * in->channels;

    /* in → out 복사 (in-place 가능) */
    if (out != in)
        memcpy(out->data, in->data, n * sizeof(float));
    out->width = in->width; out->height = in->height; out->channels = in->channels;

    /* 1. 히스토그램 정규화 + 메타 저장 */
    float mn = 1e30f, mx = -1e30f, sum = 0;
    for (size_t i = 0; i < n; i++) {
        float v = out->data[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
    }
    if (meta) {
        meta->orig_min = mn;
        meta->orig_max = mx;
        meta->orig_mean = sum / (float)n;
        meta->symmetry = 0;
        meta->sym_axis_x = 0.5f;
        meta->sym_axis_y = 0.5f;
    }

    float range = mx - mn;
    if (range < 1e-6f) range = 1.0f;
    for (size_t i = 0; i < n; i++)
        out->data[i] = clampf((out->data[i] - mn) / range, 0, 1);

    /* 2. 바이래터럴 필터 (엣지 보존 노이즈 제거) */
    SligImage *tmp = slig_image_alloc(out->width, out->height, out->channels);
    if (!tmp) return;

    float sig2 = cfg0->edge_sigma * cfg0->edge_sigma + 1e-6f;
    float range_sig2 = 0.08f * 0.08f;  /* 밝기 차이 허용 범위 */

    for (int y = 0; y < out->height; y++) {
        for (int x = 0; x < out->width; x++) {
            for (int c = 0; c < out->channels; c++) {
                float center = img_get(out, x, y, c);
                float acc = 0, wsum = 0;

                for (int dy = -2; dy <= 2; dy++) {
                    for (int dx = -2; dx <= 2; dx++) {
                        float q = img_get(out, x + dx, y + dy, c);
                        float ds = (float)(dx * dx + dy * dy);
                        float dr = q - center;
                        /* 공간 가중 × 밝기 차이 가중 */
                        float w = expf(-ds / (2 * sig2))
                                * expf(-(dr * dr) / (2 * range_sig2));
                        acc += w * q;
                        wsum += w;
                    }
                }

                float smoothed = acc / fmaxf(wsum, 1e-6f);
                /* 노이즈 임계값 이하의 변화만 부드럽게 */
                float result = (fabsf(center - smoothed) < cfg0->noise_threshold)
                             ? smoothed : center;
                img_set(tmp, x, y, c, result);
            }
        }
    }
    memcpy(out->data, tmp->data, n * sizeof(float));
    slig_image_free(tmp);

    /* 3. 대칭 검출 */
    if (meta && cfg0->detect_symmetry) {
        double lr = 0, tb = 0, total = 0;
        for (int y = 0; y < out->height; y++) {
            for (int x = 0; x < out->width; x++) {
                float a = img_luma(out, x, y);
                total += fabsf(a) + 1e-3;
                lr += fabsf(a - img_luma(out, out->width - 1 - x, y));
                tb += fabsf(a - img_luma(out, x, out->height - 1 - y));
            }
        }
        int is_lr = (lr / total) < 0.08;
        int is_tb = (tb / total) < 0.08;
        meta->symmetry = (is_lr && is_tb) ? 3 : is_lr ? 1 : is_tb ? 2 : 0;
    }
}

/* ════════════════════════════════════════════════════
 *  2. 분해 — 5단계 파이프라인
 *
 *  핵심 변경: 각 단계가 이전 잔차를 입력으로 받음
 * ════════════════════════════════════════════════════ */

/* 2-1. 구조 분해: SVD로 전역 rank-1 성분 추출 */
void slig_decompose_structure(SligDecomposed *out, const SligImage *img,
                              int max_cells, float energy_target) {
    if (!out || !img) return;
    int dim = (img->width < img->height) ? img->width : img->height;
    if (dim > SLIG_SIG_LEN) dim = SLIG_SIG_LEN;

    /* 밝기 행렬 생성 */
    float *mat = (float *)calloc(dim * dim, sizeof(float));
    float mean = 0;
    for (int y = 0; y < dim; y++)
        for (int x = 0; x < dim; x++) {
            mat[y * dim + x] = img_luma(img, x, y);
            mean += mat[y * dim + x];
        }
    mean /= (float)(dim * dim);
    for (int i = 0; i < dim * dim; i++) mat[i] -= mean;

    /* SVD: rank-1 성분을 하나씩 추출하고 잔차에서 빼기 */
    float *u = (float *)calloc(dim, sizeof(float));
    float *v = (float *)calloc(dim, sizeof(float));
    float total_energy = 0;
    for (int i = 0; i < dim * dim; i++) total_energy += mat[i] * mat[i];

    float cumul = 0;
    for (int k = 0; k < max_cells; k++) {
        float sigma;
        svd_one(mat, dim, u, v, &sigma);
        if (sigma < 0.01f) break;

        /* SligSignal 구성 */
        SligSignal sig;
        memset(&sig, 0, sizeof(sig));
        sig.dir = SLIG_DIR_HORIZONTAL;
        sig.scale = (k < 2) ? SLIG_SCALE_COARSE
                 : (k < 5) ? SLIG_SCALE_MID
                 : SLIG_SCALE_FINE;
        sig.sigma = (uint16_t)clampf(sigma * 100, 0, 65535);
        sig.frequency = (1 << (sig.scale + 2));
        sig.origin_x = 128;
        sig.origin_y = 128;
        if (sigma < 0) sig.flags |= SLIG_FLAG_NEGATIVE;

        /* u, v 신호를 int16으로 변환 */
        for (int i = 0; i < dim && i < SLIG_SIG_LEN; i++) {
            sig.u[i] = (int16_t)(u[i] * 10000.0f);
            sig.v[i] = (int16_t)(v[i] * 10000.0f);
        }

        /* CE Cell에 pack해서 저장 */
        push_signal(out, &sig);

        /* 잔차에서 이 성분 제거 */
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                mat[i * dim + j] -= sigma * u[i] * v[j];

        cumul += sigma * sigma;
        if (total_energy > 0 && cumul / total_energy >= energy_target)
            break;
    }

    free(mat); free(u); free(v);
}

/* 2-2. 엣지 분해: 소벨 그래디언트 → 방향별 CE Cell */
void slig_decompose_edges(SligDecomposed *out, const SligImage *img,
                          const SligImage *residual, int max_cells) {
    if (!out || !img) return;
    const SligImage *src = residual ? residual : img;

    /* 가장 강한 엣지 위치 N개를 찾기 */
    typedef struct { float strength; float nx; float ny; float angle; } EdgeCand;
    EdgeCand best[32];
    memset(best, 0, sizeof(best));

    int step = (int)fmaxf(4, fminf(src->width, src->height) / 16.0f);
    for (int y = 1; y < src->height - 1; y += step) {
        for (int x = 1; x < src->width - 1; x += step) {
            float gx = img_luma(src, x + 1, y) - img_luma(src, x - 1, y);
            float gy = img_luma(src, x, y + 1) - img_luma(src, x, y - 1);
            float e = sqrtf(gx * gx + gy * gy);
            if (e < 0.02f) continue;

            /* 가장 약한 후보를 교체 */
            int weakest = 0;
            for (int i = 1; i < max_cells && i < 32; i++)
                if (best[i].strength < best[weakest].strength)
                    weakest = i;

            if (e > best[weakest].strength) {
                best[weakest].strength = e;
                best[weakest].nx = (float)x / src->width;
                best[weakest].ny = (float)y / src->height;
                best[weakest].angle = atan2f(gy, gx);
            }
        }
    }

    /* 방향 양자화 → CE Cell 생성 */
    for (int i = 0; i < max_cells && i < 32; i++) {
        if (best[i].strength <= 0) continue;

        /* 각도 → 4방향 (가로/세로/대각╲/대각╱) */
        float angle = best[i].angle;
        int dir;
        float a_norm = fmodf(angle + (float)M_PI, (float)M_PI); /* 0~π */
        if (a_norm < M_PI / 8 || a_norm > 7 * M_PI / 8)
            dir = SLIG_DIR_HORIZONTAL;
        else if (a_norm < 3 * M_PI / 8)
            dir = SLIG_DIR_DIAG_UP;
        else if (a_norm < 5 * M_PI / 8)
            dir = SLIG_DIR_VERTICAL;
        else
            dir = SLIG_DIR_DIAG_DOWN;

        SligSignal sig;
        memset(&sig, 0, sizeof(sig));
        sig.dir = (uint8_t)dir;
        sig.scale = SLIG_SCALE_MID;
        sig.sigma = (uint16_t)clampf(best[i].strength * 20000, 0, 65535);
        sig.origin_x = (uint8_t)(best[i].nx * 255);
        sig.origin_y = (uint8_t)(best[i].ny * 255);
        sig.frequency = 4;

        /* 엣지 주변의 1D 프로파일 추출 → u/v 신호 */
        int cx = (int)(best[i].nx * img->width);
        int cy = (int)(best[i].ny * img->height);
        for (int k = 0; k < SLIG_SIG_LEN; k++) {
            int sx = cx - SLIG_SIG_LEN / 2 + k;
            sig.v[k] = (int16_t)(img_luma(src, sx, cy) * 10000);
            int sy = cy - SLIG_SIG_LEN / 2 + k;
            sig.u[k] = (int16_t)(img_luma(src, cx, sy) * 10000);
        }

        push_signal(out, &sig);
    }
}

/* 2-3. 텍스처 분해: 블록별 분산 → 웨이블릿 스타일 CE Cell */
void slig_decompose_texture(SligDecomposed *out, const SligImage *img,
                            const SligImage *residual, int max_cells) {
    if (!out || !img) return;
    const SligImage *src = residual ? residual : img;

    typedef struct { float var; float cx; float cy; } TexCand;
    TexCand best[32];
    memset(best, 0, sizeof(best));

    /* 블록 크기: 멀티스케일 (큰 블록 → 작은 블록 순서) */
    int block_sizes[] = { 32, 16, 8 };
    int num_scales = 3;

    for (int s = 0; s < num_scales; s++) {
        int bs = block_sizes[s];
        if (bs >= src->width || bs >= src->height) continue;

        for (int by = 0; by < src->height; by += bs) {
            for (int bx = 0; bx < src->width; bx += bs) {
                /* 블록 평균 + 분산 */
                double mean = 0;
                int cnt = 0;
                for (int yy = by; yy < by + bs && yy < src->height; yy++)
                    for (int xx = bx; xx < bx + bs && xx < src->width; xx++) {
                        mean += img_luma(src, xx, yy);
                        cnt++;
                    }
                mean /= fmax(cnt, 1);

                double var = 0;
                for (int yy = by; yy < by + bs && yy < src->height; yy++)
                    for (int xx = bx; xx < bx + bs && xx < src->width; xx++) {
                        double d = img_luma(src, xx, yy) - mean;
                        var += d * d;
                    }
                float v = (float)sqrt(var / fmax(cnt, 1));
                if (v < 0.015f) continue;

                /* Top-K 유지 */
                int weakest = 0;
                for (int i = 1; i < max_cells && i < 32; i++)
                    if (best[i].var < best[weakest].var) weakest = i;

                if (v > best[weakest].var) {
                    best[weakest].var = v;
                    best[weakest].cx = (bx + 0.5f * bs) / src->width;
                    best[weakest].cy = (by + 0.5f * bs) / src->height;
                }
            }
        }
    }

    for (int i = 0; i < max_cells && i < 32; i++) {
        if (best[i].var <= 0) continue;

        SligSignal sig;
        memset(&sig, 0, sizeof(sig));
        sig.dir = SLIG_DIR_HORIZONTAL;
        sig.scale = SLIG_SCALE_FINE;
        sig.sigma = (uint16_t)clampf(best[i].var * 30000, 0, 65535);
        sig.origin_x = (uint8_t)(best[i].cx * 255);
        sig.origin_y = (uint8_t)(best[i].cy * 255);
        sig.frequency = 2;

        /* 텍스처 블록에서 1D 프로파일 추출 */
        int cx = (int)(best[i].cx * src->width);
        int cy = (int)(best[i].cy * src->height);
        for (int k = 0; k < SLIG_SIG_LEN; k++) {
            int sx = cx - SLIG_SIG_LEN / 2 + k;
            sig.v[k] = (int16_t)(img_luma(src, sx, cy) * 10000);
            int sy = cy - SLIG_SIG_LEN / 2 + k;
            sig.u[k] = (int16_t)(img_luma(src, cx, sy) * 10000);
        }

        push_signal(out, &sig);
    }
}

/* 2-4. 색상 분해: 블록별 Cb/Cr (공간적 색 분포) */
void slig_decompose_color(SligDecomposed *out, const SligImage *img,
                          int max_cells) {
    if (!out || !img || img->channels < 3) return;

    /* 색은 공간적으로 변하므로, 2×2 격자로 나눠서 각 영역의 색 저장 */
    int grid = 2;  /* 2×2 = 4 영역 */
    int cells_per_ch = max_cells / 2;  /* Cb와 Cr에 반반 */
    int bw = img->width / grid;
    int bh = img->height / grid;

    for (int gy = 0; gy < grid; gy++) {
        for (int gx = 0; gx < grid; gx++) {
            if ((int)out->num_cells >= 64) break;

            /* 블록 내 평균 Cb, Cr 계산 */
            double sum_cb = 0, sum_cr = 0;
            int cnt = 0;

            for (int y = gy * bh; y < (gy + 1) * bh && y < img->height; y++) {
                for (int x = gx * bw; x < (gx + 1) * bw && x < img->width; x++) {
                    float r = img_get(img, x, y, 0);
                    float g = img_get(img, x, y, 1);
                    float b = img_get(img, x, y, 2);
                    /* RGB → Cb, Cr */
                    sum_cb += -0.169f * r - 0.331f * g + 0.500f * b + 0.5f;
                    sum_cr +=  0.500f * r - 0.419f * g - 0.081f * b + 0.5f;
                    cnt++;
                }
            }

            float avg_cb = (float)(sum_cb / fmax(cnt, 1));
            float avg_cr = (float)(sum_cr / fmax(cnt, 1));

            /* Cb CE Cell */
            if (fabsf(avg_cb - 0.5f) > 0.01f && cells_per_ch > 0) {
                SligSignal sig;
                memset(&sig, 0, sizeof(sig));
                sig.dir = SLIG_DIR_HORIZONTAL;
                sig.scale = SLIG_SCALE_COARSE;
                sig.sigma = (uint16_t)clampf(fabsf(avg_cb - 0.5f) * 50000, 0, 65535);
                sig.origin_x = (uint8_t)((gx + 0.5f) / grid * 255);
                sig.origin_y = (uint8_t)((gy + 0.5f) / grid * 255);
                sig.frequency = 32;
                sig.audio_bins[0] = 1;  /* channel marker: Cb */
                sig.audio_amps[0] = (uint8_t)(avg_cb * 255);
                if (avg_cb < 0.5f) sig.flags |= SLIG_FLAG_NEGATIVE;
                push_signal(out, &sig);
            }

            /* Cr CE Cell */
            if (fabsf(avg_cr - 0.5f) > 0.01f && cells_per_ch > 0) {
                SligSignal sig;
                memset(&sig, 0, sizeof(sig));
                sig.dir = SLIG_DIR_HORIZONTAL;
                sig.scale = SLIG_SCALE_COARSE;
                sig.sigma = (uint16_t)clampf(fabsf(avg_cr - 0.5f) * 50000, 0, 65535);
                sig.origin_x = (uint8_t)((gx + 0.5f) / grid * 255);
                sig.origin_y = (uint8_t)((gy + 0.5f) / grid * 255);
                sig.frequency = 32;
                sig.audio_bins[0] = 2;  /* channel marker: Cr */
                sig.audio_amps[0] = (uint8_t)(avg_cr * 255);
                if (avg_cr < 0.5f) sig.flags |= SLIG_FLAG_NEGATIVE;
                push_signal(out, &sig);
            }
        }
    }
}

/* 2-5. 이벤트 분해: 밝기 중심 + 극값 */
void slig_decompose_events(SligDecomposed *out, const SligImage *img,
                           int max_cells) {
    if (!out || !img || max_cells <= 0) return;

    /* 밝기 무게중심 → 물결 CE Cell */
    double sx = 0, sy = 0, sw = 0;
    int bright_x = 0, bright_y = 0;
    float bright_val = -1;

    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            float v = img_luma(img, x, y);
            sx += x * v;
            sy += y * v;
            sw += v + 1e-6;
            if (v > bright_val) {
                bright_val = v;
                bright_x = x;
                bright_y = y;
            }
        }
    }

    /* 무게중심에서 물결 */
    SligSignal sig;
    memset(&sig, 0, sizeof(sig));
    sig.dir = SLIG_DIR_RIPPLE;
    sig.scale = SLIG_SCALE_MID;
    sig.sigma = 3000;
    sig.origin_x = (uint8_t)clampf((float)(sx / sw) / img->width * 255, 0, 255);
    sig.origin_y = (uint8_t)clampf((float)(sy / sw) / img->height * 255, 0, 255);
    sig.frequency = img->width / 4;
    sig.speed = 3;
    sig.decay = 230;
    push_signal(out, &sig);

    /* 가장 밝은 점에서 빔 */
    if (max_cells > 1) {
        memset(&sig, 0, sizeof(sig));
        sig.dir = SLIG_DIR_BEAM;
        sig.scale = SLIG_SCALE_FINE;
        sig.sigma = (uint16_t)clampf(bright_val * 5000, 0, 65535);
        sig.origin_x = (uint8_t)((float)bright_x / img->width * 255);
        sig.origin_y = (uint8_t)((float)bright_y / img->height * 255);
        sig.frequency = 8;
        sig.speed = 4;
        sig.decay = 220;
        sig.beam_dir = 0;
        sig.beam_width = 15;
        push_signal(out, &sig);
    }
}

/* 2-통합. 5단계 분해 + 잔차 체인 */
void slig_decompose_v2(SligDecomposed *out, const SligImage *image,
                       const SligDecomposeConfig *cfg0) {
    SligDecomposeConfig def;
    if (!cfg0) { slig_decompose_config_default(&def); cfg0 = &def; }
    if (!out || !image) return;

    /* 메타데이터 보존 (전처리에서 설정된 것) */
    uint8_t keep_sym = out->symmetry;
    float keep_min = out->orig_min, keep_max = out->orig_max, keep_mean = out->orig_mean;

    memset(out, 0, sizeof(*out));
    out->symmetry = keep_sym;
    out->orig_min = keep_min;
    out->orig_max = keep_max;
    out->orig_mean = keep_mean;

    /* Stage 1: 구조 (SVD) */
    slig_decompose_structure(out, image, cfg0->max_structure_cells, cfg0->energy_target);
    out->structure_end = out->num_cells;

    /* 잔차 계산: 원본 − 구조 복원 (절대 image-scale 보존)
     *
     * 이전 버전은 SligCanvas raw int32 누적기를 min-max 로 [0,1]
     * 정규화한 뒤 차분 → 모양은 맞지만 절대 magnitude 가 사라져
     * 잔차 스케일이 깨졌다. 이 버전은 unpack(cell) → ∑σ·u·v 외적
     * + 평균 복원으로 image-scale 그대로 직접 reconstruct 한다. */
    SligImage *residual = slig_image_alloc(image->width, image->height, image->channels);
    if (residual) {
        int H = image->height, W = image->width;
        int dim_lim = (W < H) ? W : H;
        if (dim_lim > SLIG_SIG_LEN) dim_lim = SLIG_SIG_LEN;

        float *recon = (float *)calloc((size_t)W * H, sizeof(float));
        if (recon) {
            /* 각 구조 셀의 σ·u·v 외적을 픽셀에 누적 */
            for (uint32_t i = 0; i < out->structure_end && i < out->num_cells; i++) {
                SligSignal sig;
                slig_unpack(&sig, &out->cells[i]);
                float sigma_f = (float)sig.sigma / 100.0f;
                if (sig.flags & SLIG_FLAG_NEGATIVE) sigma_f = -sigma_f;
                for (int y = 0; y < H && y < dim_lim; y++) {
                    float u_y = (float)sig.u[y] / 10000.0f;
                    for (int x = 0; x < W && x < dim_lim; x++) {
                        float v_x = (float)sig.v[x] / 10000.0f;
                        recon[y * W + x] += sigma_f * u_y * v_x;
                    }
                }
            }
            /* SVD 는 mean-centered matrix 에 대해 분해됐으니
             * 복원에 이미지 평균을 더해줘야 image-scale 이 맞는다. */
            float img_mean = 0; int cnt = 0;
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    img_mean += img_luma(image, x, y); cnt++;
                }
            if (cnt) img_mean /= (float)cnt;

            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    float orig_val  = img_luma(image, x, y);
                    float recon_val = recon[y * W + x] + img_mean;
                    img_set(residual, x, y, 0, orig_val - recon_val);
                }
            free(recon);
        }

        /* Stage 2: 엣지 (잔차에서) */
        slig_decompose_edges(out, image, residual, cfg0->max_edge_cells);
        out->edge_end = out->num_cells;

        /* Stage 3: 텍스처 (잔차에서) */
        slig_decompose_texture(out, image, residual, cfg0->max_texture_cells);
        out->texture_end = out->num_cells;

        slig_image_free(residual);
    } else {
        /* 메모리 부족 시 잔차 없이 원본에서 */
        slig_decompose_edges(out, image, NULL, cfg0->max_edge_cells);
        out->edge_end = out->num_cells;
        slig_decompose_texture(out, image, NULL, cfg0->max_texture_cells);
        out->texture_end = out->num_cells;
    }

    /* Stage 4: 색상 */
    slig_decompose_color(out, image, cfg0->max_color_cells);
    out->color_end = out->num_cells;

    /* Stage 5: 이벤트 */
    slig_decompose_events(out, image, cfg0->max_event_cells);
}

/* ════════════════════════════════════════════════════
 *  3. 업스케일 (오디오 복구 기법)
 *
 *  순서: 하모닉 OR SBR (택1) → 위너 → CS (선택)
 *  동시 적용하면 간섭하므로 순서가 중요
 * ════════════════════════════════════════════════════ */

/* DCT 계수를 직접 조작: CE Cell의 inc.G/B + dec.R/G에서 추출 */
static void cell_get_coeffs(const CEUnit *cell, float coeffs[SLIG_DCT_COEFF]) {
    coeffs[0]  = (float)cell->inc.G.plus[0]  - 128;
    coeffs[1]  = (float)cell->inc.G.plus[1]  - 128;
    coeffs[2]  = (float)cell->inc.G.plus[2]  - 128;
    coeffs[3]  = (float)cell->inc.G.plus[3]  - 128;
    coeffs[4]  = (float)cell->inc.G.minus[0] - 128;
    coeffs[5]  = (float)cell->inc.G.minus[1] - 128;
    coeffs[6]  = (float)cell->inc.G.minus[2] - 128;
    coeffs[7]  = (float)cell->inc.G.minus[3] - 128;
    coeffs[8]  = (float)cell->dec.R.plus[0]  - 128;
    coeffs[9]  = (float)cell->dec.R.plus[1]  - 128;
    coeffs[10] = (float)cell->dec.R.plus[2]  - 128;
    coeffs[11] = (float)cell->dec.R.plus[3]  - 128;
    coeffs[12] = (float)cell->dec.R.minus[0] - 128;
    coeffs[13] = (float)cell->dec.R.minus[1] - 128;
    coeffs[14] = (float)cell->dec.R.minus[2] - 128;
    coeffs[15] = (float)cell->dec.R.minus[3] - 128;
}

static void cell_set_coeffs(CEUnit *cell, const float coeffs[SLIG_DCT_COEFF]) {
    cell->inc.G.plus[0]  = (uint8_t)clampf(coeffs[0]  + 128, 0, 255);
    cell->inc.G.plus[1]  = (uint8_t)clampf(coeffs[1]  + 128, 0, 255);
    cell->inc.G.plus[2]  = (uint8_t)clampf(coeffs[2]  + 128, 0, 255);
    cell->inc.G.plus[3]  = (uint8_t)clampf(coeffs[3]  + 128, 0, 255);
    cell->inc.G.minus[0] = (uint8_t)clampf(coeffs[4]  + 128, 0, 255);
    cell->inc.G.minus[1] = (uint8_t)clampf(coeffs[5]  + 128, 0, 255);
    cell->inc.G.minus[2] = (uint8_t)clampf(coeffs[6]  + 128, 0, 255);
    cell->inc.G.minus[3] = (uint8_t)clampf(coeffs[7]  + 128, 0, 255);
    cell->dec.R.plus[0]  = (uint8_t)clampf(coeffs[8]  + 128, 0, 255);
    cell->dec.R.plus[1]  = (uint8_t)clampf(coeffs[9]  + 128, 0, 255);
    cell->dec.R.plus[2]  = (uint8_t)clampf(coeffs[10] + 128, 0, 255);
    cell->dec.R.plus[3]  = (uint8_t)clampf(coeffs[11] + 128, 0, 255);
    cell->dec.R.minus[0] = (uint8_t)clampf(coeffs[12] + 128, 0, 255);
    cell->dec.R.minus[1] = (uint8_t)clampf(coeffs[13] + 128, 0, 255);
    cell->dec.R.minus[2] = (uint8_t)clampf(coeffs[14] + 128, 0, 255);
    cell->dec.R.minus[3] = (uint8_t)clampf(coeffs[15] + 128, 0, 255);
}

/* Harmonic damping: how much of the predicted harmonic to mix into
 * existing coefficients. Keep low to add subtle texture without
 * smashing PSNR. Per measurement table:
 *   0.00 → ±0 dB     2785 depth   (baseline / no harmonic)
 *   0.02 → −0.94 dB  4001 depth   "subtle texture"      ← chosen
 *   0.03 → −1.88 dB  4498 depth   "texture applied"
 * Higher values trade fidelity for perceptual richness. */
#define SLIG_HARMONIC_DAMPING 0.02f

void slig_upscale_harmonic(CEUnit *cell, int target_coeffs) {
    if (!cell) return;
    float coeffs[SLIG_DCT_COEFF];
    cell_get_coeffs(cell, coeffs);
    /* Subtle harmonic blend: every coefficient k receives a tiny
     * fraction (DAMPING) of base × 0.55^(k−h) where h = k/2. The
     * earlier "replace if existing < 30% of predicted" is too
     * aggressive — it spikes high-frequency noise the renderer
     * shows as moiré. Now we add a small bias instead. */
    for (int k = 1; k < SLIG_DCT_COEFF; k++) {
        int harmonic = k / 2;
        float base = coeffs[harmonic];
        float predicted = base * powf(0.55f, (float)(k - harmonic));
        coeffs[k] += predicted * SLIG_HARMONIC_DAMPING;
    }
    cell_set_coeffs(cell, coeffs);
    (void)target_coeffs;
}

void slig_upscale_sbr(CEUnit *cell, int target_coeffs) {
    if (!cell) return;
    float coeffs[SLIG_DCT_COEFF];
    cell_get_coeffs(cell, coeffs);
    /* SBR: 저주파(0~7) 패턴을 고주파(8~15)에 복제
     * MP3의 AAC-HE가 쓰는 기법 */
    for (int k = 8; k < SLIG_DCT_COEFF; k++) {
        int src_k = k - 8;
        float replicated = coeffs[src_k] * 0.25f;
        if (fabsf(coeffs[k]) < fabsf(replicated))
            coeffs[k] = replicated;
    }
    cell_set_coeffs(cell, coeffs);
    (void)target_coeffs;
}

void slig_upscale_wiener(CEUnit *cell) {
    if (!cell) return;
    float coeffs[SLIG_DCT_COEFF];
    cell_get_coeffs(cell, coeffs);
    /* 위너 필터: 신호 파워 / (신호 파워 + 노이즈 파워)
     * 녹음에서 배경 잡음 제거하는 표준 기법 */
    double total_power = 0;
    for (int k = 0; k < SLIG_DCT_COEFF; k++)
        total_power += coeffs[k] * coeffs[k];
    total_power /= SLIG_DCT_COEFF;

    float noise = (float)(total_power * 0.03 + 1e-6);
    for (int k = 0; k < SLIG_DCT_COEFF; k++) {
        float s2 = coeffs[k] * coeffs[k];
        float gain = s2 / (s2 + noise);
        coeffs[k] *= gain;
    }
    cell_set_coeffs(cell, coeffs);
}

void slig_upscale_cs(CEUnit *cell, int target_coeffs, int iterations) {
    if (!cell) return;
    /* 압축 센싱: L1 soft thresholding
     * 신호가 sparse하다는 가정 하에 작은 값을 0으로 수렴 */
    float coeffs[SLIG_DCT_COEFF];
    cell_get_coeffs(cell, coeffs);
    float lambda = 0.002f;
    for (int it = 0; it < iterations; it++) {
        for (int k = 0; k < SLIG_DCT_COEFF; k++) {
            float s = coeffs[k];
            coeffs[k] = copysignf(fmaxf(fabsf(s) - lambda, 0), s);
        }
    }
    cell_set_coeffs(cell, coeffs);
    (void)target_coeffs;
}

void slig_upscale(SligDecomposed *d, const SligUpscaleConfig *cfg0) {
    SligUpscaleConfig def;
    if (!cfg0) { slig_upscale_config_default(&def); cfg0 = &def; }
    if (!d) return;

    for (uint32_t i = 0; i < d->num_cells; i++) {
        /* 하모닉 OR SBR — 택1 (동시 적용하면 간섭) */
        if (cfg0->enable_harmonic)
            slig_upscale_harmonic(&d->cells[i], cfg0->target_coeffs);
        else if (cfg0->enable_sbr)
            slig_upscale_sbr(&d->cells[i], cfg0->target_coeffs);

        /* 위너: 항상 마지막에 (노이즈 정리) */
        if (cfg0->enable_wiener)
            slig_upscale_wiener(&d->cells[i]);

        /* CS: 선택적 (느림, 고품질 필요 시만) */
        if (cfg0->enable_cs)
            slig_upscale_cs(&d->cells[i], cfg0->target_coeffs, cfg0->cs_iterations);
    }
}

/* ════════════════════════════════════════════════════
 *  4. 적응형 렌더러 (애트모스 방식)
 *
 *  핵심 변경: splat 대신 slig_apply 사용
 *  프로그레시브: 5단계 콜백
 * ════════════════════════════════════════════════════ */

/* 단일 CE Cell을 캔버스에 적용 (해상도 적응) */
static void render_cell(SligCanvas *canvas, const CEUnit *cell,
                        float sigma_scale) {
    SligSignal sig;
    slig_unpack(&sig, cell);

    /* 애트모스 방식: σ를 타겟에 맞게 스케일 */
    sig.sigma = (uint16_t)clampf(sig.sigma * sigma_scale, 0, 65535);

    /* slig_signal.c의 방향별 외적 + 이벤트 렌더 사용 */
    slig_apply(canvas, &sig);
}

/* 캔버스 → SligImage (그레이스케일 스냅샷 — 프로그레시브 중간 단계용) */
static void canvas_snapshot(SligImage *out, const SligCanvas *canvas) {
    uint8_t buf[SLIG_CANVAS_DIM][SLIG_CANVAS_DIM];
    slig_canvas_to_u8(buf, canvas);

    for (int y = 0; y < out->height && y < SLIG_CANVAS_DIM; y++)
        for (int x = 0; x < out->width && x < SLIG_CANVAS_DIM; x++) {
            float v = buf[y][x] / 255.0f;
            img_set(out, x, y, 0, v);
            if (out->channels >= 3) {
                img_set(out, x, y, 1, v);
                img_set(out, x, y, 2, v);
            }
        }
}

/* 크로마 캔버스 → uint8 (128 중심, 고정 스케일).
 *  Cb/Cr 셀은 audio_amps[0]·falloff 형태로 누적된 signed 값이다 —
 *  min-max 스트레치하면 약한 chroma 도 ±127 까지 부풀어 색이 폭주한다.
 *  대신 스케일 1/256 으로 압축해 128 ± offset 으로 매핑한다. */
static void chroma_canvas_to_u8(uint8_t out[SLIG_CANVAS_DIM][SLIG_CANVAS_DIM],
                                const SligCanvas *c) {
    int N = SLIG_CANVAS_DIM;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            int v = 128 + (c->pixels[y][x] >> 8);
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            out[y][x] = (uint8_t)v;
        }
}

/* YCbCr (3 캔버스) → RGB 합성. */
static void compose_ycbcr_to_rgb(SligImage *out,
                                 const SligCanvas *y_c,
                                 const SligCanvas *cb_c,
                                 const SligCanvas *cr_c) {
    uint8_t y_buf [SLIG_CANVAS_DIM][SLIG_CANVAS_DIM];
    uint8_t cb_buf[SLIG_CANVAS_DIM][SLIG_CANVAS_DIM];
    uint8_t cr_buf[SLIG_CANVAS_DIM][SLIG_CANVAS_DIM];
    slig_canvas_to_u8(y_buf, y_c);
    chroma_canvas_to_u8(cb_buf, cb_c);
    chroma_canvas_to_u8(cr_buf, cr_c);

    int H = out->height < SLIG_CANVAS_DIM ? out->height : SLIG_CANVAS_DIM;
    int W = out->width  < SLIG_CANVAS_DIM ? out->width  : SLIG_CANVAS_DIM;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int Y  = y_buf [y][x];
            int Cb = (int)cb_buf[y][x] - 128;
            int Cr = (int)cr_buf[y][x] - 128;
            /* BT.601 inverse, integer */
            int R = Y + (( 91881 * Cr + 32768) >> 16);
            int G = Y - (( 22554 * Cb + 46802 * Cr + 32768) >> 16);
            int B = Y + ((116130 * Cb + 32768) >> 16);
            if (R < 0) R = 0; else if (R > 255) R = 255;
            if (G < 0) G = 0; else if (G > 255) G = 255;
            if (B < 0) B = 0; else if (B > 255) B = 255;
            img_set(out, x, y, 0, R / 255.0f);
            if (out->channels >= 3) {
                img_set(out, x, y, 1, G / 255.0f);
                img_set(out, x, y, 2, B / 255.0f);
            }
        }
    }
}

void slig_render_adaptive(SligImage *out, const SligDecomposed *d,
                          const SligRenderConfig *cfg0) {
    SligRenderConfig def;
    if (!cfg0) { slig_render_config_default(&def); cfg0 = &def; }
    if (!out || !out->data || !d) return;

    /* 3 canvases: Y (휘도) + Cb/Cr (크로마). Phase 4 가 audio_bins[0]
     * 마커로 디스패치한다. 컬러 셀이 없으면 Cb/Cr 은 0 유지 → 그레이
     * 출력. 있으면 마지막에 YCbCr→RGB 합성으로 진짜 색을 만든다. */
    SligCanvas y_canvas, cb_canvas, cr_canvas;
    slig_canvas_clear(&y_canvas);
    slig_canvas_clear(&cb_canvas);
    slig_canvas_clear(&cr_canvas);
    int has_color_cells = 0;

    size_t img_bytes = (size_t)out->width * out->height * out->channels * sizeof(float);
    memset(out->data, 0, img_bytes);

    /* Phase 1: 구조 (채널 베드) */
    for (uint32_t i = 0; i < d->structure_end && i < d->num_cells; i++) {
        float s = cfg0->sigma_scale;
        if (cfg0->depth_enable) s *= 1.2f;
        render_cell(&y_canvas, &d->cells[i], s);
    }
    if (cfg0->progressive && cfg0->on_progress) {
        canvas_snapshot(out, &y_canvas);
        cfg0->on_progress(out, 1, 5, cfg0->progress_ctx);
    }

    /* Phase 2: 엣지 */
    for (uint32_t i = d->structure_end; i < d->edge_end && i < d->num_cells; i++)
        render_cell(&y_canvas, &d->cells[i], cfg0->sigma_scale);
    if (cfg0->progressive && cfg0->on_progress) {
        canvas_snapshot(out, &y_canvas);
        cfg0->on_progress(out, 2, 5, cfg0->progress_ctx);
    }

    /* Phase 3: 텍스처 */
    for (uint32_t i = d->edge_end; i < d->texture_end && i < d->num_cells; i++) {
        float s = cfg0->sigma_scale;
        if (cfg0->depth_enable) s *= 0.7f;
        render_cell(&y_canvas, &d->cells[i], s);
    }
    if (cfg0->progressive && cfg0->on_progress) {
        canvas_snapshot(out, &y_canvas);
        cfg0->on_progress(out, 3, 5, cfg0->progress_ctx);
    }

    /* Phase 4: 색상 — channel marker (audio_bins[0]) 로 분기.
     *
     *  slig_decompose_color 가 채워주는 셀은 u/v 가 비어있어 그대로
     *  slig_apply 하면 0 만 누적된다. 대신 audio_amps[0] (블록 평균
     *  크로마, 0..255) 를 origin (block center) 주변 falloff 으로
     *  Cb/Cr 캔버스에 직접 누적한다. */
    int N = SLIG_CANVAS_DIM;
    for (uint32_t i = d->texture_end; i < d->color_end && i < d->num_cells; i++) {
        SligSignal sig;
        slig_unpack(&sig, &d->cells[i]);
        uint8_t marker = sig.audio_bins[0];

        if (marker == 1 || marker == 2) {
            has_color_cells = 1;
            SligCanvas *target = (marker == 1) ? &cb_canvas : &cr_canvas;
            /* Two encodings of color cells:
             *
             *  (A) v3 native — slig_decompose_color emits cells with u/v
             *      empty (memset 0) and stores the 0..255 chroma value in
             *      audio_amps[0] for a 2×2 block layout. We block-write
             *      around (origin_x, origin_y) with triangular falloff.
             *
             *  (B) v2.3 native — slig_decompose_channel emits real SVD
             *      cells on the Cb / Cr plane with u/v populated and
             *      audio_amps[0] still at 0. The adapter (v23 → v3 in
             *      spatial_image_gen.c) sets the marker but leaves the
             *      amp blank. For these, slig_apply does the right thing
             *      when targeted at the chroma canvas.
             */
            if (sig.audio_amps[0] != 0) {
                /* (A) v3 block-falloff path */
                int chroma = (int)sig.audio_amps[0] - 128;
                if (sig.flags & SLIG_FLAG_NEGATIVE)
                    chroma = -((chroma < 0) ? -chroma : chroma);
                int cx = sig.origin_x;
                int cy = sig.origin_y;
                int half = N / 4;
                int32_t scale = (int32_t)(cfg0->sigma_scale * 256);
                int32_t base  = chroma * scale;
                for (int dy = -half; dy < half; dy++) {
                    int y = cy + dy;
                    if (y < 0 || y >= N) continue;
                    for (int dx = -half; dx < half; dx++) {
                        int x = cx + dx;
                        if (x < 0 || x >= N) continue;
                        int ax = (dx < 0) ? -dx : dx;
                        int ay = (dy < 0) ? -dy : dy;
                        int max_ad = (ax > ay) ? ax : ay;
                        int falloff = 256 - (max_ad * 256 / half);
                        if (falloff < 0) falloff = 0;
                        target->pixels[y][x] += (base * falloff) >> 8;
                    }
                }
            } else {
                /* (B) v2.3 SVD path — render the cell into the chroma
                 * canvas as if it were luma. canvas accumulator is shared
                 * scale (>>20 inside apply_global), so resulting Cb/Cr
                 * canvas is in the same magnitude as y_canvas — chroma
                 * normalize step (chroma_canvas_to_u8) keeps it bounded. */
                sig.sigma = (uint16_t)clampf(sig.sigma * cfg0->sigma_scale, 0, 65535);
                slig_apply(target, &sig);
            }
        } else {
            /* 마커 없는 컬러 셀은 luma 로 fallback */
            render_cell(&y_canvas, &d->cells[i], cfg0->sigma_scale);
        }
    }
    if (cfg0->progressive && cfg0->on_progress) {
        if (has_color_cells && out->channels >= 3) {
            compose_ycbcr_to_rgb(out, &y_canvas, &cb_canvas, &cr_canvas);
        } else {
            canvas_snapshot(out, &y_canvas);
        }
        cfg0->on_progress(out, 4, 5, cfg0->progress_ctx);
    }

    /* Phase 5: 이벤트 (시간 틱 누적) — 휘도 캔버스에만 */
    for (uint32_t i = d->color_end; i < d->num_cells; i++) {
        SligSignal sig;
        slig_unpack(&sig, &d->cells[i]);
        sig.sigma = (uint16_t)clampf(sig.sigma * cfg0->sigma_scale, 0, 65535);
        for (uint8_t t = 0; t <= 20; t++)
            slig_apply_at_tick(&y_canvas, &sig, t);
    }

    /* 조건부 강조 (guidance) — Y 캔버스에만 적용 */
    if (cfg0->guidance_scale != 1.0f) {
        int32_t cmean = 0;
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
                cmean += y_canvas.pixels[y][x] / (N * N);

        float g = cfg0->guidance_scale;
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
                y_canvas.pixels[y][x] =
                    cmean + (int32_t)((y_canvas.pixels[y][x] - cmean) * g);
    }

    /* 최종 출력: 컬러 셀 있으면 RGB 합성, 없으면 그레이 */
    if (has_color_cells && out->channels >= 3) {
        compose_ycbcr_to_rgb(out, &y_canvas, &cb_canvas, &cr_canvas);
    } else {
        canvas_snapshot(out, &y_canvas);
    }

    if (cfg0->progressive && cfg0->on_progress)
        cfg0->on_progress(out, 5, 5, cfg0->progress_ctx);
}

/* 페이즈별 렌더 (디버그용) */
void slig_render_phase_structure(SligCanvas *c, const SligDecomposed *d, int w, int h) {
    if (!c || !d) return;
    for (uint32_t i = 0; i < d->structure_end && i < d->num_cells; i++)
        slig_apply_cell(c, &d->cells[i]);
    (void)w; (void)h;
}
void slig_render_phase_edges(SligCanvas *c, const SligDecomposed *d, int w, int h) {
    if (!c || !d) return;
    for (uint32_t i = d->structure_end; i < d->edge_end && i < d->num_cells; i++)
        slig_apply_cell(c, &d->cells[i]);
    (void)w; (void)h;
}
void slig_render_phase_texture(SligCanvas *c, const SligDecomposed *d, int w, int h) {
    if (!c || !d) return;
    for (uint32_t i = d->edge_end; i < d->texture_end && i < d->num_cells; i++)
        slig_apply_cell(c, &d->cells[i]);
    (void)w; (void)h;
}
void slig_render_phase_color(SligCanvas *c, const SligDecomposed *d, int w, int h) {
    if (!c || !d) return;
    for (uint32_t i = d->texture_end; i < d->color_end && i < d->num_cells; i++)
        slig_apply_cell(c, &d->cells[i]);
    (void)w; (void)h;
}
void slig_render_phase_events(SligCanvas *c, const SligDecomposed *d, int w, int h, uint8_t t) {
    if (!c || !d) return;
    for (uint32_t i = d->color_end; i < d->num_cells; i++)
        slig_apply_cell_at_tick(c, &d->cells[i], t);
    (void)w; (void)h;
}

/* ════════════════════════════════════════════════════
 *  5. 후처리
 *
 *  언샤프 마스크 + 밴딩 제거 + 컬러 보정 + 히스토그램 매칭
 * ════════════════════════════════════════════════════ */

void slig_postprocess(SligImage *out, const SligImage *in,
                      const SligDecomposed *meta,
                      const SligPostprocessConfig *cfg0) {
    SligPostprocessConfig def;
    if (!cfg0) { slig_postprocess_config_default(&def); cfg0 = &def; }
    if (!out || !in) return;

    size_t n = (size_t)in->width * in->height * in->channels;
    if (out != in) {
        memcpy(out->data, in->data, n * sizeof(float));
        out->width = in->width; out->height = in->height; out->channels = in->channels;
    }

    /* 1. 언샤프 마스크 (선명화) — tick math: 5-tap uniform blur + (orig − blur)·amount/256.
     *
     * Replaces the expf-based Gaussian kernel with a fixed 5-neighbor
     * cross average (center + N/E/S/W). Each value is uint8-quantised
     * before the int32 mix, then promoted back to float [0,1]. Coarse
     * but float-free in the inner loop. */
    if (cfg0->enable_sharpen && cfg0->sharpen_amount > 0) {
        int amount_q8 = (int)(cfg0->sharpen_amount * 256.0f);
        if (amount_q8 < 0) amount_q8 = 0;
        SligImage *blurred = slig_image_alloc(out->width, out->height, out->channels);
        if (blurred) {
            int W = out->width, H = out->height, C = out->channels;
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    for (int c = 0; c < C; c++) {
                        /* Convert center + 4 neighbours to uint8 once, average. */
                        int center = (int)(img_get(out, x,     y,     c) * 255.0f);
                        int up     = (int)(img_get(out, x,     y - 1, c) * 255.0f);
                        int down   = (int)(img_get(out, x,     y + 1, c) * 255.0f);
                        int left   = (int)(img_get(out, x - 1, y,     c) * 255.0f);
                        int right  = (int)(img_get(out, x + 1, y,     c) * 255.0f);
                        /* Q8 weights ≈ 5-tap uniform: 0.2 each → 51/256 */
                        int sum_n = up + down + left + right;
                        int blur  = (51 * sum_n + 52 * center) >> 8;  /* sum 256 */
                        if (blur < 0) blur = 0; if (blur > 255) blur = 255;
                        img_set(blurred, x, y, c, blur / 255.0f);
                    }
                }
            }

            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    for (int c = 0; c < C; c++) {
                        int orig = (int)(img_get(out,     x, y, c) * 255.0f);
                        int blur = (int)(img_get(blurred, x, y, c) * 255.0f);
                        /* sharp = orig + amount × (orig − blur), Q8 fixed */
                        int sharp = orig + ((amount_q8 * (orig - blur)) >> 8);
                        if (sharp < 0)   sharp = 0;
                        if (sharp > 255) sharp = 255;
                        img_set(out, x, y, c, sharp / 255.0f);
                    }
                }
            }
            slig_image_free(blurred);
        }
    }

    /* 2. 밴딩 제거 (블록 경계 부드럽게) */
    if (cfg0->enable_debanding) {
        /* 8픽셀 경계에서만 약한 블러 */
        for (int y = 0; y < out->height; y++) {
            for (int x = 0; x < out->width; x++) {
                if ((x % 8 == 0 || y % 8 == 0) && x > 0 && y > 0) {
                    for (int c = 0; c < out->channels; c++) {
                        float v = img_get(out, x, y, c);
                        float avg = (img_get(out, x - 1, y, c)
                                   + img_get(out, x + 1, y, c)
                                   + img_get(out, x, y - 1, c)
                                   + img_get(out, x, y + 1, c)) * 0.25f;
                        /* 경계의 50%를 이웃 평균과 블렌드 */
                        img_set(out, x, y, c, v * 0.5f + avg * 0.5f);
                    }
                }
            }
        }
    }

    /* 3. 컬러 보정 (대비 + 밝기 + 채도) */
    if (cfg0->enable_color_correct) {
        for (int y = 0; y < out->height; y++) {
            for (int x = 0; x < out->width; x++) {
                if (out->channels >= 3 && cfg0->saturation != 1.0f) {
                    /* HSV 방식 채도 조절 */
                    float r = img_get(out, x, y, 0);
                    float g = img_get(out, x, y, 1);
                    float b = img_get(out, x, y, 2);
                    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
                    img_set(out, x, y, 0, clampf(gray + (r - gray) * cfg0->saturation, 0, 1));
                    img_set(out, x, y, 1, clampf(gray + (g - gray) * cfg0->saturation, 0, 1));
                    img_set(out, x, y, 2, clampf(gray + (b - gray) * cfg0->saturation, 0, 1));
                }

                /* 대비 + 밝기 — tick: (v − 128) × contrast_q8 ≫ 8 + 128 + bright_q8 */
                {
                    int contrast_q8   = (int)(cfg0->contrast   * 256.0f);
                    int brightness_q8 = (int)(cfg0->brightness * 255.0f);
                    for (int c = 0; c < out->channels; c++) {
                        int v   = (int)(img_get(out, x, y, c) * 255.0f);
                        int cen = v - 128;
                        int sc  = (cen * contrast_q8) >> 8;
                        int rs  = sc + 128 + brightness_q8;
                        if (rs < 0)   rs = 0;
                        if (rs > 255) rs = 255;
                        img_set(out, x, y, c, rs / 255.0f);
                    }
                }
            }
        }
    }

    /* 4. 히스토그램 매칭 (전처리 역변환) */
    if (cfg0->enable_histogram_match && meta) {
        for (size_t i = 0; i < n; i++) {
            float v = out->data[i];
            out->data[i] = clampf(meta->orig_min + v * (meta->orig_max - meta->orig_min), 0, 1);
        }
    }
}

/* ════════════════════════════════════════════════════
 *  6. 통합 API
 * ════════════════════════════════════════════════════ */

void slig_learn_image(SligDecomposed *out,
                      const uint8_t *image_data,
                      int width, int height, int channels) {
    if (!out || !image_data) return;

    /* uint8 → float 변환 */
    SligImage *im = slig_image_alloc(width, height, channels);
    if (!im) return;
    size_t n = (size_t)width * height * channels;
    for (size_t i = 0; i < n; i++)
        im->data[i] = image_data[i] / 255.0f;

    /* 전처리 */
    SligImage *pre = slig_image_alloc(width, height, channels);
    if (!pre) { slig_image_free(im); return; }

    SligPreprocessConfig pc;
    slig_preprocess_config_default(&pc);
    slig_preprocess(pre, out, im, &pc);

    /* 5단계 분해 */
    SligDecomposeConfig dc;
    slig_decompose_config_default(&dc);
    slig_decompose_v2(out, pre, &dc);

    slig_image_free(im);
    slig_image_free(pre);
}

void slig_generate_image(uint8_t *out_image,
                         int target_width, int target_height,
                         const SligDecomposed *decomposed,
                         float guidance_scale) {
    if (!out_image || !decomposed) return;

    /* 업스케일 (원본 수정 안 하려고 복사) */
    SligDecomposed tmp = *decomposed;
    SligUpscaleConfig uc;
    slig_upscale_config_default(&uc);
    slig_upscale(&tmp, &uc);

    /* 렌더 */
    SligImage *im = slig_image_alloc(target_width, target_height, 3);
    if (!im) return;

    SligRenderConfig rc;
    slig_render_config_default(&rc);
    rc.target_width = target_width;
    rc.target_height = target_height;
    rc.guidance_scale = guidance_scale;
    slig_render_adaptive(im, &tmp, &rc);

    /* 후처리 */
    SligImage *post = slig_image_alloc(target_width, target_height, 3);
    if (!post) { slig_image_free(im); return; }

    SligPostprocessConfig pc;
    slig_postprocess_config_default(&pc);
    slig_postprocess(post, im, &tmp, &pc);

    /* float → uint8 */
    size_t n = (size_t)target_width * target_height * 3;
    for (size_t i = 0; i < n; i++)
        out_image[i] = (uint8_t)(clampf(post->data[i], 0, 1) * 255.0f + 0.5f);

    slig_image_free(im);
    slig_image_free(post);
}

/* ════════════════════════════════════════════════════
 *  7. 유틸리티
 * ════════════════════════════════════════════════════ */

void slig_compute_residual(SligImage *residual,
                           const SligImage *original,
                           const SligDecomposed *decomposed) {
    if (!residual || !original) return;

    /* 복원 이미지 생성 */
    SligImage *rec = slig_image_alloc(original->width, original->height, original->channels);
    if (!rec) return;

    SligRenderConfig rc;
    slig_render_config_default(&rc);
    rc.target_width = original->width;
    rc.target_height = original->height;
    slig_render_adaptive(rec, decomposed, &rc);

    /* 잔차 = 원본 - 복원 */
    size_t n = (size_t)original->width * original->height * original->channels;
    for (size_t i = 0; i < n; i++)
        residual->data[i] = original->data[i] - rec->data[i];
    residual->width = original->width;
    residual->height = original->height;
    residual->channels = original->channels;

    slig_image_free(rec);
}

int slig_image_save_ppm(const char *path, const SligImage *img) {
    if (!path || !img) return 0;
    FILE *f = fopen(path, "wb");
    if (!f) return 0;

    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            unsigned char rgb[3];
            for (int c = 0; c < 3; c++) {
                int cc = (img->channels == 1) ? 0 : c;
                rgb[c] = (unsigned char)(clampf(img_get(img, x, y, cc), 0, 1) * 255.0f + 0.5f);
            }
            fwrite(rgb, 1, 3, f);
        }
    }
    fclose(f);
    return 1;
}

float slig_measure_psnr(const SligImage *original,
                        const SligDecomposed *decomposed) {
    if (!original || !decomposed) return 0;

    SligImage *rec = slig_image_alloc(original->width, original->height, original->channels);
    if (!rec) return 0;

    SligRenderConfig rc;
    slig_render_config_default(&rc);
    rc.target_width = original->width;
    rc.target_height = original->height;
    slig_render_adaptive(rec, decomposed, &rc);

    double mse = 0;
    size_t n = (size_t)original->width * original->height * original->channels;
    for (size_t i = 0; i < n; i++) {
        double e = original->data[i] - rec->data[i];
        mse += e * e;
    }
    mse /= fmax(n, 1);

    slig_image_free(rec);
    return (mse <= 1e-12) ? 99.0f : (float)(10.0 * log10(1.0 / mse));
}
