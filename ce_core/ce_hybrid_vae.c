/* ce_hybrid_vae.c — 통합 VAE 구현
 *
 * 방법1 (블록 도장) + 방법2 (SLIG 악보) + 주파수 복원을 조합.
 *
 * Encode:
 *   이미지 → [블록 도장 1024개] + [SLIG 분해 9셋] + [오디오 링크]
 *
 * Decode:
 *   블록 도장 → 색 기반
 *   SLIG → 주파수 업스케일 → 틱 누적 → 형태 디테일
 *   블렌딩 → wave refine → 최종 이미지
 */

#include "ce_hybrid_vae.h"
#include "ce_tick.h"
#include "slig_tick_math.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ══════════════════════════════════════════════════════
 *  설정 기본값
 * ══════════════════════════════════════════════════════ */

void hybrid_vae_config_default(HybridVAEConfig *cfg) {
    if (!cfg) return;
    cfg->base_weight      = 0.4f;
    cfg->detail_weight    = 0.6f;
    cfg->enable_harmonic  = 1;
    cfg->enable_sbr       = 1;
    cfg->target_coeffs    = 32;
    cfg->structure_ticks  = 5;
    cfg->edge_ticks       = 10;
    cfg->detail_ticks     = 15;
    cfg->wave_iterations  = 100;
    cfg->guidance_scale   = 1.5f;
    cfg->residual_book    = NULL;
}

/* ══════════════════════════════════════════════════════
 *  내부 유틸
 * ══════════════════════════════════════════════════════ */

static int clamp_i(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* canvas_id를 CE Cell의 오디오트랙에 기록 (링크) */
static void stamp_audio_link(SligCellSet *set, uint32_t canvas_id) {
    for (uint32_t i = 0; i < set->num_cells; i++) {
        SligSignal sig;
        slig_unpack(&sig, &set->cells[i]);

        /* audio_bins[0..1] = canvas_id의 하위 16비트 (lo/hi byte)
         * audio_bins[2]    = scale_level
         * audio_bins[3]    = channel
         * audio_amps[0..1] = canvas_id의 상위 16비트 */
        sig.audio_bins[0] = (uint8_t)(canvas_id & 0xFF);
        sig.audio_bins[1] = (uint8_t)((canvas_id >> 8) & 0xFF);
        sig.audio_bins[2] = set->scale_level;
        sig.audio_bins[3] = set->channel;
        sig.audio_amps[0] = (uint8_t)((canvas_id >> 16) & 0xFF);
        sig.audio_amps[1] = (uint8_t)((canvas_id >> 24) & 0xFF);
        sig.flags |= SLIG_FLAG_AUDIO_LINK;

        slig_pack(&set->cells[i], &sig);
    }
}

/* slig_scale_dim 재현 (COARSE=32, MID=128, FINE=256) */
static int scale_dim(int level) {
    switch (level) {
        case SLIG_LEVEL_COARSE: return 32;
        case SLIG_LEVEL_MID:    return 128;
        case SLIG_LEVEL_FINE:   return 256;
        default:                return 256;
    }
}

/* 셀 예산 (채널별) */
static uint32_t cell_budget(int channel) {
    switch (channel) {
        case SLIG_CH_Y:  return SLIG_CELL_BUDGET_Y;
        case SLIG_CH_CB: return SLIG_CELL_BUDGET_CB;
        case SLIG_CH_CR: return SLIG_CELL_BUDGET_CR;
        default:         return SLIG_CELL_BUDGET_Y;
    }
}

/* ══════════════════════════════════════════════════════
 *  ENCODE
 * ══════════════════════════════════════════════════════ */

void hybrid_vae_encode(HybridEncodeResult *result,
                       CEStorage          *storage,
                       SligCodebook       *codebook,
                       const uint8_t      *rgba,
                       int                 width,
                       int                 height,
                       uint32_t            canvas_id) {
    if (!result || !storage || !codebook || !rgba) return;
    if (width != 256 || height != 256) return;  /* 현재 256×256 전용 */
    memset(result, 0, sizeof(*result));
    result->canvas_id = canvas_id;

    int n = width * height;

    /* ── Step 1: 방법1 — 블록 도장 ──────────────────────
     * 16×16 블록으로 잘라서 색 정보를 CEStorage에 저장.
     * 각 블록 → 4 quadrant CEUnit. 총 1024 엔트리. */

    result->block_entries = ce_storage_ingest_rgba_16(
        storage, canvas_id, rgba, width, height);

    /* ── Step 2: 방법2 — SLIG 파형 분해 ────────────────
     * RGB → YCbCr → 각 채널을 3단계 스케일로 분해.
     * 각 (scale, channel) 조합이 하나의 SligCellSet이 됨. */

    /* RGB 분리 */
    uint8_t *rgb_interleaved = (uint8_t *)malloc((size_t)n * 3);
    uint8_t *y_plane  = (uint8_t *)malloc(n);
    uint8_t *cb_plane = (uint8_t *)malloc(n);
    uint8_t *cr_plane = (uint8_t *)malloc(n);
    if (!rgb_interleaved || !y_plane || !cb_plane || !cr_plane) {
        free(rgb_interleaved); free(y_plane); free(cb_plane); free(cr_plane);
        return;
    }

    /* RGBA → RGB 인터리브드 (A 제거) */
    for (int i = 0; i < n; i++) {
        rgb_interleaved[i * 3 + 0] = rgba[i * 4 + 0];
        rgb_interleaved[i * 3 + 1] = rgba[i * 4 + 1];
        rgb_interleaved[i * 3 + 2] = rgba[i * 4 + 2];
    }

    /* RGB → YCbCr */
    slig_rgb_to_ycbcr_planes(y_plane, cb_plane, cr_plane,
                             rgb_interleaved, width, height);
    free(rgb_interleaved);

    /* 각 채널의 원본 포인터 */
    const uint8_t *ch_planes[SLIG_NUM_CHANNELS] = { y_plane, cb_plane, cr_plane };

    /* 3 scale × 3 channel = 9개 분해 */
    uint32_t total_cells = 0;

    for (int lvl = 0; lvl < SLIG_NUM_LEVELS; lvl++) {
        int dim = scale_dim(lvl);

        /* 현재 스케일에 맞게 다운샘플 */
        uint8_t *scaled = NULL;
        if (dim < 256) {
            scaled = (uint8_t *)malloc((size_t)dim * dim);
            if (!scaled) continue;
        }

        for (int ch = 0; ch < SLIG_NUM_CHANNELS; ch++) {
            const uint8_t *src;
            if (dim < 256) {
                slig_image_downsample(scaled, dim, ch_planes[ch], 256);
                src = scaled;
            } else {
                src = ch_planes[ch];
            }

            /* 분해 */
            SligCellSet *set = &result->sets[lvl][ch];
            slig_decompose_channel(set, src, dim, dim,
                                   (uint8_t)ch, (uint8_t)lvl,
                                   cell_budget(ch));

            total_cells += set->num_cells;

            /* ── Step 3: 주파수 업스케일 (학습 시) ──────
             * DCT 16계수 → 32계수로 확장 (harmonic + SBR)
             * 복원 시 고주파 디테일이 더 살아남. */
            for (uint32_t c = 0; c < set->num_cells; c++) {
                slig_upscale_harmonic(&set->cells[c], 32);
                slig_upscale_sbr(&set->cells[c], 32);
            }

            /* ── Step 4: 오디오 링크 기록 ──────────────
             * dec.A에 canvas_id + scale + channel 기록.
             * "이 악보는 저 색 지도 위에서 연주한다" */
            stamp_audio_link(set, canvas_id);

            /* ── Step 5: codebook 등록 ─────────────── */
            result->codebook_idx[lvl][ch] =
                slig_codebook_add_or_lookup(codebook, set,
                                            SLIG_CODEBOOK_DEFAULT_THRESHOLD);

            /* ── Step 6: CEStorage 영속화 (CE_TYPE_SLIG) ──
             * 각 cell이 한 entry로 들어가서 디코드 단계에서
             * canvas_id로 다시 9 셋을 재구성할 수 있게 한다. */
            ce_storage_persist_slig_set(storage, canvas_id, set);
        }

        free(scaled);
    }

    result->total_cells = total_cells;

    /* 정리 */
    free(y_plane);
    free(cb_plane);
    free(cr_plane);
}

/* ══════════════════════════════════════════════════════
 *  블록 도장 → coarse 색 기반 복원 (Step 1)
 * ══════════════════════════════════════════════════════ */

void hybrid_decode_base_only(uint8_t         *out_rgba,
                             const CEStorage *storage,
                             uint32_t         canvas_id) {
    if (!out_rgba || !storage) return;
    memset(out_rgba, 128, (size_t)256 * 256 * 4);  /* 미드그레이 초기화 */

    for (uint32_t i = 0; i < storage->count; i++) {
        const CEStorageEntry *e = &storage->entries[i];
        if (e->type != CE_TYPE_IMAGE)   continue;
        if (e->canvas_id != canvas_id)  continue;

        /* TL quadrant만 수집 (block_idx & 3 == 0) */
        if ((e->block_idx & 0x3) != 0) continue;

        uint16_t brow = e->slot;
        uint16_t bcol = (uint16_t)(e->block_idx >> 2);

        /* 같은 블록의 4 quadrant 수집 */
        CEUnit q[4];
        int found[4] = {0, 0, 0, 0};
        q[0] = e->keyframe;
        found[0] = 1;

        for (uint32_t j = 0; j < storage->count; j++) {
            const CEStorageEntry *f = &storage->entries[j];
            if (f->type != CE_TYPE_IMAGE)   continue;
            if (f->canvas_id != canvas_id)  continue;
            if (f->slot != brow)            continue;
            if ((f->block_idx >> 2) != bcol) continue;
            int qi = f->block_idx & 0x3;
            if (qi > 0 && qi < 4) {
                q[qi] = f->keyframe;
                found[qi] = 1;
            }
        }

        /* 4 quadrant 모두 있으면 16×16 패치 디코드 */
        if (found[0] && found[1] && found[2] && found[3]) {
            uint8_t patch[16 * 16 * 4];
            ce_decode_image_block_16(patch, q);

            /* 256×256 캔버스에 배치 */
            int px0 = bcol * 16;
            int py0 = brow * 16;
            for (int dy = 0; dy < 16; dy++) {
                for (int dx = 0; dx < 16; dx++) {
                    int cx = px0 + dx;
                    int cy = py0 + dy;
                    if (cx >= 256 || cy >= 256) continue;
                    int si = (dy * 16 + dx) * 4;
                    int di = (cy * 256 + cx) * 4;
                    out_rgba[di + 0] = patch[si + 0];
                    out_rgba[di + 1] = patch[si + 1];
                    out_rgba[di + 2] = patch[si + 2];
                    out_rgba[di + 3] = patch[si + 3];
                }
            }
        }
    }
}

/* ══════════════════════════════════════════════════════
 *  SLIG 셀 → 형태 디테일 복원 (Step 2-3)
 * ══════════════════════════════════════════════════════ */

void hybrid_decode_detail_only(uint8_t             *out_gray,
                               const SligCellSet    sets[SLIG_NUM_LEVELS][SLIG_NUM_CHANNELS],
                               const HybridVAEConfig *cfg) {
    if (!out_gray || !sets || !cfg) return;

    int dim_c = scale_dim(SLIG_LEVEL_COARSE);  /* 32  */
    int dim_m = scale_dim(SLIG_LEVEL_MID);      /* 128 */
    int dim_f = scale_dim(SLIG_LEVEL_FINE);     /* 256 */

    uint8_t *coarse_panel = (uint8_t *)calloc(1, (size_t)dim_c * dim_c);
    uint8_t *mid_panel    = (uint8_t *)calloc(1, (size_t)dim_m * dim_m);
    uint8_t *fine_panel   = (uint8_t *)calloc(1, (size_t)dim_f * dim_f);
    uint8_t *up_buf       = (uint8_t *)calloc(1, (size_t)dim_f * dim_f);
    if (!coarse_panel || !mid_panel || !fine_panel || !up_buf) {
        free(coarse_panel); free(mid_panel); free(fine_panel); free(up_buf);
        memset(out_gray, 128, (size_t)dim_f * dim_f);
        return;
    }

    (void)cfg;  /* 틱 파라미터는 향후 slig_render_timed 경로에서 사용 */

    /* Y 채널의 3단계 스케일을 각 dim에 맞게 복원.
     * slig_reconstruct_at_dim: SligCellSet → dim×dim 렌더.
     * 빈 셋은 mid-gray(128)로 채워져서 잔차 체인에서 "변화 없음"이 됨. */
    slig_reconstruct_at_dim(coarse_panel, dim_c,
                            &sets[SLIG_LEVEL_COARSE][SLIG_CH_Y]);
    slig_reconstruct_at_dim(mid_panel, dim_m,
                            &sets[SLIG_LEVEL_MID][SLIG_CH_Y]);
    slig_reconstruct_at_dim(fine_panel, dim_f,
                            &sets[SLIG_LEVEL_FINE][SLIG_CH_Y]);

    /* 잔차 체인 디코드:
     *   pred_mid  = upsample(coarse) + (mid − 128)
     *   pred_fine = upsample(pred_mid) + (fine − 128)
     * mid/fine은 잔차이므로 128 중심 (128 = 변화 없음). */

    /* coarse → mid 해상도로 업샘플 */
    slig_image_upsample(up_buf, 128, coarse_panel, 32);

    /* mid 잔차 합산 */
    uint8_t *pred_mid = (uint8_t *)malloc(128 * 128);
    if (!pred_mid) {
        free(coarse_panel); free(mid_panel); free(fine_panel); free(up_buf);
        memset(out_gray, 128, (size_t)dim_f * dim_f);
        return;
    }
    for (int i = 0; i < 128 * 128; i++) {
        int v = (int)up_buf[i] + (int)mid_panel[i] - 128;
        pred_mid[i] = (uint8_t)clamp_i(v, 0, 255);
    }

    /* pred_mid → fine 해상도로 업샘플 */
    slig_image_upsample(up_buf, 256, pred_mid, 128);

    /* fine 잔차 합산 → 최종 디테일 */
    for (int i = 0; i < dim_f * dim_f; i++) {
        int v = (int)up_buf[i] + (int)fine_panel[i] - 128;
        out_gray[i] = (uint8_t)clamp_i(v, 0, 255);
    }

    free(coarse_panel); free(mid_panel); free(fine_panel);
    free(up_buf); free(pred_mid);
}

/* ══════════════════════════════════════════════════════
 *  DECODE — 통합 (블록 도장 + SLIG + 주파수 복원)
 * ══════════════════════════════════════════════════════ */

void hybrid_vae_decode(uint8_t                *out_rgb,
                       const CEStorage        *storage,
                       uint32_t                canvas_id,
                       const SligCellSet       sets[SLIG_NUM_LEVELS][SLIG_NUM_CHANNELS],
                       const HybridVAEConfig  *cfg) {
    if (!out_rgb || !storage || !sets) return;

    HybridVAEConfig c;
    if (cfg) c = *cfg;
    else     hybrid_vae_config_default(&c);

    int n = 256 * 256;

    /* ── Step 1: 블록 도장 → 색 기반 (방법1) ──────── */
    uint8_t *base_rgba = (uint8_t *)calloc(1, (size_t)n * 4);
    if (!base_rgba) return;
    hybrid_decode_base_only(base_rgba, storage, canvas_id);

    /* base에서 YCbCr 분리 */
    uint8_t *base_rgb = (uint8_t *)malloc((size_t)n * 3);
    uint8_t *base_y   = (uint8_t *)malloc(n);
    uint8_t *base_cb  = (uint8_t *)malloc(n);
    uint8_t *base_cr  = (uint8_t *)malloc(n);
    if (!base_rgb || !base_y || !base_cb || !base_cr) {
        free(base_rgba); free(base_rgb);
        free(base_y); free(base_cb); free(base_cr);
        return;
    }
    for (int i = 0; i < n; i++) {
        base_rgb[i * 3 + 0] = base_rgba[i * 4 + 0];
        base_rgb[i * 3 + 1] = base_rgba[i * 4 + 1];
        base_rgb[i * 3 + 2] = base_rgba[i * 4 + 2];
    }
    slig_rgb_to_ycbcr_planes(base_y, base_cb, base_cr,
                             base_rgb, 256, 256);
    free(base_rgba);
    free(base_rgb);

    /* ── Step 2: 주파수 업스케일 (방법2 강화) ──────── */
    /* encode 시 이미 harmonic/SBR이 적용됨.
     * decode 시에는 셀 복사본에 위너 필터 추가 적용. */
    SligCellSet work_sets[SLIG_NUM_LEVELS][SLIG_NUM_CHANNELS];
    memcpy(work_sets, sets, sizeof(work_sets));

    for (int lvl = 0; lvl < SLIG_NUM_LEVELS; lvl++) {
        for (int ch = 0; ch < SLIG_NUM_CHANNELS; ch++) {
            SligCellSet *ws = &work_sets[lvl][ch];
            for (uint32_t ci = 0; ci < ws->num_cells; ci++) {
                slig_upscale_wiener(&ws->cells[ci]);
            }
        }
    }

    /* ── Step 3: SLIG 틱 렌더 (방법2 + 틱 엔진) ────── */
    /* 각 채널(Y/Cb/Cr)별로 3-scale 피라미드 렌더 + 잔차 합산 */

    uint8_t *detail_y  = (uint8_t *)malloc(n);
    uint8_t *detail_cb = (uint8_t *)malloc(n);
    uint8_t *detail_cr = (uint8_t *)malloc(n);
    if (!detail_y || !detail_cb || !detail_cr) {
        free(base_y); free(base_cb); free(base_cr);
        free(detail_y); free(detail_cb); free(detail_cr);
        return;
    }

    /* Y 채널 디테일 */
    hybrid_decode_detail_only(detail_y, work_sets, &c);

    /* Cb/Cr 채널: coarse 스케일만 사용 (색은 저해상도로 충분) */
    {
        int dim_c = 32;
        uint8_t *cb_coarse = (uint8_t *)calloc(1, (size_t)dim_c * dim_c);
        uint8_t *cr_coarse = (uint8_t *)calloc(1, (size_t)dim_c * dim_c);
        if (cb_coarse && cr_coarse) {
            slig_reconstruct_at_dim(cb_coarse, dim_c,
                                    &work_sets[SLIG_LEVEL_COARSE][SLIG_CH_CB]);
            slig_reconstruct_at_dim(cr_coarse, dim_c,
                                    &work_sets[SLIG_LEVEL_COARSE][SLIG_CH_CR]);
            slig_image_upsample(detail_cb, 256, cb_coarse, dim_c);
            slig_image_upsample(detail_cr, 256, cr_coarse, dim_c);
        } else {
            memset(detail_cb, 128, n);
            memset(detail_cr, 128, n);
        }
        free(cb_coarse);
        free(cr_coarse);
    }

    /* ── Step 4: 블렌딩 — base(색) × α + detail(형태) × β ── */
    uint8_t *blend_y  = (uint8_t *)malloc(n);
    uint8_t *blend_cb = (uint8_t *)malloc(n);
    uint8_t *blend_cr = (uint8_t *)malloc(n);
    if (!blend_y || !blend_cb || !blend_cr) {
        free(base_y); free(base_cb); free(base_cr);
        free(detail_y); free(detail_cb); free(detail_cr);
        free(blend_y); free(blend_cb); free(blend_cr);
        return;
    }

    /* Q8 고정소수점 블렌딩 */
    int alpha = (int)(c.base_weight * 256.0f);
    int beta  = (int)(c.detail_weight * 256.0f);

    for (int i = 0; i < n; i++) {
        /* Y: 블록 도장(밝기) + SLIG(형태) 조합 */
        blend_y[i] = (uint8_t)clamp_i(
            (alpha * (int)base_y[i] + beta * (int)detail_y[i]) >> 8, 0, 255);

        /* Cb/Cr: 블록 도장(색) 우선, SLIG(색 디테일) 보조
         * 색은 base가 더 정확하므로 비중을 높임 */
        int cb_alpha = (int)(0.6f * 256.0f);
        int cb_beta  = (int)(0.4f * 256.0f);
        blend_cb[i] = (uint8_t)clamp_i(
            (cb_alpha * (int)base_cb[i] + cb_beta * (int)detail_cb[i]) >> 8, 0, 255);
        blend_cr[i] = (uint8_t)clamp_i(
            (cb_alpha * (int)base_cr[i] + cb_beta * (int)detail_cr[i]) >> 8, 0, 255);
    }

    free(base_y);  free(base_cb);  free(base_cr);
    free(detail_y); free(detail_cb); free(detail_cr);

    /* ── Step 4b: tick-sorted residual codebook 적용 ──
     *
     * storage에 들어있는 CE_TYPE_RESIDUAL descriptors를 tick(B>G>R>A)
     * 순으로 정렬한 뒤 그 순서대로 codebook 패턴을 blend_y에 적용.
     * descriptor의 (x, y)를 패치 중심 좌표로 사용해 8×8 가우시안 stamp.
     * book이 NULL이면 이 단계는 no-op. */
    if (c.residual_book) {
        uint32_t resid_idx[256];
        uint32_t nresid = ce_tick_sorted_indices(
            storage, canvas_id,
            (1u << CE_TYPE_RESIDUAL),
            resid_idx, 256);

        /* 8×8 패치용 가우시안 양자화 weight (Q8 — sum / 256 = 1.0 in
         * the central row, falling off at edges). 사전계산해서 매 패치
         * 호출 시 곱셈만 한다. */
        static const uint8_t patch_w[8][8] = {
            { 16,  32,  48,  64,  64,  48,  32, 16},
            { 32,  64,  96, 128, 128,  96,  64, 32},
            { 48,  96, 144, 192, 192, 144,  96, 48},
            { 64, 128, 192, 255, 255, 192, 128, 64},
            { 64, 128, 192, 255, 255, 192, 128, 64},
            { 48,  96, 144, 192, 192, 144,  96, 48},
            { 32,  64,  96, 128, 128,  96,  64, 32},
            { 16,  32,  48,  64,  64,  48,  32, 16},
        };

        for (uint32_t k = 0; k < nresid; ++k) {
            const CEStorageEntry *e = &storage->entries[resid_idx[k]];
            uint8_t cb_idx = 0, strength = 0;
            TickRGBA tk = (TickRGBA){0,0,0,0};
            uint16_t px = 0, py = 0;
            if (ce_residual_storage_unpack(e, &cb_idx, &strength, &tk,
                                           &px, &py) != 0)
                continue;

            const CEResidualCode *code =
                ce_residual_codebook_get(c.residual_book, cb_idx);
            if (!code) continue;

            uint8_t code_amp = ce_tick_amp_from_cell(&code->unit);
            int eff = ((int)strength * (int)code_amp) >> 8;
            if (eff > 64) eff = 64;
            if (eff <= 0) continue;

            int sign = (tk.g & 1) ? -1 : 1;

            /* px / py 가 16-bit 이지만 source 패치는 256-pixel 좌표계
             * 위에서 stamping된다. 캔버스 범위 wrap 클램프. */
            int cx = (int)(px & 0xFF);
            int cy = (int)(py & 0xFF);

            /* 8×8 패치 적용: 중심 (cx, cy) 기준 -3..+4 픽셀 윈도우.
             * 가장자리는 클램프해서 잘려도 이상한 wrap 안 생기게. */
            for (int dy = -3; dy <= 4; ++dy) {
                int yy = cy + dy;
                if (yy < 0 || yy >= 256) continue;
                for (int dx = -3; dx <= 4; ++dx) {
                    int xx = cx + dx;
                    if (xx < 0 || xx >= 256) continue;
                    int wpx = patch_w[dy + 3][dx + 3];
                    int delta = sign * ((eff * wpx) >> 8);
                    int v = (int)blend_y[yy * 256 + xx] + delta;
                    if (v < 0) v = 0;
                    if (v > 255) v = 255;
                    blend_y[yy * 256 + xx] = (uint8_t)v;
                }
            }
        }
    }

    /* ── Step 5: CFG — achromatic(128) 기준 편차 증폭 ── */
    if (c.guidance_scale != 1.0f) {
        int s_q8 = (int)(c.guidance_scale * 256.0f);
        for (int i = 0; i < n; i++) {
            int dy = (int)blend_y[i] - 128;
            blend_y[i] = (uint8_t)clamp_i(128 + (s_q8 * dy) / 256, 0, 255);
        }
    }

    /* ── Step 6: Wave refine — 블렌딩 결과를 파동 정제 ── */
    if (c.wave_iterations > 0) {
        /* Y 채널에 대해 SligCanvas 기반 wave refine 수행 */
        SligCanvas canvas;
        slig_canvas_clear(&canvas);

        /* blend_y를 SligCanvas로 변환 */
        for (int y = 0; y < 256; y++)
            for (int x = 0; x < 256; x++)
                canvas.pixels[y][x] = (int32_t)blend_y[y * 256 + x] * 256;

        /* 타겟 없이 self-smoothing */
        slig_wave_refine(&canvas, NULL, (int)c.wave_iterations);

        /* 캔버스 → uint8 */
        uint8_t refined[256][256];
        slig_canvas_to_u8(refined, &canvas);
        for (int i = 0; i < n; i++)
            blend_y[i] = refined[i / 256][i % 256];
    }

    /* ── Step 7: YCbCr → RGB 최종 변환 ──────────────── */
    slig_ycbcr_planes_to_rgb(out_rgb, blend_y, blend_cb, blend_cr, 256, 256);

    free(blend_y); free(blend_cb); free(blend_cr);
}

/* ══════════════════════════════════════════════════════
 *  PSNR 계산
 * ══════════════════════════════════════════════════════ */

float hybrid_psnr(const uint8_t *original_rgb,
                  const uint8_t *decoded_rgb,
                  int width, int height) {
    if (!original_rgb || !decoded_rgb) return 0.0f;
    int n = width * height * 3;
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)original_rgb[i] - (double)decoded_rgb[i];
        mse += d * d;
    }
    mse /= (double)n;
    if (mse < 1e-10) return 99.0f;
    return (float)(10.0 * log10(255.0 * 255.0 / mse));
}

/* ══════════════════════════════════════════════════════
 *  Roundtrip — encode + decode + PSNR
 * ══════════════════════════════════════════════════════ */

float hybrid_vae_roundtrip(uint8_t            *out_rgb,
                           CEStorage          *storage,
                           SligCodebook       *codebook,
                           const uint8_t      *rgba,
                           int                 width,
                           int                 height,
                           uint32_t            canvas_id,
                           const HybridVAEConfig *cfg) {
    if (!out_rgb || !storage || !codebook || !rgba) return 0.0f;

    /* Encode */
    HybridEncodeResult enc;
    hybrid_vae_encode(&enc, storage, codebook, rgba, width, height, canvas_id);

    /* Decode */
    HybridVAEConfig c;
    if (cfg) c = *cfg;
    else     hybrid_vae_config_default(&c);
    hybrid_vae_decode(out_rgb, storage, canvas_id, enc.sets, &c);

    /* PSNR: 원본 RGB와 복원 RGB 비교 */
    uint8_t *orig_rgb = (uint8_t *)malloc((size_t)width * height * 3);
    if (!orig_rgb) return 0.0f;
    for (int i = 0; i < width * height; i++) {
        orig_rgb[i * 3 + 0] = rgba[i * 4 + 0];
        orig_rgb[i * 3 + 1] = rgba[i * 4 + 1];
        orig_rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    float psnr = hybrid_psnr(orig_rgb, out_rgb, width, height);
    free(orig_rgb);

    return psnr;
}
