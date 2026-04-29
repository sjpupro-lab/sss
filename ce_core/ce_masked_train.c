/* ce_masked_train.c — 마스크 학습 구현
 *
 * 핵심 루프:
 *   1. 이미지 분해 → basis / residual / correction
 *   2. correction 마스크 → CEStorage에 basis+residual만 넣음
 *   3. denoise로 correction 예측
 *   4. 예측 vs 실제 → loss
 *   5. 파라미터 조정 → 반복
 *   6. 수렴 시 전체 셀(예측된 correction 포함)을 CEStorage에 저장
 */

#include "ce_masked_train.h"
#include "ce_tick.h"
#include "slig_signal.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ══════════════════════════════════════════════════════
 *  설정 기본값
 * ══════════════════════════════════════════════════════ */

void masked_train_config_default(MaskedTrainConfig *cfg) {
    if (!cfg) return;
    cfg->epochs            = 10;
    cfg->denoise_steps     = 8;
    cfg->target_loss       = 8.0f;
    cfg->loss_patience     = 3;
    cfg->strategy          = MASK_PROGRESSIVE;
    cfg->mask_seed         = 42;
    cfg->store_on_converge = 1;
    cfg->residual_book     = NULL;
    cfg->residual_threshold = 0;
    /* Segment weights default to 0 → use legacy mean loss. Callers who
     * want the structure/texture/color split set positive weights. */
    cfg->loss_w_structure  = 0.0f;
    cfg->loss_w_texture    = 0.0f;
    cfg->loss_w_color      = 0.0f;
    cfg->on_epoch          = NULL;
    cfg->cb_ctx            = NULL;
}

/* ══════════════════════════════════════════════════════
 *  내부: SplitMix64 PRNG (랜덤 마스크용)
 * ══════════════════════════════════════════════════════ */

static uint64_t splitmix(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ══════════════════════════════════════════════════════
 *  내부: SligDecomposed의 셀을 CEStorage에 넣기
 *
 *  basis/residual/correction 경계를 mask 배열로 제어.
 *  mask[i] = 0 → 저장 (context로 사용)
 *  mask[i] = 1 → 스킵 (예측 대상)
 * ══════════════════════════════════════════════════════ */

/* Internal: store one SligDecomposed bucket as raw CE_TYPE_SLIG
 * entries. `type_override` lets callers pin the modality (e.g. keep
 * CE_TYPE_IMAGE for the legacy mask-only path that needs to stay
 * compatible with test_masked_train's prompt-from-storage behaviour). */
static uint32_t store_raw_cells(CEStorage *storage,
                                uint32_t canvas_id,
                                const SligDecomposed *dec,
                                const uint8_t *mask,
                                uint8_t type_override) {
    uint32_t added = 0;
    for (uint32_t i = 0; i < dec->num_cells; i++) {
        if (mask && mask[i]) continue;

        CEUnit zero;
        ce_init(&zero);
        CEUnit delta;
        ce_delta(&delta, &zero, &dec->cells[i]);

        ce_storage_add_typed(storage, canvas_id,
                             (uint16_t)(i / 8),
                             (uint16_t)(i % 8),
                             (CEType)type_override,
                             &dec->cells[i], &delta);
        ++added;
    }
    return added;
}

/* Backwards-compatible legacy path: every cell is stored raw with
 * CE_TYPE_IMAGE. No mask filter, no codebook bucketing. The default
 * masked_train_image flow keeps using this when residual_book is
 * NULL so existing callers (and test_masked_train) see no change. */
static void store_cells_masked(CEStorage *storage,
                               uint32_t canvas_id,
                               const SligDecomposed *dec,
                               const uint8_t *mask) {
    (void)store_raw_cells(storage, canvas_id, dec, mask, CE_TYPE_IMAGE);
}

/* Map a flat SligDecomposed cell index to a (scale_level, channel)
 * bucket so masked-train output lands in the same 3×3 grid that
 * hybrid_vae_decode reads via ce_storage_load_slig_sets. The mapping
 * follows the natural decomposition split:
 *
 *   [0           .. structure_end)  → COARSE × Y    (basis luminance)
 *   [structure_end .. edge_end)     → MID    × Y    (luma edges)
 *   [edge_end    .. texture_end)    → FINE   × Y    (luma textures)
 *   [texture_end .. color_end)      → COARSE × Cb   (color base, even idx)
 *                                     COARSE × Cr   (color base, odd  idx)
 *   [color_end   .. num_cells)      → MID    × Cr   (transient events)
 *
 * The decoder reads Y at all three scales for shape and Cb/Cr at
 * COARSE for chroma, so this distribution exposes every region of the
 * masked-train output to the decode pipeline. */
static void map_decomposed_to_bucket(const SligDecomposed *dec, uint32_t i,
                                     uint8_t *out_scale, uint8_t *out_channel) {
    if (i < dec->structure_end) {
        *out_scale = SLIG_LEVEL_COARSE; *out_channel = SLIG_CH_Y;
    } else if (i < dec->edge_end) {
        *out_scale = SLIG_LEVEL_MID;    *out_channel = SLIG_CH_Y;
    } else if (i < dec->texture_end) {
        *out_scale = SLIG_LEVEL_FINE;   *out_channel = SLIG_CH_Y;
    } else if (i < dec->color_end) {
        *out_scale = SLIG_LEVEL_COARSE;
        *out_channel = (i & 1u) ? SLIG_CH_CR : SLIG_CH_CB;
    } else {
        *out_scale = SLIG_LEVEL_MID;    *out_channel = SLIG_CH_CR;
    }
}

/* Pattern-bucketed path: split SligDecomposed cells across two
 * storage layouts.
 *   - basis + residual edge + residual texture (cells[0..texture_end))
 *     → CE_TYPE_SLIG entries deposited into (scale, channel) buckets
 *       so ce_storage_load_slig_sets picks them up at decode time.
 *       Cells stack BEHIND any pre-existing entries in each bucket
 *       (encoder runs first, masked-train output appends).
 *   - correction = color + event (cells[texture_end..num_cells))
 *     → ce_residual_codebook lookup-or-add → CE_TYPE_RESIDUAL
 *       descriptor only (idx + strength + tick), patch payload lives
 *       in the codebook.
 *
 * `out_residuals` is incremented per descriptor written. Returns
 * total entries added across both paths. */
static uint32_t store_cells_with_codebook(CEStorage *storage,
                                          uint32_t canvas_id,
                                          const SligDecomposed *dec,
                                          CEResidualCodebook *book,
                                          uint32_t threshold,
                                          uint32_t *out_residuals) {
    if (!storage || !dec || !book) return 0;
    if (out_residuals) *out_residuals = 0;
    if (threshold == 0) threshold = CE_RESIDUAL_DEFAULT_THRESHOLD;

    uint32_t total = 0;
    uint32_t basis_end = dec->texture_end;
    if (basis_end > dec->num_cells) basis_end = dec->num_cells;

    /* Part 1: basis + residual cells deposited into (scale, channel)
     * buckets. We accumulate into a local 3×3 SligCellSet grid first
     * so we can compute each bucket's append base just once and stamp
     * channel/scale_level tags consistently. */
    SligCellSet buckets[SLIG_NUM_LEVELS][SLIG_NUM_CHANNELS];
    for (uint8_t s = 0; s < SLIG_NUM_LEVELS; ++s) {
        for (uint8_t c = 0; c < SLIG_NUM_CHANNELS; ++c) {
            memset(&buckets[s][c], 0, sizeof(SligCellSet));
            buckets[s][c].scale_level = s;
            buckets[s][c].channel     = c;
        }
    }
    for (uint32_t i = 0; i < basis_end; ++i) {
        uint8_t s = 0, c = 0;
        map_decomposed_to_bucket(dec, i, &s, &c);
        SligCellSet *b = &buckets[s][c];
        if (b->num_cells >= SLIG_MAX_CELLS) continue;
        b->cells[b->num_cells++] = dec->cells[i];
    }
    for (uint8_t s = 0; s < SLIG_NUM_LEVELS; ++s) {
        for (uint8_t c = 0; c < SLIG_NUM_CHANNELS; ++c) {
            SligCellSet *b = &buckets[s][c];
            if (b->num_cells == 0) continue;
            uint32_t base = ce_storage_slig_bucket_count(
                storage, canvas_id, s, c);
            total += ce_storage_append_slig_set(storage, canvas_id, b, base);
        }
    }

    /* Part 2: correction cells reduced to codebook descriptors. */
    for (uint32_t i = basis_end; i < dec->num_cells; ++i) {
        CEResidualCode probe;
        memset(&probe, 0, sizeof(probe));
        probe.unit        = dec->cells[i];
        /* SligDecomposed doesn't carry a per-cell scale tag, so we
         * use the wildcard (0xFF) — matches everything. Future passes
         * can stamp the actual scale here once it's available. */
        probe.scale_level = 0xFF;
        probe.direction   = 0;
        /* Cell strength derived the same way ce_tick_amp_from_cell
         * does — max audio amp + sigma upper byte. Keep it cheap. */
        probe.strength    = ce_tick_amp_from_cell(&dec->cells[i]);
        /* TickRGBA: R = local index in correction segment, G/B/A
         * derived from cell index + amp. The amp byte is the
         * meaningful payload; G/B start at 0 and are bumped during
         * decode if multiple corrections share the same R. */
        probe.tick.r = (uint8_t)((i - basis_end) & 0xFF);
        probe.tick.g = (uint8_t)((i / 16) & 0xFF);
        probe.tick.b = 0;
        probe.tick.a = probe.strength;

        uint8_t idx = ce_residual_codebook_add_or_lookup(
            book, &probe, threshold);

        /* Position: read origin_x / origin_y from the cell's SligSignal —
         * slig_decompose_color/events stamp where in the source plane
         * the patch was sampled (0..255 for a 256×256 image). Scale to
         * 16-bit so the descriptor can carry sub-pixel positions in
         * future high-resolution passes. */
        SligSignal sig;
        slig_unpack(&sig, &dec->cells[i]);
        uint16_t pos_x = (uint16_t)sig.origin_x;
        uint16_t pos_y = (uint16_t)sig.origin_y;

        ce_residual_storage_add(storage, canvas_id,
                                /*slot=*/(uint16_t)(i / 8),
                                /*block_idx=*/(uint16_t)(i % 8),
                                idx, probe.strength, probe.tick,
                                pos_x, pos_y);
        ++total;
        if (out_residuals) ++(*out_residuals);
    }

    return total;
}

/* ══════════════════════════════════════════════════════
 *  내부: 마스크 생성
 *
 *  SligDecomposed의 경계 인덱스를 사용해
 *  어떤 셀을 가릴지 결정.
 *
 *  구간:
 *    [0 .. structure_end)       = basis      (큰 윤곽)
 *    [structure_end .. edge_end)  = edge       (엣지)
 *    [edge_end .. texture_end)    = texture    (질감)
 *    [texture_end .. color_end)   = color      (색상)
 *    [color_end .. num_cells)     = event      (이벤트)
 *
 *  basis = structure
 *  residual = edge + texture
 *  correction = color + event
 * ══════════════════════════════════════════════════════ */

static void build_mask(uint8_t *mask, uint32_t num_cells,
                       const SligDecomposed *dec,
                       MaskStrategy strategy,
                       int epoch, int total_epochs,
                       uint64_t *rng_state) {
    memset(mask, 0, num_cells);

    /* progressive: epoch 비율에 따라 자동 전환 */
    MaskStrategy effective = strategy;
    if (strategy == MASK_PROGRESSIVE) {
        float ratio = (float)epoch / (float)(total_epochs > 1 ? total_epochs : 1);
        if (ratio < 0.33f)      effective = MASK_CORRECTION_ONLY;
        else if (ratio < 0.67f) effective = MASK_RESIDUAL_UP;
        else                    effective = MASK_RANDOM_HALF;
    }

    switch (effective) {
        case MASK_CORRECTION_ONLY:
            /* color + event 셀만 가림 */
            for (uint32_t i = dec->texture_end; i < num_cells; i++)
                mask[i] = 1;
            break;

        case MASK_RESIDUAL_UP:
            /* edge + texture + color + event 가림 (basis만 남김) */
            for (uint32_t i = dec->structure_end; i < num_cells; i++)
                mask[i] = 1;
            break;

        case MASK_RANDOM_HALF:
            /* 랜덤 50% 가림 (basis 첫 2개는 보존) */
            for (uint32_t i = 0; i < num_cells; i++) {
                if (i < 2) continue;  /* 최소 구조는 보존 */
                if (splitmix(rng_state) % 2 == 0)
                    mask[i] = 1;
            }
            break;

        default:
            break;
    }
}

/* ══════════════════════════════════════════════════════
 *  내부: 마스크된 셀을 denoise로 예측
 *
 *  basis+residual을 CEStorage에 넣고,
 *  마스크된 correction 셀 위치에 noise를 초기화한 뒤
 *  denoise loop을 돌려서 예측합니다.
 * ══════════════════════════════════════════════════════ */

/* Output struct mirroring SligDecomposed's natural segments:
 *   structure = cells[0 .. structure_end)
 *   texture   = cells[structure_end .. texture_end)
 *   color     = cells[texture_end .. num_cells)         */
typedef struct {
    float total;        /* mean over all masked cells (legacy) */
    float structure;    /* mean over masked structure cells   */
    float texture;      /* mean over masked texture cells     */
    float color;        /* mean over masked color cells       */
} SegLosses;

static float predict_and_measure(SligDecomposed *predicted,
                                 const SligDecomposed *original,
                                 const uint8_t *mask,
                                 uint32_t canvas_id,
                                 const CEGenConfig *gen_cfg,
                                 uint64_t seed,
                                 SegLosses *out_seg) {
    /* 임시 storage: 마스크되지 않은 셀만 넣음 */
    CEStorage tmp_storage;
    ce_storage_init(&tmp_storage, 128);
    store_cells_masked(&tmp_storage, canvas_id, original, mask);

    /* 예측용 latent grid 구성 */
    CELatentGrid z;
    z.width = CE_GRID_W;
    z.height = CE_GRID_H;
    z.current_step = 0;
    z.total_steps = 0;

    /* 마스크되지 않은 셀 → latent에 그대로 배치
     * 마스크된 셀 → noise로 초기화 (예측 대상) */
    for (uint32_t i = 0; i < (uint32_t)CE_GRID_N; i++) {
        if (i < original->num_cells) {
            if (mask[i]) {
                /* 마스크된 위치: noise로 시작 */
                uint64_t sub = seed ^ ((uint64_t)(i + 1) * 0x9E3779B97F4A7C15ULL);
                uint8_t buf[64];
                for (int k = 0; k < 8; k++) {
                    uint64_t r = splitmix(&sub);
                    for (int j = 0; j < 8; j++) buf[k*8+j] = (uint8_t)(r >> (j*8));
                }
                ce_init(&z.cells[i]);
                ce_feed(&z.cells[i], buf, 64);
            } else {
                /* 알려진 위치: 원본 셀 그대로 */
                z.cells[i] = original->cells[i];
            }
        } else {
            ce_init(&z.cells[i]);
        }
    }

    /* denoise loop: storage의 알려진 셀을 context로 사용하며 예측 */
    CEUnit prompt_cell;
    ce_init(&prompt_cell);
    /* 프롬프트 = 알려진 셀들의 합성 (첫 번째 비마스크 셀) */
    for (uint32_t i = 0; i < original->num_cells; i++) {
        if (!mask[i]) { prompt_cell = original->cells[i]; break; }
    }

    ce_denoise_loop(&z, &tmp_storage, &prompt_cell, 1, gen_cfg);

    /* 예측 결과 추출 */
    *predicted = *original;  /* 경계 인덱스 복사 */
    for (uint32_t i = 0; i < original->num_cells && i < (uint32_t)CE_GRID_N; i++) {
        if (mask[i]) {
            predicted->cells[i] = z.cells[i];  /* 예측된 셀 */
        } else {
            predicted->cells[i] = original->cells[i];  /* 원본 유지 */
        }
    }

    /* loss 계산: 마스크된 셀을 segment별로 누적
     *   structure = cells[0..structure_end)
     *   texture   = cells[structure_end..texture_end)  (edge + texture)
     *   color     = cells[texture_end..num_cells)      (color + event)
     * 각 segment 별 평균을 따로 반환해서 호출자가 가중치를 줄 수 있게 한다. */
    float sum_struct = 0, sum_tex = 0, sum_col = 0;
    int   n_struct   = 0, n_tex   = 0, n_col   = 0;
    for (uint32_t i = 0; i < original->num_cells; i++) {
        if (!mask[i]) continue;
        CELoss cell_loss;
        ce_compute_loss(&cell_loss, &predicted->cells[i], &original->cells[i]);
        if      (i < original->structure_end) { sum_struct += cell_loss.total_loss; n_struct++; }
        else if (i < original->texture_end  ) { sum_tex    += cell_loss.total_loss; n_tex++;    }
        else                                  { sum_col    += cell_loss.total_loss; n_col++;    }
    }

    ce_storage_free(&tmp_storage);

    SegLosses seg = {0};
    seg.structure = (n_struct > 0) ? (sum_struct / (float)n_struct) : 0.0f;
    seg.texture   = (n_tex    > 0) ? (sum_tex    / (float)n_tex   ) : 0.0f;
    seg.color     = (n_col    > 0) ? (sum_col    / (float)n_col   ) : 0.0f;

    int total_count = n_struct + n_tex + n_col;
    seg.total = (total_count > 0)
              ? ((sum_struct + sum_tex + sum_col) / (float)total_count)
              : 0.0f;

    if (out_seg) *out_seg = seg;
    return seg.total;
}

/* ══════════════════════════════════════════════════════
 *  단일 이미지 학습
 * ══════════════════════════════════════════════════════ */

void masked_train_image(MaskedTrainResult      *result,
                        CEStorage              *storage,
                        const uint8_t          *rgba,
                        int                     width,
                        int                     height,
                        uint32_t                canvas_id,
                        const MaskedTrainConfig *cfg) {
    if (!result || !storage || !rgba) return;
    memset(result, 0, sizeof(*result));

    MaskedTrainConfig c;
    if (cfg) c = *cfg;
    else     masked_train_config_default(&c);

    if (width != 256 || height != 256) return;

    int n = width * height;

    /* ── Step 1: 이미지 분해 ─────────────────────────── */
    SligImage *img = slig_image_alloc(width, height, 3);
    if (!img) return;
    for (int i = 0; i < n; i++) {
        img->data[i * 3 + 0] = (float)rgba[i * 4 + 0] / 255.0f;
        img->data[i * 3 + 1] = (float)rgba[i * 4 + 1] / 255.0f;
        img->data[i * 3 + 2] = (float)rgba[i * 4 + 2] / 255.0f;
    }

    SligDecomposed original;
    SligDecomposeConfig dcfg;
    slig_decompose_config_default(&dcfg);
    slig_decompose_v2(&original, img, &dcfg);
    slig_image_free(img);

    if (original.num_cells == 0) return;

    /* ── Step 2: 마스크 학습 루프 ────────────────────── */
    uint8_t mask[64];
    uint64_t rng = c.mask_seed;

    CEGenConfig gen_cfg = ce_gen_config_default();
    gen_cfg.total_steps = c.denoise_steps;

    /* If the caller set any segment weight, switch into weighted-loss
     * mode. Otherwise keep the legacy mean-of-all-masked behaviour so
     * existing callers see the same numbers. */
    int weighted = (c.loss_w_structure > 0.0f) ||
                   (c.loss_w_texture   > 0.0f) ||
                   (c.loss_w_color     > 0.0f);

    float best_loss = 1e9f;
    int patience_counter = 0;
    SligDecomposed best_predicted;
    memset(&best_predicted, 0, sizeof(best_predicted));
    SegLosses best_seg = {0};

    for (int epoch = 0; epoch < c.epochs; epoch++) {
        /* 마스크 생성 */
        build_mask(mask, original.num_cells, &original,
                   c.strategy, epoch, c.epochs, &rng);

        /* 마스크된 셀 수 확인 */
        int masked_count = 0;
        for (uint32_t i = 0; i < original.num_cells; i++)
            if (mask[i]) masked_count++;
        if (masked_count == 0) continue;  /* 가릴 게 없으면 스킵 */

        /* 예측 + loss 측정 (segment-aware) */
        SligDecomposed predicted;
        uint64_t epoch_seed = c.mask_seed ^ ((uint64_t)(epoch + 1) * 0xDEADBEEFULL);
        SegLosses seg;
        float mean_loss = predict_and_measure(&predicted, &original, mask,
                                              canvas_id, &gen_cfg, epoch_seed,
                                              &seg);

        /* Aggregate: weighted sum if any segment weight set, else mean. */
        float loss;
        if (weighted) {
            loss = c.loss_w_structure * seg.structure
                 + c.loss_w_texture   * seg.texture
                 + c.loss_w_color     * seg.color;
        } else {
            loss = mean_loss;
        }

        /* 콜백 */
        if (c.on_epoch)
            c.on_epoch(epoch, loss, c.cb_ctx);

        /* 최고 기록 갱신 */
        if (loss < best_loss) {
            best_loss = loss;
            best_predicted = predicted;
            best_seg = seg;
            patience_counter = 0;
        } else {
            patience_counter++;
        }

        /* loss 기반 파라미터 조정 */
        CELoss dummy_loss;
        dummy_loss.total_loss = loss;
        dummy_loss.channel_loss[0] = loss;
        dummy_loss.channel_loss[1] = loss;
        dummy_loss.channel_loss[2] = loss;
        dummy_loss.channel_loss[3] = loss;
        ce_update_params(&gen_cfg, &dummy_loss);

        /* 조기 종료 */
        if (best_loss <= c.target_loss) break;
        if (patience_counter >= (int)c.loss_patience) break;

        result->epochs_run = epoch + 1;
    }

    /* ── Step 3: 결과 기록 ───────────────────────────── */
    result->final_loss     = best_loss;
    result->converged      = (best_loss <= c.target_loss) ? 1 : 0;
    result->learned        = best_predicted;
    result->loss_structure = best_seg.structure;
    result->loss_texture   = best_seg.texture;
    result->loss_color     = best_seg.color;

    /* ── Step 4: 수렴된 셀을 CEStorage에 저장 ────────── */
    if (c.store_on_converge && best_predicted.num_cells > 0) {
        if (c.residual_book) {
            /* Pattern-bucketed: basis+residual stored raw as
             * CE_TYPE_SLIG, correction reduced to ce_residual_codebook
             * descriptors (CE_TYPE_RESIDUAL). */
            uint32_t residuals = 0;
            uint32_t total = store_cells_with_codebook(
                storage, canvas_id, &best_predicted,
                c.residual_book, c.residual_threshold, &residuals);
            result->cells_stored   = total;
            result->residual_cells = residuals;
        } else {
            /* Legacy: every cell stored raw with CE_TYPE_IMAGE. */
            store_cells_masked(storage, canvas_id, &best_predicted, NULL);
            result->cells_stored   = best_predicted.num_cells;
            result->residual_cells = 0;
        }
    }
}

/* ══════════════════════════════════════════════════════
 *  배치 학습
 * ══════════════════════════════════════════════════════ */

void masked_train_batch(BatchTrainResult        *result,
                        CEStorage               *storage,
                        const TrainImageEntry   *images,
                        int                      num_images,
                        const MaskedTrainConfig  *cfg) {
    if (!result || !storage || !images || num_images <= 0) return;
    memset(result, 0, sizeof(*result));
    result->total_images = num_images;
    result->best_loss = 1e9f;
    result->worst_loss = 0.0f;

    float loss_sum = 0.0f;

    for (int i = 0; i < num_images; i++) {
        const TrainImageEntry *e = &images[i];

        MaskedTrainResult img_result;
        masked_train_image(&img_result, storage,
                           e->rgba, e->width, e->height,
                           e->canvas_id, cfg);

        loss_sum += img_result.final_loss;
        if (img_result.final_loss < result->best_loss)
            result->best_loss = img_result.final_loss;
        if (img_result.final_loss > result->worst_loss)
            result->worst_loss = img_result.final_loss;
        if (img_result.converged)
            result->total_converged++;
    }

    result->avg_loss = loss_sum / (float)num_images;
}
