#include "ce_denoise.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Audio energy lookup is provided by ce_extend.c when available. We keep
 * a weak reference here so denoise can stay in Phase 3 without dragging
 * in Phase 4 headers. */
struct CEAudioTrack;
extern float ce_audio_energy_at(const struct CEAudioTrack *track,
                                int step, int total_steps);

CEGenConfig ce_gen_config_default(void) {
    CEGenConfig c;
    c.total_steps = 8;
    c.cfg_scale = 1.0f;
    c.conv_radius = 1;
    c.attn_topk = 4;
    c.residual_weight = 0.7f;
    c.skip_weight = 0.5f;
    c.sampler_mode = CE_SAMPLE_CONFIDENCE;
    return c;
}

void ce_denoise_loop(CELatentGrid *z,
                     const CEStorage *storage,
                     const CEUnit *prompt_cells, int prompt_count,
                     const struct CEAudioTrack *audio,
                     const CEGenConfig *config) {
    if (!config || config->total_steps <= 0) return;

    int total = config->total_steps;
    z->total_steps = total;

    /* Precompute retrieved pairs from a query that summarises the whole
     * grid. Use cell 0 (top-left) as the query — keeps things deterministic;
     * Phase 4 sampler can replace this with per-cell selection. */
    CESearchResult res[8];
    int k = 8;
    int found = ce_search_topk(storage, &z->cells[0], k, res);
    CERetrievedPair pairs[8];
    if (found > 0) ce_select_kd_pair(storage, res, found, pairs);

    static CEUnit skip_buf[CE_GRID_N];
    static CEAttentionLink links[16];
    int topk = config->attn_topk;
    if (topk > 16) topk = 16;

    for (int t = 0; t < total; ++t) {
        z->current_step = t;

        /* Snapshot for residual gating. */
        static CEUnit before[CE_GRID_N];
        memcpy(before, z->cells, sizeof(before));

        /* 1. Down */
        ce_down(z, pairs, found, skip_buf);
        /* 2. Conv */
        ce_conv(z, NULL, config->conv_radius);
        /* 3. Self-attention */
        if (topk > 0) ce_self_attention(z, topk, links);
        /* 4. Cross-attention */
        ce_cross_attention(z, prompt_cells, prompt_count, config->cfg_scale);

        /* 5. Residual */
        for (int i = 0; i < CE_GRID_N; ++i) {
            CEUnit r;
            ce_residual(&r, &before[i], &z->cells[i], config->residual_weight);
            z->cells[i] = r;
        }
        /* 6. Skip connect */
        ce_skip_connect(z, skip_buf, config->skip_weight);
        /* 7. Up */
        ce_up(z, skip_buf, pairs, found);

        /* 8. Audio sync — modulate global magnitude by audio energy.
         *    Without an audio track the function returns 1.0 (no-op). */
        float e = ce_audio_energy_at(audio, t, total);
        if (e != 1.0f) {
            for (int i = 0; i < CE_GRID_N; ++i) {
                CEUnit zero; ce_init(&zero);
                CEUnit d, ds, out;
                ce_delta(&d, &zero, &z->cells[i]);
                ce_delta_scale(&ds, &d, e);
                ce_apply(&out, &zero, &ds);
                z->cells[i] = out;
            }
        }
    }
    z->current_step = total;
}

void ce_compute_loss(CELoss *loss,
                     const CEUnit *predicted, const CEUnit *target) {
    /* Per-channel ce_distance projected onto the channel's bytes (8 bytes).
     * total_loss is normalised SAD across the full 64 bytes. */
    const uint8_t *pp = ce_cbytes(predicted);
    const uint8_t *tp = ce_cbytes(target);
    int channel_bytes_offset[4][2];
    /* inc.{R,G,B,A} occupy bytes 0..7, 8..15, 16..23, 24..31. */
    /* dec.{R,G,B,A} occupy bytes 32..39, ..., 56..63. */
    for (int c = 0; c < 4; ++c) {
        channel_bytes_offset[c][0] = c * 8;       /* inc half */
        channel_bytes_offset[c][1] = 32 + c * 8;  /* dec half */
    }
    uint32_t total = 0;
    for (int c = 0; c < 4; ++c) {
        uint32_t s = 0;
        for (int half = 0; half < 2; ++half) {
            int off = channel_bytes_offset[c][half];
            for (int j = 0; j < 8; ++j) {
                int d = (int)pp[off + j] - (int)tp[off + j];
                s += (d < 0) ? (uint32_t)(-d) : (uint32_t)d;
            }
        }
        loss->channel_loss[c] = (float)s / 16.0f; /* avg per byte */
        total += s;
    }
    loss->total_loss = (float)total / 64.0f;
}

void ce_update_params(CEGenConfig *config, const CELoss *loss) {
    /* Simple gradient-free update:
     *   - high loss -> more steps, more attention, smaller residual carry
     *   - low loss  -> fewer steps to save compute */
    if (loss->total_loss > 96.0f) {
        if (config->total_steps < 64) config->total_steps += 1;
        config->cfg_scale *= 1.05f;
        if (config->cfg_scale > 4.0f) config->cfg_scale = 4.0f;
    } else if (loss->total_loss < 16.0f) {
        if (config->total_steps > 4) config->total_steps -= 1;
    }
    /* keep within sensible bounds */
    if (config->residual_weight < 0.0f) config->residual_weight = 0.0f;
    if (config->residual_weight > 1.0f) config->residual_weight = 1.0f;
    if (config->skip_weight < 0.0f) config->skip_weight = 0.0f;
    if (config->skip_weight > 1.0f) config->skip_weight = 1.0f;
}
