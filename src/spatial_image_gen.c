/* spatial_image_gen.c — SLIG v2.2 integrated image pipeline
 * (color × multi-scale pyramid).
 *
 * The legacy ai_generate_image (spatial_image.c) is kept as a refine-
 * driven feasibility prototype. This file adds the v2 path:
 *
 *   prompt
 *     ↓ layers_encode_clause + update_rgb_directional + apply_ema
 *   text grid (256²)
 *     ↓ ai_match_morphemes  (keyframe_id only — no cell copying)
 *     ↓ slig_grid_to_spectrum  →  slig_spectrum_weight
 *   render-weight rescaled by audio-track overlap
 *     ↓ render coarse / mid / fine for each Y / Cb / Cr
 *     ↓ upsample + blend per channel  (0.4 coarse + 0.3 mid + 0.3 fine)
 *     ↓ slig_ycbcr_planes_to_rgb
 *   PPM output
 *
 * ai_learn_image is the trivial PPM ingestion path: load the file
 * into a SpatialGrid via image_to_grid (which keeps RGB) and hand it
 * to ai_store_grid, which now splits RGB→YCbCr internally and
 * decomposes the image at three scales (32 / 128 / 256) for each
 * plane.
 */

#include "spatial_image.h"
#include "spatial_keyframe.h"
#include "spatial_match.h"
#include "spatial_layers.h"
#include "spatial_grid.h"
#include "slig_signal.h"
#include "slig_codebook.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ai_generate_animation overrides this so each frame's events evolve
 * cumulatively. Single-image generation leaves it at 0 (renderer
 * default = 15 ticks). Module-local to avoid plumbing the value
 * through five layers of wrapper functions. */
static uint8_t s_event_max_tick = 0;

uint32_t ai_learn_image(SpatialAI*  ai,
                        const char* image_path,
                        const char* label) {
    if (!ai || !image_path) return UINT32_MAX;
    SpatialGrid* g = image_to_grid(image_path);
    if (!g) return UINT32_MAX;
    uint32_t id = ai_store_grid(ai, g, label);
    grid_destroy(g);
    return id;
}

static uint8_t pos_to_render_mode(PartOfSpeech p) {
    switch (p) {
        case POS_NOUN: return SLIG_RENDER_GLOBAL;
        case POS_ADJ:  return SLIG_RENDER_LOCAL;
        case POS_VERB: return SLIG_RENDER_EVENT;
        default:       return 0xFF;
    }
}

static int write_ppm_rgb(const char* path, const uint8_t* rgb_interleaved) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P6\n%d %d\n255\n", SLIG_CANVAS_DIM, SLIG_CANVAS_DIM);
    size_t n = (size_t)SLIG_CANVAS_DIM * SLIG_CANVAS_DIM * 3;
    size_t w = fwrite(rgb_interleaved, 1, n, f);
    fclose(f);
    return (w == n) ? 1 : 0;
}

/* Render one (scale, channel) layer for all matches.
 * Builds a SligRenderItem array pointing at each match's cells in
 * ai->codebook.patterns[ai->keyframes[…].image_idx[scale][channel]],
 * optionally tagged with a per-match spatial mask, runs
 * slig_render_sequential into a 256² canvas (cells decomposed at
 * smaller dim concentrate signal in the top-left dim×dim region
 * because their u/v vectors are zero outside [0, dim)), and extracts
 * the top-left dim×dim into out_panel. */
static void render_layer(uint8_t                *out_panel,    /* dim×dim uint8 */
                         int                     dim,
                         const SpatialAI        *ai,
                         const MorphemeMatch    *matches, uint32_t num_matches,
                         const uint8_t * const  *match_masks,  /* may be NULL */
                         int                     scale_level,
                         int                     channel) {
    SligRenderItem items[SLIG_MAX_CELLS];
    uint32_t nitems = 0;
    for (uint32_t i = 0; i < num_matches; i++) {
        uint8_t mode = pos_to_render_mode(matches[i].morpheme.pos);
        if (mode == 0xFF) continue;
        if (matches[i].render_weight == 0) continue;
        uint32_t kf_id = matches[i].keyframe_id;
        if (kf_id >= ai->kf_count) continue;
        const Keyframe *kf = &ai->keyframes[kf_id];
        if (!kf->has_image) continue;
        /* Resolve the codebook index → pattern. NULL ⇒ idx is past
         * pattern_count (corrupt or pre-codebook engine). */
        const SligCellSet *src = slig_codebook_get(
            &ai->codebook, kf->image_idx[scale_level][channel]);
        if (!src || src->num_cells == 0) continue;

        items[nitems].cells          = src->cells;
        items[nitems].cell_count     = src->num_cells;
        items[nitems].render_mode    = mode;
        items[nitems].render_order   = matches[i].render_order;
        items[nitems].render_weight  = matches[i].render_weight;
        items[nitems].grid_x         = 128;
        items[nitems].grid_y         = 128;
        items[nitems].mask           = match_masks ? match_masks[i] : NULL;
        items[nitems].event_max_tick = s_event_max_tick;
        nitems++;
    }

    if (nitems == 0) {
        memset(out_panel, 0, (size_t)dim * dim);
        return;
    }
    /* Render to int32 canvas, then min-max normalise ONLY over the
     * top-left dim×dim region. The rest of the canvas is zero (the
     * cells were decomposed at `dim` so their u/v vectors are zero
     * outside [0, dim)) — including those zeros in normalisation
     * would compress the active region's dynamic range. */
    SligCanvas canvas;
    slig_render_sequential_canvas(&canvas, items, nitems, NULL);
    slig_canvas_to_u8_sub(out_panel, dim, &canvas, 0, 0);
}

/* Build one full-resolution channel panel by rendering coarse / mid /
 * fine, upsampling the smaller two, and weighted-summing all three.
 * Weights bias toward the coarse layer for stability. */
static void render_channel_pyramid(uint8_t                *panel_256,
                                   const SpatialAI        *ai,
                                   const MorphemeMatch    *matches, uint32_t num_matches,
                                   const uint8_t * const  *match_masks,
                                   int                     channel) {
    int dim_c = slig_scale_dim(SLIG_LEVEL_COARSE);
    int dim_m = slig_scale_dim(SLIG_LEVEL_MID);
    int dim_f = slig_scale_dim(SLIG_LEVEL_FINE);

    uint8_t *coarse = (uint8_t*)malloc((size_t)dim_c * dim_c);
    uint8_t *mid    = (uint8_t*)malloc((size_t)dim_m * dim_m);
    uint8_t *fine   = (uint8_t*)malloc((size_t)dim_f * dim_f);
    uint8_t *coarse_up = (uint8_t*)malloc((size_t)dim_f * dim_f);
    uint8_t *mid_up    = (uint8_t*)malloc((size_t)dim_f * dim_f);
    if (!coarse || !mid || !fine || !coarse_up || !mid_up) {
        free(coarse); free(mid); free(fine); free(coarse_up); free(mid_up);
        memset(panel_256, 0, (size_t)dim_f * dim_f);
        return;
    }

    render_layer(coarse, dim_c, ai, matches, num_matches, match_masks, SLIG_LEVEL_COARSE, channel);
    render_layer(mid,    dim_m, ai, matches, num_matches, match_masks, SLIG_LEVEL_MID,    channel);
    render_layer(fine,   dim_f, ai, matches, num_matches, match_masks, SLIG_LEVEL_FINE,   channel);

    /* Residual chain decode (mirrors ai_store_grid encode):
     *   pred128 = clamp(upsample(coarse) + (mid − 128), 0, 255)
     *   final   = clamp(upsample(pred128) + (fine − 128), 0, 255)
     * mid and fine were decomposed from residuals shifted to uint8
     * with offset 128, so subtract 128 to recover signed deltas. */
    int n_m = dim_m * dim_m;
    uint8_t *up_128_buf = mid_up;          /* reuse */
    slig_image_upsample(up_128_buf, dim_m, coarse, dim_c);
    uint8_t *pred_128 = (uint8_t*)malloc((size_t)n_m);
    if (!pred_128) {
        memset(panel_256, 0, (size_t)dim_f * dim_f);
        free(coarse); free(mid); free(fine); free(coarse_up); free(mid_up);
        return;
    }
    for (int i = 0; i < n_m; i++) {
        int v = (int)up_128_buf[i] + (int)mid[i] - 128;
        if (v < 0) v = 0; else if (v > 255) v = 255;
        pred_128[i] = (uint8_t)v;
    }
    slig_image_upsample(coarse_up, dim_f, pred_128, dim_m);
    int n_f = dim_f * dim_f;
    for (int i = 0; i < n_f; i++) {
        int v = (int)coarse_up[i] + (int)fine[i] - 128;
        if (v < 0) v = 0; else if (v > 255) v = 255;
        panel_256[i] = (uint8_t)v;
    }

    free(pred_128);
    free(coarse); free(mid); free(fine); free(coarse_up); free(mid_up);
}

/* Blend chroma toward neutral gray (128) when the rendered chroma
 * panel is essentially noise — the YCbCr → RGB mapping amplifies
 * any chroma deviation, so unbiased noise produces wildly saturated
 * output. */
static void chroma_neutral_bias(uint8_t *panel, uint8_t bias) {
    int n = SLIG_CANVAS_DIM * SLIG_CANVAS_DIM;
    uint16_t inv = 255 - bias;
    for (int i = 0; i < n; i++) {
        panel[i] = (uint8_t)(((uint16_t)panel[i] * inv + 128 * bias) / 255);
    }
}

/* Push a single uint8 panel away from the 128 baseline by `scale`
 * (Q8 inner loop). scale=1 is a no-op; scale=2 doubles the panel's
 * deviation from achromatic. Used by the guided variant of
 * ai_generate_image_v2 to amplify both luminance contrast and
 * chroma saturation in one pass. */
static void apply_guidance(uint8_t *panel, int n, float scale) {
    if (scale == 1.0f) return;
    int s_q8 = (int)(scale * 256.0f);
    if (s_q8 < 0) s_q8 = 0;
    for (int i = 0; i < n; i++) {
        int diff = (int)panel[i] - 128;
        int v    = 128 + (s_q8 * diff) / 256;
        if (v < 0)   v = 0;
        if (v > 255) v = 255;
        panel[i] = (uint8_t)v;
    }
}

int ai_generate_image_v2(SpatialAI*  ai,
                         const char* prompt_text,
                         const char* out_path) {
    return ai_generate_image_v2_guided(ai, prompt_text, out_path, 1.0f);
}

int ai_generate_image_v2_guided(SpatialAI*  ai,
                                const char* prompt_text,
                                const char* out_path,
                                float       guidance_scale) {
    if (!ai || !prompt_text || !out_path) return 0;

    SpatialGrid* grid = grid_create();
    if (!grid) return 0;
    layers_encode_clause(prompt_text, NULL, grid);
    update_rgb_directional(grid);
    apply_ema_to_grid(ai, grid);

    MorphemeMatch matches[SLIG_MAX_CELLS];
    uint32_t num_matches = ai_match_morphemes(ai, prompt_text,
                                              matches, SLIG_MAX_CELLS);
    if (num_matches == 0) { grid_destroy(grid); return 0; }

    uint32_t spectrum[256];
    slig_grid_to_spectrum(spectrum, (const uint16_t (*)[256])grid->A);

    /* Spectrum-weight each match using its FINE-Y cells as the
     * shape-bearing reference — same scalar applies to every layer/
     * channel since they all belong to the same morpheme. */
    for (uint32_t i = 0; i < num_matches; i++) {
        uint32_t kf_id = matches[i].keyframe_id;
        if (kf_id >= ai->kf_count) continue;
        const SligCellSet *yfine = slig_codebook_get(
            &ai->codebook,
            ai->keyframes[kf_id].image_idx[SLIG_LEVEL_FINE][SLIG_CH_Y]);
        if (!yfine || yfine->num_cells == 0) continue;
        uint8_t cell_w[SLIG_MAX_CELLS];
        slig_spectrum_weight(cell_w, (int)yfine->num_cells,
                             yfine->cells, spectrum);
        uint8_t max_w = 0;
        for (uint32_t c = 0; c < yfine->num_cells; c++) {
            if (cell_w[c] > max_w) max_w = cell_w[c];
        }
        if (max_w < 32) max_w = 32;
        uint32_t blended = (uint32_t)matches[i].render_weight * max_w / 255;
        matches[i].render_weight = (uint8_t)blended;
    }

    int panel_bytes = SLIG_CANVAS_DIM * SLIG_CANVAS_DIM;
    uint8_t *y_panel  = (uint8_t*)malloc(panel_bytes);
    uint8_t *cb_panel = (uint8_t*)malloc(panel_bytes);
    uint8_t *cr_panel = (uint8_t*)malloc(panel_bytes);
    uint8_t *rgb      = (uint8_t*)malloc(panel_bytes * 3);
    if (!y_panel || !cb_panel || !cr_panel || !rgb) {
        free(y_panel); free(cb_panel); free(cr_panel); free(rgb);
        grid_destroy(grid);
        return 0;
    }

    /* Direction-keyword routing: scan each match's predecessor token
     * for layout cues ("왼쪽/오른쪽/위/아래") and tag the noun match
     * with a half-plane mask. Allocated lazily so prompts without
     * direction words pay nothing.
     *
     * The lookup is by-substring on the prior morpheme — so "왼편",
     * "왼쪽", "좌측" all resolve to mask_left. The morpheme analyser
     * splits these out as POS_UNKNOWN since they aren't in the noun
     * dictionary; that's fine — we only need their string. */
    uint8_t *m_left = NULL, *m_right = NULL, *m_top = NULL, *m_bottom = NULL;
    const uint8_t *match_masks[SLIG_MAX_CELLS] = {0};
    for (uint32_t i = 1; i < num_matches; i++) {
        const char *prev = matches[i-1].morpheme.token;
        if (!prev || !*prev) continue;
        const uint8_t *m = NULL;
        if (strstr(prev, "왼") || strstr(prev, "좌")) {
            if (!m_left) {
                m_left = (uint8_t*)malloc(panel_bytes);
                if (m_left) slig_make_mask_left(m_left, 128, 32);
            }
            m = m_left;
        } else if (strstr(prev, "오른") || strstr(prev, "우측")) {
            if (!m_right) {
                m_right = (uint8_t*)malloc(panel_bytes);
                if (m_right) slig_make_mask_right(m_right, 128, 32);
            }
            m = m_right;
        } else if (strcmp(prev, "위") == 0 || strcmp(prev, "위에") == 0 ||
                   strstr(prev, "상단") || strstr(prev, "위쪽")) {
            if (!m_top) {
                m_top = (uint8_t*)malloc(panel_bytes);
                if (m_top) slig_make_mask_top(m_top, 128, 32);
            }
            m = m_top;
        } else if (strcmp(prev, "아래") == 0 || strcmp(prev, "아래에") == 0 ||
                   strstr(prev, "하단") || strstr(prev, "아래쪽")) {
            if (!m_bottom) {
                m_bottom = (uint8_t*)malloc(panel_bytes);
                if (m_bottom) slig_make_mask_bottom(m_bottom, 128, 32);
            }
            m = m_bottom;
        }
        match_masks[i] = m;
    }

    render_channel_pyramid(y_panel,  ai, matches, num_matches, match_masks, SLIG_CH_Y);
    render_channel_pyramid(cb_panel, ai, matches, num_matches, match_masks, SLIG_CH_CB);
    render_channel_pyramid(cr_panel, ai, matches, num_matches, match_masks, SLIG_CH_CR);

    chroma_neutral_bias(cb_panel, 192);
    chroma_neutral_bias(cr_panel, 192);

    /* CFG: amplify deviation from the achromatic baseline (128) on
     * all three planes after pyramid blend + chroma bias. Applies to
     * luma (intensifies contrast) and chroma (intensifies saturation)
     * symmetrically — driven by guidance_scale. */
    apply_guidance(y_panel,  panel_bytes, guidance_scale);
    apply_guidance(cb_panel, panel_bytes, guidance_scale);
    apply_guidance(cr_panel, panel_bytes, guidance_scale);

    slig_ycbcr_planes_to_rgb(rgb, y_panel, cb_panel, cr_panel,
                             SLIG_CANVAS_DIM, SLIG_CANVAS_DIM);
    int ok = write_ppm_rgb(out_path, rgb);

    free(y_panel); free(cb_panel); free(cr_panel); free(rgb);
    free(m_left); free(m_right); free(m_top); free(m_bottom);
    grid_destroy(grid);
    return ok;
}

uint32_t ai_generate_animation(SpatialAI*  ai,
                               const char* prompt_text,
                               const char* output_dir,
                               uint32_t    num_frames) {
    if (!ai || !prompt_text || !output_dir || num_frames == 0) return 0;

    /* mkdir is best-effort: if the dir already exists or can't be
     * created, fopen on the per-frame path will surface the real error. */
    mkdir(output_dir, 0755);

    uint32_t written = 0;
    uint8_t prev_tick = s_event_max_tick;

    for (uint32_t frame = 0; frame < num_frames; frame++) {
        char frame_path[512];
        int n = snprintf(frame_path, sizeof frame_path,
                         "%s/frame_%03u.ppm", output_dir, frame);
        if (n < 0 || (size_t)n >= sizeof frame_path) break;

        /* Frame index → cumulative tick bound. Clamp into uint8_t,
         * since slig_apply_at_tick takes uint8_t ticks. Past 255 we
         * stop advancing (further frames look the same as frame 255). */
        uint8_t tick = (frame > 255) ? 255 : (uint8_t)frame;
        s_event_max_tick = tick;

        if (ai_generate_image_v2_guided(ai, prompt_text, frame_path, 1.0f)) {
            written++;
        } else {
            break;
        }
    }

    s_event_max_tick = prev_tick;
    return written;
}
