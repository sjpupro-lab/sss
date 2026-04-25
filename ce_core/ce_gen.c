#include "ce_gen.h"
#include <string.h>
#include <stdlib.h>

/* Build a small array of prompt CE cells from a UTF-8 prompt string.
 * Each whitespace-separated token (or 8-char window for the wordless case)
 * becomes one CEUnit. Caller must free the returned buffer. */
static CEUnit *prompt_to_cells(const char *prompt, int *out_count) {
    if (!prompt || !*prompt) { *out_count = 0; return NULL; }
    int len = (int)strlen(prompt);
    int cap = 16;
    CEUnit *cells = (CEUnit *)calloc(cap, sizeof(CEUnit));
    int cnt = 0;

    int i = 0;
    while (i < len) {
        while (i < len && (prompt[i] == ' ' || prompt[i] == '\t' || prompt[i] == '\n')) ++i;
        if (i >= len) break;
        int j = i;
        while (j < len && prompt[j] != ' ' && prompt[j] != '\t' && prompt[j] != '\n') ++j;
        if (cnt == cap) {
            cap *= 2;
            cells = (CEUnit *)realloc(cells, cap * sizeof(CEUnit));
        }
        ce_init(&cells[cnt]);
        ce_feed(&cells[cnt], (const uint8_t *)(prompt + i), (uint32_t)(j - i));
        ++cnt;
        i = j;
    }
    /* Always include a whole-prompt cell as a fallback context. */
    if (cnt == cap) {
        cap *= 2;
        cells = (CEUnit *)realloc(cells, cap * sizeof(CEUnit));
    }
    ce_init(&cells[cnt]);
    ce_feed(&cells[cnt], (const uint8_t *)prompt, (uint32_t)len);
    ++cnt;

    *out_count = cnt;
    return cells;
}

static void build_initial_latent(CELatentGrid *z,
                                 const CEStorage *storage,
                                 const CEUnit *prompt_cells, int prompt_count,
                                 uint64_t seed) {
    CEUnit noise; ce_noise_init(&noise, seed);

    /* Pick top-1 keyframe/delta from storage to seed initial latent. */
    CEUnit kf, dl;
    ce_init(&kf);
    ce_init(&dl);
    if (storage && storage->count > 0) {
        CESearchResult r;
        ce_search_topk(storage, &noise, 1, &r);
        kf = storage->entries[r.entry_idx].keyframe;
        dl = storage->entries[r.entry_idx].delta;
    }
    /* Mix the prompt cells into a single representative for init. */
    CEUnit prompt_avg; ce_init(&prompt_avg);
    if (prompt_count > 0 && prompt_cells) {
        prompt_avg = prompt_cells[prompt_count - 1]; /* whole-prompt cell */
    }
    float w[5] = { 0.30f, 0.25f, 0.20f, 0.25f, 0.0f };
    ce_latent_init(z, &noise, &kf, &dl,
                   (prompt_count > 0) ? &prompt_avg : NULL,
                   NULL, w);
}

void ce_generate_image(
    CEImage *output,
    const CEStorage *storage,
    const char *prompt,
    uint64_t seed,
    const CEGenConfig *config,
    const CEMemoLayer *memo,
    const CEHintLayer *hint,
    const CEAudioTrack *audio) {
    CEGenConfig cfg = config ? *config : ce_gen_config_default();

    int prompt_count = 0;
    CEUnit *prompt_cells = prompt_to_cells(prompt, &prompt_count);

    CELatentGrid z;
    build_initial_latent(&z, storage, prompt_cells, prompt_count, seed);

    if (memo) ce_memo_apply(&z, memo);
    if (hint) ce_hint_apply(&z, hint);

    ce_denoise_loop(&z, storage, prompt_cells, prompt_count, audio, &cfg);

    ce_decode_image(output, &z);

    free(prompt_cells);
}

void ce_generate_text(
    uint8_t *output, uint32_t *output_len,
    const CEStorage *storage,
    const char *prompt,
    uint64_t seed,
    const CEGenConfig *config) {
    CEGenConfig cfg = config ? *config : ce_gen_config_default();

    int prompt_count = 0;
    CEUnit *prompt_cells = prompt_to_cells(prompt, &prompt_count);

    CELatentGrid z;
    build_initial_latent(&z, storage, prompt_cells, prompt_count, seed);
    ce_denoise_loop(&z, storage, prompt_cells, prompt_count, NULL, &cfg);

    ce_decode_text(output, output_len, &z);

    free(prompt_cells);
}

void ce_generate_inpaint(
    CEImage *output,
    const CEImage *original,
    const CEInpaintMask *mask,
    const CEStorage *storage,
    const char *prompt,
    uint64_t seed,
    const CEGenConfig *config) {
    CEGenConfig cfg = config ? *config : ce_gen_config_default();

    int prompt_count = 0;
    CEUnit *prompt_cells = prompt_to_cells(prompt, &prompt_count);

    /* Reverse-encode the original image into a latent grid so anchor cells
     * carry the original signal. Each 8x8 block is fed back into a CE cell. */
    CELatentGrid z;
    z.width = CE_GRID_W; z.height = CE_GRID_H;
    z.current_step = 0; z.total_steps = 0;
    for (int cy = 0; cy < CE_GRID_H; ++cy) {
        for (int cx = 0; cx < CE_GRID_W; ++cx) {
            uint8_t buf[CE_BLOCK * CE_BLOCK * 4];
            int k = 0;
            for (int by = 0; by < CE_BLOCK; ++by) {
                for (int bx = 0; bx < CE_BLOCK; ++bx) {
                    int px = cx * CE_BLOCK + bx;
                    int py = cy * CE_BLOCK + by;
                    const CEPixel *p = &original->pixels[py * CE_IMAGE_W + px];
                    buf[k++] = p->r; buf[k++] = p->g; buf[k++] = p->b; buf[k++] = p->a;
                }
            }
            CEUnit *u = &z.cells[cy * CE_GRID_W + cx];
            ce_init(u);
            ce_feed(u, buf, sizeof(buf));
        }
    }
    /* Add seed-driven noise into masked cells only. */
    CEUnit noise; ce_noise_init(&noise, seed);
    for (int i = 0; i < CE_GRID_N; ++i) {
        if (mask->mask[i]) {
            CEUnit d, ds, out;
            ce_delta(&d, &z.cells[i], &noise);
            ce_delta_scale(&ds, &d, 0.5f);
            ce_apply(&out, &z.cells[i], &ds);
            z.cells[i] = out;
        }
    }
    ce_inpaint(&z, mask, storage, prompt_cells, prompt_count, &cfg);
    ce_decode_image(output, &z);

    free(prompt_cells);
}

void ce_generate_upscale(
    CEHiresGrid *output_hires,
    const CELatentGrid *input_lores,
    const CEStorage *storage,
    const CEGenConfig *config) {
    CEGenConfig cfg = config ? *config : ce_gen_config_default();
    ce_upscale(output_hires, input_lores, storage, &cfg);
}
