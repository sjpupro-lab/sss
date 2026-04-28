#include "ce_gen.h"
#include "ce_image_wave_refine.h"
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

/* Build a single CEImageLink64 whose 256x256 target is the 8x8-tiled decode
 * of one CEStorage entry's keyframe + delta (so the whole image plane is
 * reconstructed from the retrieved entry, not from a single block). The
 * 64x64 semantic plane is left zeroed — it is reserved for future
 * morpheme/canvas embedding cross-talk and not exercised by the wave loop
 * itself, which only consumes `target` and `score`. */
static void link_from_entry(CEImageLink64 *link,
                            const CEStorageEntry *e,
                            float score) {
    if (!link) return;
    link->score = score;
    memset(link->semantic, 0, sizeof(link->semantic));

    /* Apply the entry's delta on top of its keyframe before decoding so the
     * stored signal is round-trippable. */
    CEUnit recon;
    ce_apply(&recon, &e->keyframe, &e->delta);

    uint8_t block[CE_BLOCK * CE_BLOCK * 4];
    ce_decode_image_block(block, &recon);

    /* Tile the single 8x8 reconstruction across the 256x256 target plane.
     * The wave-refine loop blends the top-k targets into the canvas, so
     * tiling yields a uniform color contribution from this entry. */
    for (int y = 0; y < CE_WAVE_IMG_H; ++y) {
        for (int x = 0; x < CE_WAVE_IMG_W; ++x) {
            int bx = x % CE_BLOCK;
            int by = y % CE_BLOCK;
            const uint8_t *p = block + (size_t)(by * CE_BLOCK + bx) * 4u;
            CEPixelF *d = &link->target[(size_t)y * CE_WAVE_IMG_W + (size_t)x];
            d->r = (float)p[0];
            d->g = (float)p[1];
            d->b = (float)p[2];
            d->a = (float)p[3];
        }
    }
}

/* Convert CEImage (uint8 RGBA) <-> CEImageCanvas (float RGBA). */
static void image_to_canvas(CEImageCanvas *out, const CEImage *img) {
    for (int i = 0; i < CE_WAVE_IMG_W * CE_WAVE_IMG_H; ++i) {
        out->px[i].r = (float)img->pixels[i].r;
        out->px[i].g = (float)img->pixels[i].g;
        out->px[i].b = (float)img->pixels[i].b;
        out->px[i].a = (float)img->pixels[i].a;
    }
}

static uint8_t f_to_u8(float v) {
    if (v < 0.0f)   v = 0.0f;
    if (v > 255.0f) v = 255.0f;
    return (uint8_t)(v + 0.5f);
}

static void canvas_to_image(CEImage *out, const CEImageCanvas *c) {
    for (int i = 0; i < CE_WAVE_IMG_W * CE_WAVE_IMG_H; ++i) {
        out->pixels[i].r = f_to_u8(c->px[i].r);
        out->pixels[i].g = f_to_u8(c->px[i].g);
        out->pixels[i].b = f_to_u8(c->px[i].b);
        out->pixels[i].a = f_to_u8(c->px[i].a);
    }
    out->width = CE_IMAGE_W;
    out->height = CE_IMAGE_H;
}

void ce_generate_image_typed(
    CEImage *output,
    const CEStorage *storage,
    CEType type,
    const char *prompt,
    uint64_t seed,
    const CEGenConfig *config,
    uint32_t wave_refine_iters) {

    CEGenConfig cfg = config ? *config : ce_gen_config_hq();

    int prompt_count = 0;
    CEUnit *prompt_cells = prompt_to_cells(prompt, &prompt_count);

    /* Initial latent — same construction as ce_generate_image, but seeded
     * by a typed top-1 search so the anchor cell is from the requested
     * modality. */
    CEUnit noise; ce_noise_init(&noise, seed);
    CEUnit kf, dl;
    ce_init(&kf);
    ce_init(&dl);
    if (storage && storage->count > 0) {
        CESearchResult r;
        int n = ce_search_by_type(storage, &noise, type, 1, &r);
        if (n > 0) {
            kf = storage->entries[r.entry_idx].keyframe;
            dl = storage->entries[r.entry_idx].delta;
        }
    }
    CEUnit prompt_avg; ce_init(&prompt_avg);
    if (prompt_count > 0 && prompt_cells) {
        prompt_avg = prompt_cells[prompt_count - 1];
    }
    float w[5] = { 0.30f, 0.25f, 0.20f, 0.25f, 0.0f };
    CELatentGrid z;
    ce_latent_init(&z, &noise, &kf, &dl,
                   (prompt_count > 0) ? &prompt_avg : NULL,
                   NULL, w);

    ce_denoise_loop(&z, storage, prompt_cells, prompt_count, NULL, &cfg);
    ce_decode_image(output, &z);

    /* Wave-refine pass: pull the top-CE_WAVE_TOPK typed entries (using the
     * whole-prompt cell when available, else the noise seed as the query)
     * and blend them into the decoded canvas. */
    if (wave_refine_iters > 0 && storage && storage->count > 0) {
        const CEUnit *query = (prompt_count > 0) ? &prompt_avg : &noise;
        CESearchResult sres[CE_WAVE_TOPK];
        int found = ce_search_by_type(storage, query, type, CE_WAVE_TOPK, sres);
        if (found > 0) {
            CEImageLink64 *links = (CEImageLink64 *)calloc((size_t)found, sizeof(CEImageLink64));
            CEImageLink64 *top3[CE_WAVE_TOPK] = { NULL, NULL, NULL };
            for (int i = 0; i < found; ++i) {
                /* Convert ascending distance into a positive score (closer
                 * entries get larger weight). +1 keeps zero-distance hits
                 * from collapsing the score to zero. */
                float score = 1.0f / (1.0f + (float)sres[i].distance);
                link_from_entry(&links[i], &storage->entries[sres[i].entry_idx], score);
                top3[i] = &links[i];
            }
            CEImageCanvas canvas;
            image_to_canvas(&canvas, output);
            ce_image_wave_refine(&canvas, top3, wave_refine_iters);
            canvas_to_image(output, &canvas);
            free(links);
        }
    }

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
