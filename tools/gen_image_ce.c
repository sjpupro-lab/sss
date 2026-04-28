/* gen_image_ce.c — E2E image generation from a CEStorage (.ces) file.
 *
 * Pipeline:
 *   1. ce_storage_load(.ces)
 *   2. morpheme_tokenize_clause(prompt) -> per-morpheme CEUnits via ce_feed
 *   3. For each morpheme: ce_search_by_type(CE_TYPE_TEXT, ...) -> winning
 *      canvas_id (weighted vote across morphemes by 1/(1+distance)).
 *   4. ce_generate_image_canvas_routed(winner_cid, ...) — denoise + decode
 *      + 16x16 atomic wave-refine on entries with the matching canvas_id.
 *      Falls back to ce_generate_image_typed(CE_TYPE_IMAGE) when no TEXT
 *      bridges exist or no morphemes survive tokenisation.
 *   5. write 256x256 PPM P6 (RGB only).
 *
 * Usage:
 *   gen_image_ce <model.ces> <prompt> <out.ppm> [seed] [steps] [wave_iters]
 *
 * Defaults: seed=0, steps=8 (fast preset; pass >= 50 for HQ), wave_iters=200.
 */
#include "spatial_morpheme.h"

#include "ce_storage.h"
#include "ce_storage_io.h"
#include "ce_search.h"
#include "ce_gen.h"
#include "ce_decode.h"
#include "ce_denoise.h"
#include "ce_type.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_MORPHS 256
#define MAX_VOTES  16

static int write_ppm(const char *path, const CEImage *img) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    for (int i = 0; i < img->width * img->height; ++i) {
        unsigned char rgb[3] = {
            img->pixels[i].r, img->pixels[i].g, img->pixels[i].b
        };
        if (fwrite(rgb, 1, 3, f) != 3) { fclose(f); return 0; }
    }
    fclose(f);
    return 1;
}

/* Vote canvas_id by per-morpheme TEXT search. Each morpheme contributes
 * weight 1/(1+distance) to the canvas_id of its nearest TEXT entry.
 * Returns winning canvas_id and sets *found = 1 on success. */
static uint32_t vote_canvas_id(const CEStorage *storage,
                               const Morpheme *morphs, uint32_t n,
                               int *found) {
    *found = 0;
    if (!storage || storage->count == 0 || n == 0) return 0;

    struct { uint32_t cid; double weight; } votes[MAX_VOTES];
    int vote_count = 0;

    for (uint32_t i = 0; i < n; ++i) {
        const char *tok = morphs[i].token;
        size_t len = strlen(tok);
        if (len == 0) continue;

        CEUnit q;
        ce_init(&q);
        ce_feed(&q, (const uint8_t *)tok, (uint32_t)len);

        CESearchResult res;
        int got = ce_search_by_type(storage, &q, CE_TYPE_TEXT, 1, &res);
        if (got <= 0) continue;

        uint32_t cid = storage->entries[res.entry_idx].canvas_id;
        double w = 1.0 / (1.0 + (double)res.distance);

        int slot = -1;
        for (int v = 0; v < vote_count; ++v) {
            if (votes[v].cid == cid) { slot = v; break; }
        }
        if (slot < 0 && vote_count < MAX_VOTES) {
            slot = vote_count++;
            votes[slot].cid = cid;
            votes[slot].weight = 0.0;
        }
        if (slot >= 0) votes[slot].weight += w;
    }

    if (vote_count == 0) return 0;

    int best = 0;
    for (int v = 1; v < vote_count; ++v) {
        if (votes[v].weight > votes[best].weight) best = v;
    }
    *found = 1;
    return votes[best].cid;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <model.ces> <prompt> <out.ppm> [seed] [steps] [wave_iters]\n",
                argv[0]);
        return 2;
    }
    const char *ces_path  = argv[1];
    const char *prompt    = argv[2];
    const char *out_path  = argv[3];
    uint64_t seed         = (argc > 4) ? (uint64_t)strtoull(argv[4], NULL, 0) : 0u;
    int steps             = (argc > 5) ? atoi(argv[5]) : 8;
    uint32_t wave_iters   = (argc > 6) ? (uint32_t)strtoul(argv[6], NULL, 0) : 200u;

    CEStorage S;
    if (!ce_storage_load(&S, ces_path)) {
        fprintf(stderr, "[gen_image_ce] load %s failed\n", ces_path);
        return 1;
    }
    fprintf(stderr, "[gen_image_ce] loaded %s (%u entries)\n", ces_path, S.count);

    morpheme_init();

    CEGenConfig cfg = (steps >= 50) ? ce_gen_config_hq() : ce_gen_config_default();
    if (steps > 0) cfg.total_steps = steps;

    CEImage *img = (CEImage *)calloc(1, sizeof(CEImage));
    if (!img) { ce_storage_free(&S); return 1; }

    fprintf(stderr,
            "[gen_image_ce] prompt=\"%s\" seed=0x%llx steps=%d wave_iters=%u\n",
            prompt, (unsigned long long)seed, cfg.total_steps, wave_iters);

    /* Tokenise the prompt into morphemes and vote a canvas_id. */
    Morpheme morphs[MAX_MORPHS];
    uint32_t n_morph = morpheme_tokenize_clause(prompt, morphs, MAX_MORPHS);
    fprintf(stderr, "[gen_image_ce] morphemes=%u  ", n_morph);
    for (uint32_t i = 0; i < n_morph; ++i) {
        fprintf(stderr, "%s/%s ", morphs[i].token, pos_name(morphs[i].pos));
    }
    fputc('\n', stderr);

    int routed = 0;
    uint32_t cid = vote_canvas_id(&S, morphs, n_morph, &routed);

    if (routed) {
        fprintf(stderr, "[gen_image_ce] routed canvas_id=0x%08x\n", cid);
        ce_generate_image_canvas_routed(img, &S, cid,
                                        prompt, seed, &cfg, wave_iters);
    } else {
        fprintf(stderr,
                "[gen_image_ce] no TEXT bridge match -> typed IMAGE fallback\n");
        ce_generate_image_typed(img, &S, CE_TYPE_IMAGE,
                                prompt, seed, &cfg, wave_iters);
    }

    /* Summary stats so an automated pipeline test can verify the
     * generated image without parsing the PPM. */
    double sr = 0, sg = 0, sb = 0;
    int npx = img->width * img->height;
    for (int i = 0; i < npx; ++i) {
        sr += img->pixels[i].r;
        sg += img->pixels[i].g;
        sb += img->pixels[i].b;
    }
    fprintf(stderr,
            "[gen_image_ce] mean RGB = (%.1f, %.1f, %.1f)\n",
            sr / npx, sg / npx, sb / npx);

    if (!write_ppm(out_path, img)) {
        fprintf(stderr, "[gen_image_ce] write %s failed\n", out_path);
        free(img);
        ce_storage_free(&S);
        return 1;
    }
    fprintf(stderr, "[gen_image_ce] wrote %s\n", out_path);

    free(img);
    ce_storage_free(&S);
    return 0;
}
