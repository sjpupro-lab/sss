/* test_gen_routed.c — E2E morpheme-routed image generation.
 *
 * Builds a tiny in-memory dataset with two rows:
 *   - "red apple"  + a solid red 64x64 RGBA
 *   - "blue sky"   + a solid blue 64x64 RGBA
 *
 * Ingest path (matches tools/train_demo.c):
 *   - ce_storage_ingest_rgba_16 for the image (CE_TYPE_IMAGE).
 *   - per-morpheme ce_feed -> ce_storage_add_typed CE_TYPE_TEXT,
 *     sharing canvas_id with the image entries.
 *
 * Generation:
 *   - tokenise prompt via morpheme_tokenize_clause.
 *   - per-morpheme TEXT search -> winning canvas_id.
 *   - ce_generate_image_canvas_routed.
 *
 * Asserts:
 *   - prompt "red apple"  routes to the red row's canvas_id and
 *     mean R > mean B in the generated image.
 *   - prompt "blue sky"   routes to the blue row's canvas_id and
 *     mean B > mean R.
 *   - Determinism: two runs with same seed produce identical pixels.
 */
#include "spatial_morpheme.h"

#include "ce_core.h"
#include "ce_storage.h"
#include "ce_search.h"
#include "ce_decode.h"
#include "ce_denoise.h"
#include "ce_gen.h"
#include "ce_type.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while (0)

#define MAX_MORPHS 256
#define MAX_VOTES 16

static uint8_t *make_solid(int dim, uint8_t r, uint8_t g, uint8_t b) {
    uint8_t *p = (uint8_t *)malloc((size_t)dim * dim * 4u);
    for (int i = 0; i < dim * dim; ++i) {
        p[i * 4 + 0] = r;
        p[i * 4 + 1] = g;
        p[i * 4 + 2] = b;
        p[i * 4 + 3] = 255;
    }
    return p;
}

static uint32_t ingest_clause(CEStorage *s, uint32_t cid, const char *clause) {
    Morpheme morphs[MAX_MORPHS];
    uint32_t n = morpheme_tokenize_clause(clause, morphs, MAX_MORPHS);
    CEUnit prev; ce_init(&prev);
    int has_prev = 0;
    uint32_t added = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (morphs[i].token[0] == '\0') continue;
        CEUnit kf; ce_init(&kf);
        ce_feed(&kf, (const uint8_t *)morphs[i].token,
                (uint32_t)strlen(morphs[i].token));
        CEUnit delta;
        if (!has_prev) { CEUnit z; ce_init(&z); ce_delta(&delta, &z, &kf); }
        else            { ce_delta(&delta, &prev, &kf); }
        ce_storage_add_typed(s, cid, (uint16_t)i,
                             (uint16_t)morphs[i].pos,
                             CE_TYPE_TEXT, &kf, &delta);
        prev = kf;
        has_prev = 1;
        ++added;
    }
    return added;
}

static uint32_t vote_canvas(const CEStorage *s,
                            const Morpheme *morphs, uint32_t n,
                            int *found) {
    *found = 0;
    if (s->count == 0 || n == 0) return 0;
    struct { uint32_t cid; double w; } v[MAX_VOTES];
    int vc = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (morphs[i].token[0] == '\0') continue;
        CEUnit q; ce_init(&q);
        ce_feed(&q, (const uint8_t *)morphs[i].token,
                (uint32_t)strlen(morphs[i].token));
        CESearchResult r;
        if (ce_search_by_type(s, &q, CE_TYPE_TEXT, 1, &r) <= 0) continue;
        uint32_t cid = s->entries[r.entry_idx].canvas_id;
        double w = 1.0 / (1.0 + (double)r.distance);
        int slot = -1;
        for (int j = 0; j < vc; ++j) if (v[j].cid == cid) { slot = j; break; }
        if (slot < 0 && vc < MAX_VOTES) { slot = vc++; v[slot].cid = cid; v[slot].w = 0; }
        if (slot >= 0) v[slot].w += w;
    }
    if (vc == 0) return 0;
    int best = 0;
    for (int j = 1; j < vc; ++j) if (v[j].w > v[best].w) best = j;
    *found = 1;
    return v[best].cid;
}

static double mean_r(const CEImage *img) {
    double s = 0; int n = img->width * img->height;
    for (int i = 0; i < n; ++i) s += img->pixels[i].r;
    return s / n;
}
static double mean_b(const CEImage *img) {
    double s = 0; int n = img->width * img->height;
    for (int i = 0; i < n; ++i) s += img->pixels[i].b;
    return s / n;
}

static int images_equal(const CEImage *a, const CEImage *b) {
    int n = a->width * a->height;
    for (int i = 0; i < n; ++i) {
        if (a->pixels[i].r != b->pixels[i].r) return 0;
        if (a->pixels[i].g != b->pixels[i].g) return 0;
        if (a->pixels[i].b != b->pixels[i].b) return 0;
    }
    return 1;
}

int main(void) {
    printf("=== test_gen_routed (E2E morpheme routing) ===\n");

    morpheme_init();

    CEStorage S;
    ce_storage_init(&S, 256);

    /* Row 0: red 64x64 + clause "red apple". */
    uint32_t cid_red = 0xAAAAu;
    uint8_t *rgba_red = make_solid(64, 200, 30, 30);
    uint32_t img_red = ce_storage_ingest_rgba_16(&S, cid_red, rgba_red, 64, 64);
    uint32_t txt_red = ingest_clause(&S, cid_red, "red apple");
    free(rgba_red);

    /* Row 1: blue 64x64 + clause "blue sky". */
    uint32_t cid_blu = 0xBBBBu;
    uint8_t *rgba_blu = make_solid(64, 30, 30, 200);
    uint32_t img_blu = ce_storage_ingest_rgba_16(&S, cid_blu, rgba_blu, 64, 64);
    uint32_t txt_blu = ingest_clause(&S, cid_blu, "blue sky");
    free(rgba_blu);

    printf("  ingested: red(IMG=%u TXT=%u) + blue(IMG=%u TXT=%u) = %u entries\n",
           img_red, txt_red, img_blu, txt_blu, S.count);
    CHECK(img_red == 64 && img_blu == 64,
          "16x16 ingest: 64 entries each (4x4 blocks * 4 quadrants)");
    CHECK(txt_red > 0 && txt_blu > 0, "morpheme bridge added entries for both clauses");

    /* Vote on prompts. */
    Morpheme morphs[MAX_MORPHS];
    int found;

    uint32_t n1 = morpheme_tokenize_clause("red apple", morphs, MAX_MORPHS);
    uint32_t cid1 = vote_canvas(&S, morphs, n1, &found);
    CHECK(found && cid1 == cid_red,
          "prompt \"red apple\" -> red canvas_id");

    uint32_t n2 = morpheme_tokenize_clause("blue sky", morphs, MAX_MORPHS);
    uint32_t cid2 = vote_canvas(&S, morphs, n2, &found);
    CHECK(found && cid2 == cid_blu,
          "prompt \"blue sky\" -> blue canvas_id");

    /* Generate. Use a small step count so the test runs fast. */
    CEGenConfig cfg = ce_gen_config_default();
    cfg.total_steps = 4;

    CEImage *img_a = (CEImage *)calloc(1, sizeof(CEImage));
    CEImage *img_b = (CEImage *)calloc(1, sizeof(CEImage));

    ce_generate_image_canvas_routed(img_a, &S, cid1,
                                    "red apple", 0xC0DEu, &cfg, 30);
    ce_generate_image_canvas_routed(img_b, &S, cid2,
                                    "blue sky", 0xC0DEu, &cfg, 30);

    double red_R = mean_r(img_a), red_B = mean_b(img_a);
    double blu_R = mean_r(img_b), blu_B = mean_b(img_b);
    printf("  red prompt   mean R=%.1f B=%.1f\n", red_R, red_B);
    printf("  blue prompt  mean R=%.1f B=%.1f\n", blu_R, blu_B);

    CHECK(red_R > red_B + 30.0,
          "red-routed image: mean R > mean B by >30");
    CHECK(blu_B > blu_R + 30.0,
          "blue-routed image: mean B > mean R by >30");

    /* Determinism. */
    CEImage *img_c = (CEImage *)calloc(1, sizeof(CEImage));
    ce_generate_image_canvas_routed(img_c, &S, cid1,
                                    "red apple", 0xC0DEu, &cfg, 30);
    CHECK(images_equal(img_a, img_c),
          "two routed runs with same seed produce identical pixels");

    free(img_a); free(img_b); free(img_c);
    ce_storage_free(&S);

    if (fails) {
        printf("\nFAIL: %d check(s) failed\n", fails);
        return 1;
    }
    printf("\n=== ALL PASS ===\n");
    return 0;
}
