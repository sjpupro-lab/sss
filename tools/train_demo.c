/* train_demo.c — joint text+image trainer (16×16 atomic + per-morpheme).
 *
 * Reads a TSV produced by make_demo_dataset:
 *   <label>\t<clause>\t<image_path_relative_to_dataset_dir>
 *
 * For each row:
 *   1. image_to_grid(<dataset>/<image>) -> SpatialGrid (256x256 RGB).
 *   2. ai_store_auto_with_image(ai, clause, grid, label) — SpatialAI
 *      keyframe with text+image features (Mat-1..4 included).
 *   3. ce_storage_ingest_rgba_16(ces, canvas_id, rgba, w, h) —
 *      16x16 atomic blocks via ce_feed_image_16. 4 quadrant CEUnits per
 *      block (TL/TR/BL/BR), 16x16 grid of blocks across 256x256 image
 *      → 1024 CE_TYPE_IMAGE entries per image, all sharing canvas_id.
 *   4. hybrid_vae_encode(ces, codebook, rgba, cid) — adds the SLIG
 *      decomposition (3 scale × 3 channel) on top of the block stamps,
 *      writing canvas_id into each SLIG cell's audio_bins so block-stamp
 *      and SLIG entries cross-link.
 *   5. (optional, --masked-epochs N) masked_train_image runs the
 *      mask→denoise→loss loop and pushes the converged Slig cells back
 *      into the storage so the latent learns to fill in missing parts.
 *   6. morpheme bridge — `morpheme_tokenize_clause(clause, ...)` then,
 *      per morpheme: `ce_feed(.., m.token, len)` → one CE_TYPE_TEXT
 *      entry. All TEXT entries share canvas_id with the image entries
 *      so generation can: prompt → morpheme tokenize → per-morpheme
 *      ce_search_by_type(TEXT) → vote canvas_id → IMAGE entries with
 *      matching canvas_id.
 *
 * Outputs:
 *   <out_base>.spai   SpatialAI joint text+image keyframes.
 *   <out_base>.ces    CEStorage: per-row IMAGE + TEXT entries sharing
 *                     canvas_id = fnv1a(image_path).
 *
 * Usage:
 *   train_demo <dataset_dir> <out_base> [--masked-epochs N]
 */

#include "spatial_grid.h"
#include "spatial_image.h"
#include "spatial_keyframe.h"
#include "spatial_morpheme.h"
#include "spatial_io.h"

#include "ce_core.h"
#include "ce_storage.h"
#include "ce_storage_io.h"
#include "ce_type.h"
#include "ce_hybrid_vae.h"
#include "ce_masked_train.h"
#include "ce_residual_codebook.h"
#include "slig_codebook.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MAX_MORPHS 256

static uint32_t fnv1a(const char *s) {
    uint32_t h = 0x811C9DC5u;
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        h ^= *p;
        h *= 0x01000193u;
    }
    return h;
}

static uint8_t *grid_to_rgba(const SpatialGrid *g, int *w, int *h) {
    int W = (int)GRID_SIZE, H = (int)GRID_SIZE;
    uint8_t *rgba = (uint8_t *)malloc((size_t)W * H * 4u);
    if (!rgba) return NULL;
    for (int i = 0; i < W * H; ++i) {
        rgba[i * 4 + 0] = g->R[i];
        rgba[i * 4 + 1] = g->G[i];
        rgba[i * 4 + 2] = g->B[i];
        rgba[i * 4 + 3] = 255;
    }
    *w = W; *h = H;
    return rgba;
}

/* Append one CE_TYPE_TEXT entry per morpheme of `clause`. canvas_id
 * matches the paired image's canvas_id; slot = morpheme index;
 * block_idx = morpheme part-of-speech tag. delta of the first morpheme
 * is taken against the zero CEUnit, subsequent morphemes chain the
 * previous morpheme's keyframe. Returns morphemes added. */
static uint32_t ingest_clause_morphemes(CEStorage *s,
                                        uint32_t canvas_id,
                                        const char *clause) {
    Morpheme morphs[MAX_MORPHS];
    uint32_t n = morpheme_tokenize_clause(clause, morphs, MAX_MORPHS);
    CEUnit prev; ce_init(&prev);
    int has_prev = 0;
    uint32_t added = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const char *tok = morphs[i].token;
        size_t len = strlen(tok);
        if (len == 0) continue;
        CEUnit kf;
        ce_init(&kf);
        ce_feed(&kf, (const uint8_t *)tok, (uint32_t)len);
        CEUnit delta;
        if (!has_prev) {
            CEUnit zero; ce_init(&zero);
            ce_delta(&delta, &zero, &kf);
        } else {
            ce_delta(&delta, &prev, &kf);
        }
        ce_storage_add_typed(s, canvas_id,
                             (uint16_t)i,
                             (uint16_t)morphs[i].pos,
                             CE_TYPE_TEXT, &kf, &delta);
        prev = kf;
        has_prev = 1;
        ++added;
    }
    return added;
}

/* strip leading/trailing whitespace in-place */
static char *trim(char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') ++s;
    char *e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n')) --e;
    *e = '\0';
    return s;
}

static int parse_mask_strategy(const char *s, MaskStrategy *out) {
    if (!s || !out) return 0;
    if (strcmp(s, "correction") == 0) { *out = MASK_CORRECTION_ONLY; return 1; }
    if (strcmp(s, "residual"  ) == 0) { *out = MASK_RESIDUAL_UP;     return 1; }
    if (strcmp(s, "random"    ) == 0) { *out = MASK_RANDOM_HALF;     return 1; }
    if (strcmp(s, "progressive") == 0){ *out = MASK_PROGRESSIVE;     return 1; }
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "usage: %s <dataset_dir> <out_base> [options]\n"
            "  Masked-train control (forwarded to MaskedTrainConfig):\n"
            "    --masked-epochs N         epoch budget per image (default 0 = off)\n"
            "    --target-loss F           early stop when best loss <= F (default 8.0)\n"
            "    --loss-patience N         stop after N epochs with no improvement (default 3)\n"
            "    --denoise-steps N         denoise steps per epoch (default 8)\n"
            "    --mask-strategy NAME      correction|residual|random|progressive (default progressive)\n"
            "    --mask-seed U64           PRNG seed for random strategy (default 42)\n"
            "    --residual-codebook       reduce correction cells to codebook descriptors\n"
            "  Segment loss weights (all 0 = legacy mean loss):\n"
            "    --loss-w-struct F         structure (basis) loss weight\n"
            "    --loss-w-texture F        texture (edge+texture) loss weight\n"
            "    --loss-w-color F          color (color+event) loss weight\n",
            prog);
}

int main(int argc, char **argv) {
    /* --help works even without positional args, so users can discover
     * the new flags without first having to supply a dataset path. */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
    }
    if (argc < 3) {
        usage(argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    const char *out = argv[2];

    /* Defaults match masked_train_config_default — we only overwrite
     * fields the user passes on the CLI. */
    int       masked_epochs   = 0;
    int       use_residual_cb = 0;
    float     cli_target_loss = -1.0f;     /* sentinel: unset */
    int       cli_patience    = -1;
    int       cli_denoise     = -1;
    MaskStrategy cli_strategy = MASK_PROGRESSIVE;
    int       have_strategy   = 0;
    uint64_t  cli_seed        = 0;
    int       have_seed       = 0;
    float     cli_w_struct    = 0.0f;
    float     cli_w_texture   = 0.0f;
    float     cli_w_color     = 0.0f;

    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--masked-epochs") == 0 && i + 1 < argc) {
            masked_epochs = atoi(argv[++i]);
            if (masked_epochs < 0) masked_epochs = 0;
        } else if (strcmp(argv[i], "--target-loss") == 0 && i + 1 < argc) {
            cli_target_loss = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--loss-patience") == 0 && i + 1 < argc) {
            cli_patience = atoi(argv[++i]);
            if (cli_patience < 0) cli_patience = 0;
        } else if (strcmp(argv[i], "--denoise-steps") == 0 && i + 1 < argc) {
            cli_denoise = atoi(argv[++i]);
            if (cli_denoise < 1) cli_denoise = 1;
        } else if (strcmp(argv[i], "--mask-strategy") == 0 && i + 1 < argc) {
            if (!parse_mask_strategy(argv[++i], &cli_strategy)) {
                fprintf(stderr,
                        "[train_demo] unknown mask strategy '%s'"
                        " (expected correction|residual|random|progressive)\n",
                        argv[i]);
                return 2;
            }
            have_strategy = 1;
        } else if (strcmp(argv[i], "--mask-seed") == 0 && i + 1 < argc) {
            cli_seed = strtoull(argv[++i], NULL, 0);
            have_seed = 1;
        } else if (strcmp(argv[i], "--loss-w-struct") == 0 && i + 1 < argc) {
            cli_w_struct  = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--loss-w-texture") == 0 && i + 1 < argc) {
            cli_w_texture = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--loss-w-color") == 0 && i + 1 < argc) {
            cli_w_color   = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--residual-codebook") == 0) {
            use_residual_cb = 1;
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h"    ) == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "[train_demo] ignoring unknown arg: %s\n", argv[i]);
        }
    }

    char tsv_path[1024];
    snprintf(tsv_path, sizeof(tsv_path), "%s/labels.tsv", dir);
    FILE *tsv = fopen(tsv_path, "r");
    if (!tsv) { perror(tsv_path); return 1; }

    /* idempotent — also called from spatial_keyframe.c. */
    morpheme_init();

    SpatialAI *ai = spatial_ai_create();
    if (!ai) { fclose(tsv); return 1; }

    CEStorage ces;
    ce_storage_init(&ces, 256);

    SligCodebook codebook;
    slig_codebook_init(&codebook);

    CEResidualCodebook residual_cb;
    ce_residual_codebook_init(&residual_cb);

    int rows = 0, ai_ok = 0;
    uint32_t total_image_entries  = 0;
    uint32_t total_text_entries   = 0;
    uint32_t total_hybrid_blocks  = 0;
    uint32_t total_hybrid_cells   = 0;
    uint32_t total_masked_stored  = 0;
    uint32_t total_residual_descriptors = 0;
    int      masked_converged     = 0;

    char line[1024];
    while (fgets(line, sizeof(line), tsv)) {
        char *p = trim(line);
        if (!*p || *p == '#') continue;

        char *t1 = strchr(p, '\t');
        if (!t1) continue;
        *t1 = '\0';
        char *label = p;
        char *t2 = strchr(t1 + 1, '\t');
        if (!t2) continue;
        *t2 = '\0';
        char *clause = t1 + 1;
        char *img_rel = t2 + 1;

        char img_path[1024];
        snprintf(img_path, sizeof(img_path), "%s/%s", dir, img_rel);

        SpatialGrid *g = image_to_grid(img_path);
        if (!g) {
            fprintf(stderr, "[train_demo] skip %s (image_to_grid failed)\n", img_path);
            continue;
        }

        ++rows;
        uint32_t cid = fnv1a(img_path);

        /* 1. SpatialAI joint text+image store. */
        uint32_t kf = ai_store_auto_with_image(ai, clause, g, label);
        if (kf != UINT32_MAX) {
            ++ai_ok;
            fprintf(stderr, "[train_demo] +KF id=%u  label=%-12s  clause=\"%s\"\n",
                    kf, label, clause);
        } else {
            fprintf(stderr, "[train_demo] !! ai_store_auto_with_image failed for %s\n",
                    label);
        }

        /* 2. CEStorage 16x16 atomic image ingest + hybrid VAE encode. */
        int w = 0, h = 0;
        uint8_t *rgba = grid_to_rgba(g, &w, &h);
        if (rgba) {
            uint32_t added_img = ce_storage_ingest_rgba_16(&ces, cid, rgba, w, h);
            total_image_entries += added_img;
            fprintf(stderr, "[train_demo]  +IMG entries=%u (16x16 quadrants)  cid=0x%08x\n",
                    added_img, cid);

            /* 2b. Hybrid VAE encode: SLIG decomposition + audio link. */
            HybridEncodeResult henc;
            hybrid_vae_encode(&henc, &ces, &codebook, rgba, w, h, cid);
            total_hybrid_blocks += henc.block_entries;
            total_hybrid_cells  += henc.total_cells;
            fprintf(stderr,
                    "[train_demo]  +HYB block_entries=%u slig_cells=%u  cid=0x%08x\n",
                    henc.block_entries, henc.total_cells, cid);

            /* 2c. Optional masked-training loop. */
            if (masked_epochs > 0) {
                MaskedTrainConfig mcfg;
                masked_train_config_default(&mcfg);
                mcfg.epochs = masked_epochs;
                if (cli_target_loss >= 0)  mcfg.target_loss   = cli_target_loss;
                if (cli_patience    >= 0)  mcfg.loss_patience = (float)cli_patience;
                if (cli_denoise     >= 1)  mcfg.denoise_steps = cli_denoise;
                if (have_strategy)         mcfg.strategy      = cli_strategy;
                if (have_seed)             mcfg.mask_seed     = cli_seed;
                mcfg.loss_w_structure      = cli_w_struct;
                mcfg.loss_w_texture        = cli_w_texture;
                mcfg.loss_w_color          = cli_w_color;
                if (use_residual_cb) {
                    mcfg.residual_book = &residual_cb;
                }
                MaskedTrainResult mres;
                masked_train_image(&mres, &ces, rgba, w, h, cid, &mcfg);
                total_masked_stored += mres.cells_stored;
                total_residual_descriptors += mres.residual_cells;
                if (mres.converged) ++masked_converged;
                fprintf(stderr,
                        "[train_demo]  +MSK epochs_run=%d loss=%.2f "
                        "(struct=%.2f tex=%.2f color=%.2f) converged=%d "
                        "cells_stored=%u residuals=%u\n",
                        mres.epochs_run, mres.final_loss,
                        mres.loss_structure, mres.loss_texture, mres.loss_color,
                        mres.converged, mres.cells_stored, mres.residual_cells);
            }

            free(rgba);
        }

        /* 3. Per-morpheme TEXT bridge sharing canvas_id with IMAGE entries. */
        uint32_t added_txt = ingest_clause_morphemes(&ces, cid, clause);
        total_text_entries += added_txt;
        fprintf(stderr, "[train_demo]  +TXT morphemes=%u  cid=0x%08x\n",
                added_txt, cid);

        grid_destroy(g);
    }
    fclose(tsv);

    if (rows == 0) {
        fprintf(stderr, "[train_demo] no valid rows in %s\n", tsv_path);
        ce_storage_free(&ces);
        spatial_ai_destroy(ai);
        return 1;
    }

    char spai_path[1024];
    snprintf(spai_path, sizeof(spai_path), "%s.spai", out);
    SpaiStatus st = ai_save(ai, spai_path);
    if (st != SPAI_OK) {
        fprintf(stderr, "[train_demo] ai_save failed (status=%d)\n", (int)st);
        ce_storage_free(&ces);
        spatial_ai_destroy(ai);
        return 1;
    }
    fprintf(stderr, "[train_demo] saved %s (%d/%d rows)\n",
            spai_path, ai_ok, rows);

    char ces_path[1024];
    snprintf(ces_path, sizeof(ces_path), "%s.ces", out);
    /* Always write v3 — codebook is empty when --residual-codebook is
     * off, populated when on. The trailer adds 8 bytes (magic + count)
     * even with zero codes, which is negligible. */
    if (!ce_storage_save_with_codebook(&ces,
                                       use_residual_cb ? &residual_cb : NULL,
                                       ces_path)) {
        fprintf(stderr, "[train_demo] ce_storage_save failed\n");
        ce_storage_free(&ces);
        spatial_ai_destroy(ai);
        return 1;
    }
    fprintf(stderr,
            "[train_demo] saved %s (IMG=%u TEXT=%u HYB_blocks=%u HYB_cells=%u "
            "total=%u entries across %d rows)\n",
            ces_path, total_image_entries, total_text_entries,
            total_hybrid_blocks, total_hybrid_cells,
            ces.count, rows);
    if (masked_epochs > 0) {
        fprintf(stderr,
                "[train_demo] masked-train summary: epochs/img=%d "
                "stored_cells=%u residuals=%u (codebook=%u patterns) converged=%d/%d\n",
                masked_epochs, total_masked_stored,
                total_residual_descriptors,
                use_residual_cb ? residual_cb.count : 0u,
                masked_converged, rows);
    }

    ce_storage_free(&ces);
    spatial_ai_destroy(ai);
    return 0;
}
