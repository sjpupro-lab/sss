/* test_material.c — slig_material_harmonic auto-analyzer + cell I/O. */

#include "slig_material_harmonic.h"
#include "slig_signal.h"
#include "ce_core.h"
#include "spatial_keyframe.h"  /* SpatialAI + DeltaFrame.cell_deltas */
#include "spatial_io.h"        /* ai_save / ai_load (.spai v6) */
#include "spatial_grid.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int pass = 0, fail = 0;
static void check(const char *name, int cond) {
    if (cond) { printf("  [PASS] %s\n", name); pass++; }
    else      { printf("  [FAIL] %s\n", name); fail++; }
}

static void make_uniform(uint8_t *buf, int dim, uint8_t v) {
    for (int i = 0; i < dim * dim; i++) buf[i] = v;
}

static void make_random(uint8_t *buf, int dim) {
    /* deterministic LCG */
    uint32_t s = 1234567u;
    for (int i = 0; i < dim * dim; i++) {
        s = s * 1103515245u + 12345u;
        buf[i] = (uint8_t)((s >> 16) & 0xFF);
    }
}

static void make_vstripes(uint8_t *buf, int dim) {
    for (int y = 0; y < dim; y++)
        for (int x = 0; x < dim; x++)
            buf[y * dim + x] = ((x / 4) & 1) ? 220 : 30;   /* vertical stripes */
}

static void make_hstripes(uint8_t *buf, int dim) {
    for (int y = 0; y < dim; y++)
        for (int x = 0; x < dim; x++)
            buf[y * dim + x] = ((y / 4) & 1) ? 220 : 30;   /* horizontal stripes */
}

int main(void) {
    printf("=== slig_material_harmonic ===\n");

    /* Presets — sanity values bracket analyzer output. */
    {
        SligMaterialTick m;
        slig_mat_preset(&m, SLIG_MAT_SKIN);
        check("preset SKIN roughness=90", m.roughness == 90);
        check("preset SKIN pore=30",      m.pore     == 30);
        slig_mat_preset(&m, SLIG_MAT_METAL);
        check("preset METAL specular=240", m.specular == 240);
        slig_mat_preset(&m, SLIG_MAT_WOOD);
        check("preset WOOD dir_angle=64",   m.dir_angle == 64);
    }

    /* Uniform 8×8 → roughness ≈ 0, anisotropy ≈ 0, pore = 0 */
    {
        uint8_t buf[64];
        make_uniform(buf, 8, 128);
        SligMaterialTick m;
        slig_mat_analyze_block(&m, buf, 8);
        printf("  uniform: rough=%u aniso=%u pore=%u h_str=%u\n",
               m.roughness, m.anisotropy, m.pore, m.h_strength);
        check("uniform → roughness ≤ 1",  m.roughness  <= 1);
        check("uniform → pore = 0",       m.pore       == 0);
        check("uniform → specular = 0",   m.specular   == 0);
    }

    /* Random 8×8 → roughness > 60 */
    {
        uint8_t buf[64];
        make_random(buf, 8);
        SligMaterialTick m;
        slig_mat_analyze_block(&m, buf, 8);
        printf("  random:  rough=%u aniso=%u pore=%u h_str=%u\n",
               m.roughness, m.anisotropy, m.pore, m.h_strength);
        check("random → roughness > 60", m.roughness > 60);
    }

    /* Vertical stripes → high anisotropy, dir_angle = 0 (horizontal Δ wins
     * because vertical stripes have most of the change ALONG the x axis). */
    {
        uint8_t buf[64];
        make_vstripes(buf, 8);
        SligMaterialTick m;
        slig_mat_analyze_block(&m, buf, 8);
        printf("  v-stripes: rough=%u aniso=%u dir=%u\n",
               m.roughness, m.anisotropy, m.dir_angle);
        check("v-stripes → anisotropy > 50", m.anisotropy > 50);
        check("v-stripes → dir_angle = 0 (H-axis Δ peak)", m.dir_angle == 0);
    }

    /* Horizontal stripes → mirror case: vertical Δ peak, dir_angle = 64. */
    {
        uint8_t buf[64];
        make_hstripes(buf, 8);
        SligMaterialTick m;
        slig_mat_analyze_block(&m, buf, 8);
        printf("  h-stripes: rough=%u aniso=%u dir=%u\n",
               m.roughness, m.anisotropy, m.dir_angle);
        check("h-stripes → anisotropy > 50", m.anisotropy > 50);
        check("h-stripes → dir_angle = 64 (V-axis Δ peak)", m.dir_angle == 64);
    }

    /* to_cell / from_cell roundtrip */
    {
        SligMaterialTick m;
        slig_mat_preset(&m, SLIG_MAT_FABRIC);

        SligSignal sig;
        memset(&sig, 0, sizeof sig);
        sig.dir = SLIG_DIR_HORIZONTAL;
        for (int i = 0; i < SLIG_SIG_LEN; i++) sig.u[i] = 1;
        CEUnit cell;
        slig_pack(&cell, &sig);

        check("before write: has_material = 0", slig_mat_has(&cell) == 0);
        slig_mat_to_cell(&cell, &m);
        check("after write: has_material = 1",  slig_mat_has(&cell) == 1);

        SligMaterialTick recovered;
        slig_mat_from_cell(&recovered, &cell);
        check("roundtrip roughness",  recovered.roughness  == m.roughness);
        check("roundtrip anisotropy", recovered.anisotropy == m.anisotropy);
        check("roundtrip h_strength", recovered.h_strength == m.h_strength);
        check("roundtrip grain",      recovered.grain      == m.grain);
    }

    /* Decompose path: Y cells get material, Cb/Cr cells don't. */
    {
        int dim = 32;
        uint8_t img[32 * 32];
        make_random(img, dim);

        SligCellSet y_set, cb_set, cr_set;
        slig_decompose_channel(&y_set,  img, dim, dim,
                               SLIG_CH_Y,  SLIG_LEVEL_COARSE, SLIG_MAX_CELLS);
        slig_decompose_channel(&cb_set, img, dim, dim,
                               SLIG_CH_CB, SLIG_LEVEL_COARSE, 6);
        slig_decompose_channel(&cr_set, img, dim, dim,
                               SLIG_CH_CR, SLIG_LEVEL_COARSE, 6);

        printf("  decompose: Y=%u Cb=%u Cr=%u cells\n",
               y_set.num_cells, cb_set.num_cells, cr_set.num_cells);

        int y_with_mat = 0;
        for (uint32_t i = 0; i < y_set.num_cells; i++)
            if (slig_mat_has(&y_set.cells[i])) y_with_mat++;
        check("Y cells carry material", y_with_mat > 0 &&
                                        y_with_mat == (int)y_set.num_cells);

        int cb_with_mat = 0, cr_with_mat = 0;
        for (uint32_t i = 0; i < cb_set.num_cells; i++)
            if (slig_mat_has(&cb_set.cells[i])) cb_with_mat++;
        for (uint32_t i = 0; i < cr_set.num_cells; i++)
            if (slig_mat_has(&cr_set.cells[i])) cr_with_mat++;
        check("Cb cells skip material (chroma marker compat)",
              cb_with_mat == 0);
        check("Cr cells skip material",
              cr_with_mat == 0);
    }

    /* Mat-S3: cell_delta roundtrip — ce_delta(parent, current) packed
     * into DeltaFrame.cell_deltas[], save → load via .spai v6, verify
     * ce_apply(parent, delta) reconstructs current exactly. */
    {
        SpatialAI *ai = spatial_ai_create();
        if (!ai) {
            check("spatial_ai_create (Mat-S3)", 0);
        } else {
            /* Build two fixture grids and ingest as keyframe + delta. */
            SpatialGrid *g_parent = grid_create();
            SpatialGrid *g_child  = grid_create();
            if (g_parent && g_child) {
                /* Parent: gradient. Child: gradient + slight perturb so
                 * the text grid match still falls within the delta
                 * threshold but the image cells are detectably different. */
                for (uint32_t i = 0; i < GRID_TOTAL; i++) {
                    g_parent->A[i] = (uint16_t)((i & 0xFF));
                    g_parent->R[i] = (uint8_t) ((i >> 4) & 0xFF);
                    g_parent->G[i] = (uint8_t) ( i       & 0xFF);
                    g_parent->B[i] = (uint8_t) ((i >> 2) & 0xFF);
                    g_child->A[i]  = (uint16_t)((i & 0xFF) + 5);
                    g_child->R[i]  = (uint8_t) ((i >> 4) & 0xFE);
                    g_child->G[i]  = (uint8_t) ( i       & 0xFE);
                    g_child->B[i]  = (uint8_t) ((i >> 2) & 0xFE);
                }
                uint32_t kf_id = ai_store_grid(ai, g_parent, "parent");
                check("Mat-S3 parent stored", kf_id != UINT32_MAX);

                /* Synthetic delta — mirror the parent's CE Cells with a
                 * known perturbation so we can verify ce_apply roundtrip
                 * after save/load. Make sure the engine has a delta. */
                if (ai->df_count == 0 && ai->kf_count > 0 &&
                    ai->keyframes[kf_id].has_image) {
                    /* Append a delta ourselves */
                    if (ai->df_count >= ai->df_capacity) {
                        uint32_t new_cap = ai->df_capacity * 2;
                        DeltaFrame *new_df = (DeltaFrame*)realloc(
                            ai->deltas, new_cap * sizeof(DeltaFrame));
                        if (new_df) {
                            memset(&new_df[ai->df_capacity], 0,
                                   (new_cap - ai->df_capacity) * sizeof(DeltaFrame));
                            ai->deltas = new_df;
                            ai->df_capacity = new_cap;
                        }
                    }
                    DeltaFrame *df = &ai->deltas[ai->df_count];
                    memset(df, 0, sizeof(*df));
                    df->id = ai->df_count;
                    df->parent_id = kf_id;
                    strncpy(df->label, "child-delta", 63);
                    df->count = 0;
                    df->entries = NULL;

                    /* Take parent's FINE-Y cells as anchor, perturb
                     * each by a known offset, store ce_delta(anchor,
                     * perturbed). After load, ce_apply must recover
                     * the perturbed cells. */
                    const SligCellSet *anchor = slig_codebook_get(
                        &ai->codebook,
                        ai->keyframes[kf_id].image_idx[SLIG_LEVEL_FINE][SLIG_CH_Y]);
                    if (anchor && anchor->num_cells > 0) {
                        CEUnit perturbed[4];
                        uint32_t n = anchor->num_cells < 4 ? anchor->num_cells : 4;
                        for (uint32_t i = 0; i < n; i++) {
                            perturbed[i] = anchor->cells[i];
                            ce_bytes(&perturbed[i])[0]
                                = (uint8_t)(ce_cbytes(&anchor->cells[i])[0] + 7);
                            ce_delta(&df->cell_deltas[i],
                                     &anchor->cells[i],
                                     &perturbed[i]);
                        }
                        df->cell_delta_count = n;
                        ai->df_count++;

                        /* Roundtrip via .spai v6 */
                        SpaiStatus s = ai_save(ai, "/tmp/mat_s3.spai");
                        check("Mat-S3 ai_save (v6)", s == SPAI_OK);
                        SpaiStatus ls;
                        SpatialAI *loaded = ai_load("/tmp/mat_s3.spai", &ls);
                        check("Mat-S3 ai_load",
                              loaded != NULL && ls == SPAI_OK);
                        if (loaded) {
                            check("Mat-S3 df_count survives",
                                  loaded->df_count == 1);
                            if (loaded->df_count == 1) {
                                DeltaFrame *ldf = &loaded->deltas[0];
                                check("Mat-S3 cell_delta_count survives",
                                      ldf->cell_delta_count == n);

                                /* Verify ce_apply recovers perturbed */
                                int recovered = 1;
                                for (uint32_t i = 0; i < n; i++) {
                                    CEUnit r;
                                    ce_apply(&r, &anchor->cells[i],
                                             &ldf->cell_deltas[i]);
                                    if (ce_distance(&r, &perturbed[i]) != 0)
                                        recovered = 0;
                                }
                                check("Mat-S3 ce_apply recovers perturbed",
                                      recovered);
                            }
                            spatial_ai_destroy(loaded);
                        }
                    }
                }
                grid_destroy(g_parent);
                grid_destroy(g_child);
            }
            spatial_ai_destroy(ai);
        }
    }

    printf("=== %d PASS / %d FAIL ===\n", pass, fail);
    return fail;
}
