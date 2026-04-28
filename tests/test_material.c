/* test_material.c — slig_material_harmonic auto-analyzer + cell I/O. */

#include "slig_material_harmonic.h"
#include "slig_signal.h"
#include "ce_core.h"
#include <stdio.h>
#include <string.h>

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

    printf("=== %d PASS / %d FAIL ===\n", pass, fail);
    return fail;
}
