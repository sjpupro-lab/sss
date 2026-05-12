/* sss_io.c — .sss v9 binary loader (with v8 backward-compat).
 *
 * Layout (little-endian, host writes/reads must agree):
 *   Header (28 bytes):
 *     uint32 magic      = 'SSX9' (v9) or 'SSX8' (v8 read-only)
 *     uint32 version    = 9 (or 8 for legacy)
 *     uint32 height
 *     uint32 width
 *     uint32 nf
 *     uint32 nf_low
 *     uint32 num_cells
 *   Cells (repeated num_cells times):
 *     uint32 type
 *     uint32 label_len
 *     bytes  label[label_len]
 *     bytes  ce_key[64]      <-- v9 only; v8 regenerates via ce_feed
 *     float  fp[256]
 *     uint32 amp_len    (in floats)
 *     float  amp[amp_len]
 *     uint32 phase_len  (in floats)  <-- DEPRECATED, see below
 *     float  phase[phase_len]
 *
 * Phase deprecation (Phase 1, 2026-05):
 *   The trainer no longer writes phase data — `scripts/sss_train.py`
 *   emits phase_len == 0 for every cell, and the generator
 *   (`ce_core/sss_rowvae.c`) samples a per-seed random phase at
 *   noise-init time. To stay backward-compatible with v9 files
 *   produced before the change, this reader still consumes the
 *   `phase_len` u32 and `phase[]` floats when they are present:
 *   they land in `SSSCell.phase` so the legacy gentle low-band phase
 *   relaxation in `sss_rowvae.c` can still run. Cells with
 *   `phase_len == 0` leave `cell->phase = NULL`, which the generator
 *   treats as "no relaxation, free-evolve from random init".
 *
 *   Phase 2 (planned) will introduce a `.sfb` (SSS Feature Bank)
 *   format that drops the phase field entirely. Until then the on-
 *   disk layout above stays stable.
 */
#include "sss_rowvae.h"
#include "ce_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_u32(FILE *f, uint32_t *v)
{
    return fread(v, sizeof(uint32_t), 1, f) == 1 ? 0 : -1;
}

static int read_floats(FILE *f, float *v, size_t n)
{
    return fread(v, sizeof(float), n, f) == n ? 0 : -1;
}

int sss_model_load(const char *path, SSSModel *out)
{
    if (!path || !out) return -1;
    memset(out, 0, sizeof(*out));

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    /* Accept both the current SSX9 layout and the legacy SSX8 layout
     * (which lacks the per-cell ce_key block). The cell loop below
     * checks `is_v8` to know whether to read+skip or synthesise it. */
    uint32_t magic, version;
    if (read_u32(f, &magic)) { fclose(f); return -2; }
    if (magic != SSS_MAGIC && magic != 0x53535838u /* SSX8 */) {
        fclose(f); return -2;
    }
    if (read_u32(f, &version)) { fclose(f); return -3; }
    int is_v8 = (magic == 0x53535838u && version == 8u);
    if (!is_v8 && (magic != SSS_MAGIC || version != SSS_VERSION)) {
        fclose(f); return -3;
    }
    if (read_u32(f, &out->height)
     || read_u32(f, &out->width)
     || read_u32(f, &out->nf)
     || read_u32(f, &out->nf_low)
     || read_u32(f, &out->num_cells)) {
        fclose(f); return -4;
    }

    if (out->num_cells == 0) {
        fclose(f);
        return 0;
    }

    out->cells = (SSSCell *)calloc(out->num_cells, sizeof(SSSCell));
    if (!out->cells) { fclose(f); return -5; }

    for (uint32_t i = 0; i < out->num_cells; ++i) {
        SSSCell *c = &out->cells[i];
        uint32_t label_len;
        if (read_u32(f, &c->type) || read_u32(f, &label_len)) {
            fclose(f); sss_model_free(out); return -6;
        }

        /* Read the full on-disk label into a heap buffer. We need
         * every byte for the v8 ce_feed regeneration (truncating
         * c->label to SSS_LABEL_MAX-1 would change the ce_key for
         * long labels — feeding only the prefix gives a different
         * morpheme link than the trainer used). c->label still gets
         * the truncated NUL-terminated prefix for display. */
        uint8_t *full_label = NULL;
        if (label_len > 0) {
            full_label = (uint8_t *)malloc(label_len);
            if (!full_label) {
                fclose(f); sss_model_free(out); return -7;
            }
            if (fread(full_label, 1, label_len, f) != label_len) {
                free(full_label); fclose(f);
                sss_model_free(out); return -7;
            }
        }
        uint32_t to_keep = (label_len < SSS_LABEL_MAX)
                           ? label_len : (uint32_t)(SSS_LABEL_MAX - 1);
        if (to_keep > 0) memcpy(c->label, full_label, to_keep);
        c->label[to_keep] = '\0';

        /* v9 stores the ce_key explicitly; v8 files don't, so we
         * regenerate it from the full on-disk label bytes via
         * ce_feed — same morpheme, same ce_key, search key stays
         * consistent across versions even when the human-readable
         * label was longer than SSS_LABEL_MAX. */
        if (is_v8) {
            ce_init(&c->ce_key);
            if (label_len > 0) {
                ce_feed(&c->ce_key, full_label, label_len);
            }
        } else {
            if (fread(ce_bytes(&c->ce_key), 1, 64, f) != 64) {
                free(full_label); fclose(f);
                sss_model_free(out); return -8;
            }
        }
        free(full_label);

        if (read_floats(f, c->fp, SSS_FP_LEN)) {
            fclose(f); sss_model_free(out); return -8;
        }

        if (read_u32(f, &c->amp_len)) { fclose(f); sss_model_free(out); return -9; }
        if (c->amp_len > 0) {
            c->amp = (float *)malloc(c->amp_len * sizeof(float));
            if (!c->amp || read_floats(f, c->amp, c->amp_len)) {
                fclose(f); sss_model_free(out); return -10;
            }
        }
        if (read_u32(f, &c->phase_len)) { fclose(f); sss_model_free(out); return -11; }
        if (c->phase_len > 0) {
            c->phase = (float *)malloc(c->phase_len * sizeof(float));
            if (!c->phase || read_floats(f, c->phase, c->phase_len)) {
                fclose(f); sss_model_free(out); return -12;
            }
        }
    }

    fclose(f);
    return 0;
}

void sss_model_free(SSSModel *m)
{
    if (!m) return;
    if (m->cells) {
        for (uint32_t i = 0; i < m->num_cells; ++i) {
            free(m->cells[i].amp);
            free(m->cells[i].phase);
        }
        free(m->cells);
    }
    memset(m, 0, sizeof(*m));
}
