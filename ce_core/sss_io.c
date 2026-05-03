/* sss_io.c — .sss v8 binary loader.
 *
 * Layout (little-endian, host writes/reads must agree):
 *   Header (28 bytes):
 *     uint32 magic      = 'SSX8'
 *     uint32 version    = 8
 *     uint32 height
 *     uint32 width
 *     uint32 nf
 *     uint32 nf_low
 *     uint32 num_cells
 *   Cells (repeated num_cells times):
 *     uint32 type
 *     uint32 label_len
 *     bytes  label[label_len]
 *     float  fp[256]
 *     uint32 amp_len    (in floats)
 *     float  amp[amp_len]
 *     uint32 phase_len  (in floats)
 *     float  phase[phase_len]
 */
#include "sss_rowvae.h"

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

    uint32_t magic, version;
    if (read_u32(f, &magic) || magic != SSS_MAGIC) {
        fclose(f); return -2;
    }
    if (read_u32(f, &version) || version != SSS_VERSION) {
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
        if (label_len >= SSS_LABEL_MAX) {
            /* Truncate label but still consume the bytes. */
            char drop[256];
            uint32_t to_keep = SSS_LABEL_MAX - 1;
            if (fread(c->label, 1, to_keep, f) != to_keep) {
                fclose(f); sss_model_free(out); return -7;
            }
            c->label[to_keep] = '\0';
            uint32_t rest = label_len - to_keep;
            while (rest > 0) {
                uint32_t chunk = rest > sizeof(drop) ? (uint32_t)sizeof(drop) : rest;
                if (fread(drop, 1, chunk, f) != chunk) {
                    fclose(f); sss_model_free(out); return -7;
                }
                rest -= chunk;
            }
        } else {
            if (label_len > 0 && fread(c->label, 1, label_len, f) != label_len) {
                fclose(f); sss_model_free(out); return -7;
            }
            c->label[label_len] = '\0';
        }

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
