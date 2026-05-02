/* sss_io.c — .sss v9 binary loader.
 *
 * Layout (little-endian; trainer and runtime must agree):
 *   Header (28 bytes):
 *     uint32 magic      = 'SSX9'   (0x53535839)
 *     uint32 version    = 9
 *     uint32 height
 *     uint32 width
 *     uint32 channels   = 3
 *     uint32 num_cells
 *     uint32 reserved   = 0
 *   Cells (repeated num_cells times):
 *     uint32 type
 *     uint32 label_len
 *     bytes  label[label_len]
 *     float  fp[256]
 *     uint32 num_imgs
 *     float  imgs[num_imgs * height * width * channels]
 *
 * Each cell holds a pool of training images for one attribute, not
 * an averaged fingerprint of pixels. The sculpt engine walks the
 * pool row-by-row at generate time.
 */
#include "sss_rowvae.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Hard caps used to reject obviously malformed or hostile model files
 * before we trust their length fields for allocation. The numbers are
 * deliberately generous (the canonical sculpt resolution is 64) but
 * still well below sizes that would overflow a 32-bit count. */
#define SSS_MAX_DIM         1024u
#define SSS_MAX_CELLS       4096u
#define SSS_MAX_IMGS_CELL    256u

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

    uint32_t magic, version, reserved;
    if (read_u32(f, &magic) || magic != SSS_MAGIC) {
        fclose(f); return -2;
    }
    if (read_u32(f, &version) || version != SSS_VERSION) {
        fclose(f); return -3;
    }
    if (read_u32(f, &out->height)
     || read_u32(f, &out->width)
     || read_u32(f, &out->channels)
     || read_u32(f, &out->num_cells)
     || read_u32(f, &reserved)) {
        fclose(f); return -4;
    }
    if (out->channels != 3) {
        fclose(f); return -5;
    }
    if (out->height == 0 || out->height > SSS_MAX_DIM
     || out->width  == 0 || out->width  > SSS_MAX_DIM
     || out->num_cells > SSS_MAX_CELLS) {
        fclose(f); return -5;
    }

    if (out->num_cells == 0) {
        fclose(f);
        return 0;
    }

    out->cells = (SSSCell *)calloc(out->num_cells, sizeof(SSSCell));
    if (!out->cells) { fclose(f); return -6; }

    /* height/width/channels are bounded above; this product cannot
     * overflow size_t on any supported platform. */
    size_t pixels_per_img = (size_t)out->height * (size_t)out->width
                          * (size_t)out->channels;

    for (uint32_t i = 0; i < out->num_cells; ++i) {
        SSSCell *c = &out->cells[i];
        uint32_t label_len;
        if (read_u32(f, &c->type) || read_u32(f, &label_len)) {
            fclose(f); sss_model_free(out); return -7;
        }
        if (label_len >= SSS_LABEL_MAX) {
            char drop[256];
            uint32_t to_keep = SSS_LABEL_MAX - 1;
            if (fread(c->label, 1, to_keep, f) != to_keep) {
                fclose(f); sss_model_free(out); return -8;
            }
            c->label[to_keep] = '\0';
            uint32_t rest = label_len - to_keep;
            while (rest > 0) {
                uint32_t chunk = rest > sizeof(drop) ? (uint32_t)sizeof(drop) : rest;
                if (fread(drop, 1, chunk, f) != chunk) {
                    fclose(f); sss_model_free(out); return -8;
                }
                rest -= chunk;
            }
        } else {
            if (label_len > 0 && fread(c->label, 1, label_len, f) != label_len) {
                fclose(f); sss_model_free(out); return -8;
            }
            c->label[label_len] = '\0';
        }

        if (read_floats(f, c->fp, SSS_FP_LEN)) {
            fclose(f); sss_model_free(out); return -9;
        }

        if (read_u32(f, &c->num_imgs)) {
            fclose(f); sss_model_free(out); return -10;
        }
        if (c->num_imgs > SSS_MAX_IMGS_CELL) {
            fclose(f); sss_model_free(out); return -11;
        }
        if (c->num_imgs > 0) {
            /* Reject any (num_imgs * pixels_per_img * sizeof(float))
             * that would overflow size_t. With both factors clamped
             * above this can't happen for legitimate files; the check
             * is here so a corrupted on-disk count never reaches
             * malloc with an under-sized request. */
            if (pixels_per_img > SIZE_MAX / sizeof(float) / c->num_imgs) {
                fclose(f); sss_model_free(out); return -12;
            }
            size_t total = (size_t)c->num_imgs * pixels_per_img;
            c->imgs = (float *)malloc(total * sizeof(float));
            if (!c->imgs || read_floats(f, c->imgs, total)) {
                fclose(f); sss_model_free(out); return -13;
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
            free(m->cells[i].imgs);
        }
        free(m->cells);
    }
    memset(m, 0, sizeof(*m));
}
