#include "ce_ingest.h"
#include "ce_feed_image.h"
#include "ce_type.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include <sys/stat.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#define STBI_FAILURE_USERMSG
#include "third_party/stb_image.h"

#define CE_BLOCK_PX  8
#define CE_BLOCK_RGBA_BYTES (CE_BLOCK_PX * CE_BLOCK_PX * 4) /* 256 */

uint32_t ce_path_hash(const char *path) {
    /* FNV-1a 32-bit. */
    uint32_t h = 0x811C9DC5u;
    for (const unsigned char *p = (const unsigned char *)path; *p; ++p) {
        h ^= *p;
        h *= 0x01000193u;
    }
    return h;
}

static int has_image_ext(const char *name) {
    const char *dot = strrchr(name, '.');
    if (!dot) return 0;
    char ext[8];
    size_t n = strlen(dot);
    if (n >= sizeof(ext)) return 0;
    for (size_t i = 0; i < n; ++i) ext[i] = (char)tolower((unsigned char)dot[i]);
    ext[n] = 0;
    return strcmp(ext, ".png")  == 0 ||
           strcmp(ext, ".jpg")  == 0 ||
           strcmp(ext, ".jpeg") == 0 ||
           strcmp(ext, ".bmp")  == 0 ||
           strcmp(ext, ".tga")  == 0;
}

/* Copy an 8x8 RGBA tile from (bx*8, by*8) into `out_block`. Pixels falling
 * outside the image dimensions are zero-filled (right/bottom edge padding). */
static void copy_block(const uint8_t *img, int w, int h,
                       int bx, int by, uint8_t out_block[CE_BLOCK_RGBA_BYTES]) {
    memset(out_block, 0, CE_BLOCK_RGBA_BYTES);
    int x0 = bx * CE_BLOCK_PX;
    int y0 = by * CE_BLOCK_PX;
    for (int dy = 0; dy < CE_BLOCK_PX; ++dy) {
        int y = y0 + dy;
        if (y >= h) break;
        for (int dx = 0; dx < CE_BLOCK_PX; ++dx) {
            int x = x0 + dx;
            if (x >= w) break;
            const uint8_t *src = img + ((size_t)y * (size_t)w + (size_t)x) * 4u;
            uint8_t *dst = out_block + ((size_t)dy * CE_BLOCK_PX + (size_t)dx) * 4u;
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
        }
    }
}

int ce_ingest_file(CEStorage *s, const char *path, CEIngestStats *stats) {
    if (stats) ++stats->images_seen;

    int w = 0, h = 0, ch = 0;
    /* Force RGBA so block layout is uniform. */
    uint8_t *img = stbi_load(path, &w, &h, &ch, 4);
    if (!img) {
        if (stats) ++stats->errors;
        return 0;
    }

    if (w <= 0 || h <= 0) {
        stbi_image_free(img);
        if (stats) ++stats->errors;
        return 0;
    }

    int blocks_x = (w + CE_BLOCK_PX - 1) / CE_BLOCK_PX;
    int blocks_y = (h + CE_BLOCK_PX - 1) / CE_BLOCK_PX;
    if (blocks_x <= 0 || blocks_y <= 0) {
        stbi_image_free(img);
        if (stats) ++stats->errors;
        return 0;
    }

    uint32_t canvas_id = ce_path_hash(path);
    int added = 0;
    CEUnit prev; ce_init(&prev);

    uint8_t block_buf[CE_BLOCK_RGBA_BYTES];
    for (int by = 0; by < blocks_y; ++by) {
        for (int bx = 0; bx < blocks_x; ++bx) {
            copy_block(img, w, h, bx, by, block_buf);

            CEUnit fresh;
            /* Image-modality encoder: per-channel sums in inc.plus,
             * 4-direction gradients in dec.{plus,minus}. Distinct
             * from the text path's ce_feed (no channel mixing, no
             * position-mixing salt). */
            ce_feed_image(&fresh, block_buf);

            CEUnit delta;
            if (added == 0) {
                /* first block: delta is just the keyframe (anchor=0). */
                CEUnit zero; ce_init(&zero);
                ce_delta(&delta, &zero, &fresh);
            } else {
                ce_delta(&delta, &prev, &fresh);
            }

            ce_storage_add_typed(s, canvas_id,
                                 (uint16_t)(by & 0xFFFF),
                                 (uint16_t)(bx & 0xFFFF),
                                 CE_TYPE_IMAGE,
                                 &fresh, &delta);
            prev = fresh;
            ++added;
            if (stats) ++stats->blocks_added;
        }
    }

    stbi_image_free(img);
    if (stats && added > 0) ++stats->images_decoded;
    return added > 0;
}

static int is_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return S_ISDIR(st.st_mode);
}

static int join_path(char *out, size_t out_cap,
                     const char *dir, const char *name) {
    int n = snprintf(out, out_cap, "%s/%s", dir, name);
    return (n > 0 && (size_t)n < out_cap);
}

int ce_ingest_directory(CEStorage *s, const char *dir, CEIngestStats *stats) {
    if (!s || !dir) return 0;
    DIR *d = opendir(dir);
    if (!d) return 0;

    int n_ok = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (de->d_name[0] == '.') continue;
        if (!has_image_ext(de->d_name)) continue;

        char path[4096];
        if (!join_path(path, sizeof(path), dir, de->d_name)) continue;
        if (is_dir(path)) continue;
        if (ce_ingest_file(s, path, stats)) ++n_ok;
    }
    closedir(d);
    return n_ok;
}

static int ingest_recursive(CEStorage *s, const char *dir, CEIngestStats *stats) {
    DIR *d = opendir(dir);
    if (!d) return 0;
    int n_ok = 0;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        if (de->d_name[0] == '.') continue;
        char path[4096];
        if (!join_path(path, sizeof(path), dir, de->d_name)) continue;
        if (is_dir(path)) {
            n_ok += ingest_recursive(s, path, stats);
            continue;
        }
        if (!has_image_ext(de->d_name)) continue;
        if (ce_ingest_file(s, path, stats)) ++n_ok;
    }
    closedir(d);
    return n_ok;
}

int ce_ingest_directory_recursive(CEStorage *s, const char *dir, CEIngestStats *stats) {
    if (!s || !dir) return 0;
    return ingest_recursive(s, dir, stats);
}
