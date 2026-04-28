#include "ce_storage_io.h"
#include <stdio.h>
#include <string.h>

static int write_u32(FILE *fp, uint32_t v) {
    uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8),
                     (uint8_t)(v >> 16), (uint8_t)(v >> 24) };
    return fwrite(b, 1, 4, fp) == 4;
}
static int write_u16(FILE *fp, uint16_t v) {
    uint8_t b[2] = { (uint8_t)v, (uint8_t)(v >> 8) };
    return fwrite(b, 1, 2, fp) == 2;
}
static int read_u32(FILE *fp, uint32_t *out) {
    uint8_t b[4];
    if (fread(b, 1, 4, fp) != 4) return 0;
    *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8)
         | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}
static int read_u16(FILE *fp, uint16_t *out) {
    uint8_t b[2];
    if (fread(b, 1, 2, fp) != 2) return 0;
    *out = (uint16_t)((uint16_t)b[0] | ((uint16_t)b[1] << 8));
    return 1;
}

int ce_storage_save(const CEStorage *s, const char *path) {
    if (!s || !path) return 0;
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;

    int ok = 1;
    ok &= write_u32(fp, CE_STORAGE_MAGIC);
    ok &= write_u32(fp, CE_STORAGE_VERSION);
    ok &= write_u32(fp, s->count);
    ok &= write_u32(fp, 0); /* reserved */

    for (uint32_t i = 0; i < s->count && ok; ++i) {
        const CEStorageEntry *e = &s->entries[i];
        ok &= write_u32(fp, e->canvas_id);
        ok &= write_u16(fp, e->slot);
        ok &= write_u16(fp, e->block_idx);
        uint8_t type_block[4] = { e->type, 0, 0, 0 };
        ok &= (fwrite(type_block, 1, 4, fp) == 4);
        ok &= (fwrite(&e->keyframe, sizeof(CEUnit), 1, fp) == 1);
        ok &= (fwrite(&e->delta,    sizeof(CEUnit), 1, fp) == 1);
    }
    fclose(fp);
    return ok;
}

int ce_storage_load(CEStorage *out, const char *path) {
    ce_storage_init(out, 16);
    if (!out || !path) return 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;

    uint32_t magic = 0, version = 0, count = 0, reserved = 0;
    if (!read_u32(fp, &magic)   || magic != CE_STORAGE_MAGIC ||
        !read_u32(fp, &version) ||
        !read_u32(fp, &count)   ||
        !read_u32(fp, &reserved)) {
        fclose(fp);
        return 0;
    }
    /* Accept current and previous versions. */
    if (version != CE_STORAGE_VERSION && version != CE_STORAGE_VERSION_V1) {
        fclose(fp);
        return 0;
    }

    int ok = 1;
    for (uint32_t i = 0; i < count && ok; ++i) {
        CEStorageEntry e; memset(&e, 0, sizeof(e));
        ok &= read_u32(fp, &e.canvas_id);
        ok &= read_u16(fp, &e.slot);
        ok &= read_u16(fp, &e.block_idx);
        if (version >= CE_STORAGE_VERSION) {
            uint8_t type_block[4];
            ok &= (fread(type_block, 1, 4, fp) == 4);
            if (ok) e.type = type_block[0];
        } else {
            /* v1 entries pre-date the type tag — treat as TEXT. */
            e.type = (uint8_t)CE_TYPE_TEXT;
        }
        ok &= (fread(&e.keyframe, sizeof(CEUnit), 1, fp) == 1);
        ok &= (fread(&e.delta,    sizeof(CEUnit), 1, fp) == 1);
        if (ok) ce_storage_add_typed(out, e.canvas_id, e.slot, e.block_idx,
                                     (CEType)e.type, &e.keyframe, &e.delta);
    }
    fclose(fp);
    return ok;
}
