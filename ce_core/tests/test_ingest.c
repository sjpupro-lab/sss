/* test_ingest.c — directory ingest + storage save/load.
 *
 * Builds a few synthetic 16x16 TGA images (top-left origin, 24-bit BGR) in
 * a temp directory, ingests them into a CEStorage, saves to disk, reloads,
 * and verifies bit-exact round-trip.
 */
#include "../ce_core.h"
#include "../ce_storage.h"
#include "../ce_storage_io.h"
#include "../ce_ingest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else         { printf("  PASS: %s\n", msg); } \
} while(0)

/* Write a 16x16 24-bit TGA at `path`. `colourise(x, y, &r, &g, &b)` decides
 * the pixel value at each location. Returns 1 on success. */
typedef void (*pixel_fn)(int x, int y, uint8_t *r, uint8_t *g, uint8_t *b);

static int write_tga_16(const char *path, pixel_fn fn) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    uint8_t hdr[18] = {0};
    hdr[2]  = 2;     /* uncompressed truecolor */
    hdr[12] = 16; hdr[13] = 0;   /* width  = 16 */
    hdr[14] = 16; hdr[15] = 0;   /* height = 16 */
    hdr[16] = 24;    /* pixel depth */
    hdr[17] = 0x20;  /* top-left origin */
    if (fwrite(hdr, 1, 18, fp) != 18) { fclose(fp); return 0; }
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            uint8_t r, g, b;
            fn(x, y, &r, &g, &b);
            uint8_t bgr[3] = { b, g, r };
            if (fwrite(bgr, 1, 3, fp) != 3) { fclose(fp); return 0; }
        }
    }
    fclose(fp);
    return 1;
}

/* Pixel painters for our synthetic fixtures. */
static void p_red   (int x, int y, uint8_t *r, uint8_t *g, uint8_t *b) {
    (void)x; (void)y; *r = 255; *g = 0;   *b = 0;
}
static void p_grad  (int x, int y, uint8_t *r, uint8_t *g, uint8_t *b) {
    *r = (uint8_t)(x * 16); *g = (uint8_t)(y * 16); *b = (uint8_t)((x + y) * 8);
}
static void p_check (int x, int y, uint8_t *r, uint8_t *g, uint8_t *b) {
    int v = ((x / 4) + (y / 4)) & 1;
    *r = v ? 255 : 0; *g = *r; *b = *r;
}

int main(void) {
    printf("=== ce_ingest tests ===\n");

    /* Create a unique temp dir under /tmp. */
    char dir[256];
    snprintf(dir, sizeof(dir), "/tmp/ce_ingest_test_%d", (int)getpid());
    /* clean stale dir */
    char rm[512]; snprintf(rm, sizeof(rm), "rm -rf %s", dir);
    int rc = system(rm); (void)rc;
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) {
        printf("  FAIL: mkdir %s\n", dir);
        return 1;
    }

    /* Write three synthetic 16x16 images. */
    char p1[400], p2[400], p3[400], p_bad[400];
    snprintf(p1, sizeof(p1), "%s/red.tga", dir);
    snprintf(p2, sizeof(p2), "%s/grad.tga", dir);
    snprintf(p3, sizeof(p3), "%s/check.tga", dir);
    snprintf(p_bad, sizeof(p_bad), "%s/notes.txt", dir);
    CHECK(write_tga_16(p1, p_red),   "write fixture red.tga");
    CHECK(write_tga_16(p2, p_grad),  "write fixture grad.tga");
    CHECK(write_tga_16(p3, p_check), "write fixture check.tga");
    /* non-image file that should be ignored */
    FILE *fp = fopen(p_bad, "w");
    if (fp) { fputs("not an image", fp); fclose(fp); }

    /* --- Single-file ingest --- */
    CEStorage s1; ce_storage_init(&s1, 8);
    CEIngestStats st1 = {0,0,0,0};
    int ok = ce_ingest_file(&s1, p1, &st1);
    CHECK(ok == 1, "ce_ingest_file returns 1 on success");
    /* 16x16 / 8x8 = 2x2 = 4 blocks per image */
    CHECK(s1.count == 4, "16x16 image -> 4 blocks (8x8 each)");
    CHECK(st1.images_decoded == 1, "stats: images_decoded == 1");
    CHECK(st1.blocks_added == 4, "stats: blocks_added == 4");
    CHECK(st1.errors == 0, "stats: no errors");
    /* All blocks of a constant-red image should be identical keyframes. */
    int identical = 1;
    for (uint32_t i = 1; i < s1.count; ++i) {
        if (!ce_equal(&s1.entries[0].keyframe, &s1.entries[i].keyframe)) {
            identical = 0; break;
        }
    }
    CHECK(identical, "constant-color image -> identical keyframes across blocks");
    ce_storage_free(&s1);

    /* --- Directory ingest --- */
    CEStorage s; ce_storage_init(&s, 32);
    CEIngestStats st = {0,0,0,0};
    int n = ce_ingest_directory(&s, dir, &st);
    CHECK(n == 3, "ingest 3 .tga files (text file ignored)");
    CHECK(s.count == 12, "3 images x 4 blocks = 12 entries");
    CHECK(st.errors == 0, "directory walk: no decode errors");

    /* canvas_id must differ across files (path-hash based). */
    uint32_t ids[3];
    int seen = 0;
    for (uint32_t i = 0; i < s.count && seen < 3; ++i) {
        int dup = 0;
        for (int j = 0; j < seen; ++j) if (ids[j] == s.entries[i].canvas_id) { dup = 1; break; }
        if (!dup) ids[seen++] = s.entries[i].canvas_id;
    }
    CHECK(seen == 3, "3 distinct canvas_ids (one per image)");

    /* slot/block layout: each image must have slot 0/1 x block 0/1. */
    int slot_ok = 1;
    for (uint32_t i = 0; i < s.count; ++i) {
        if (s.entries[i].slot > 1 || s.entries[i].block_idx > 1) { slot_ok = 0; break; }
    }
    CHECK(slot_ok, "slot,block in {0,1} x {0,1} for 16x16 inputs");

    /* --- Save / Load roundtrip --- */
    char out[400]; snprintf(out, sizeof(out), "%s/storage.ces", dir);
    int saved = ce_storage_save(&s, out);
    CHECK(saved == 1, "ce_storage_save returns 1");

    CEStorage loaded;
    int loaded_ok = ce_storage_load(&loaded, out);
    CHECK(loaded_ok == 1, "ce_storage_load returns 1");
    CHECK(loaded.count == s.count, "loaded count matches saved count");

    int eq = 1;
    for (uint32_t i = 0; i < s.count && eq; ++i) {
        const CEStorageEntry *a = &s.entries[i];
        const CEStorageEntry *b = &loaded.entries[i];
        if (a->canvas_id != b->canvas_id) eq = 0;
        else if (a->slot != b->slot)      eq = 0;
        else if (a->block_idx != b->block_idx) eq = 0;
        else if (!ce_equal(&a->keyframe, &b->keyframe)) eq = 0;
        else if (!ce_equal(&a->delta,    &b->delta))    eq = 0;
    }
    CHECK(eq, "save/load round-trip is byte-exact");

    /* --- Negative cases --- */
    CEStorage bad;
    int bad_load = ce_storage_load(&bad, "/no/such/file.ces");
    CHECK(bad_load == 0 && bad.count == 0, "load missing file -> 0, empty storage");
    ce_storage_free(&bad);

    /* corrupted file: write garbage, expect load failure. */
    char bad_path[400]; snprintf(bad_path, sizeof(bad_path), "%s/bad.ces", dir);
    FILE *bf = fopen(bad_path, "wb");
    if (bf) { fputs("not a real ces file", bf); fclose(bf); }
    CEStorage bad2;
    int bad_load2 = ce_storage_load(&bad2, bad_path);
    CHECK(bad_load2 == 0, "load garbage file -> 0");
    ce_storage_free(&bad2);

    /* --- Hash determinism --- */
    uint32_t h1 = ce_path_hash("hello/world.png");
    uint32_t h2 = ce_path_hash("hello/world.png");
    uint32_t h3 = ce_path_hash("hello/world.jpg");
    CHECK(h1 == h2, "ce_path_hash deterministic");
    CHECK(h1 != h3, "ce_path_hash sensitive to extension");

    ce_storage_free(&s);
    ce_storage_free(&loaded);

    /* cleanup */
    rc = system(rm); (void)rc;

    if (fails) { printf("=== %d FAIL ===\n", fails); return 1; }
    printf("=== ALL PASS ===\n");
    return 0;
}
