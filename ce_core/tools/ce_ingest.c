/* ce_ingest — CLI: directory of PNG/JPG -> .ces storage file.
 *
 * Usage:
 *   ce_ingest <dir> -o <out.ces> [--recursive]
 *   ce_ingest --info <file.ces>          # print summary of a saved storage
 */
#include "../ce_core.h"
#include "../ce_storage.h"
#include "../ce_storage_io.h"
#include "../ce_ingest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(void) {
    fprintf(stderr,
        "usage:\n"
        "  ce_ingest <dir> -o <out.ces> [--recursive]\n"
        "  ce_ingest --info <file.ces>\n");
}

static int cmd_info(const char *path) {
    CEStorage s;
    if (!ce_storage_load(&s, path)) {
        fprintf(stderr, "error: failed to load %s\n", path);
        return 1;
    }
    printf("file        : %s\n", path);
    printf("entries     : %u\n", s.count);
    printf("bytes/entry : %zu\n", sizeof(uint32_t) + sizeof(uint16_t) * 2 + sizeof(CEUnit) * 2);

    /* Group by canvas_id to count distinct images. */
    uint32_t canvases = 0;
    uint32_t prev_id = 0;
    int prev_set = 0;
    /* We do a simple O(n^2) distinct count for clarity (storage usually <100k). */
    for (uint32_t i = 0; i < s.count; ++i) {
        uint32_t id = s.entries[i].canvas_id;
        int seen = 0;
        for (uint32_t j = 0; j < i; ++j) {
            if (s.entries[j].canvas_id == id) { seen = 1; break; }
        }
        if (!seen) ++canvases;
        prev_id = id; prev_set = 1;
    }
    (void)prev_id; (void)prev_set;
    printf("images      : %u (distinct canvas_id)\n", canvases);
    if (s.count > 0) {
        uint16_t max_slot = 0, max_block = 0;
        for (uint32_t i = 0; i < s.count; ++i) {
            if (s.entries[i].slot > max_slot) max_slot = s.entries[i].slot;
            if (s.entries[i].block_idx > max_block) max_block = s.entries[i].block_idx;
        }
        printf("max slot    : %u  (block_y, => image height %u-%u px)\n",
               max_slot, max_slot * 8, (max_slot + 1) * 8);
        printf("max block   : %u  (block_x, => image width  %u-%u px)\n",
               max_block, max_block * 8, (max_block + 1) * 8);
    }
    ce_storage_free(&s);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 3 && strcmp(argv[1], "--info") == 0) {
        return cmd_info(argv[2]);
    }
    if (argc < 4) {
        print_usage();
        return 2;
    }

    const char *dir = argv[1];
    const char *out_path = NULL;
    int recursive = 0;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--recursive") == 0) {
            recursive = 1;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            print_usage();
            return 2;
        }
    }
    if (!out_path) { print_usage(); return 2; }

    CEStorage s;
    ce_storage_init(&s, 1024);

    CEIngestStats stats = { 0, 0, 0, 0 };
    int n = recursive
        ? ce_ingest_directory_recursive(&s, dir, &stats)
        : ce_ingest_directory(&s, dir, &stats);

    printf("ingested      : %d images from %s%s\n",
           n, dir, recursive ? " (recursive)" : "");
    printf("  seen        : %u\n", stats.images_seen);
    printf("  decoded     : %u\n", stats.images_decoded);
    printf("  blocks      : %u\n", stats.blocks_added);
    printf("  errors      : %u\n", stats.errors);

    if (s.count == 0) {
        fprintf(stderr, "warning: no blocks were added — output not written\n");
        ce_storage_free(&s);
        return 1;
    }

    if (!ce_storage_save(&s, out_path)) {
        fprintf(stderr, "error: failed to write %s\n", out_path);
        ce_storage_free(&s);
        return 1;
    }
    printf("saved         : %s (%u entries)\n", out_path, s.count);

    ce_storage_free(&s);
    return 0;
}
