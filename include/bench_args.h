#ifndef BENCH_ARGS_H
#define BENCH_ARGS_H

#include <string.h>

/*
 * Tiny shared CLI-flag parser for benchmark binaries.
 *
 * Recognizes:
 *   --save <path>         persist model after training
 *   --load <path>         load existing model, continue training
 *   --load-only <path>    load existing model, skip training
 *
 * Everything else is treated as a positional argument, preserving
 * order. Typical usage:
 *
 *   ./build/test_wiki data/sample_ko.txt 500 --save model_ko.spai
 *   ./build/test_wiki data/sample_en.txt --load model_ko.spai --save mixed.spai
 *   ./build/bench_qa  data/qa.tsv --load-only mixed.spai
 */

#define BENCH_MAX_POSITIONAL 8

typedef struct {
    const char* positional[BENCH_MAX_POSITIONAL];
    int         positional_count;
    const char* save_path;
    const char* load_path;
    int         load_only;   /* 0 or 1 */
    int         cascade;     /* 0 or 1 — enables cascade matching mode */
} BenchArgs;

static inline void bench_args_init(BenchArgs* a) {
    a->positional_count = 0;
    a->save_path = NULL;
    a->load_path = NULL;
    a->load_only = 0;
    a->cascade   = 0;
}

/* Returns 0 on success, -1 on bad arguments (missing value after a flag
   that expects one). */
static inline int bench_parse_args(int argc, char** argv, BenchArgs* a) {
    bench_args_init(a);
    int i = 1;
    while (i < argc) {
        const char* cur = argv[i];
        if (strcmp(cur, "--save") == 0) {
            if (i + 1 >= argc) return -1;
            a->save_path = argv[++i];
        } else if (strcmp(cur, "--load") == 0) {
            if (i + 1 >= argc) return -1;
            a->load_path = argv[++i];
            a->load_only = 0;
        } else if (strcmp(cur, "--load-only") == 0) {
            if (i + 1 >= argc) return -1;
            a->load_path = argv[++i];
            a->load_only = 1;
        } else if (strcmp(cur, "--cascade") == 0) {
            a->cascade = 1;
        } else if (cur[0] == '-' && cur[1] == '-') {
            /* Unknown long flag — skip it and its value if present */
            if (i + 1 < argc && argv[i + 1][0] != '-') i++;
        } else {
            if (a->positional_count < BENCH_MAX_POSITIONAL)
                a->positional[a->positional_count++] = cur;
        }
        i++;
    }
    return 0;
}

#endif /* BENCH_ARGS_H */
