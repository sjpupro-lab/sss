#ifndef BENCH_UTF8_H
#define BENCH_UTF8_H

/*
 * UTF-8 console initialization for benchmarks.
 * Ensures Korean/Japanese/Chinese output renders correctly on Windows.
 * Also disables stdout buffering so per-line progress prints don't stall.
 */

#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#include <fcntl.h>
#include <io.h>
static inline void utf8_console_init(void) {
    SetConsoleOutputCP(65001);   /* 65001 = UTF-8 code page */
    SetConsoleCP(65001);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
}
#else
static inline void utf8_console_init(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
}
#endif

#endif /* BENCH_UTF8_H */
