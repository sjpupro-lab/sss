#include "spatial_morpheme.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { tests_total++; printf("  [TEST] %s ... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)

static void test_noun_particle(void) {
    TEST("noun + particle: goyangi-ga");
    morpheme_init();
    Morpheme m[8];
    /* 고양이가 */
    uint32_t n = morpheme_analyze(
        "\xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80", m, 8);
    assert(n >= 2);
    assert(m[0].pos == POS_NOUN);
    assert(m[1].pos == POS_PARTICLE);
    PASS();
}

static void test_verb_ending(void) {
    TEST("verb + ending: meok-neunda");
    Morpheme m[8];
    /* 먹는다 */
    uint32_t n = morpheme_analyze(
        "\xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4", m, 8);
    assert(n >= 2);
    assert(m[0].pos == POS_VERB);
    assert(m[1].pos == POS_ENDING);
    PASS();
}

static void test_adjective(void) {
    TEST("adjective: gwiyeoun");
    Morpheme m[8];
    /* 귀여운 */
    uint32_t n = morpheme_analyze(
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4", m, 8);
    assert(n >= 1);
    assert(m[0].pos == POS_ADJ);
    PASS();
}

static void test_punct(void) {
    TEST("verb + ending + punct: meok-neunda.");
    Morpheme m[8];
    /* 먹는다. */
    uint32_t n = morpheme_analyze(
        "\xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.", m, 8);
    assert(n >= 3);
    /* Should have verb, ending, punct */
    int has_punct = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (m[i].pos == POS_PUNCT) has_punct = 1;
    }
    assert(has_punct);
    PASS();
}

static void test_clause_tokenize(void) {
    TEST("clause tokenize: full sentence");
    Morpheme m[32];
    /* "귀여운 고양이가 밥을 먹는다." */
    uint32_t n = morpheme_tokenize_clause(
        "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4 "
        "\xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4\xea\xb0\x80 "
        "\xeb\xb0\xa5\xec\x9d\x84 "
        "\xeb\xa8\xb9\xeb\x8a\x94\xeb\x8b\xa4.",
        m, 32);
    assert(n >= 6); /* adj + noun + particle + noun + particle + verb + ending + punct */

    /* Print for debugging */
    for (uint32_t i = 0; i < n; i++) {
        printf("\n    [%s] %s", pos_name(m[i].pos), m[i].token);
    }
    printf("\n");
    PASS();
}

int main(void) {
    printf("=== test_morpheme ===\n");

    test_noun_particle();
    test_verb_ending();
    test_adjective();
    test_punct();
    test_clause_tokenize();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
