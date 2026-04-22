#ifndef SPATIAL_MORPHEME_H
#define SPATIAL_MORPHEME_H

#include <stdint.h>

/* Part-of-speech tags */
typedef enum {
    POS_NOUN,
    POS_VERB,
    POS_ADJ,
    POS_PARTICLE,
    POS_ENDING,
    POS_PUNCT,
    POS_UNKNOWN
} PartOfSpeech;

/* A single morpheme token */
typedef struct {
    PartOfSpeech pos;
    char token[64];   /* UTF-8 token string */
} Morpheme;

/* Initialize the morpheme analyzer (load built-in dictionaries).
   Call once at startup. */
void morpheme_init(void);

/* Analyze a single word (eojeol) into morphemes.
   word: UTF-8 Korean word (no spaces)
   out: array to receive morphemes
   max: max number of morphemes to output
   Returns: number of morphemes produced */
uint32_t morpheme_analyze(const char* word, Morpheme* out, uint32_t max);

/* Tokenize a full clause by spaces, then analyze each word.
   clause: UTF-8 Korean clause
   out: array to receive all morphemes
   max: max morphemes
   Returns: total morphemes */
uint32_t morpheme_tokenize_clause(const char* clause, Morpheme* out, uint32_t max);

/* Get POS name string */
const char* pos_name(PartOfSpeech pos);

#endif /* SPATIAL_MORPHEME_H */
