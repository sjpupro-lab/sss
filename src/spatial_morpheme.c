#include "spatial_morpheme.h"
#include <string.h>
#include <stdio.h>

/* ── Built-in dictionaries (embedded, no file I/O needed) ── */

/* Nouns */
static const char* dict_nouns[] = {
    /* Animals */
    "\xea\xb3\xa0\xec\x96\x91\xec\x9d\xb4",       /* 고양이 */
    "\xea\xb0\x95\xec\x95\x84\xec\xa7\x80",         /* 강아지 */
    "\xeb\xac\xbc\xea\xb3\xa0\xea\xb8\xb0",         /* 물고기 */
    "\xeb\x8f\x8c\xea\xb3\xa0\xeb\x9e\x98",         /* 돌고래 */
    "\xed\x98\xb8\xeb\x9e\x91\xec\x9d\xb4",         /* 호랑이 */
    "\xec\xbd\x94\xeb\x81\xbc\xeb\xa6\xac",         /* 코끼리 */
    "\xec\x84\xa0\xec\x83\x9d\xeb\x8b\x98",         /* 선생님 */
    "\xec\xa7\x80\xed\x95\x98\xec\xb2\xa0",         /* 지하철 */
    "\xec\x9e\x90\xeb\x8f\x99\xec\xb0\xa8",         /* 자동차 */
    "\xec\xbb\xb4\xed\x93\xa8\xed\x84\xb0",         /* 컴퓨터 */
    "\xeb\xb9\x84\xed\x96\x89\xea\xb8\xb0",         /* 비행기 */
    "\xed\x8e\xad\xea\xb7\x84",                     /* 펭귄 */
    "\xea\xb3\xa0\xeb\x9e\x98",                     /* 고래 */
    "\xed\x86\xa0\xeb\x81\xbc",                     /* 토끼 */
    "\xec\x82\xac\xec\x9e\x90",                     /* 사자 */
    "\xec\x82\xac\xeb\x9e\x8c",                     /* 사람 */
    "\xec\x82\xac\xeb\x9e\x91",                     /* 사랑 */
    "\xec\x82\xac\xea\xb3\xbc",                     /* 사과 */
    "\xec\x83\x9d\xec\x84\xa0",                     /* 생선 */
    "\xec\x83\x9d\xea\xb0\x81",                     /* 생각 */
    "\xea\xb8\xb0\xec\x96\xb5",                     /* 기억 */
    "\xed\x9d\xac\xeb\xa7\x9d",                     /* 희망 */
    "\xed\x96\x89\xeb\xb3\xb5",                     /* 행복 */
    "\xec\x8a\xac\xed\x94\x94",                     /* 슬픔 */
    "\xec\x8b\x9c\xea\xb0\x84",                     /* 시간 */
    "\xec\x84\xb8\xec\x9b\x94",                     /* 세월 */
    "\xec\x98\xa4\xeb\x8a\x98",                     /* 오늘 */
    "\xeb\x82\xb4\xec\x9d\xbc",                     /* 내일 */
    "\xec\x96\xb4\xec\xa0\x9c",                     /* 어제 */
    "\xec\x95\x84\xec\xb9\xa8",                     /* 아침 */
    "\xec\xa0\x80\xeb\x85\x81",                     /* 저녁 */
    "\xec\x9d\x8c\xec\x8b\x9d",                     /* 음식 */
    "\xea\xb0\x84\xec\x8b\x9d",                     /* 간식 */
    "\xeb\x9d\xbc\xeb\xa9\xb4",                     /* 라면 */
    "\xec\xb0\xbd\xeb\xac\xb8",                     /* 창문 */
    "\xec\x9d\x98\xec\x9e\x90",                     /* 의자 */
    "\xec\xb1\x85\xec\x83\x81",                     /* 책상 */
    "\xec\xa0\x84\xed\x99\x94",                     /* 전화 */
    "\xed\x95\x98\xeb\x8a\x98",                     /* 하늘 */
    "\xeb\x82\x98\xeb\xac\xb4",                     /* 나무 */
    "\xeb\xb0\x94\xeb\x9e\x8c",                     /* 바람 */
    "\xea\xb5\xac\xeb\xa6\x84",                     /* 구름 */
    "\xeb\x8f\x99\xec\x83\x9d",                     /* 동생 */
    "\xec\xb9\x9c\xea\xb5\xac",                     /* 친구 */
    "\xec\x97\x84\xeb\xa7\x88",                     /* 엄마 */
    "\xec\x95\x84\xeb\xb9\xa0",                     /* 아빠 */
    "\xed\x95\x99\xec\x83\x9d",                     /* 학생 */
    "\xec\x96\xb4\xeb\xa5\xb8",                     /* 어른 */
    "\xeb\xa7\x88\xec\x9d\x8c",                     /* 마음 */
    "\xec\x95\x84\xec\x9d\xb4",                     /* 아이 */
    "\xec\x95\x84\xea\xb8\xb0",                     /* 아기 */
    "\xeb\xb0\x94\xeb\x8b\xa4",                     /* 바다 */
    "\xea\xb3\xbc\xec\x9d\xbc",                     /* 과일 */
    "\xec\xbb\xa4\xed\x94\xbc",                     /* 커피 */
    "\xeb\xb2\x84\xec\x8a\xa4",                     /* 버스 */
    "\xec\x9a\xb0\xeb\xa6\xac",                     /* 우리 */
    "\xec\x9a\xb0\xec\x9c\xa0",                     /* 우유 */
    "\xea\xb3\xa0\xea\xb8\xb0",                     /* 고기 */
    "\xea\xb0\x9c",                                   /* 개 */
    "\xec\x83\x88",                                   /* 새 */
    "\xea\xb3\xb0",                                   /* 곰 */
    "\xeb\x8b\xad",                                   /* 닭 */
    "\xec\x98\xa4\xeb\xa6\xac",                     /* 오리 */
    "\xeb\xb0\xa5",                                   /* 밥 */
    "\xeb\xac\xbc",                                   /* 물 */
    "\xeb\xb9\xb5",                                   /* 빵 */
    "\xea\xb5\xad",                                   /* 국 */
    "\xeb\x96\xa1",                                   /* 떡 */
    "\xea\xb7\xa4",                                   /* 귤 */
    "\xeb\xb0\xb0",                                   /* 배 */
    "\xec\xb0\xa8",                                   /* 차 */
    "\xec\xa7\x91",                                   /* 집 */
    "\xeb\xb0\xa9",                                   /* 방 */
    "\xeb\xac\xb8",                                   /* 문 */
    "\xec\xb1\x85",                                   /* 책 */
    "\xec\x82\xb0",                                   /* 산 */
    "\xea\xb0\x95",                                   /* 강 */
    "\xea\xbd\x83",                                   /* 꽃 */
    "\xed\x92\x80",                                   /* 풀 */
    "\xeb\xb9\x84",                                   /* 비 */
    "\xeb\x88\x88",                                   /* 눈 */
    "\xed\x95\xb4",                                   /* 해 */
    "\xeb\x8b\xac",                                   /* 달 */
    "\xeb\xb3\x84",                                   /* 별 */
    "\xed\x98\x95",                                   /* 형 */
    "\xeb\x88\x84\xeb\x82\x98",                     /* 누나 */
    "\xea\xbf\x88",                                   /* 꿈 */
    "\xeb\xb0\xa4",                                   /* 밤 */
    NULL
};

/* Verb stems */
static const char* dict_verbs[] = {
    "\xec\x9d\xbc\xec\x96\xb4\xeb\x82\x98",         /* 일어나 */
    "\xeb\xa7\x90\xed\x95\x98",                     /* 말하 */
    "\xeb\xa7\x8c\xeb\x93\xa4",                     /* 만들 */
    "\xeb\xaa\xa8\xeb\xa5\xb4",                     /* 모르 */
    "\xeb\x82\xb4\xeb\xa6\xac",                     /* 내리 */
    "\xec\x98\xac\xeb\xa6\xac",                     /* 올리 */
    "\xeb\xa7\x88\xec\x8b\x9c",                     /* 마시 */
    "\xeb\xa8\xb9",                                   /* 먹 */
    "\xea\xb0\x80",                                   /* 가 */
    "\xec\x98\xa4",                                   /* 오 */
    "\xeb\xb3\xb4",                                   /* 보 */
    "\xed\x95\x98",                                   /* 하 */
    "\xeb\x90\x98",                                   /* 되 */
    "\xec\xa3\xbc",                                   /* 주 */
    "\xeb\xb0\x9b",                                   /* 받 */
    "\xec\x93\xb0",                                   /* 쓰 */
    "\xec\x9d\xbd",                                   /* 읽 */
    "\xeb\x93\xa3",                                   /* 듣 */
    "\xea\xb1\xb7",                                   /* 걷 */
    "\xeb\x9b\xb0",                                   /* 뛰 */
    "\xec\x95\x89",                                   /* 앉 */
    "\xec\x84\x9c",                                   /* 서 */
    "\xec\x9e\x90",                                   /* 자 */
    "\xec\xb0\xbe",                                   /* 찾 */
    "\xec\x95\x8c",                                   /* 알 */
    "\xec\x82\xb4",                                   /* 살 */
    "\xec\xa3\xbd",                                   /* 죽 */
    "\xeb\x86\x80",                                   /* 놀 */
    "\xec\x9a\xb8",                                   /* 울 */
    "\xec\x9b\x83",                                   /* 웃 */
    "\xec\x82\xac",                                   /* 사 */
    "\xed\x8c\x94",                                   /* 팔 */
    "\xec\x97\xb4",                                   /* 열 */
    "\xeb\x8b\xab",                                   /* 닫 */
    "\xeb\x84\xa3",                                   /* 넣 */
    "\xeb\xb9\xbc",                                   /* 빼 */
    "\xed\x83\x80",                                   /* 타 */
    "\xec\x9e\x85",                                   /* 입 */
    "\xeb\xb2\x97",                                   /* 벗 */
    NULL
};

/* Adjectives (prenominal forms) */
static const char* dict_adjectives[] = {
    "\xec\x95\x84\xeb\xa6\x84\xeb\x8b\xa4\xec\x9a\xb4", /* 아름다운 */
    "\xea\xb7\x80\xec\x97\xac\xec\x9a\xb4",             /* 귀여운 */
    "\xec\x96\xb4\xeb\x91\x90\xec\x9a\xb4",             /* 어두운 */
    "\xec\x98\x88\xec\x81\x9c",                         /* 예쁜 */
    "\xeb\xa9\x8b\xec\xa7\x84",                         /* 멋진 */
    "\xec\xb0\xa9\xed\x95\x9c",                         /* 착한 */
    "\xec\xa2\x8b\xec\x9d\x80",                         /* 좋은 */
    "\xeb\x82\x98\xec\x81\x9c",                         /* 나쁜 */
    "\xec\x9e\x91\xec\x9d\x80",                         /* 작은 */
    "\xeb\x86\x92\xec\x9d\x80",                         /* 높은 */
    "\xeb\x82\xae\xec\x9d\x80",                         /* 낮은 */
    "\xec\xa7\xa7\xec\x9d\x80",                         /* 짧은 */
    "\xeb\xb9\xa0\xeb\xa5\xb8",                         /* 빠른 */
    "\xeb\x8a\x90\xeb\xa6\xb0",                         /* 느린 */
    "\xeb\xb0\x9d\xec\x9d\x80",                         /* 밝은 */
    "\xec\x98\xa4\xeb\x9e\x9c",                         /* 오랜 */
    "\xec\x8a\xac\xed\x94\x88",                         /* 슬픈 */
    "\xea\xb8\xb0\xec\x81\x9c",                         /* 기쁜 */
    "\xed\x81\xb0",                                       /* 큰 */
    "\xea\xb8\xb4",                                       /* 긴 */
    NULL
};

/* Particles (longest first for longest-match) */
static const char* dict_particles[] = {
    "\xec\x97\x90\xec\x84\x9c\xeb\x8a\x94",   /* 에서는 */
    "\xec\x97\x90\xec\x84\x9c\xeb\x8f\x84",   /* 에서도 */
    "\xec\x9c\xbc\xeb\xa1\x9c\xeb\x8a\x94",   /* 으로는 */
    "\xec\x9c\xbc\xeb\xa1\x9c\xeb\x8f\x84",   /* 으로도 */
    "\xec\x97\x90\xea\xb2\x8c\xec\x84\x9c",   /* 에게서 */
    "\xec\x97\x90\xec\x84\x9c",               /* 에서 */
    "\xec\x9c\xbc\xeb\xa1\x9c",               /* 으로 */
    "\xec\x97\x90\xea\xb2\x8c",               /* 에게 */
    "\xea\xb9\x8c\xec\xa7\x80",               /* 까지 */
    "\xeb\xb6\x80\xed\x84\xb0",               /* 부터 */
    "\xec\xb2\x98\xeb\x9f\xbc",               /* 처럼 */
    "\xeb\xa7\x8c\xed\x81\xbc",               /* 만큼 */
    "\xec\x9d\x80",                             /* 은 */
    "\xeb\x8a\x94",                             /* 는 */
    "\xec\x9d\xb4",                             /* 이 */
    "\xea\xb0\x80",                             /* 가 */
    "\xec\x9d\x84",                             /* 을 */
    "\xeb\xa5\xbc",                             /* 를 */
    "\xec\x97\x90",                             /* 에 */
    "\xec\x9d\x98",                             /* 의 */
    "\xec\x99\x80",                             /* 와 */
    "\xea\xb3\xbc",                             /* 과 */
    "\xeb\x8f\x84",                             /* 도 */
    "\xeb\xa1\x9c",                             /* 로 */
    "\xec\x84\x9c",                             /* 서 */
    "\xeb\xa7\x8c",                             /* 만 */
    NULL
};

/* Endings (longest first) */
static const char* dict_endings[] __attribute__((used)) = {
    "\xeb\x8a\x94\xeb\x8b\xa4",   /* 는다 */
    "\xe3\x84\xb4\xeb\x8b\xa4",   /* ㄴ다 */
    "\xec\x97\x88\xeb\x8b\xa4",   /* 었다 */
    "\xec\x95\x98\xeb\x8b\xa4",   /* 았다 */
    "\xea\xb2\xa0\xeb\x8b\xa4",   /* 겠다 */
    "\xed\x95\x9c\xeb\x8b\xa4",   /* 한다 */
    "\xeb\x8a\x94",               /* 는 */
    "\xec\x9d\x80",               /* 은 */
    "\xec\x9d\x84",               /* 을 */
    "\xe3\x84\xb4",               /* ㄴ */
    "\xe3\x84\xb9",               /* ㄹ */
    "\xea\xb3\xa0",               /* 고 */
    "\xeb\xa9\xb0",               /* 며 */
    "\xeb\xa9\xb4",               /* 면 */
    "\xec\xa7\x80",               /* 지 */
    "\xea\xb2\x8c",               /* 게 */
    "\xec\x84\x9c",               /* 서 */
    "\xeb\x8b\x88",               /* 니 */
    "\xec\x9e\x90",               /* 자 */
    "\xeb\x8b\xa4",               /* 다 */
    NULL
};

/* Punctuation characters */
static const char punct_chars[] = ".!?,;:~";

/* ── Helpers ── */

/* Check if str starts with prefix */
static int starts_with(const char* str, const char* prefix) {
    while (*prefix) {
        if (*str != *prefix) return 0;
        str++;
        prefix++;
    }
    return 1;
}

/* Find longest match in a dictionary. Returns byte length of match, 0 if none. */
static uint32_t find_longest_match(const char* text, const char** dict, int* match_idx) {
    uint32_t best_len = 0;
    int best_idx = -1;

    for (int i = 0; dict[i] != NULL; i++) {
        uint32_t dlen = (uint32_t)strlen(dict[i]);
        if (dlen > best_len && starts_with(text, dict[i])) {
            best_len = dlen;
            best_idx = i;
        }
    }

    if (match_idx) *match_idx = best_idx;
    return best_len;
}

/* Try to strip a particle from the end of text[0..len).
   Returns byte length of the particle, or 0. */
static uint32_t strip_particle_suffix(const char* text, uint32_t len) {
    /* Try longest particles first */
    for (int i = 0; dict_particles[i] != NULL; i++) {
        uint32_t plen = (uint32_t)strlen(dict_particles[i]);
        if (plen <= len && memcmp(text + len - plen, dict_particles[i], plen) == 0) {
            return plen;
        }
    }
    return 0;
}

const char* pos_name(PartOfSpeech pos) {
    switch (pos) {
        case POS_NOUN:     return "NOUN";
        case POS_VERB:     return "VERB";
        case POS_ADJ:      return "ADJ";
        case POS_PARTICLE: return "PARTICLE";
        case POS_ENDING:   return "ENDING";
        case POS_PUNCT:    return "PUNCT";
        case POS_UNKNOWN:  return "UNKNOWN";
    }
    return "?";
}

void morpheme_init(void) {
    /* Dictionaries are embedded and there is no I/O to perform.
     * The guard is kept so a future switch to loaded dictionaries
     * (e.g. runtime .txt reload) only runs once per process. */
    static int ready = 0;
    if (ready) return;
    ready = 1;
}

uint32_t morpheme_analyze(const char* word, Morpheme* out, uint32_t max) {
    if (!word || !out || max == 0) return 0;

    uint32_t count = 0;
    uint32_t wlen = (uint32_t)strlen(word);
    const char* p = word;

    /* 1. Strip leading punctuation */
    while (*p && strchr(punct_chars, *p)) {
        if (count >= max) return count;
        out[count].pos = POS_PUNCT;
        out[count].token[0] = *p;
        out[count].token[1] = '\0';
        count++;
        p++;
        wlen--;
    }

    /* Strip trailing punctuation */
    char clean[256];
    uint32_t clean_len = 0;
    char trailing_punct[32];
    uint32_t trailing_count = 0;

    /* Copy word without trailing punct */
    const char* end = p + wlen;
    while (end > p) {
        /* Check if last byte is ASCII punct */
        char last = *(end - 1);
        if (strchr(punct_chars, last)) {
            trailing_punct[trailing_count++] = last;
            end--;
        } else {
            break;
        }
    }

    clean_len = (uint32_t)(end - p);
    if (clean_len > 0 && clean_len < sizeof(clean)) {
        memcpy(clean, p, clean_len);
        clean[clean_len] = '\0';
    } else {
        clean[0] = '\0';
        clean_len = 0;
    }

    if (clean_len == 0) goto add_trailing;

    /* 2. Try adjective (prenominal) match */
    {
        uint32_t alen = find_longest_match(clean, dict_adjectives, NULL);
        if (alen > 0 && alen == clean_len) {
            if (count < max) {
                out[count].pos = POS_ADJ;
                memcpy(out[count].token, clean, clean_len);
                out[count].token[clean_len] = '\0';
                count++;
            }
            goto add_trailing;
        }
    }

    /* 3. Try noun match (longest) */
    {
        uint32_t nlen = find_longest_match(clean, dict_nouns, NULL);
        if (nlen > 0) {
            if (count < max) {
                out[count].pos = POS_NOUN;
                memcpy(out[count].token, clean, nlen);
                out[count].token[nlen] = '\0';
                count++;
            }
            /* Remainder is particle */
            if (nlen < clean_len && count < max) {
                uint32_t rem = clean_len - nlen;
                out[count].pos = POS_PARTICLE;
                memcpy(out[count].token, clean + nlen, rem);
                out[count].token[rem] = '\0';
                count++;
            }
            goto add_trailing;
        }
    }

    /* 4. Try adjective match (partial, with ending) */
    {
        uint32_t alen = find_longest_match(clean, dict_adjectives, NULL);
        if (alen > 0) {
            if (count < max) {
                out[count].pos = POS_ADJ;
                memcpy(out[count].token, clean, alen);
                out[count].token[alen] = '\0';
                count++;
            }
            if (alen < clean_len && count < max) {
                uint32_t rem = clean_len - alen;
                out[count].pos = POS_ENDING;
                memcpy(out[count].token, clean + alen, rem);
                out[count].token[rem] = '\0';
                count++;
            }
            goto add_trailing;
        }
    }

    /* 5. Try verb stem match */
    {
        uint32_t vlen = find_longest_match(clean, dict_verbs, NULL);
        if (vlen > 0) {
            if (count < max) {
                out[count].pos = POS_VERB;
                memcpy(out[count].token, clean, vlen);
                out[count].token[vlen] = '\0';
                count++;
            }
            if (vlen < clean_len && count < max) {
                uint32_t rem = clean_len - vlen;
                out[count].pos = POS_ENDING;
                memcpy(out[count].token, clean + vlen, rem);
                out[count].token[rem] = '\0';
                count++;
            }
            goto add_trailing;
        }
    }

    /* 6. Unknown word: try to strip particle from end */
    {
        uint32_t plen = strip_particle_suffix(clean, clean_len);
        if (plen > 0 && plen < clean_len) {
            uint32_t stem_len = clean_len - plen;
            if (count < max) {
                out[count].pos = POS_UNKNOWN;
                memcpy(out[count].token, clean, stem_len);
                out[count].token[stem_len] = '\0';
                count++;
            }
            if (count < max) {
                out[count].pos = POS_PARTICLE;
                memcpy(out[count].token, clean + stem_len, plen);
                out[count].token[plen] = '\0';
                count++;
            }
        } else {
            /* Fully unknown */
            if (count < max) {
                out[count].pos = POS_UNKNOWN;
                memcpy(out[count].token, clean, clean_len);
                out[count].token[clean_len] = '\0';
                count++;
            }
        }
    }

add_trailing:
    /* Add trailing punctuation */
    for (uint32_t i = 0; i < trailing_count && count < max; i++) {
        /* Reverse order (was collected backwards) */
        out[count].pos = POS_PUNCT;
        out[count].token[0] = trailing_punct[trailing_count - 1 - i];
        out[count].token[1] = '\0';
        count++;
    }

    return count;
}

uint32_t morpheme_tokenize_clause(const char* clause, Morpheme* out, uint32_t max) {
    if (!clause || !out || max == 0) return 0;

    uint32_t total = 0;
    char buf[256];
    const char* p = clause;

    /* Skip leading whitespace */
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;

    while (*p && total < max) {
        /* Extract next word (space-separated) */
        uint32_t wi = 0;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r' && wi < sizeof(buf) - 1) {
            buf[wi++] = *p++;
        }
        buf[wi] = '\0';

        if (wi > 0) {
            uint32_t n = morpheme_analyze(buf, out + total, max - total);
            total += n;
        }

        /* Skip whitespace */
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    }

    return total;
}
