/* sss_feature_bank.c — Phase 2 .sfb on-disk format I/O.
 *
 * Pack(1) + little-endian fixed. The build is statically asserted to
 * run on a little-endian host (every platform CE targets — x86_64,
 * aarch64, riscv64 default little — passes) so save/load is a pair of
 * fwrite/fread calls without any byte-swapping. Record sizes are
 * stored in the header so v2 can grow records by bumping
 * `*_record_size` without breaking v1 readers; this v1 implementation
 * accepts only the exact sizes it understands and rejects mismatches
 * with SFB_ERR_VERSION.
 */
#include "sss_feature_bank.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Static assertions on the wire layout ──────────────────────── */

/* Compile-time fail if struct packing slipped (e.g. a future
 * #pragma pack mismatch). 2864 / 20 / 104 are the canonical v1
 * record sizes; the Python side reads identical bytes. */
#ifdef __STDC_VERSION__
# if __STDC_VERSION__ >= 201112L
#  define SFB_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
# endif
#endif
#ifndef SFB_STATIC_ASSERT
# define SFB_STATIC_ASSERT(cond, msg) \
    typedef char sfb_static_assert_##__LINE__[(cond) ? 1 : -1]
#endif

/* v1 motif size — kept as a named constant so the load path can
 * accept legacy v1 files (motif_record_size = 2864 on disk) and
 * zero-pad them into the v2 SSSMotif struct in memory. */
#define SFB_V1_MOTIF_SIZE 2864

SFB_STATIC_ASSERT(sizeof(SSSFeatureBankHeader) == 40,
                  "SSSFeatureBankHeader must be 40 bytes");
SFB_STATIC_ASSERT(sizeof(SSSMotif)    == 3744,
                  "SSSMotif v2 must be 3744 bytes");
SFB_STATIC_ASSERT(sizeof(SSSRelation) == 20,
                  "SSSRelation must be 20 bytes");
SFB_STATIC_ASSERT(sizeof(SSSIdentity) == 104,
                  "SSSIdentity must be 104 bytes");
SFB_STATIC_ASSERT(sizeof(SSSCondition) == 1332,
                  "SSSCondition must be 1332 bytes");
SFB_STATIC_ASSERT(sizeof(SSSInteractionResponse) == 1808,
                  "SSSInteractionResponse must be 1808 bytes");

/* Little-endian host check. The .sfb format is fixed LE so callers
 * on a hypothetical big-endian host would read garbage; we'd rather
 * fail to compile than ship silently broken files. */
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
SFB_STATIC_ASSERT(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
                  ".sfb requires a little-endian host");
#endif

/* ── Validation helpers ────────────────────────────────────────── */

static int validate_relation(const SSSRelation *r, uint32_t motif_count)
{
    if (r->src_motif_id >= motif_count) return SFB_ERR_INVALID_REL;
    if (r->dst_motif_id >= motif_count) return SFB_ERR_INVALID_REL;
    if (r->relation_type > SFB_RELATION_TYPE_MAX) return SFB_ERR_INVALID_REL;
    return SFB_OK;
}

static int validate_identity(const SSSIdentity *id, uint32_t motif_count)
{
    if (id->motif_count < 1
        || id->motif_count > SFB_MAX_IDENTITY_MOTIFS) {
        return SFB_ERR_INVALID_IDENT;
    }
    for (uint8_t k = 0; k < id->motif_count; ++k) {
        if (id->motif_ids[k] >= motif_count) return SFB_ERR_INVALID_IDENT;
    }
    return SFB_OK;
}

static int validate_condition(const SSSCondition *c)
{
    if (c->condition_type > SFB_CONDITION_TYPE_MAX) {
        return SFB_ERR_INVALID_COND;
    }
    return SFB_OK;
}

static int validate_response(const SSSInteractionResponse *r,
                             uint32_t motif_count,
                             uint32_t condition_count)
{
    if (r->motif_id     >= motif_count)     return SFB_ERR_INVALID_RESP;
    if (r->condition_id >= condition_count) return SFB_ERR_INVALID_RESP;
    return SFB_OK;
}

/* ── Save ──────────────────────────────────────────────────────── */

int sss_feature_bank_save(const char *path, const SSSFeatureBank *bank)
{
    if (!path || !bank) return SFB_ERR_IO;

    /* Validate every relation / identity / condition / response
     * up-front so we don't write a partially-corrupt file on the
     * way to a late failure. */
    for (uint32_t i = 0; i < bank->relation_count; ++i) {
        int rc = validate_relation(&bank->relations[i], bank->motif_count);
        if (rc != SFB_OK) return rc;
    }
    for (uint32_t i = 0; i < bank->identity_count; ++i) {
        int rc = validate_identity(&bank->identities[i], bank->motif_count);
        if (rc != SFB_OK) return rc;
    }
    for (uint32_t i = 0; i < bank->condition_count; ++i) {
        int rc = validate_condition(&bank->conditions[i]);
        if (rc != SFB_OK) return rc;
    }
    for (uint32_t i = 0; i < bank->response_count; ++i) {
        int rc = validate_response(&bank->responses[i],
                                   bank->motif_count,
                                   bank->condition_count);
        if (rc != SFB_OK) return rc;
    }

    FILE *f = fopen(path, "wb");
    if (!f) return SFB_ERR_IO;

    SSSFeatureBankHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, SFB_MAGIC, 4);
    hdr.version              = SFB_VERSION;
    hdr.motif_count          = bank->motif_count;
    hdr.relation_count       = bank->relation_count;
    hdr.identity_count       = bank->identity_count;
    hdr.motif_record_size    = (uint16_t)sizeof(SSSMotif);
    hdr.relation_record_size = (uint16_t)sizeof(SSSRelation);
    hdr.identity_record_size = (uint16_t)sizeof(SSSIdentity);
    hdr.condition_count      = (uint16_t)bank->condition_count;
    hdr.response_count       = bank->response_count;
    /* reserved1 / reserved2 stay zero by memset above. */

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f); return SFB_ERR_IO;
    }
    if (bank->motif_count > 0
        && fwrite(bank->motifs, sizeof(SSSMotif),
                  bank->motif_count, f) != bank->motif_count) {
        fclose(f); return SFB_ERR_IO;
    }
    if (bank->relation_count > 0
        && fwrite(bank->relations, sizeof(SSSRelation),
                  bank->relation_count, f) != bank->relation_count) {
        fclose(f); return SFB_ERR_IO;
    }
    if (bank->identity_count > 0
        && fwrite(bank->identities, sizeof(SSSIdentity),
                  bank->identity_count, f) != bank->identity_count) {
        fclose(f); return SFB_ERR_IO;
    }
    /* v2: conditions follow identities, responses follow conditions.
     * Both arrays are pack(1) structs of fixed sizes, so a single
     * fwrite per array is byte-exact with the Python writer. */
    if (bank->condition_count > 0
        && fwrite(bank->conditions, sizeof(SSSCondition),
                  bank->condition_count, f) != bank->condition_count) {
        fclose(f); return SFB_ERR_IO;
    }
    if (bank->response_count > 0
        && fwrite(bank->responses, sizeof(SSSInteractionResponse),
                  bank->response_count, f) != bank->response_count) {
        fclose(f); return SFB_ERR_IO;
    }
    if (fclose(f) != 0) return SFB_ERR_IO;
    return SFB_OK;
}

/* ── Load ──────────────────────────────────────────────────────── */

/* Helper: read N bytes per record into a malloc'd buffer of N *
 * count bytes, striding by `record_size` from the on-disk layout.
 * For v1 the on-disk record_size equals sizeof(T), so this is a
 * single fread; this indirection is what lets v2 grow records
 * without bumping the file version. */
static int read_records(FILE *f, void *dst, size_t record_size,
                        size_t struct_size, uint32_t count)
{
    if (count == 0) return SFB_OK;
    if (record_size == struct_size) {
        if (fread(dst, struct_size, count, f) != count) return SFB_ERR_IO;
        return SFB_OK;
    }
    /* v2 hook: read record_size bytes, copy struct_size into the
     * v1 struct, discard the trailing extension. Activated only
     * when a future writer ships larger records. */
    if (record_size < struct_size) return SFB_ERR_VERSION;
    uint8_t *scratch = (uint8_t *)malloc(record_size);
    if (!scratch) return SFB_ERR_ALLOC;
    uint8_t *out = (uint8_t *)dst;
    for (uint32_t i = 0; i < count; ++i) {
        if (fread(scratch, 1, record_size, f) != record_size) {
            free(scratch); return SFB_ERR_IO;
        }
        memcpy(out + (size_t)i * struct_size, scratch, struct_size);
    }
    free(scratch);
    return SFB_OK;
}

/* Helper: zero-pad load helper for v1 motifs. The on-disk record is
 * 2864 B (v1) but the in-memory struct is 3744 B (v2). For each
 * record we read 2864 B and leave the v2 extension at zero — which
 * means `temporal_length = 0`, `response_count_in_motif = 0`, and
 * empty cross_resonance / warp_self fields. This matches "treat
 * the v1 motif as a static motif with no learned interactions". */
static int read_motifs_v1_into_v2(FILE *f, SSSMotif *dst, uint32_t count)
{
    if (count == 0) return SFB_OK;
    /* Per-record read of SFB_V1_MOTIF_SIZE bytes into the head of
     * the v2 struct; the rest stays zero (caller calloc'd). */
    for (uint32_t i = 0; i < count; ++i) {
        if (fread((uint8_t *)(dst + i), 1, SFB_V1_MOTIF_SIZE, f)
                != SFB_V1_MOTIF_SIZE) {
            return SFB_ERR_IO;
        }
    }
    return SFB_OK;
}

int sss_feature_bank_load(const char *path, SSSFeatureBank *out)
{
    if (!path || !out) return SFB_ERR_IO;
    memset(out, 0, sizeof(*out));

    FILE *f = fopen(path, "rb");
    if (!f) return SFB_ERR_IO;

    SSSFeatureBankHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f); return SFB_ERR_IO;
    }
    if (memcmp(hdr.magic, SFB_MAGIC, 4) != 0) {
        fclose(f); return SFB_ERR_MAGIC;
    }
    /* v2 loader accepts both v2 and v1 (motif backward-compat path).
     * v1 → v2 lifting is handled per-record below; v2-or-bust is
     * the rule. */
    int is_v1 = 0;
    if (hdr.version == 1u) {
        is_v1 = 1;
        /* v1 files report flags / reserved / reserved2 where v2
         * has condition_count / response_count / reserved1 /
         * reserved2. The byte layout is identical, so the field
         * values are simply meaningless to v2 — clear them so the
         * load path doesn't try to consume condition / response
         * arrays that don't exist on disk. */
        hdr.condition_count = 0;
        hdr.response_count  = 0;
    } else if (hdr.version != SFB_VERSION) {
        fclose(f); return SFB_ERR_VERSION;
    }

    /* Record-size sanity. v1 motifs are SFB_V1_MOTIF_SIZE on disk;
     * everything else must match the current struct size or be
     * larger (the read_records helper handles "larger" by trimming).
     */
    int v1_motif_layout =
        (is_v1 && hdr.motif_record_size == SFB_V1_MOTIF_SIZE);
    if (!v1_motif_layout && hdr.motif_record_size < sizeof(SSSMotif)) {
        fclose(f); return SFB_ERR_VERSION;
    }
    if (hdr.relation_record_size < sizeof(SSSRelation)
     || hdr.identity_record_size < sizeof(SSSIdentity)) {
        fclose(f); return SFB_ERR_VERSION;
    }

    SSSMotif               *motifs     = NULL;
    SSSRelation            *relations  = NULL;
    SSSIdentity            *identities = NULL;
    SSSCondition           *conditions = NULL;
    SSSInteractionResponse *responses  = NULL;

    if (hdr.motif_count > 0) {
        motifs = (SSSMotif *)calloc(hdr.motif_count, sizeof(SSSMotif));
        if (!motifs) { fclose(f); return SFB_ERR_ALLOC; }
    }
    if (hdr.relation_count > 0) {
        relations = (SSSRelation *)calloc(hdr.relation_count, sizeof(SSSRelation));
        if (!relations) { free(motifs); fclose(f); return SFB_ERR_ALLOC; }
    }
    if (hdr.identity_count > 0) {
        identities = (SSSIdentity *)calloc(hdr.identity_count, sizeof(SSSIdentity));
        if (!identities) {
            free(motifs); free(relations); fclose(f); return SFB_ERR_ALLOC;
        }
    }
    if (hdr.condition_count > 0) {
        conditions = (SSSCondition *)calloc(hdr.condition_count,
                                            sizeof(SSSCondition));
        if (!conditions) {
            free(motifs); free(relations); free(identities);
            fclose(f); return SFB_ERR_ALLOC;
        }
    }
    if (hdr.response_count > 0) {
        responses = (SSSInteractionResponse *)calloc(
            hdr.response_count, sizeof(SSSInteractionResponse));
        if (!responses) {
            free(motifs); free(relations); free(identities); free(conditions);
            fclose(f); return SFB_ERR_ALLOC;
        }
    }

    int rc;
    if (v1_motif_layout) {
        rc = read_motifs_v1_into_v2(f, motifs, hdr.motif_count);
    } else {
        rc = read_records(f, motifs, hdr.motif_record_size,
                          sizeof(SSSMotif), hdr.motif_count);
    }
    if (rc == SFB_OK) {
        rc = read_records(f, relations, hdr.relation_record_size,
                          sizeof(SSSRelation), hdr.relation_count);
    }
    if (rc == SFB_OK) {
        rc = read_records(f, identities, hdr.identity_record_size,
                          sizeof(SSSIdentity), hdr.identity_count);
    }
    /* v2-only: conditions + responses sit at the tail. v1 readers
     * stop after identities; v2 readers continue. */
    if (rc == SFB_OK && hdr.condition_count > 0) {
        if (fread(conditions, sizeof(SSSCondition),
                  hdr.condition_count, f) != hdr.condition_count) {
            rc = SFB_ERR_IO;
        }
    }
    if (rc == SFB_OK && hdr.response_count > 0) {
        if (fread(responses, sizeof(SSSInteractionResponse),
                  hdr.response_count, f) != hdr.response_count) {
            rc = SFB_ERR_IO;
        }
    }
    fclose(f);
    if (rc != SFB_OK) {
        free(motifs); free(relations); free(identities);
        free(conditions); free(responses);
        return rc;
    }

    /* Post-load validation: relation / identity indices must point
     * inside the motif array, conditions / responses must validate.
     * Catches stale .sfb files (older writer produced inconsistent
     * indices) and outright corruption. */
    for (uint32_t i = 0; i < hdr.relation_count; ++i) {
        if (validate_relation(&relations[i], hdr.motif_count) != SFB_OK) {
            free(motifs); free(relations); free(identities);
            free(conditions); free(responses);
            return SFB_ERR_INVALID_REL;
        }
    }
    for (uint32_t i = 0; i < hdr.identity_count; ++i) {
        if (validate_identity(&identities[i], hdr.motif_count) != SFB_OK) {
            free(motifs); free(relations); free(identities);
            free(conditions); free(responses);
            return SFB_ERR_INVALID_IDENT;
        }
    }
    for (uint32_t i = 0; i < hdr.condition_count; ++i) {
        if (validate_condition(&conditions[i]) != SFB_OK) {
            free(motifs); free(relations); free(identities);
            free(conditions); free(responses);
            return SFB_ERR_INVALID_COND;
        }
    }
    for (uint32_t i = 0; i < hdr.response_count; ++i) {
        if (validate_response(&responses[i], hdr.motif_count,
                              hdr.condition_count) != SFB_OK) {
            free(motifs); free(relations); free(identities);
            free(conditions); free(responses);
            return SFB_ERR_INVALID_RESP;
        }
    }

    out->motif_count     = hdr.motif_count;
    out->motifs          = motifs;
    out->relation_count  = hdr.relation_count;
    out->relations       = relations;
    out->identity_count  = hdr.identity_count;
    out->identities      = identities;
    out->condition_count = hdr.condition_count;
    out->conditions      = conditions;
    out->response_count  = hdr.response_count;
    out->responses       = responses;
    return SFB_OK;
}

void sss_feature_bank_free(SSSFeatureBank *bank)
{
    if (!bank) return;
    free(bank->motifs);
    free(bank->relations);
    free(bank->identities);
    free(bank->conditions);
    free(bank->responses);
    memset(bank, 0, sizeof(*bank));
}

int sss_feature_bank_find_motif(const SSSFeatureBank *bank,
                                const char *label)
{
    if (!bank || !label) return -1;
    /* Compare up to SFB_LABEL_LEN bytes. The on-disk label may be
     * NUL-terminated or fill the full 32-byte field; we match either
     * way by capping strncmp at SFB_LABEL_LEN and confirming the
     * stored bytes after the match are NUL (label fits) or that the
     * query is exactly 32 bytes (label fills the slot). Hand-rolled
     * length scan (instead of strnlen) so the file compiles without
     * _POSIX_C_SOURCE on every CE target. */
    size_t qlen = 0;
    while (qlen <= SFB_LABEL_LEN && label[qlen] != '\0') ++qlen;
    if (qlen > SFB_LABEL_LEN) return -1;
    for (uint32_t i = 0; i < bank->motif_count; ++i) {
        const char *stored = bank->motifs[i].label;
        if (memcmp(stored, label, qlen) != 0) continue;
        /* qlen == 32: full field, no terminator required.
         * qlen <  32: the next stored byte must be NUL. */
        if (qlen == SFB_LABEL_LEN || stored[qlen] == '\0') {
            return (int)i;
        }
    }
    return -1;
}
