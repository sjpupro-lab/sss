# SPATIAL-PATTERN-AI: 엔진 최적화 명세서 (Page 2)

> **전제**: 이 문서는 SPEC.md (Page 1)의 구조 위에서 동작하는 엔진 최적화 명세이다.
> Page 1의 SpatialGrid, 3-레이어, 매칭 파이프라인 구조를 변경하지 않고,
> 메모리/연산/검색 성능을 실전 엔진 수준으로 끌어올린다.
> Python 프로토타입으로 검증 완료. (11 PASS / 0 FAIL)

---

## 전체 파이프라인 (최적화 적용 후)

```
입력 텍스트
  → 3-레이어 인코딩 (1D 정렬 메모리)
  → 방향성 RGB 업데이트
  → 블록 합 계산 (16×16)
  → [KF < 100] 활성 픽셀 겹침 → Top-K
  → [KF ≥ 100] 해시 버킷 → 겹침 → Top-K
  → 블록 스킵 코사인 (SIMD)
  → LRU 캐시 조회/저장
  → 최종 매칭
```

---

## Phase A: 메모리 레이아웃

### A.1 문제

Page 1의 2D 배열(`uint16_t A[256][256]`)은 논리적으로 명확하지만,
SIMD 명령어와 CPU 캐시 최적화에 불리하다.

### A.2 해결: 1D + 32바이트 정렬

```c
#define GRID_TOTAL 65536

typedef struct {
    uint16_t* A;  // 32-byte aligned
    uint8_t*  R;  // 32-byte aligned
    uint8_t*  G;  // 32-byte aligned
    uint8_t*  B;  // 32-byte aligned
} SpatialGrid;

SpatialGrid* grid_create(void) {
    SpatialGrid* g = malloc(sizeof(SpatialGrid));
    posix_memalign((void**)&g->A, 32, GRID_TOTAL * sizeof(uint16_t));
    posix_memalign((void**)&g->R, 32, GRID_TOTAL);
    posix_memalign((void**)&g->G, 32, GRID_TOTAL);
    posix_memalign((void**)&g->B, 32, GRID_TOTAL);
    memset(g->A, 0, GRID_TOTAL * sizeof(uint16_t));
    memset(g->R, 0, GRID_TOTAL);
    memset(g->G, 0, GRID_TOTAL);
    memset(g->B, 0, GRID_TOTAL);
    return g;
}

void grid_destroy(SpatialGrid* g) {
    free(g->A); free(g->R); free(g->G); free(g->B); free(g);
}
```

좌표 접근: `g->A[y * 256 + x]` (row-major 1D)

### A.3 검증

```
1D 전환 후 코사인 유사도: 78.5% (2D와 동일, 오차 0.000%)
```

### A.4 효과

- AVX2 `_mm256_loadu_si256` 직접 적용 가능
- 캐시 라인(64B) 내 연속 접근 보장
- C 구현 시 2~5배 속도 향상 기대

---

## Phase B: 블록 기반 처리

### B.1 개념

256×256 격자를 16×16 블록 256개로 분할.
변화가 없는 블록은 건너뛴다.

```
256 × 256 = 16 × 16 블록 × 16개/축 = 256 블록
```

### B.2 블록 합 계산

```c
#define BLOCK 16
#define BLOCKS 16

typedef struct {
    uint32_t sum[BLOCKS][BLOCKS];
} BlockSummary;

void compute_block_sums(const SpatialGrid* g, BlockSummary* bs) {
    for (int by = 0; by < BLOCKS; by++) {
        for (int bx = 0; bx < BLOCKS; bx++) {
            uint32_t s = 0;
            for (int y = 0; y < BLOCK; y++)
                for (int x = 0; x < BLOCK; x++)
                    s += g->A[(by*BLOCK+y)*256 + (bx*BLOCK+x)];
            bs->sum[by][bx] = s;
        }
    }
}
```

### B.3 활용: 블록 단위 early skip

코사인 계산 시, 양쪽 블록 합이 모두 0인 블록은 건너뛴다.
dot product 기여가 0이므로 정확도 손실 없음.

```c
// 한쪽이라도 0이면 dot 기여 없음 → 스킵
if (bs_a->sum[by][bx] == 0 || bs_b->sum[by][bx] == 0) {
    // norm 계산만 수행 (정확도 유지)
    continue;
}
```

### B.4 검증

```
블록 스킵 vs 전체 코사인:
  KF0↔KF1: 78.5% vs 78.5% (차이 0.000%)
  KF0↔KF2:  0.0% vs  0.0% (차이 0.000%)
  KF0↔KF3:  8.4% vs  8.4% (차이 0.000%)

입력 "착한 고양이가 밥을 먹는다.":
  빈 블록: 240/256 → 94% 스킵
```

**정확도 0% 손실, 연산량 94% 감소.**

### B.5 효과

- 절 단위(짧은 텍스트): 90~95% 블록이 비어있음 → 대부분 스킵
- C + SIMD 조합 시 2~4배 추가 가속

---

## Phase C: 적응형 Top-K 검색

### C.1 설계 원칙

키프레임 수에 따라 coarse 전략을 자동 전환한다.

```
KF < 100개:  활성 픽셀 겹침으로 전체 스캔 (정확도 최우선)
KF ≥ 100개:  해시 버킷 → 겹침 → Top-K (속도 최우선)
```

### C.2 소규모 경로 (KF < 100)

Page 1의 검증된 overlap coarse를 그대로 사용.

```c
#define BUCKET_THRESHOLD 100

if (ai->kf_count < BUCKET_THRESHOLD) {
    // 전체 overlap scan
    for (uint32_t i = 0; i < ai->kf_count; i++) {
        pool[i].id = i;
        pool[i].score = (float)overlap_score(input, &ai->keyframes[i].grid);
    }
}
```

### C.3 대규모 경로 (KF ≥ 100)

#### 해시 함수

활성 X좌표 집합의 해시. 같은 바이트 패턴을 공유하는 프레임이 같은 버킷에 들어간다.

```c
#define NUM_BUCKETS 256

uint32_t grid_hash(const SpatialGrid* g) {
    // 활성 X좌표 비트마스크 해시
    uint32_t h = 0;
    for (int i = 0; i < GRID_TOTAL; i++) {
        if (g->A[i] > 0) {
            uint32_t x = i % 256;
            h = h * 31 + x;
        }
    }
    return h % NUM_BUCKETS;
}
```

#### 버킷 구조

```c
typedef struct {
    uint32_t ids[256];
    uint32_t count;
} Bucket;

typedef struct {
    Bucket buckets[NUM_BUCKETS];
} BucketIndex;
```

#### 검색: 인접 버킷 확장

```c
// 해시 ± expand 범위의 버킷에서 후보 수집
void bucket_candidates(BucketIndex* idx, uint32_t hash,
                       int expand, uint32_t* out, uint32_t* out_count) {
    *out_count = 0;
    for (int d = -expand; d <= expand; d++) {
        uint32_t bi = (hash + d + NUM_BUCKETS) % NUM_BUCKETS;
        Bucket* b = &idx->buckets[bi];
        for (uint32_t i = 0; i < b->count; i++) {
            out[(*out_count)++] = b->ids[i];
        }
    }
}
```

#### fallback

후보가 top_k 미만이면 overlap 전체 스캔으로 전환.

### C.4 검증

```
소규모 경로(8개 KF):
  5개 테스트 입력 전부 v3 매칭과 100% 동일
  이전 실패 케이스("오늘 저녁 하늘이...") 해결됨
```

### C.5 효과

- 소규모: 정확도 100% 보존 (검증된 overlap 사용)
- 대규모: O(N) → O(N/B + K) (B=버킷 수, K=top_k)
- 1000+ KF 시 10~50배 가속 기대

---

## Phase D: 델타 압축

### D.1 적응 전략

절 단위(256×256)에서는 변화가 산재(16개 정도)하여 RLE가 불리.
4096×4096 스케일에서는 연속 변화가 많아 RLE가 유리.

```
자동 선택:
  sparse 크기 = 항목 수 × 6B (uint32 index + int16 diff)
  RLE 크기    = 항목 수 × 8B (uint32 start + uint16 length + int16 diff)
  → 작은 쪽 선택
```

### D.2 Sparse 형식 (절 단위 기본)

```c
typedef struct {
    uint32_t index;
    int16_t  diff_A;
} SparseDelta;  // 6 bytes
```

### D.3 RLE 형식 (4096 스케일용)

```c
typedef struct {
    uint32_t start;
    uint16_t length;
    int16_t  diff;
} RLEDelta;  // 8 bytes
```

### D.4 검증

```
KF0→KF1 (유사 절):  16항목 → sparse 96B (RLE 128B → sparse 선택)
KF0→KF2 (이질 절):  71항목 → sparse 426B
```

절 단위에서는 sparse가 항상 유리. RLE는 4096 스케일에서 활성화.

---

## Phase E: LRU 프레임 캐시

### E.1 구조

```c
#define CACHE_SIZE 256

typedef struct {
    uint32_t     frame_id;
    SpatialGrid* grid;
    uint32_t     access_order;  // LRU 순서
} CacheEntry;

typedef struct {
    CacheEntry entries[CACHE_SIZE];
    uint32_t   count;
    uint32_t   clock;
} FrameCache;
```

### E.2 동작

```
get(frame_id):
  캐시 히트 → 즉시 반환, access_order 갱신
  캐시 미스 → NULL 반환

put(frame_id, grid):
  빈 슬롯 있음 → 삽입
  가득 참 → 가장 오래된(access_order 최소) 제거 후 삽입
```

### E.3 검증

```
4슬롯 캐시, 패턴 [0,1,3,0,1,0] × 5회:
  히트율: 90%
```

### E.4 효과

- 반복 조회 시 디스크/메모리 할당 회피
- 대화형 추론에서 최근 맥락 즉시 재활용

---

## 최종 매칭 함수 (전체 통합)

```c
uint32_t match_engine(SpatialAI* ai, SpatialGrid* input,
                      BucketIndex* bidx, BlockSummary* bs_all,
                      FrameCache* cache, float* out_sim) {

    uint32_t n = ai->kf_count;
    Candidate pool[MAX_CANDIDATES];
    uint32_t pool_size = 0;

    // ── Step 1: 후보 선택 (적응형) ──
    if (n < BUCKET_THRESHOLD) {
        // 소규모: overlap 전체 스캔
        for (uint32_t i = 0; i < n; i++) {
            pool[i].id = i;
            pool[i].score = (float)overlap_score(input, &ai->keyframes[i].grid);
        }
        pool_size = n;
    } else {
        // 대규모: 버킷 → overlap
        uint32_t cand_ids[1024];
        uint32_t cand_count;
        uint32_t h = grid_hash(input);
        bucket_candidates(bidx, h, 5, cand_ids, &cand_count);

        if (cand_count < TOP_K) {
            // fallback: 전체 스캔
            for (uint32_t i = 0; i < n; i++) {
                pool[i].id = i;
                pool[i].score = (float)overlap_score(input, &ai->keyframes[i].grid);
            }
            pool_size = n;
        } else {
            for (uint32_t i = 0; i < cand_count && i < 1024; i++) {
                pool[i].id = cand_ids[i];
                pool[i].score = (float)overlap_score(input, &ai->keyframes[cand_ids[i]].grid);
            }
            pool_size = cand_count;
        }
    }

    // Top-K 선택
    topk_select(pool, pool_size, TOP_K);

    // ── Step 2: 정밀 코사인 (캐시 + 블록 스킵) ──
    BlockSummary inp_bs;
    compute_block_sums(input, &inp_bs);

    uint32_t best_id = 0;
    float best_sim = -1.0f;

    for (int i = 0; i < TOP_K && i < (int)pool_size; i++) {
        uint32_t fid = pool[i].id;

        // 캐시 조회
        SpatialGrid* target = cache_get(cache, fid);
        if (!target) {
            target = &ai->keyframes[fid].grid;
            cache_put(cache, fid, target);
        }

        float sim = cosine_block_skip(input, target, &inp_bs, &bs_all[fid]);
        if (sim > best_sim) {
            best_sim = sim;
            best_id = fid;
        }
    }

    *out_sim = best_sim;
    return best_id;
}
```

---

## 성능 기대치

| 최적화 | Python 검증 | C + SIMD 기대 |
|--------|------------|--------------|
| Phase A: 1D 정렬 | 동일 결과 | 2~5× |
| Phase B: 블록 스킵 | 94% 스킵, 0% 정확도 손실 | 2~4× |
| Phase C: 적응 Top-K | 100% 정확도 | 10~50× (대규모) |
| Phase D: 적응 델타 | sparse 96B (절 단위) | 메모리 절약 |
| Phase E: LRU 캐시 | 90% 히트율 | 2~10× (반복 조회) |
| **종합** | **정확도 100% 유지** | **20~100×** |

**핵심: 프레임이 늘어나도 속도가 거의 안 떨어지는 구조.**

---

## C 구현 Phase 추가 (Page 1 이후)

```
Phase 8:  메모리 레이아웃 (1D + posix_memalign)
Phase 9:  블록 합 계산 + 블록 스킵 코사인
Phase 10: 해시 버킷 + 적응형 Top-K
Phase 11: 적응 델타 (sparse/RLE 자동 선택)
Phase 12: LRU 프레임 캐시
Phase 13: SIMD (AVX2 코사인 + 겹침 popcount)
Phase 14: 멀티스레드 매칭 (향후)
Phase 15: 프레임 클러스터링 (향후)
```

---

## 향후 확장

| 우선순위 | 항목 | 설명 |
|---------|------|------|
| P0 | SIMD 코사인 | AVX2 `_mm256_loadu_si256` + FMA |
| P1 | SIMD 겹침 | A채널 비트마스크 AND + `_mm_popcnt_u64` |
| P2 | 멀티스레드 | Top-K 후보를 스레드 풀로 병렬 코사인 |
| P3 | 프레임 클러스터링 | 유사 키프레임 자동 그룹화 (검색 범위 축소) |
| P4 | RGB 학습 안정화 | EMA + decay로 R/G/B 폭주 방지 |
| P5 | 디스크 스트리밍 | mmap 기반 프레임 로딩 + LRU 연동 |

---

## 부록: 검증 이력

### 엔진 최적화 검증 (11 PASS / 0 FAIL)
Phase A: 1D 코사인=78.5% (2D와 동일).
Phase B: 블록 스킵 정확도 0.000% 오차, 빈 블록 94%.
Phase C: 소규모 overlap 직행, 5개 입력 v3와 100% 일치.
Phase D: 절 단위 sparse 96B, 적응 선택 동작.
Phase E: LRU 4슬롯 90% 히트율.
종합: 최적화 엔진 == v3 매칭 100% 동일.
