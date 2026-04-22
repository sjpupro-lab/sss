# 03 — Task A: ContextPool (별도 작업 기억 공간)

**우선순위**: 최우선 (가장 먼저 구현)
**의존성**: 없음
**브랜치 권장**: main 직접 작업 가능

---

## 목적

장기 기억은 그대로 두고, 세션 문맥만 별도 축적한다.

---

## 설계 원칙

- 장기 prior는 기본적으로 `agg_build(ai)`로부터 얻는다 (keyframes 집계).
- 단기 prior는 `agg_build_from_pool(ai->context_pool)`로부터 얻는다.
- `ContextPool`은 기본적으로 `.spai` 저장 대상이 아니다.

---

## 헤더 변경 (기존 구조 비파괴)

`include/spatial_keyframe.h` 또는 적절한 공개 헤더에 신규 필드/함수 추가:

```c
typedef struct SpatialAI_ {
    /* existing fields remain unchanged */
    struct SpatialCanvasPool_* context_pool;  /* optional working memory */
} SpatialAI;

struct SpatialCanvasPool_* ai_get_context_pool(SpatialAI* ai);
void                       ai_clear_context_pool(SpatialAI* ai);
void                       ai_release_context_pool(SpatialAI* ai);
```

**주의**: 기존 필드의 순서·크기 변경 금지. 신규 포인터는 구조체 끝에 추가.

---

## 조회 결합

```c
AggTables* agg_L = agg_build(ai);  /* long-term prior from keyframes */
AggTables* agg_S = (ai->context_pool != NULL)
                 ? agg_build_from_pool(ai->context_pool)
                 : NULL;

score(y, v) = W_LONG  * score_long(...)
            + W_SHORT * score_short(...);
```

### 권장 초기값

- `W_LONG = 0.6`
- `W_SHORT = 0.4`

---

## 주의사항

- `agg_S == NULL`이면 long-term only로 동작.
- context는 **검색 bias / refinement bias**로만 사용하고, **장기 기억을 mutate하지 말 것**.
- 수명 관리:
  - `ai_get_context_pool(ai)` — lazy create. 호출 시점에 풀이 없으면 생성.
  - `ai_clear_context_pool(ai)` — 풀 내용만 비우고 포인터는 유지(선택).
  - `ai_release_context_pool(ai)` — 풀을 파괴하고 포인터 NULL로.

---

## `.spai` 직렬화 정책

- 기본: `context_pool`은 저장하지 않는다.
- 명시 플래그(예: `ai_save_with_context(ai, path)`)로만 포함.
- 포함 시 **신규 `SPAI_TAG_*` 태그**로 추가 (기존 태그 순서 변경 금지).

---

## 테스트

Task A 완료 후:

```
[ ] make all         경고 없이 빌드
[ ] make test        기존 69 tests PASS (회귀 없음)
[ ] 수동 probe       ai_get_context_pool() → pool_add_clause() → 조회 동작 확인
```

---

## 다음 단계

Task A 안정화 후 `07_TASK_E_REPL_INTEGRATION.md`의 `:ctx on/off/clear` 명령부터 먼저 통합한다. 그 다음 `04_TASK_B_SEQUENCE_METADATA.md`로 진행.
