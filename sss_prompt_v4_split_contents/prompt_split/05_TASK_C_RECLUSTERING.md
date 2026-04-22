# 05 — Task C: Periodic Re-clustering (offline / maintenance)

**우선순위**: Task B 이후
**의존성**: Task B의 `sequence_id` 필수
**브랜치 권장**: main 직접 작업 가능 (추가 경로, offline)

---

## 목적

유사한 canvases의 **물리 배치**를 개선해 delta/RLE 압축률 및 국소 문맥 품질을 높인다.

---

## 신규 파일

- `include/spatial_recluster.h`
- `src/spatial_recluster.c`
- `tests/test_recluster.c`

---

## API

```c
typedef struct {
    uint32_t canvases_reordered;
    uint32_t bytes_before;
    uint32_t bytes_after;
    float    compression_gain;
} ReclusterReport;

ReclusterReport pool_recluster_by_topic(SpatialCanvasPool* p,
                                        float min_gain_ratio);

void pool_iterate_in_sequence_order(
    const SpatialCanvasPool* p,
    void (*visit)(const SpatialCanvas*, void* user),
    void* user);
```

---

## 알고리즘

1. **`topic_hash` exact match 기준으로 1차 grouping**
   - 같은 `topic_hash`를 가진 캔버스들을 하나의 그룹으로 묶는다.
   - **bit distance / 해밍 거리는 사용하지 말 것** (해시는 설계상 비트가 무작위 분산되므로 비트 차이는 의미 거리와 무관).

2. **같은 group 내부에서 실제 delta 비용으로 인접 순서 결정**
   - `canvas_delta_sparse()` / `canvas_delta_rle_bytes()`로 실제 비용 측정
   - greedy: 현재 캔버스에 대해 남은 후보 중 RLE 비용이 가장 작은 것을 다음 위치로

3. **압축 이득 판정**
   - 전체 재배치 결과의 총 RLE 바이트 수 측정
   - `(bytes_before - bytes_after) / bytes_before >= min_gain_ratio`인 경우에만 commit
   - 이득이 부족하면 원본 순서 유지

4. **sequence metadata로 원래 순서 복원 가능해야 함**
   - 캔버스 자체의 `SlotMeta.sequence_id`는 변경하지 않음
   - `pool_iterate_in_sequence_order()`는 물리 배열 순서와 무관하게 `sequence_id` 오름차순으로 방문

---

## 주의사항

- **`topic_hash` bit distance를 의미 거리로 사용하지 말 것.** 해시 함수의 비트 분산은 의미적 유사성을 반영하지 않는다.
- **실시간 hot path보다 checkpoint / offline maintenance 경로를 우선할 것.** 재클러스터링은 비용이 크므로 학습 종료 시 또는 명시적 호출 시에만 실행.
- **subtitle track / canvas_id / slot_id 일관성 유지 필수.** 재배치 후 `SubtitleTrack.entries[i].canvas_id`가 새 물리 인덱스를 가리키도록 동기 업데이트.

---

## 검증 (`tests/test_recluster.c`)

- 10개 토픽 × 10절씩 의도적으로 섞어서 삽입
- `pool_recluster_by_topic(p, 0.10f)` 호출
- 검증 항목:
  - `compression_gain >= 0.10`일 때 `canvases_reordered > 0`
  - `pool_iterate_in_sequence_order()`가 원래 발화 순서와 정확히 일치
  - `SubtitleTrack` 일관성 유지 (canvas_id / slot_id가 재배치 후에도 유효)

```
[ ] make all         경고 없이 빌드
[ ] make test        기존 69 tests PASS + test_recluster PASS
[ ] compression_gain 임계치 미달 시 원본 순서 보존 확인
```

---

## 다음 단계

Task C 완료 후 `06_TASK_D_DRAFT_REFINEMENT.md`로 진행. Task D는 **별도 실험 브랜치**에서 작업할 것을 강력히 권장한다.
