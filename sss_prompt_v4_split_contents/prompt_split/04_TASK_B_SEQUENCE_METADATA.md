# 04 — Task B: Sequence Metadata (필수) + Reserved Anchor Slot (선택)

**우선순위**: Task C의 선행 조건
**의존성**: Task A 완료 권장
**브랜치 권장**: main 직접 작업 가능 (메타데이터 추가만)

---

## 목적

재배치/재클러스터링 후에도 **논리적 순서를 복원 가능**하게 한다.

---

## 현재 `SlotMeta` 상태

현재 존재하는 필드:

```c
typedef struct {
    DataType type;
    float    boundary_weight;
    uint32_t byte_length;
    uint32_t topic_hash;
    int      occupied;
} SlotMeta;
```

여기에 시퀀스 복원용 메타데이터를 추가한다.

---

## 필수 변경

```c
typedef struct {
    DataType type;
    float    boundary_weight;
    uint32_t byte_length;
    uint32_t topic_hash;
    int      occupied;
    uint32_t sequence_id;      /* new */
    uint64_t timestamp_us;     /* new */
} SlotMeta;
```

`sequence_id`는 풀 전체에서 단조 증가. `timestamp_us`는 슬롯이 채워진 시각(microseconds).

---

## 선택 기능: Reserved Anchor Slot

slot 0을 시퀀스 메타 앵커로 예약하는 방식은 **실험 옵션으로만** 구현 가능하다.

기본 동작에서는 **32 content slots 유지**. slot 0을 비워두지 말 것.

### 권장 순서

1. **먼저** `sequence_id`, `timestamp_us`를 메타데이터로만 구현
2. reserved slot이 정말 필요한지 별도 실험
3. 기본 경로에서는 slot 0를 비워두지 말 것

### Reserved slot을 실험하는 경우

별도 플래그(`SpatialCanvasPool.reserve_slot_zero = 1`)로만 활성화. 활성 시:

- slot 0의 A채널에 `sequence_id`를 little-endian 4바이트로 기록
- 콘텐츠는 slot 1~31에 배치
- 기존 테스트는 reserve_slot_zero = 0 (기본값)에서 돌아가야 한다

---

## 주의사항

- `sequence_id`가 없어도 풀 배열 인덱스로 임시 순서를 추정할 수 있지만, Task C의 재클러스터링 후에는 풀 인덱스가 섞인다. 그래서 sequence metadata는 **필수**.
- `timestamp_us`는 디버깅 및 멀티 세션 병합에 필요. 시스템 시계 사용(`clock_gettime(CLOCK_MONOTONIC, ...)` 권장).
- `.spai` 직렬화에 `sequence_id` / `timestamp_us`를 포함하려면 **기존 `SlotMeta` 직렬화 코드를 직접 수정하지 말고**, 신규 `SPAI_TAG_SEQMETA` 같은 optional trailing tag로 추가.

---

## 테스트

- `tests/test_canvas.c`에 sequence metadata 검증 케이스 추가:
  - 슬롯 추가 시 `sequence_id` 단조 증가 확인
  - `timestamp_us > 0` 확인

```
[ ] make all         경고 없이 빌드
[ ] make test        기존 69 tests PASS + 신규 케이스 PASS
[ ] reserve_slot_zero = 0 기본 경로에서 기존 동작 불변 확인
```

---

## 다음 단계

Task B 완료 후 `05_TASK_C_RECLUSTERING.md`로 진행. sequence metadata가 있어야 재배치 후 순서 복원이 가능하다.
