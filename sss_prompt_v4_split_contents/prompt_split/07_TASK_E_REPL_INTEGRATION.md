# 07 — Task E: REPL / Tools Integration

**우선순위**: 분할 진행 (A 후 일부, D 후 나머지)
**의존성**: Task A (context), Task D (refine)
**브랜치 권장**: main (단, refine 관련 명령은 Task D 브랜치 머지 후)

---

## 목적

`tools/chat.c`에 신규 명령을 추가해 **baseline과 새 경로를 쉽게 비교 가능**하게 한다.

---

## 신규 명령 목록

`tools/chat.c` 또는 equivalent tool에 다음 명령 추가:

| 명령 | 효과 | 의존 Task |
|---|---|---|
| `:ctx on` | context_pool 사용 활성화 | A |
| `:ctx off` | context_pool 조회 비활성화 | A |
| `:ctx clear` | context_pool 내용 초기화 | A |
| `:gen next` | baseline `ai_generate_next()` 사용 | — |
| `:gen refine` | `ai_generate_refine()` 사용 | D |
| `:refine cfg` | 현재 `RefineConfig` 출력 | D |
| `:refine trace` | 레벨별 수렴 그래프 ASCII 출력 | D |
| `:recluster` | 수동 재클러스터링 실행 + 리포트 | C |

---

## 설계 권장

### baseline과 refine 비교 용이성

`:gen next`와 `:gen refine`은 토글이어야 하며, 동일 입력에 대해 즉시 비교 가능하도록 두 모드의 출력을 나란히 표시하는 `:compare` 명령도 고려.

### context on/off 비교 용이성

`:ctx on`, `:ctx off` 전환이 **즉시 반영**되어야 하며, 전환 후 다음 프롬프트부터 다른 결과를 볼 수 있어야 한다.

### `:refine trace`

레벨별 promotion curve를 ASCII로 시각화. 예:

```
Level 0 (B dominant, large radius):
  iter 01: n_cand=4523 promoted=312 rate=6.9%
  iter 02: n_cand=4211 promoted=189 rate=4.5%
  iter 03: n_cand=4022 promoted=82  rate=2.0%  [converged]

Level 1 (G dominant, medium radius):
  iter 01: n_cand=4022 promoted=410 rate=10.2%
  ...

Level 2 (R dominant, small radius):
  ...

Total iterations: 42
Final confidence: 0.73
Fallback used: no
```

---

## 신규 bench

`tests/bench_context.c`:

- 10턴 멀티턴 대화 시나리오 정의 (예: 이야기 이어가기)
- 측정:
  - context OFF에서 단일 턴 생성
  - context ON에서 동일 입력 생성
  - 두 결과의 retrieval consistency 비교
- 리포트만 출력. 구체적 수치 gate는 두지 말 것.

---

## 구현 순서 (분할)

### Task A 완료 직후

`:ctx on`, `:ctx off`, `:ctx clear`만 먼저 통합. Task A 검증 목적.

### Task D 완료 후

`:gen next`, `:gen refine`, `:refine cfg`, `:refine trace` 통합.

### Task C 완료 후

`:recluster` 통합.

---

## 검증 체크리스트

```
[ ] make all              경고 없이 빌드
[ ] make test             기존 69 tests PASS
[ ] 수동 REPL probe       모든 신규 명령이 crash 없이 동작
[ ] bench_context         context on/off 비교 리포트 출력
[ ] :refine trace         수렴 그래프 정상 출력
```

---

## 주의사항

- 기존 REPL 명령 (`:q`, `:topk`, `:gen`, `:both`, `:retr`, `:help`)은 **동작 유지**.
- `:gen`은 기존대로 `:gen retr` 혹은 `:gen gen`일 텐데, 신규 `:gen next`, `:gen refine`과 네임스페이스 충돌하지 않도록 주의.
- `tools/chat.c`의 argv 파싱은 단순 tokenization이므로, `:refine cfg`처럼 2-토큰 명령을 처리할 때 기존 파서를 확장할 수 있음.

---

## 다음 단계

Task E의 refine 부분까지 통합되면 `08_TASK_F_IMAGE_PROTOTYPE.md`로 진행. Task F는 A~E 안정화 후에만 착수.
