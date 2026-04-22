# CANVAS / SPAI 벤치마크 v2 — `wiki20k.spai` (재학습 X, 로드만)

- 실행 시각: 2026-04-15
- 벤치 바이너리: `build/bench_full_v2.exe` (로드 전용 모드)
- 모델: `build/models/wiki20k.spai` (1,298 MB, KF=4,076 / Δ=2,727)
- 입력: `data/kaggle_train.txt` — 6,803 clause (train 4,762 / test 2,041)
- 원본 TXT 리포트: `build/models/wiki20k_v2_report.txt`

> v2 소스에서 `ai_load()` 시그니처 불일치 2곳 + 스택 512KB 배열 1곳만 수정 (최소 변경). 로직은 손대지 않음.

---

## 1. 스코어카드

| 영역 | 값 | 기준표 등급 |
|---|---|---|
| Matching (unseen avg sim) | **65.3 %** | **GOOD** (≥60%) |
| Self-Recall@1 (≥90%) | 123/200 = **61.5 %** | **FAIL** (<80%) |
| Word Top-1 (all bytes) | 0/500 = **0.0 %** | **BELOW BASELINE** (<5%) |
| Word Top-3 (any byte) | 0/500 = **0.0 %** | — |
| Generation UTF-8 | 88/100 = **88.0 %** | **MINIMAL** |
| Generation Unique | 84/100 = **84.0 %** | — |
| RGB Channels | std R 77.6 · G 15.5 · B 72.0 | **GOOD** |
| Efficiency | 195.5 KB/clause · Δ 40.1 % | **NORMAL** |
| Encoding speed | 3,866 c/s | **VERY FAST** |
| Matching speed | 10.91 ms/query | 보통 |
| Generation speed | 10.62 ms/gen | 보통 |

---

## 2. Matching & Retrieval (unseen 500 쿼리)

```text
Avg similarity:    65.3%
Min / Max:         6.6% / 100.0%
Hits >= 90%:       295 (59.0%)
Hits >= 50%:       295 (59.0%)
Hits >= 10%:       491 (98.2%)
ms/query:          10.91
GRADE: GOOD
```

**해석**: `≥50%` 와 `≥90%` 가 완전히 동일한 295 → **분포가 이분화**. 매칭이 잡히면 거의 무조건 90%+, 못 잡으면 10%대. 중간 유사도가 없음. 의미적 근접 매칭(부분 재사용)이 약하다는 신호.

## 2b. Self-Recall@1 (학습 절 재검색)

```text
Recall@1 (≥90%):   123/200 (61.5%)
GRADE: FAIL (기준 80%)
```

> **주목**: 학습한 절을 다시 쿼리해도 61.5% 만 Recall@1 만족. Delta 프레임으로 흡수된 clause 들이 원본 KF 로 완벽 복원되지 않거나, EMA blending 이 쿼리 시 그리드를 흐리게 만들 가능성. Threshold 0.146 calibration 과 연관 가능성 높음.

---

## 3. Word Prediction (A-channel Top-1)  — **0.0 %**

```text
Words tested:      500
Top-1 (all bytes): 0 (0.0%)
Top-3 (any byte):  0 (0.0%)
GRADE: BELOW BASELINE (랜덤 0.4% 기준)
```

### 왜 0% 인가

v2 의 단어 예측 알고리즘:

```c
for each KF k: agg_A[y, v] += kf[k].grid.A[y, v]
for each position y in masked word:
    predicted_byte = argmax_v agg_A[y, v]
```

즉 **전체 코퍼스에서 "행 y 에 가장 많이 등장하는 바이트"** 를 무조건 뽑습니다. 맥락(clause 내 주변 바이트, RGB 컨텍스트)을 전혀 쓰지 않으므로, 테스트 단어의 모든 바이트가 이 "전역 최빈값"과 일치해야만 Top-1 이 맞음. 사실상 불가능 → 0 %.

앞서 논의한 "A 에 3 레이어가 섞여 있는데 예측에서만 분해"가 **왜 안 되는지를 정확히 증명**하는 결과입니다. v2 는 A 만 쓰면서도 맥락 없는 argmax 라 → 전역 최빈값 예측기로 수렴.

### 진짜 통하는 경로

1. **RGB 컨텍스트 재결합** — `agg_score_byte` 처럼 context R/G/B 와 cell R/G/B mean 을 매칭. 단 곱셈 말고 가중합.
2. **3 레이어 분리 저장** — 학습 시 `AggTables.A_base / A_word / A_morph` 분리. 예측 시 3 랭킹 투표.
3. **KF 매칭 후 next-KF 바이트** — 매칭된 KF 의 "다음 KF" 그리드에서 해당 row 의 argmax (이미 `ai_generate_next` 가 이 방식. 84% unique 출력 → 생성은 꽤 동작).

---

## 4. Text Generation

```text
Prompts:           100
Non-empty:         100 (100.0%)
Valid UTF-8:       88 (88.0%)
Unique outputs:    84 (84.0%)
Avg length:        236.3 bytes
ms/gen:            10.62
GRADE: MINIMAL
```

### 샘플 출력

| # | 출력 (80 bytes) |
|---|---|
| 0 | `Quality, efficiency, and access` |
| 1 | `Chinaphascoenolied iterlaedrbooderinwihe 12souandfS14theCghbarSeascosnchias,thav` |
| 2 | `CehLdidonThMeeareaiIePaicValNeflYcrilCeahaaBu(c.rhA3r00aBCEaaghac.MTd1300EBCE)ad` |
| 3 | `Serefalsneologerasoesior teahnicaelythiIhlercapablersseNetifenh(aInier"citidegio` |
| 4 | `Theekethessrepicstandsthe pheapenegianio(see ocean#Tcidefacare)n is the way the ` |

**해석**:
- [0] 은 완전한 영문 단어 시퀀스로 떨어짐 — KF 매칭이 정확히 걸린 경우.
- [1]~[4] 는 부분적으로 영단어 조각이 보이지만 대부분 깨진 토큰. row-argmax 디코딩의 한계.
- UTF-8 88% 는 **깨진 바이트 12%** — 디코더에서 멀티바이트 경계를 일부 잃음.
- Unique 84% → 프롬프트별로 다른 KF 매칭 작동 중. 좋은 신호.

---

## 5. 공간 / RGB 분석

```text
Active cells:      1,335,075
A max / mean:      36 / 7.95
R mean/std:        175.4 / 77.6
G mean/std:         19.0 / 15.5
B mean/std:        168.1 / 72.0
EMA active:        24,489 (max count 2,501)
Weights:           A=3.371  R=0.033  G=0.597  B=0.000
GRADE: GOOD
```

- **채널 분화 양호**: R·B 가 표준편차 70+ 로 넓게 사용됨 → 의미/맥락 정보 실제로 학습됨.
- **학습된 채널 가중치**: `A=3.371 R=0.033 G=0.597 B=0.000` — 여전히 A 중심이지만 bench25k 때 (`A=4.049, R=G=B=0`) 보다 G 가 살아남. B 는 여전히 죽음.
- **EMA 수렴도**: 활성 24,489 셀 (전체 65,536 의 37%), 최대 count 2,501 → 긴 꼬리 분포.

> 가중치 재조정(경쟁 → 합산)을 하면 R·B 를 매칭에 되살릴 여지 있음.

---

## 6. 속도 / 자원

| 항목 | 값 | 등급 |
|---|---|---|
| Encoding | 0.26 ms/clause · 3,866 c/s | VERY FAST |
| Matching | 10.91 ms/query | NORMAL |
| Generation | 10.62 ms/gen | NORMAL |
| Memory (peak) | 1,316.5 MB | — |
| Model file | 1,298 MB | — |
| KF / Δ | 4,076 / 2,727 (Δ 40.1%) | — |
| KB/clause | 195.5 | NORMAL |

인코딩은 매우 빠름. 매칭/생성은 10 ms 대로 실시간 응답 가능한 수준.

---

## 7. 종합 판단

| 측면 | 상태 |
|---|---|
| 검색 / 매칭 | **좋음** — 65.3% 평균, 59% 는 ≥90% 일치 |
| 재현성 (Self-Recall) | **문제** — 학습 절도 61.5% 만 재현. 저장/인덱스 튜닝 필요 |
| 단어 예측 | **실패** — 현재 A-only argmax 로는 원리적 불가능 |
| 텍스트 생성 | **부분 동작** — UTF-8 88%, 고유 84%, 간혹 완전한 문장 출력 |
| 공간 표현력 | **좋음** — R·B 채널이 실제로 정보를 담음 |
| 속도 | **매우 빠름** — 인코딩 3866 c/s |
| 효율 | **보통** — 195 KB/clause |

### 다음 개선 지점 (우선순위)

1. **단어 예측**: A-channel argmax + RGB 컨텍스트 매칭 병합 (합산 방식)
2. **Self-Recall**: calibration threshold 0.146 재검토 또는 EMA blending 의 쿼리 시점 제외
3. **채널 가중치**: w_B=0 상태 탈출. 경쟁→합산 정규화 + 초기값 조정
4. **Generation UTF-8**: row-argmax 디코더에 멀티바이트 경계 보호 추가 (`grid_decode_text_utf8` 이미 존재 — 사용 여부 확인)

---

*원본 TXT 리포트*: `C:\Users\devil\JH\CANVAS\build\models\wiki20k_v2_report.txt`
*수정된 벤치 소스*: `C:\Users\devil\Downloads\TEST\bench_full_v2.c`, `C:\Users\devil\JH\CANVAS\tests\bench_full_v2.c`
