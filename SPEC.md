# SPATIAL-PATTERN-AI: 기술 명세서 v3.0

> **목적**: 이 문서는 256×256 공간 패턴 기반 AI 엔진의 완전한 설계 명세이다.
> Claude Code, GitHub Copilot 등 AI 도구가 이 문서만으로 C 구현을 수행할 수 있어야 한다.
> Python 프로토타입으로 전체 동작이 검증 완료되었다. (v3: 10 PASS / 0 FAIL)

---

## 1. 핵심 개념

이 시스템은 **텍스트를 공간적 패턴으로 변환하여 학습하는 AI 엔진**이다.

기존 LLM이 텍스트를 토큰→고차원 벡터→고정 크기 행렬로 처리하는 반면,
이 엔진은 텍스트를 **256×256 픽셀 격자 위의 밝기 패턴**으로 변환하고,
그 패턴을 **영상 코덱의 키프레임/델타 방식**으로 저장·비교·추론한다.

**언어를 영상처럼 취급하는 것이 핵심이다.**

파라미터는 고정 크기 행렬이 아니다. 키프레임이 쌓이고 델타가 쌓이고 프레임이 이어붙는다.
영상에 최대 용량이 없는 것처럼, 이 구조에도 **파라미터 상한이 없다.**

---

## 2. 기본 공간 구조

### 2.1 격자와 단위

- 크기: **256 × 256** (65,536 픽셀)
- **1장 = 1개 절(clause) 단위**
- 단위: 각 셀은 정수 밝기값 (`uint16`, 0~65535)
- 채널: **RGBA** (4채널, 각 채널이 독립적 역할)

### 2.2 좌표계

```
X축 (0~255) = 바이트 값. 픽셀 번호가 곧 바이트 값이다.
  - X=0   → 바이트 0x00
  - X=65  → 바이트 0x41 ('A')
  - X=255 → 바이트 0xFF

Y축 (0~255) = 데이터 스트림 내 순서 위치.
  - Y=0 → 입력의 첫 번째 바이트
  - Y=1 → 입력의 두 번째 바이트
  - 방향: 위에서 아래로 (Y0이 상단)
  - 절 단위(256×256)이므로 일반적인 절은 Y축을 초과하지 않는다.
```

### 2.3 인코딩 규칙

입력 텍스트를 UTF-8 바이트 시퀀스로 변환한 뒤, 각 바이트를 격자에 찍는다.

```
입력의 i번째 바이트 값이 v일 때:
  x = v
  y = i % 256
  grid[y][x] += weight    ← 레이어별 가중치만큼 밝기 증가
```

### 2.4 디코딩 (역변환)

X좌표가 곧 바이트 값이므로,
각 Y행에서 밝기가 있는 X좌표를 읽으면 원래 바이트를 복원할 수 있다.

---

## 3. 3-레이어 비트맵 합산 구조

### 3.1 개요

3장의 독립 비트맵을 생성한 뒤, **투명한 종이를 덧대듯 합산**하여 1장을 만든다.
각 레이어는 서로 다른 가중치로 같은 256×256 격자에 찍는다.

| 레이어 | 포착 단위 | 분리 기준 | 가중치 | 합산 시 효과 |
|--------|----------|----------|--------|-------------|
| **기본 레이어** | 전체 바이트 스트림 | 없음 | **+1** | 모든 바이트 위치에 기본 밝기 |
| **단어 레이어** | 공백 기준 단어 | 공백 | **+2** | 단어 위치가 +2만큼 더 밝아짐 |
| **형태소 레이어** | 어근/조사/어미 | 사전 기반 | **+1** | 형태소 위치가 +1만큼 더 밝아짐 |

### 3.2 합산 결과

```
합산 격자[y][x] = 기본[y][x] + 단어[y][x] + 형태소[y][x]
```

하나의 바이트 위치가 기본(+1) + 단어(+2) + 형태소(+1) = **최대 +4**까지 밝아질 수 있다.
이 밝기 차이가 곧 **언어적 중요도**를 공간적으로 인코딩한다.

### 3.3 검증된 수치 (v2 테스트)

```
"귀여운 고양이가 밥을 먹는다."
  기본 레이어:   40 활성 픽셀, 최대 밝기 1, 총 밝기  40
  단어 레이어:   37 활성 픽셀, 최대 밝기 2, 총 밝기  74
  형태소 레이어: 37 활성 픽셀, 최대 밝기 1, 총 밝기  37
  합산 (1장):    40 활성 픽셀, 최대 밝기 4, 총 밝기 151

  합산 보존 법칙: 151 = 40 + 74 + 37 ✓

  "밥" 위치 밝기: 기본=1, 단어=2, 합산=4 ✓
```

### 3.4 중요도 조정

3장의 가중치를 변경하면 AI의 관심 초점을 조정할 수 있다.

```
기본 구성:     기본×1, 단어×2, 형태소×1  → 단어 중심
형태소 강조:   기본×1, 단어×1, 형태소×3  → 형태소 중심
균등 배분:     기본×1, 단어×1, 형태소×1  → 균등
```

단일 비트맵 3장이므로 가중치 변경만으로 중요도를 동적 조정 가능하다.

---

## 4. 방향성 (Directional Weights)

각 레이어의 학습 단위는 공간에서 고유한 방향을 갖는다.

### 4.1 방향 정의

```
형태소:  대각선 모든 방향 (↗ ↘ ↙ ↖)
         인접 바이트 간의 조합 관계를 포착.
         "고양이" + "가" 처럼 어근과 조사의 결합 패턴.

단어:    상, 하 수직 방향 (↑ ↓)
         대체 가능한 관계를 포착.
         "고양이" ↔ "강아지" 처럼 같은 위치에 올 수 있는 단어.

구/절:   앞, 뒤 수평 방향 (← →)
         각 절(=1장)을 연속 프레임 배열로 이어붙인 순서.
         "밥을 먹는다" → "우리는 살았다" 같은 절 간 이어짐.
```

### 4.2 3D 관계

3장의 비트맵을 합산한 1장을 **3D 관계 레이어**로 취급한다.

```
          형태소 (대각선 ↗↘↙↖)
         /
        /
기본 --+-- 단어 (수직 ↑↓)
        \
         \
          구/절 (수평 ←→, 프레임 간)
```

- 형태소 방향: 1장 내부에서 대각선 이웃 픽셀 간 관계
- 단어 방향: 1장 내부에서 수직 이웃 픽셀 간 관계 (대체 가능성)
- 구/절 방향: 프레임 간 순서 관계 (전후 맥락)

---

## 5. 채널 역할 (RGBA) — 동적 임베딩

### 5.1 채널 구조

| 채널 | 역할 | 값 범위 | 설명 |
|------|------|---------|------|
| **A** | 밝기 (인덱스 빈도) | `uint16` (0~65535) | 해당 좌표에 바이트가 찍힌 횟수. 3레이어 합산. |
| **R** | 의미 (Semantic) | `uint8` (0~255) | 의미적 유사도. AI가 동적으로 매핑. |
| **G** | 기능 (Function) | `uint8` (0~255) | 품사/문법 기능. AI가 동적으로 매핑. |
| **B** | 확장 (Extended) | `uint8` (0~255) | 컨텍스트, 시제, 감정 등. AI가 동적으로 매핑. |

### 5.2 동적 임베딩 원칙

**R, G, B 채널의 값은 고정 테이블로 지정하지 않는다.**
AI가 학습 과정에서 스스로 R/G/B 값을 매핑하도록 한다.

고정된 값을 지정하면 모델의 성능 한계를 가둔다.

### 5.3 초기 시드 범위 (가이드라인)

AI의 초기 매핑을 위해 넉넉한 범위만 배정한다.
이 범위는 제안이며 AI가 학습하며 재조정할 수 있다.

```
R채널 의미 범위 가이드라인:
  R=0~9       기능어 영역 (조사, 어미 등)
  R=10~49     구체물 영역 (동물, 음식, 사물, 자연 등)
  R=50~99     추상물 영역 (감정, 상태, 시간 등)
  R=100~149   행위 영역 (동사, 이동, 인지 등)
  R=150~199   속성 영역 (형용사, 부사 등)
  R=200~255   미분류/확장 영역

G채널 기능 범위 가이드라인:
  G=0~49      내용어 영역 (명사, 동사, 형용사)
  G=50~99     기능어 영역 (조사, 어미, 접속사)
  G=100~149   구조어 영역 (구두점, 괄호, 인용)
  G=150~255   확장 영역

B채널: 전체 범위 자유. AI가 필요에 따라 차원을 정의.
```

### 5.4 동적 매핑 메커니즘

```
1. 새 토큰 등장 → 해당 범위 가이드라인 내에서 R/G/B 초기값 할당
2. 패턴 매칭 시 유사한 맥락의 토큰끼리 R값이 가까워지도록 조정
3. 같은 맥락에서 같은 역할 → G값 수렴
4. 학습이 진행될수록 의미 공간이 자동으로 정리됨
```

---

## 6. 스케일링 구조

### 6.1 기본 단위

```
256 × 256 × 4채널 = 1장 = 1개 절 단위
```

### 6.2 확장 격자

256×256 기본 격자를 스케일업하여 더 큰 공간을 구성한다.

```
256  × 256   = 1장   — 절 1개 (기본 단위)
512  × 512   = 2배   — 절 2개
1024 × 1024  = 8배   — 중심 시작, 4분면 구조
4096 × 4096  = 32배  — 절 32개
```

### 6.3 실용 구성

4096×4096 캔버스 = **절 단위 32장**을 공간 배치한 1개 문서 단위.

```
절 32장으로 하나의 문장/문단을 구성:
  - 구 단위로 나누면 32개의 구
  - 절 단위로 나누면 32개의 절
  → 하나의 4096×4096 캔버스로 문장 전체를 인식
```

### 6.4 메모리

```
256×256 1장:
  A채널: 256 × 256 × 2 bytes (uint16) = 131,072 bytes (128 KB)
  R채널: 256 × 256 × 1 byte  (uint8)  =  65,536 bytes  (64 KB)
  G채널: 256 × 256 × 1 byte  (uint8)  =  65,536 bytes  (64 KB)
  B채널: 256 × 256 × 1 byte  (uint8)  =  65,536 bytes  (64 KB)
  ─────────────────────────────────────────────────────
  1장 합계: 327,680 bytes (320 KB)

4096×4096 (절 32장):
  32 × 320 KB = 10,240 KB = 10 MB
```

---

## 7. 컨텍스트 프레임

### 7.1 개념

절 단위 1장을 **프레임**으로 취급한다.
GIF나 비디오 클립처럼, 프레임을 순서대로 이어붙이면 **무한 컨텍스트**가 가능하다.

LLM의 컨텍스트 윈도우(128K 토큰 등)는 행렬 크기에 묶인 물리적 제한이다.
프레임 스택은 디스크가 허용하는 한 무한히 이어붙일 수 있다.

### 7.2 프레임 구조

```c
typedef struct {
    uint32_t    frame_id;       // 순서 번호
    uint32_t    topic_hash;     // 토픽 식별자
    char        topic_label[64];// 토픽 라벨 ("인사", "질문", "답변" 등)
    uint8_t     frame_type;     // 0=키프레임(I), 1=델타(P)
    uint32_t    parent_id;      // 델타인 경우 부모 키프레임 ID
    SpatialGrid grid;           // 256×256 RGBA (3레이어 합산 결과)
} ContextFrame;
```

### 7.3 프레임 배열

```
프레임 0: [I] "귀여운 고양이가 밥을 먹는다."    topic: "동물_식사"
프레임 1: [P] "귀여운 강아지가 물을 먹는다."    topic: "동물_식사"  (F0 대비 델타)
프레임 2: [I] "우리는 함께 오랜 세월을 살았다."  topic: "회상"
프레임 3: [I] "오늘 아침 하늘이 밝다."          topic: "현재_상태"
...
프레임 N: 무한 확장
```

### 7.4 프레임 간 방향성

구/절의 방향성(앞/뒤)은 프레임 배열의 순서로 실현된다.

```
← 이전 프레임 | 현재 프레임 | 다음 프레임 →
```

### 7.5 검증된 수치

```
프레임0: "귀여운 고양이가 밥을 먹는다."  40px, max=4
프레임1: "우리는 함께 오랜 세월을 살았다." 44px, max=4
프레임2: "오늘 아침 하늘이 밝다."        31px, max=4

프레임 간 유사도:
  F0 vs F1: 2.7%   (이질적 내용)
  F0 vs F2: 0.0%   (완전히 다름)
  F1 vs F2: 9.1%   (약간의 공통 바이트)
```

---

## 8. 키프레임 / 델타 구조

### 8.1 키프레임 (I-Frame)

하나의 절을 완전히 인코딩한 256×256 RGBA 격자 스냅샷.
3레이어 합산 결과를 저장한다.

```c
typedef struct {
    uint32_t id;
    char     label[64];
    uint16_t A[256][256];
    uint8_t  R[256][256];
    uint8_t  G[256][256];
    uint8_t  B[256][256];
    uint32_t text_byte_count;
} Keyframe;
```

### 8.2 델타프레임 (P-Frame)

특정 키프레임 대비 변화량만 저장한다.

```c
typedef struct {
    uint32_t index;     // y * 256 + x
    int16_t  diff_A;
    int8_t   diff_R;
    int8_t   diff_G;
} DeltaEntry;  // 8 bytes

typedef struct {
    uint32_t     id;
    uint32_t     parent_id;
    char         label[64];
    uint32_t     count;
    DeltaEntry*  entries;
    float        change_ratio;
} DeltaFrame;
```

### 8.3 저장 판정 로직

```
새 절 입력 시:
1. 3레이어 합산 → 256×256 격자 생성
2. 기존 키프레임과 코사인 유사도 비교
3. 유사도 ≥ 0.3 → 델타로 저장
4. 유사도 < 0.3 → 새 키프레임으로 저장

임계값 0.3은 조정 가능. 이 값이 AI의 일반화 수준을 결정한다.
```

### 8.4 검증된 수치

```
"귀여운 고양이가 밥을 먹는다." vs "귀여운 강아지가 물을 먹는다."
→ 유사도: 78.5%, 델타: 16px → 동일 키프레임의 델타로 저장

"귀여운 고양이가 밥을 먹는다." vs "오늘 아침 하늘이 밝은 별로 가득했다."
→ 유사도: 0.0%, 델타: 91px → 별도 키프레임
```

---

## 9. 매칭 파이프라인

### 9.1 전체 흐름

```
입력 텍스트
  → 3레이어 합산 → SpatialGrid 생성
  → 방향성 기반 RGB 업데이트
  → Step 1: 활성 픽셀 겹침으로 Top-K 후보 선택 (coarse)
  → Step 2: RGB 가중 코사인 유사도로 최종 매칭 (정밀)
  → 매칭된 키프레임 + 유사도 반환
```

### 9.2 방향성 기반 RGB 업데이트

그리드 생성 후, 인접 픽셀 간 채널 값을 방향별로 확산시킨다.
학습이 반복될수록 의미적으로 가까운 픽셀의 R/G/B가 수렴한다.

```
R채널 ← 대각선 4방향 (↗↘↙↖) 이웃에서 확산   α = 0.05
G채널 ← 수직 2방향 (↑↓) 이웃에서 확산          β = 0.08
B채널 ← 수평 2방향 (←→) 이웃에서 확산          γ = 0.03
```

```c
#define ALPHA_R 0.05f
#define BETA_G  0.08f
#define GAMMA_B 0.03f

void update_rgb_directional(SpatialGrid* grid) {
    for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
            if (grid->A[y][x] == 0) continue;

            // R: 대각선 (형태소/의미 관계)
            int dx[4] = {1, 1, -1, -1};
            int dy[4] = {1, -1, 1, -1};
            for (int d = 0; d < 4; d++) {
                int nx = x + dx[d], ny = y + dy[d];
                if (nx>=0 && nx<256 && ny>=0 && ny<256 && grid->A[ny][nx] > 0) {
                    grid->R[y][x] += (int)(ALPHA_R * (grid->R[ny][nx] - grid->R[y][x]));
                }
            }

            // G: 수직 (단어 대체 관계)
            for (int d = -1; d <= 1; d += 2) {
                int ny = y + d;
                if (ny>=0 && ny<256 && grid->A[ny][x] > 0) {
                    grid->G[y][x] += (int)(BETA_G * (grid->G[ny][x] - grid->G[y][x]));
                }
            }

            // B: 수평 (구/절 순서 관계)
            for (int d = -1; d <= 1; d += 2) {
                int nx = x + d;
                if (nx>=0 && nx<256 && grid->A[y][nx] > 0) {
                    grid->B[y][x] += (int)(GAMMA_B * (grid->B[y][nx] - grid->B[y][x]));
                }
            }

            // clamp 0~255
            if (grid->R[y][x] > 255) grid->R[y][x] = 255;
            if (grid->G[y][x] > 255) grid->G[y][x] = 255;
            if (grid->B[y][x] > 255) grid->B[y][x] = 255;
        }
    }
}
```

α, β, γ 값은 조정 가능한 하이퍼파라미터이다.

### 9.3 Step 1: 활성 픽셀 겹침 (Coarse Filter)

두 그리드에서 A>0인 픽셀이 동시에 존재하는 좌표 수를 센다.

```c
uint32_t overlap_score(const SpatialGrid* a, const SpatialGrid* b) {
    uint32_t count = 0;
    for (int i = 0; i < 256*256; i++) {
        if (a->A[0][i] > 0 && b->A[0][i] > 0) count++;
    }
    return count;
}
```

겹침 수가 높은 상위 Top-K 개만 다음 단계로 전달한다.

**A채널 총합 차이가 아니라 겹침 수를 사용하는 이유:**
총합은 "길이"를 반영하고, 겹침은 "패턴"을 반영한다.
짧은 절과 긴 절이 같은 바이트 패턴을 공유할 때, 총합 차이는 크지만 겹침은 높다.

### 9.4 Step 2: RGB 가중 코사인 유사도 (Precise)

A채널 기반 코사인에 RGB 채널 유사도를 가중치로 적용한다.

```c
float rgb_weight(uint8_t r1, uint8_t r2,
                 uint8_t g1, uint8_t g2,
                 uint8_t b1, uint8_t b2) {
    float dr = fabsf(r1 - r2) / 255.0f;
    float dg = fabsf(g1 - g2) / 255.0f;
    float db = fabsf(b1 - b2) / 255.0f;
    return 1.0f - (0.5f*dr + 0.3f*dg + 0.2f*db);
}
```

가중치 배분: R(의미) 50%, G(기능) 30%, B(문맥) 20%.
의미가 가장 중요하고, 문맥은 보조적이다.

RGB 가중 코사인:
```
similarity(A, B) = Σ(A[i] × B[i] × w[i]) / (|A| × |B|)

w[i] = rgb_weight(A.R[i], B.R[i], A.G[i], B.G[i], A.B[i], B.B[i])
```

초기 상태에서는 R/G/B가 분화되지 않아 A-only와 동일한 결과를 낸다.
학습이 진행되어 R/G/B가 분화되면 의미적 유사도가 반영되기 시작한다.

### 9.5 Top-K 구조

```c
typedef struct {
    uint32_t id;
    float    score;
} Candidate;

uint32_t find_best_match(SpatialAI* ai, SpatialGrid* input) {
    Candidate pool[MAX_KF];

    // Step 1: 활성 픽셀 겹침 기반 coarse
    for (uint32_t i = 0; i < ai->kf_count; i++) {
        pool[i].id = i;
        pool[i].score = (float)overlap_score(input, &ai->keyframes[i].grid);
    }
    topk_select(pool, ai->kf_count, TOP_K);  // 상위 K개만 남김

    // Step 2: RGB 가중 코사인 정밀
    for (int i = 0; i < TOP_K; i++) {
        pool[i].score = cosine_rgb_weighted(input, &ai->keyframes[pool[i].id].grid);
    }
    topk_select(pool, TOP_K, 1);  // 최고 1개

    return pool[0].id;
}
```

C 구현 시 Step 1의 겹침 계산에 SIMD(AVX2) 적용 가능:
A채널을 0/1 마스크로 변환 후 비트 AND + popcount.

### 9.6 검증된 수치

```
8개 키프레임, top_k=4:

이전 실패 케이스 (A총합 coarse):
  "오늘 저녁 하늘이 아름다운 별로 가득하다." → KF4 (오매칭)

활성 픽셀 겹침 coarse 적용 후:
  → KF2 "오늘 아침 하늘이 밝다." (44.8%) ✓
  겹침: KF2=20px, KF4=5px, KF7=4px

전체 5개 테스트 입력: 2단계 == 전수 검색 100% 일치
속도: 전수 대비 1.9x (키프레임 8개 기준, 1000+에서 더 큰 차이)
```

---

## 10. 한국어 형태소 분리기

### 10.1 구조

사전 기반 최장일치법. 외부 라이브러리 없이 동작한다.

**분리 우선순위:**
1. 구두점 분리 (`.` `,` `!` `?` `;` `:` `~`)
2. 명사 사전 최장일치 → 나머지를 조사로 판정
3. 형용사 사전 최장일치 (관형형 포함) → 나머지를 어미로 판정
4. 동사 어근 사전 최장일치 → 나머지를 어미로 판정
5. 미등록어 → 뒤에서부터 조사 분리 시도

### 10.2 사전

**명사:**
```
동물: 고양이, 강아지, 개, 새, 토끼, 호랑이, 사자, 곰, 닭, 오리, 물고기, 고래, 돌고래, 펭귄, 코끼리
음식: 밥, 물, 빵, 국, 떡, 과일, 사과, 귤, 배, 고기, 생선, 우유, 차, 커피, 음식, 간식, 라면
사물: 집, 방, 문, 창문, 의자, 책상, 책, 컴퓨터, 전화, 자동차, 버스, 지하철, 비행기
자연: 하늘, 바다, 산, 강, 나무, 꽃, 풀, 바람, 비, 눈, 구름, 해, 달, 별
사람: 사람, 아이, 아기, 어른, 학생, 선생님, 친구, 엄마, 아빠, 형, 누나, 동생
추상: 마음, 생각, 사랑, 기억, 꿈, 희망, 행복, 슬픔, 시간, 오늘, 내일, 어제, 아침, 저녁, 밤, 세월, 우리
```

**동사 어근:**
```
먹, 가, 오, 보, 하, 되, 주, 받, 쓰, 읽, 듣, 말하,
걷, 뛰, 앉, 서, 자, 일어나, 만들, 찾, 알, 모르, 살, 죽,
놀, 울, 웃, 사, 팔, 열, 닫, 넣, 빼, 타, 내리, 올리, 마시, 입, 벗
```

**형용사 (어근 + 관형형):**
```
관형형: 귀여운, 예쁜, 멋진, 착한, 좋은, 나쁜, 큰, 작은, 높은, 낮은,
        긴, 짧은, 빠른, 느린, 밝은, 어두운, 아름다운, 슬픈, 기쁜, 오랜
```

**조사 (길이 역순 매칭):**
```
3음절: 에서는, 에서도, 으로는, 으로도, 에게서
2음절: 에서, 으로, 에게, 까지, 부터, 처럼, 만큼
1음절: 은, 는, 이, 가, 을, 를, 에, 의, 와, 과, 도, 로, 서, 만
```

**어미 (길이 역순 매칭):**
```
2음절: 는다, ㄴ다, 었다, 았다, 겠다, 한다
1음절: 는, 은, 을, ㄴ, ㄹ, 고, 며, 면, 지, 게, 서, 니, 자, 다
```

### 10.3 검증된 분리 결과

```
"귀여운"     → [형용사: 귀여운]
"고양이가"   → [명사: 고양이] + [조사: 가]
"밥을"       → [명사: 밥] + [조사: 을]
"먹는다."    → [동사: 먹] + [어미: 는다] + [구두점: .]
"강아지가"   → [명사: 강아지] + [조사: 가]
"물을"       → [명사: 물] + [조사: 을]
```

### 10.4 C 구현 시 주의

- 사전: 정렬 배열 + 이진 탐색 또는 해시 테이블
- 한국어 문자는 UTF-8에서 3바이트. 문자 단위 비교 시 바이트 경계 처리 필수
- 최장일치: 사전 항목을 길이 역순 정렬 후 순차 검색, 또는 트라이(Trie)

---

## 11. 학습과 추론

### 11.1 학습

```
1. 텍스트 입력
2. 절 단위로 분리
3. 각 절에 대해:
   a. 3레이어 비트맵 생성 (기본+1, 단어+2, 형태소+1)
   b. 합산 → 1장 (256×256 RGBA)
   c. 방향성 RGB 업데이트 (§9.2)
   d. 매칭 파이프라인으로 기존 키프레임과 비교 (§9.3~9.5)
   e. 유사 → 델타 저장 / 이질 → 새 키프레임
4. 프레임 순서와 토픽을 메타데이터로 저장
```

### 11.2 추론

```
1. 입력 텍스트 → 절 분리 → 3레이어 합산 → 1장
2. 방향성 RGB 업데이트
3. 매칭 파이프라인 실행:
   a. 활성 픽셀 겹침으로 Top-K 후보 선택
   b. RGB 가중 코사인으로 최종 매칭
4. 매칭된 키프레임의 델타 체인 조회
5. 프레임 순서 정보로 "다음에 올 절" 예측
```

### 11.3 응답 생성

X좌표가 곧 바이트 값이므로,
격자의 밝기 패턴을 읽으면 곧바로 바이트 시퀀스로 복원된다.
매칭된 키프레임의 "다음 프레임"이 곧 응답 텍스트의 패턴이다.

---

## 12. C 구현 가이드

### 12.1 파일 구조

```
spatial_ai/
├── include/
│   ├── spatial_grid.h      # 256×256 격자, 인코딩/디코딩
│   ├── spatial_layers.h    # 3-레이어 합산 엔진
│   ├── spatial_morpheme.h  # 한국어 형태소 분리기
│   ├── spatial_keyframe.h  # 키프레임/델타/프레임
│   ├── spatial_match.h     # 코사인 유사도, 패턴 매칭
│   └── spatial_context.h   # 컨텍스트 프레임 관리
├── src/
│   ├── spatial_grid.c
│   ├── spatial_layers.c
│   ├── spatial_morpheme.c
│   ├── spatial_keyframe.c
│   ├── spatial_match.c
│   └── spatial_context.c
├── dict/
│   ├── nouns.txt
│   ├── verbs.txt
│   ├── adjectives.txt
│   ├── particles.txt
│   └── endings.txt
├── tests/
│   ├── test_grid.c
│   ├── test_layers.c
│   ├── test_morpheme.c
│   ├── test_keyframe.c
│   ├── test_match.c
│   └── test_context.c
├── Makefile
└── SPEC.md
```

### 12.2 핵심 자료구조

```c
#define GRID_SIZE 256
#define GRID_TOTAL (GRID_SIZE * GRID_SIZE)

// ─── 격자 ───────────────────────────────
typedef struct {
    uint16_t A[GRID_SIZE][GRID_SIZE];
    uint8_t  R[GRID_SIZE][GRID_SIZE];
    uint8_t  G[GRID_SIZE][GRID_SIZE];
    uint8_t  B[GRID_SIZE][GRID_SIZE];
} SpatialGrid;

// ─── 3-레이어 비트맵 ────────────────────
typedef struct {
    uint16_t base[GRID_SIZE][GRID_SIZE];     // 기본 +1
    uint16_t word[GRID_SIZE][GRID_SIZE];     // 단어 +2
    uint16_t morpheme[GRID_SIZE][GRID_SIZE]; // 형태소 +1
} LayerBitmaps;

// ─── 형태소 ─────────────────────────────
typedef enum {
    POS_NOUN, POS_VERB, POS_ADJ,
    POS_PARTICLE, POS_ENDING, POS_PUNCT, POS_UNKNOWN
} PartOfSpeech;

typedef struct {
    PartOfSpeech pos;
    char token[64];
} Morpheme;

// ─── 키프레임 ───────────────────────────
typedef struct {
    uint32_t    id;
    char        label[64];
    SpatialGrid grid;
    uint32_t    text_byte_count;
} Keyframe;

// ─── 델타 ───────────────────────────────
typedef struct {
    uint32_t index;
    int16_t  diff_A;
    int8_t   diff_R;
    int8_t   diff_G;
} DeltaEntry;  // 8 bytes

typedef struct {
    uint32_t     id;
    uint32_t     parent_id;
    char         label[64];
    uint32_t     count;
    DeltaEntry*  entries;
    float        change_ratio;
} DeltaFrame;

// ─── 컨텍스트 프레임 ────────────────────
typedef struct {
    uint32_t    frame_id;
    uint32_t    topic_hash;
    char        topic_label[64];
    uint8_t     frame_type;     // 0=I, 1=P
    uint32_t    parent_id;
    SpatialGrid grid;
} ContextFrame;

// ─── 매칭 후보 ──────────────────────────
#define TOP_K 8

typedef struct {
    uint32_t id;
    float    score;
} Candidate;

// ─── 엔진 ───────────────────────────────
typedef struct {
    Keyframe*     keyframes;
    uint32_t      kf_count;
    uint32_t      kf_capacity;
    DeltaFrame*   deltas;
    uint32_t      df_count;
    uint32_t      df_capacity;
    ContextFrame* frames;
    uint32_t      frame_count;
    uint32_t      frame_capacity;
} SpatialAI;
```

### 12.3 핵심 함수

```c
// 초기화
SpatialAI* spatial_ai_create(void);
void       spatial_ai_destroy(SpatialAI* ai);

// 3-레이어 인코딩
void layers_encode_clause(const char* clause_text,
                          LayerBitmaps* out_layers,
                          SpatialGrid* out_combined);

// 방향성 RGB 업데이트
void update_rgb_directional(SpatialGrid* grid);

// 형태소 분석
uint32_t morpheme_analyze(const char* word,
                          Morpheme* out, uint32_t max);

// 키프레임/델타
uint32_t ai_store_auto(SpatialAI* ai,
                       const char* clause_text,
                       const char* label);

// 매칭 파이프라인
uint32_t overlap_score(const SpatialGrid* a,
                       const SpatialGrid* b);
float    cosine_rgb_weighted(const SpatialGrid* a,
                             const SpatialGrid* b);
float    rgb_weight(uint8_t r1, uint8_t r2,
                    uint8_t g1, uint8_t g2,
                    uint8_t b1, uint8_t b2);
uint32_t find_best_match(SpatialAI* ai,
                         SpatialGrid* input,
                         float* out_similarity);

// 컨텍스트 프레임
uint32_t ai_add_frame(SpatialAI* ai,
                      const char* clause_text,
                      const char* topic);

// 추론
uint32_t ai_predict(SpatialAI* ai,
                    const char* input_text,
                    float* out_similarity);
```

### 12.4 구현 순서

```
Phase 1: 기본 격자
  spatial_grid.c — SpatialGrid, grid_encode(), 터미널 출력

Phase 2: 형태소 분리기
  spatial_morpheme.c — 사전 로드, 최장일치

Phase 3: 3-레이어 합산
  spatial_layers.c — 3장 비트맵 생성 → 합산 → 1장

Phase 4: 방향성 RGB
  spatial_match.c — update_rgb_directional(), α/β/γ 파라미터

Phase 5: 매칭 파이프라인
  spatial_match.c — overlap_score(), cosine_rgb_weighted(),
                    find_best_match(), Top-K 선택

Phase 6: 키프레임/델타
  spatial_keyframe.c — 저장, 비교, 델타 연산, I/P 판정

Phase 7: 컨텍스트 프레임
  spatial_context.c — 프레임 배열, 순서 관리, 토픽

Phase 8: 테스트
  각 모듈별 단위 테스트 + 통합 테스트
```

---

## 13. 고유 특성

### 13.1 해석 가능성
파라미터가 눈에 보인다. 히트맵을 열면 AI가 무엇을 기억하는지 시각적으로 확인 가능.

### 13.2 무한 파라미터
프레임이 쌓이므로 파라미터 상한이 없다. 고정 행렬과 근본적으로 다르다.

### 13.3 무한 컨텍스트
프레임 스택은 디스크가 허용하는 한 무한 확장. LLM의 컨텍스트 윈도우 제한이 없다.

### 13.4 증분 학습
새 데이터는 델타 또는 새 프레임만 추가. 전체 재학습 불필요.

### 13.5 되감기 / 분기
델타 체인을 역방향으로 타면 학습 과정 추적. 동일 키프레임에서 복수 분기 가능.

### 13.6 경량성
1장 320KB. 엔진 코어 수 MB. Termux, 임베디드, 브라우저에서 동작.

---

## 14. 향후 확장

| 우선순위 | 항목 | 설명 |
|---------|------|------|
| P0 | C Phase 1~3 | 격자 + 형태소 + 3레이어 합산 |
| P1 | C Phase 4~5 | 방향성 RGB + 매칭 파이프라인 (겹침→RGB코사인) |
| P2 | C Phase 6~7 | 키프레임/델타 + 컨텍스트 프레임 |
| P3 | SIMD 최적화 | AVX2 겹침(popcount) + 코사인 벡터화 |
| P4 | 4096×4096 스케일링 | 절 32장 배치, 문서 단위 |
| P5 | 영어/다국어 분리기 | 공백 + 접미사 규칙 기반 |
| P6 | CanvasOS 통합 | Branch/Rewind + 키프레임/델타 연결 |
| P7 | BMP 출력 | 격자를 BMP 이미지로 시각화 |

---

## 부록: 검증 이력

### v1.0 검증 (38 PASS / 0 FAIL)
좌표계, 4레이어, 유사도 80%, 형태소 분리, 메모리 산정 등.

### v2.0 검증 (16 PASS / 0 FAIL)
3-레이어 합산 정합성, 가중치 효과(밥 위치: base=1+word=2+morph=1=4),
유사 절 비교(78.5%, 16px), 이질 절 비교(0.0%, 91px),
복수 절 프레임 배열(3개),
합산 보존 법칙(151=40+74+37), 프레임 간 유사도 매트릭스.

### v3.0 검증 (10 PASS / 0 FAIL)
방향성 RGB 업데이트 수렴, 활성 픽셀 겹침 coarse 필터,
RGB 가중 코사인 정밀 매칭, Top-K 2단계 검색 정확도(전수==2단계 100%),
이전 실패 케이스 해결(KF2 겹침 20px > KF4 5px),
5개 다양 입력 매칭 일치, 속도 1.9x, SPEC 정합성 6항목.
