# AI 생성 리뷰 탐지 — Handcrafted Feature × RoBERTa Cross-Attention

> **정형 수치 피처(23차원)와 비정형 텍스트 임베딩(RoBERTa)을 Cross-Attention으로 융합하는 이종(heterogeneous) 신호 융합 분류기.**
> Yelp 리뷰 20,000건(Human 10K / LLM 생성 10K)에서 **Test F1 0.9832 (5-Run 평균), AUROC 0.9989** 달성.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?logo=tensorflow&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-FFD21E)
![License](https://img.shields.io/badge/License-MIT-green)

한성대학교 빅데이터프로그래밍 산학협력프로젝트 (26-1학기) · **Neural Nexus** 팀 프로젝트의 개인 포트폴리오 정리본입니다.
본인 담당 범위와 팀 공동 산출물을 [기여 범위](#기여-범위) 절에서 명시적으로 구분했습니다.

---

## 문제 정의

LLM이 생성한 텍스트 탐지가 어려운 이유는, **판별 신호가 의미(semantic)가 아니라 통계적 문체(stylometric)에 있기** 때문입니다.
LLM이 재작성한 리뷰는 내용·감성·평점 정합성이 모두 자연스러워 의미 수준에서는 진짜와 구분되지 않습니다.
실제로 구분되는 지점은 "문장별 예측 난이도의 분산이 작다", "가독성 지표가 균일하다" 같은 **표층 통계량**인데,
이 신호는 트랜스포머 임베딩이 의미 중심으로 압축하는 과정에서 상당 부분 손실됩니다.
반대로 수치 피처만 쓰면 문맥 정보를 통째로 버리게 됩니다.
즉 이 문제의 본질은 탐지 그 자체가 아니라, **성질이 전혀 다른 두 신호(저차원 정형 스칼라 vs 고차원 비정형 시퀀스)를 어느 지점에서 어떻게 섞을 것인가**입니다.

이 구조는 리뷰 탐지에만 국한되지 않습니다.
센서 계측값(온도·진동 RMS·전류) + 정비 로그 텍스트로 장비 이상을 판정하는 **스마트팩토리 예지보전**,
레이더/EO-IR 트랙 파라미터 + 전술 메시지 텍스트를 결합하는 **방산 다중센서 상황인식**,
로봇 상태 벡터 + 자연어 명령을 융합하는 **로보틱스 인터페이스** 모두 동일한 형식의 문제입니다.
공통 과제는 세 가지 — (1) 차원 스케일이 100배 차이 나는 신호를 한 잠재공간에 정렬하는 것,
(2) 가변 길이 시퀀스의 패딩이 라벨 누수가 되지 않게 마스킹하는 것,
(3) 판단 근거를 어텐션 가중치로 추적해 **설명 가능한 판정**을 만드는 것.
본 저장소는 이 세 가지를 실제 코드와 대조 실험으로 다룹니다.

---

## 모델 구조

### 설계 축 — 융합 지점을 3단계로 나눠 대조 실험

| 단계 | 융합 지점 | 표현 | 한 줄 요약 |
|------|-----------|------|-----------|
| ① Simple Concat | 분류기 직전 | CLS 벡터 ⊕ HC 벡터 | 두 신호가 **섞이지 않고 나란히** 들어감 |
| ② Cross-Attention | 인코딩 후 | CLS(Q) ↔ HC(K,V) | 텍스트 요약이 **HC를 조회**해 보정 |
| ③ 양방향 Co-Attention | 시퀀스 레벨 | 토큰 256개 ↔ HC 토큰 23개 | **토큰 ↔ 피처 단위 상호 참조** (팀 제안 모델) |

### ② HC + RoBERTa Cross-Attention (`src` 외부 · `notebooks/08`)

```
리뷰 텍스트 ──▶ roberta-base (frozen) ──▶ CLS [B, 768] ──▶ Linear(d) ──▶ Q [B, 1, d]
                                                                            │
HC 23개 ──▶ log(1+|x|)·sign(x) ──▶ [B, 23] ──▶ Linear(d) ──▶ K, V [B, 1, d]─┤
                                                                            ▼
                                              MultiHeadAttention(heads=8, d=128)
                                                                            │
                                            Residual(Q + Dropout(attn)) → LayerNorm
                                                                            │
                                              Linear(d→64) → ReLU → Dropout → Linear(64→2)
```

`HCRoBertaCrossAttnModel` — Q는 텍스트, K/V는 HC. 즉 **"텍스트 요약 벡터가 어떤 문체 지표를 참조할지"** 를 학습합니다.
최적 하이퍼파라미터: `lr=1e-4 · dropout=0.1 · d_model=128 · batch=64 · num_heads=8` (Greedy Search)

### ③ 양방향 Co-Attention — 시퀀스↔시퀀스 (`src/model.py`)

②의 구조적 한계는 **양쪽 모두 길이-1 시퀀스**라는 점입니다. 벡터 하나가 벡터 하나를 어텐션하면 사실상 게이팅에 가깝고, 어텐션 가중치에서 얻을 해석 정보가 없습니다.
이를 해소하기 위해 팀 제안 모델은 양쪽을 진짜 시퀀스로 만들었습니다 — 텍스트는 마지막 hidden layer 전체(256 토큰), HC는 **FT-Transformer 방식의 FeatureTokenizer** 로 피처 1개당 토큰 1개(23 토큰)를 생성합니다.

```
 text_emb [B,256,768]                                  hc [B,23]
 (frozen RoBERTa last_hidden_state, float16 캐시)           │
        │                                        FeatureTokenizer
   Dense(256)                             out[b,i,:] = x[b,i]·W[i,:] + B[i,:]
        │                                                  │
   T [B,256,256]                                      F [B,23,256]
        │                                                  │
        ├──── K,V ────▶ ┌──────────────────┐ ◀──── Q ──────┤
        │               │ 방향1  MHA(h=4)   │               │   HC → Text 참조
        │               │ + padding K/V 마스크│              │
        │               └────────┬─────────┘               │
        │                 GlobalAvgPool1D → co_text [B,256] │
        │                                                  │
        ├──── Q ──────▶ ┌──────────────────┐ ◀─── K,V ─────┤
        │               │ 방향2  MHA(h=4)   │               │   Text → HC 참조
        │               └────────┬─────────┘               │
        │               MaskedMeanPool(mask) → co_hc [B,256]│
        ▼                                                  ▼
              Concat[co_text ‖ co_hc]  [B, 512]
              → Dense(256, ReLU) → Dropout(0.3) → Dense(2, softmax)
```

**설계 포인트 3가지**

1. **FeatureTokenizer로 피처 정체성 보존** — HC 23개를 MLP로 한 번에 요약하면 개별 피처가 사라집니다.
   피처마다 고유 임베딩 `W[i], B[i]` 를 두어 `perplexity` 토큰과 `adverb` 토큰이 독립적인 어텐션 대상이 되게 했습니다.
2. **패딩 마스크 이중 적용** — 본 데이터셋은 human 리뷰가 AI 리뷰보다 길어, **패딩 양 자체가 라벨 누수**가 됩니다.
   그래서 (a) 방향1에서는 패딩 토큰을 K/V에서 차단하고, (b) 방향2에서는 패딩 쿼리의 출력을 풀링에서 제외합니다(`MaskedMeanPool`).
3. **RoBERTa frozen + 임베딩 1회 캐시** — 파인튜닝 없이 `last_hidden_state`를 float16으로 디스크 캐시(`.npy`, memmap 부분 적재)하여,
   모든 비교 실험이 **완전히 동일한 텍스트 표현** 위에서 돌아가도록 통제했습니다. 융합 구조의 차이만 성능 차이로 남습니다.

> 수식 참고: `perplexity`, `flesch_reading_ease`, `gunning_fog` 등 HC 피처의 정의식은
> 원 데이터셋 논문(AiGen-FoodReview, ICWSM 2024)의 `formulas_supplement.md`를 따릅니다.

---

## 실험 결과

**공통 조건** — Yelp 20,000건(Human 10K / AI 10K), stratified 70/15/15 split(seed=42) → Train 14,000 / Val 3,000 / **Test 3,000**.
아래 수치는 모두 **held-out test set** 기준이며, 노트북 실행 출력에서 직접 확인한 값만 기재했습니다.

### 주 실험 — 융합 구조 비교 (동일 데이터 · 동일 split)

| 모델 | 입력 | Accuracy | Precision | Recall | **F1** | AUROC | 실행 |
|------|------|----------|-----------|--------|--------|-------|------|
| HC only (MLP) | HC 23개 | 0.9217 | 0.9191 | 0.9249 | **0.9220** ±0.0053 | 0.9780 | 5-Run (seed 42~46) |
| ① HC + RoBERTa **Simple Concat** | CLS[768] ⊕ HC[23] | 0.9831 | 0.9824 | 0.9839 | **0.9831** ±0.0011 | 0.9990 | 5-Run (seed 42~46) |
| ② HC + RoBERTa **Cross-Attention** | CLS(Q) ↔ HC(K,V) | 0.9832 | 0.9808 | 0.9857 | **0.9832** ±0.0007 | 0.9989 | 5-Run (seed 42~46) |
| ③ **양방향 Co-Attention** (팀 제안) | 토큰 256 ↔ HC 토큰 23 | **0.9870** | **0.9893** | 0.9847 | **0.9870** | — | 단일 실행 (EarlyStopping) |

<sub>근거 파일 — HC only: `notebooks/05_baseline_hc_mlp.ipynb` 셀 13 (SUMMARY ACROSS SEEDS) ·
①: `notebooks/07_hc_roberta_simple_concat.ipynb` 셀 19 (결과 요약 표) ·
②: `notebooks/08_hc_roberta_cross_attention.ipynb` 셀 17 (5-Run 출력; Accuracy/Precision/Recall은 seed별 출력값의 산술평균) ·
③: `notebooks/04_bidirectional_coattention.ipynb` 셀 19 (`TEST: {...}` 출력). ③은 AUROC를 산출하지 않았고 5-Run 반복 없이 단일 실행이므로, ①②와 표준편차·AUROC 기준의 직접 비교는 불가합니다.</sub>

**읽는 법 — 무엇이 성능을 만들었는가**

- **HC만으로도 F1 0.9220.** `perplexity`·`burstiness` 같은 LM 기반 통계 7~8개가 이미 강한 판별력을 갖습니다. 즉 이 문제는 "정형 피처가 실제로 정보를 갖는" 융합 문제입니다.
- **텍스트를 더하면 +0.061.** Simple Concat만으로 0.9831까지 올라가며, 두 신호가 상호보완적임이 확인됩니다.
- **CLS 레벨에서는 Concat ≈ Cross-Attention (0.9831 vs 0.9832).** 성능차는 표준편차(±0.001) 안이라 **유의미한 차이가 아닙니다.**
  다만 Cross-Attention 쪽이 5-Run 표준편차가 더 작고(0.0007 vs 0.0011) Recall이 높습니다(0.9857 vs 0.9839).
  이 결과의 해석은 "어텐션이 우월하다"가 아니라 **"길이-1 시퀀스끼리의 어텐션은 구조적으로 Concat 이상을 하기 어렵다"** 입니다.
- **시퀀스 레벨로 올리면 0.9870.** 융합 지점을 CLS에서 토큰↔피처 단위로 내리자 test F1이 0.9832 → 0.9870로 올랐습니다. 오분류 건수로는 3,000건 중 약 50건 → 39건 수준입니다.
  (단일 실행 값이므로 반복 실험으로 재확인이 필요합니다 — [한계](#한계-및-향후-과제) 참고.)

### 참고 — 텍스트 단독 백본 베이스라인 (직접 비교 불가)

| 모델 | Accuracy | Precision | Recall | F1 | AUROC |
|------|----------|-----------|--------|----|-------|
| BERT-base-uncased (frozen) + MLP | 0.9767 | 0.9753 | 0.9766 | 0.9760 | 0.9975 |
| ModernBERT-base (frozen, mean-pool) + MLP | 0.9773 | 0.9671 | 0.9879 | 0.9774 | 0.9981 |

<sub>근거: `notebooks/06_baseline_bert_modernbert.ipynb` 셀 4 · 셀 10.
**주의** — 이 두 실험은 20,000건 전체가 아니라 **5,000건 부분집합** 위에서 단일 시드로 수행되었고 split 구성도 주 실험과 다릅니다.
따라서 위 주 실험 표와 같은 축에서 비교하면 안 되며, "텍스트 단독 백본의 대략적 수준" 참고용으로만 기재합니다.</sub>

### 하이퍼파라미터 탐색

전체 격자(3×3×3×3×3 = 243조합) 대신 **Greedy Search**(LR → Dropout → d_model → Batch → Heads 순차 고정, 15회 학습)를 사용했습니다.
Cross-Attention 기준 탐색 결과 — LR이 가장 민감(3e-5: 0.9815 → 1e-4: 0.9831), d_model은 512에서 뚜렷하게 악화(0.9799)되어 과파라미터화 구간을 확인했습니다.
(`notebooks/08_hc_roberta_cross_attention.ipynb` 셀 16)

---

## 피처 엔지니어링

`src/features.py`의 `FeatureExtractor`가 리뷰 1건 → **23차원 벡터**를 생성합니다.
원본 28개 중 LIWC 계열 감성 지표 5개(`anger`, `sadness`, `posemo`, `anx`, `negate`)는 제거했습니다.

| 카테고리 | 개수 | 피처 | 도구 |
|----------|------|------|------|
| 카운트 | 7 | `syllable`, `lexicon`, `sentence`, `char`, `letter`, `polysyllab`, `monosyllab` | textstat |
| 가독성 | 6 | `smog_index`, `flesch_reading_ease`, `flesch_kincaid_grade`, `fog_scale`, `dale_chall`, `reading_time` | textstat |
| 감성 | 2 | `sentiment`(polarity), `subjectivity` | TextBlob |
| **LM 기반** | 2 | `perplexity`, `burstiness` | GPT-2 |
| 품사 | 6 | `nouns`, `adj`, `verbs`, `pronoun`, `adverb`, `article` | spaCy (`en_core_web_sm`) |

**핵심 피처 — `perplexity` / `burstiness`**

단순 텍스트 통계가 아니라 **GPT-2를 문장 단위 측정 도구로 사용한 파생 신호**입니다.
리뷰를 문장으로 분할(`(?<=[.!?])\s+`, 3단어 미만 제외) → 각 문장의 GPT-2 손실 → `exp(loss)`로 문장별 perplexity 배열 `P`를 만든 뒤:

- `perplexity  = mean(P)` — 텍스트 전체의 평균 "의외성"
- `burstiness  = std(P) / mean(P)` — 변동계수(CV). **문장 간 의외성이 얼마나 들쭉날쭉한지.** 사람 글은 쉬운 문장과 어려운 문장이 섞여 CV가 크고, LLM 생성문은 균일해 CV가 작습니다.

> 엣지 케이스 처리: 유효 문장이 1개뿐이면 `std=0`이 되어 `burstiness=0`이라는 인위적 값이 나옵니다(한 문장 리뷰의 artifact). 코드 주석에 명시해 두었습니다.

**스케일링 — 모델별로 다르게 적용**

| 모델 | 변환 | 이유 |
|------|------|------|
| ①② Concat / Cross-Attention (PyTorch) | `log(1+\|x\|)·sign(x)` | `char`(수백) ↔ `sentiment`(−1~1) 간 스케일 격차를 로그로 압축 |
| ③ 양방향 Co-Attention (TF/Keras) | `StandardScaler` (**train split에만 fit**) | FeatureTokenizer가 선형 임베딩이라 z-score가 적합. val/test에는 transform만 적용해 **정보 누수 차단** |

**데이터 누수 방지 장치** — ⓐ scaler는 train에만 fit, ⓑ 임베딩 캐시 생성 후 `df.iloc[i]` ↔ `MASK[i]` 행 순서 정합성을 `assert`로 재검증(`src/dataset.py`), ⓒ split은 `SEED=42` 고정 stratified.

---

## 저장소 구조

```
hc-roberta-cross-attention/
├── src/                                    # 양방향 Co-Attention 파이프라인 (패키지)
│   ├── config.py                           #   경로 · 하이퍼파라미터 상수
│   ├── features.py                         #   FeatureExtractor — HC 23개 추출
│   ├── dataset.py                          #   로드 · stratified split · 스케일링 · RoBERTa 임베딩 캐시
│   ├── model.py                            #   FeatureTokenizer · MaskedMeanPool · build_model()
│   ├── train.py                            #   학습 + val/test 평가 진입점
│   ├── attention.py                        #   학습된 모델의 attention weight 분석
│   └── seed.py                             #   random/numpy/torch/tf 시드 일괄 고정
├── notebooks/
│   ├── 01_preprocessing.ipynb              #   Yelp 원본 필터링 · 정제
│   ├── 02_augmentation.ipynb               #   LLM 재작성으로 AI 리뷰 10K 생성
│   ├── 03_feature_extraction.ipynb         #   HC 23개 추출 → parquet
│   ├── 04_bidirectional_coattention.ipynb  #   ③ 양방향 Co-Attention (학습·평가·attention 분석)
│   ├── 05_baseline_hc_mlp.ipynb            #   베이스라인: HC only MLP
│   ├── 06_baseline_bert_modernbert.ipynb   #   베이스라인: BERT / ModernBERT (5K 부분집합)
│   ├── 07_hc_roberta_simple_concat.ipynb   #   ① Simple Concat  ★본인 담당
│   └── 08_hc_roberta_cross_attention.ipynb #   ② Cross-Attention ★본인 담당
├── Makefile                                # setup / lint / verify / reproduce / clean
├── Dockerfile                              # CPU 재현 환경 (TF + PyTorch + spaCy 모델)
├── DATA.md                                 # Yelp 데이터 확보 및 parquet 구성 안내
├── requirements.txt                        # 버전 고정 의존성
├── LICENSE                                 # MIT
└── README.md
```

> `data/`와 `models/`는 `.gitignore` 대상입니다. 원본 Yelp JSON(약 5GB), 최종 `data_yelp.parquet`,
> RoBERTa 임베딩 캐시(약 7.8GB), 학습 가중치는 저장소에 포함되지 않습니다.

---

## 실행 방법

```bash
git clone <this-repo> && cd hc-roberta-cross-attention
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**③ 양방향 Co-Attention 학습 · 평가**

```bash
mkdir -p data && cp /path/to/data_yelp.parquet data/    # 20,000행 × (text + label + HC 23)

python -m src.train --smoke   # 2K 서브샘플 · 2 epochs — 파이프라인 동작만 빠르게 확인
python -m src.train           # 전체 학습 (첫 실행 시 RoBERTa 임베딩 캐시 자동 생성, 약 20분 / 7.8GB)
python -m src.attention       # 학습된 가중치 로드 → HC feature별 attention weight 표 출력
```

산출물은 `models/`에 저장됩니다 — `coattn_best.weights.h5`, `coattn_grid_results.csv`(val), `coattn_best_test.csv`(test).

**①② 비교 모델** — `notebooks/07`, `notebooks/08`을 순서대로 실행합니다 (PyTorch / GPU 권장, Colab T4 기준 RoBERTa CLS 임베딩 추출 약 30~40분 + Greedy Search 약 30~40분).

> ⚠️ **macOS 주의** — `src/*` 진입점은 **TensorFlow를 pandas/pyarrow보다 먼저** import합니다.
> pandas 3.x가 로드하는 pyarrow와 TF가 각자 내장한 abseil 심볼이 충돌하면, 첫 `fit()`의 `absl::Mutex`가
> Arrow 쪽 구현에 바인딩되어 **CPU 0%인 채로 영구 데드락**에 빠집니다. import 순서를 바꾸지 마십시오.

---

## 재현

위 "실행 방법"이 스크립트를 직접 호출하는 경로라면, 이 절은 **환경째로 고정해 한 명령으로 돌리는** 경로입니다.

### 데이터 준비

Yelp Open Dataset은 라이선스상 재배포할 수 없어, 원본도 파생 `data_yelp.parquet` 도 저장소에 없습니다.
내려받는 곳, parquet 스키마, 노트북 01→02→03 으로 데이터셋을 구성하는 방법은 [DATA.md](DATA.md)에 정리했습니다.

### Makefile

```bash
make help       # 타깃 목록
make setup      # requirements.txt + spaCy en_core_web_sm + TextBlob 코퍼스
make lint       # src/ 문법 검사
make verify     # 데이터 없이 ③ 제안 모델 빌드·forward 확인
make reproduce  # ③ 학습 → attention 분석 (데이터 필요)
make clean      # 캐시·산출물 정리 (임베딩 캐시는 보존)
```

| 타깃 | 하는 일 | 데이터 필요 | 소요 시간 |
|---|---|---|---|
| `make verify` | `build_model()` 로 FeatureTokenizer + 양방향 Co-Attention 을 빌드하고 난수 `(text_emb, mask, hc)` 로 forward 통과 확인 | ✗ | 수십 초 |
| `make lint` | `compileall` + `ruff`(설치 시, 문법·미정의 이름 규칙만) | ✗ | 1초 미만 |
| `make smoke` | `python -m src.train --smoke` — 2K 서브샘플 · 2 epochs | **✓** | 수 분 |
| `make train` | ③ 전체 학습. 첫 실행 시 RoBERTa 임베딩 캐시 자동 생성(약 20분 / 7.8GB) | **✓** | GPU 기준 수십 분 |
| `make attention` | 학습된 가중치로 HC 피처별 attention weight 표 출력 | 가중치 필요 | 수 분 |
| `make reproduce` | `train` → `attention` 순차 실행 | **✓** | 위 둘의 합 |
| `make notebooks` | ①② 비교 모델(Simple Concat / Cross-Attention) 재현 방법 안내 | — | — |

> **소요 시간 주의** — 위 학습 시간은 원 실험 당시 Colab(T4) 기록에 근거한 추정치입니다.
> 본 재현 레이어를 정비하는 시점에 `data_yelp.parquet` 이 로컬에 없어
> `make reproduce` 를 끝까지 돌려 실측하지 못했습니다.
> 데이터 없이 검증한 것은 `make lint` 와 `make verify` 입니다.

### Docker

파이썬·TensorFlow·PyTorch·spaCy 모델 버전까지 한 번에 고정합니다.

```bash
make docker-build          # = docker build -t hc-roberta:cpu .
make docker-shell          # data/models 볼륨을 붙여 셸 진입
```

이미지에는 **소스만** 들어갑니다. 데이터·임베딩 캐시·가중치는 볼륨으로 마운트합니다.

```bash
docker run --rm -it \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  hc-roberta:cpu \
  make reproduce
```

- `roberta-base` · `gpt2` 등 사전학습 가중치는 이미지에 굽지 않고 최초 실행 시 HF 허브에서 내려받습니다(`HF_HOME=/app/.cache/huggingface`). **실행 시 네트워크가 필요합니다.**
- ③(TensorFlow)과 ①②(PyTorch)를 모두 담기 때문에 이미지가 큽니다. 한쪽만 필요하면 `requirements.txt` 의 해당 블록을 지우고 빌드하세요.
- `tf-keras` 가 `requirements.txt` 에 들어 있습니다. `transformers` 의 TensorFlow 연동 경로가 Keras 3를 아직 지원하지 않아, 이 패키지가 없으면 `notebooks/06` 의 `import sentence_transformers` 가 `ValueError` 로 실패합니다. Keras 3 자체는 그대로 유지되므로 `src/model.py` 의 `keras.ops` 사용에는 영향이 없습니다.

### 시드 고정

`src/seed.py` 의 `set_seed()` 가 `random` · `numpy` · `torch`(+cuDNN deterministic) · `tensorflow` 를
한 번에 고정하고, **실제로 고정된 항목을 dict 로 돌려줍니다.**
기본값은 `src/config.py` 의 `SEED = 42` 입니다.

```python
import tensorflow as tf          # ← 반드시 pandas 보다 먼저 (아래 주의)
import pandas as pd
from src.seed import set_seed
set_seed()        # {'random': True, 'numpy': True, 'torch': True, 'tensorflow': True}
```

> ⚠️ **`src/seed.py` 는 TensorFlow 를 직접 import 하지 않습니다.**
> 위 "macOS 주의"의 import 순서 규칙(TF를 pandas/pyarrow보다 먼저)을 깨지 않기 위해,
> TF가 **이미 로드되어 있을 때만** `tf.random.set_seed()` 를 겁니다.
> TF 시드까지 걸렸는지는 반환값의 `'tensorflow'` 키로 확인하세요.

기존 `src/train.py` 의 `set_seed()`(random / numpy / tf)는 **그대로 두었습니다.**
`src/seed.py` 는 그것을 대체하지 않으며, 노트북 등 `src/train.py` 를 거치지 않는
PyTorch 경로(①②)에서 쓰라고 추가한 것입니다.

---

## 기술 스택

| 영역 | 사용 기술 |
|------|-----------|
| 언어 | Python 3.10+ |
| 딥러닝 | **PyTorch 2.x** (①② 비교 모델), **TensorFlow 2.16+ / Keras 3** (③ 제안 모델) |
| 사전학습 모델 | `roberta-base`, `gpt2` (perplexity 측정), `bert-base-uncased`, `answerdotai/ModernBERT-base` — 전부 **frozen** |
| NLP · 피처 | HuggingFace Transformers, spaCy, textstat, TextBlob |
| 데이터 | pandas, NumPy, PyArrow(parquet), `np.memmap` 기반 대용량 임베딩 부분 적재 |
| 실험 | scikit-learn(split·metrics), Greedy hyperparameter search, 5-Run seed 반복, EarlyStopping |
| 환경 | Jupyter / Google Colab (T4), macOS(MPS) · CUDA 양쪽 대응 |

---

## 한계 및 향후 과제

1. **제안 모델(③)의 반복 실험 부재.** 0.9870은 단일 실행 값입니다. ①②는 5-Run으로 ±0.001 수준의 분산을 확인했지만 ③은 그렇지 않아, 0.9832 → 0.9870의 개선폭이 시드 분산을 넘어선다고 단정할 수 없습니다. **최소 5-Run 반복 + AUROC 산출**이 1순위 후속 작업입니다.
2. **AI 리뷰 생성원의 편중.** AI 라벨 10K는 Qwen3-1.7B와 Llama-3.2-1B-Instruct 두 모델로만 생성되었습니다. 98%대 성능에는 "두 소형 모델의 고유 문체를 학습한" 부분이 섞여 있을 수 있으며, **미학습 생성기(예: GPT-4급)에 대한 cross-generator 일반화 성능은 검증되지 않았습니다.**
3. **frozen 백본.** 계산 비용과 비교 공정성(모든 모델 동일 임베딩) 때문에 RoBERTa를 동결했습니다. 파인튜닝 시 성능·해석성이 어떻게 변하는지는 미측정입니다.
4. **성능 포화로 인한 구조 비교의 한계.** 전 모델이 F1 0.98대에 몰려 있어 융합 구조 간 차이를 분해하기 어렵습니다. **난이도를 올린 평가셋**(짧은 리뷰, 사람이 부분 편집한 리뷰, 도메인 외 리뷰)을 따로 구성해야 구조의 우열이 드러날 것으로 봅니다.
5. **어텐션 해석의 정량화 미완.** `src/attention.py`가 방향1/방향2 가중치를 집계하는 probe를 제공하지만, 저장소 내 노트북에는 해당 셀의 출력이 남아 있지 않습니다. 재실행 시 산출되는 피처별 attention 분포를 **permutation importance 등 독립 지표와 교차 검증**하는 작업이 남아 있습니다.
6. **도메인 전이 미실험.** 서론에서 언급한 산업 신호(센서 시계열 + 정비 로그) 적용은 아직 가설 단계입니다. 텍스트 토큰 시퀀스를 시계열 윈도우로 치환했을 때 FeatureTokenizer·마스킹 설계가 그대로 유효한지 확인이 필요합니다.

---

## 기여 범위

본 저장소는 **Neural Nexus 팀(4인)** 프로젝트를 개인 포트폴리오 형태로 재정리한 것입니다. 팀원 실명은 기재하지 않습니다.

| 구분 | 항목 | 파일 |
|------|------|------|
| **본인 설계·구현** | ① HC + RoBERTa **Simple Concat** 비교 모델 — 모델 정의, Greedy Search, 5-Run 반복 실험 | `notebooks/07_hc_roberta_simple_concat.ipynb` |
| **본인 설계·구현** | ② HC + RoBERTa **Cross-Attention** 비교 모델 — `HCRoBertaCrossAttnModel`, num_heads 탐색 포함 Greedy Search, 5-Run 반복 실험 | `notebooks/08_hc_roberta_cross_attention.ipynb` |
| **본인 작성** | 본 README의 문제 정의 · 구조 분석 · 결과 해석 · 한계 정리 | `README.md` |
| 팀 공동 (타 팀원 주도) | 전처리 · LLM 증강 · 데이터셋 구성 | `notebooks/01`, `notebooks/02` |
| 팀 공동 (타 팀원 주도) | HC 23개 피처 추출 모듈 | `notebooks/03`, `src/features.py` |
| 팀 공동 (타 팀원 주도) | ③ 양방향 Co-Attention 제안 모델 설계·구현 및 모듈화 | `notebooks/04`, `src/model.py`, `src/dataset.py`, `src/train.py`, `src/attention.py`, `src/config.py` |
| 팀 공동 (타 팀원 주도) | HC only MLP · BERT / ModernBERT 베이스라인 | `notebooks/05`, `notebooks/06` |

> 팀 공동 산출물은 전체 파이프라인의 재현성과 비교 맥락을 위해 포함했습니다.
> 원본 팀 저장소: `striker13j/BigdataProgramming` (팀 대표 계정). 본 저장소는 그 포크가 아니라, 개인 기여분을 중심으로 재구성한 정리본입니다.

---

## 데이터셋 · 참고 문헌

**데이터셋** — Yelp Open Dataset의 레스토랑 리뷰를 필터링한 human 10,000건 + 이를 LLM(`Qwen/Qwen3-1.7B` 5,000건, `meta-llama/Llama-3.2-1B-Instruct` 5,000건)으로 재작성한 AI 10,000건, 총 **20,000건 균형 데이터셋**.
피처 설계는 AiGen-FoodReview(ICWSM 2024)의 readability/perplexity 속성 체계를 참고했습니다.

- Gambetti & Han, *AiGen-FoodReview: A Multimodal Dataset of Machine-Generated Restaurant Reviews and Images on Social Media*, ICWSM 2024
- Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, arXiv 2019
- Lu et al., *ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations*, NeurIPS 2019 — 양방향 co-attention(co-TRM) 구조
- Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*, NeurIPS 2021 — FeatureTokenizer(FT-Transformer)
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019

---

## License

MIT — 자세한 내용은 [LICENSE](LICENSE) 참고.
