# AI 생성 리뷰 탐지 — Handcrafted Feature × RoBERTa Cross-Attention

> **정형 수치 피처(23차원)와 비정형 텍스트 임베딩(RoBERTa)을 Cross-Attention으로 융합하는 이종(heterogeneous) 신호 융합 분류기.**
> Yelp 리뷰 20,000건(Human 10K / LLM 생성 10K)에서 **Test F1 0.9832 (5-Run 평균), AUROC 0.9989** 달성.
>
> **다만 이 수치는 모델 성능이 아니라 벤치마크 난이도를 반영합니다.** 같은 test set에서 TF-IDF bag-of-words만으로 F1 0.9744,
> 파이썬 문자열 통계 11개만으로 F1 0.8867이 나옵니다. 실측 근거는 [누수 감사](docs/누수감사_상세.md) 절에 정리했습니다.

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

## 핵심 결과

**공통 조건** — Yelp 20,000건(Human 10K / AI 10K), stratified 70/15/15 split(seed=42) → held-out **Test 3,000건** 기준.
전체 표·근거 파일·하이퍼파라미터 탐색은 [docs/실험결과_상세.md](docs/실험결과_상세.md)에 있습니다.

| 모델 | 입력 | Accuracy | Precision | Recall | **F1** | AUROC | 실행 |
|------|------|----------|-----------|--------|--------|-------|------|
| HC only (MLP) | HC 23개 | 0.9217 | 0.9191 | 0.9249 | **0.9220** ±0.0053 | 0.9780 | 5-Run (seed 42~46) |
| ① HC + RoBERTa **Simple Concat** | CLS[768] ⊕ HC[23] | 0.9831 | 0.9824 | 0.9839 | **0.9831** ±0.0011 | 0.9990 | 5-Run (seed 42~46) |
| ② HC + RoBERTa **Cross-Attention** | CLS(Q) ↔ HC(K,V) | 0.9832 | 0.9808 | 0.9857 | **0.9832** ±0.0007 | 0.9989 | 5-Run (seed 42~46) |
| ③ **양방향 Co-Attention** (팀 제안) | 토큰 256 ↔ HC 토큰 23 | **0.9870** | **0.9893** | 0.9847 | **0.9870** | — | 단일 실행 (EarlyStopping) |

### 알아둘 것 세 가지

**1. F1 0.9832 중 융합 구조의 고유 기여는 약 0.009입니다.** 같은 test set에서 TF-IDF bag-of-words + 로지스틱 회귀만으로
F1 0.9744가 나옵니다(오분류 76건 vs 50건). 그 아래로는 문자열 통계 11개만으로 0.8867, HC 23개로 0.9114입니다.
0.98이라는 숫자는 모델의 우수성이 아니라 **벤치마크의 난이도**를 읽는 값입니다.
근거: [docs/누수감사_상세.md](docs/누수감사_상세.md).

**2. 생성기가 바뀌면 F1 0.7228까지 떨어집니다.** AI 라벨은 Qwen3-1.7B와 Llama-3.2-1B-Instruct 두 모델로만 생성했습니다.
Llama로 학습해 Qwen으로 평가하면 TF-IDF 분류기가 0.9744 → **0.7228**로 무너집니다.
어휘 신호의 상당 부분이 일반적인 "AI 문체"가 아니라 특정 생성기의 지문이라는 뜻입니다.

**3. 표에서 가장 높은 0.9870(③ 양방향 Co-Attention)은 본인 작업이 아닙니다.** 팀의 다른 팀원이 주도한 제안 모델이며,
본인 담당은 ① Simple Concat과 ② Cross-Attention 비교 모델(`notebooks/07`, `notebooks/08`)입니다.
구분은 [기여 범위](#기여-범위) 절에 명시했습니다.

---

## 빠른 시작

```bash
git clone https://github.com/gawbi/hc-roberta-cross-attention.git && cd hc-roberta-cross-attention
pip install -r requirements.txt && python -m spacy download en_core_web_sm
make verify        # 데이터 없이 제안 모델 빌드·forward 통과 확인 (수십 초)
```

데이터(`data/data_yelp.parquet`)가 있으면 `make reproduce`(③ 학습 → attention 분석), `make audit`(누수 감사, CPU 수 분).
데이터 확보 방법은 [DATA.md](DATA.md), 전체 실행 절차·Docker·시드 고정·기술 스택은 [docs/실행재현_상세.md](docs/실행재현_상세.md)에 있습니다.

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
├── analysis/
│   ├── leakage_audit.py                    #   피처 누수 · 데이터셋 아티팩트 감사 (학습 파이프라인과 독립)
│   └── leakage_results.json                #   감사 실행 결과 (전체 수치)
├── docs/
│   ├── 실험결과_상세.md                    #   융합 구조 비교 전체 표 · 텍스트 단독 백본 · 하이퍼파라미터 탐색
│   ├── 누수감사_상세.md                    #   사소한 베이스라인 · 단일 피처 AUC · bag-of-words · 생성기 교차 전이
│   ├── 피처엔지니어링_상세.md              #   HC 23개 정의 · perplexity/burstiness · 스케일링
│   ├── 실행재현_상세.md                    #   실행 절차 · Makefile · Docker · 시드 고정 · 기술 스택
│   └── images/                             #   감사 플롯 (audit_*.png)
├── notebooks/
│   ├── 01_preprocessing.ipynb              #   Yelp 원본 필터링 · 정제
│   ├── 02_augmentation.ipynb               #   LLM 재작성으로 AI 리뷰 10K 생성
│   ├── 03_feature_extraction.ipynb         #   HC 23개 추출 → parquet
│   ├── 04_bidirectional_coattention.ipynb  #   ③ 양방향 Co-Attention (학습·평가·attention 분석)
│   ├── 05_baseline_hc_mlp.ipynb            #   베이스라인: HC only MLP
│   ├── 06_baseline_bert_modernbert.ipynb   #   베이스라인: BERT / ModernBERT (5K 부분집합)
│   ├── 07_hc_roberta_simple_concat.ipynb   #   ① Simple Concat  ★본인 담당
│   └── 08_hc_roberta_cross_attention.ipynb #   ② Cross-Attention ★본인 담당
├── Makefile                                # setup / lint / verify / reproduce / audit / clean
├── Dockerfile                              # CPU 재현 환경 (TF + PyTorch + spaCy 모델)
├── DATA.md                                 # Yelp 데이터 확보 및 parquet 구성 안내
├── requirements.txt                        # 버전 고정 의존성
├── LICENSE                                 # MIT
└── README.md
```

> `data/`와 `models/`는 `.gitignore` 대상입니다. 원본 Yelp JSON(약 5GB), 최종 `data_yelp.parquet`,
> RoBERTa 임베딩 캐시(약 7.8GB), 학습 가중치는 저장소에 포함되지 않습니다.

---

## 한계 및 향후 과제

> 이 절의 1~6번은 구조·실험 설계상의 한계이고, 데이터셋 아티팩트에 관한 한계는
> [누수 감사](docs/누수감사_상세.md) 절에 실측 수치와 함께 별도로 정리했습니다.
> **요약: F1 0.9832 중 융합 구조가 기여한 몫은 약 0.009이고, 나머지는 데이터셋의 표층 단서에서 옵니다.**

1. **제안 모델(③)의 반복 실험 부재.** 0.9870은 단일 실행 값입니다. ①②는 5-Run으로 ±0.001 수준의 분산을 확인했지만 ③은 그렇지 않아, 0.9832 → 0.9870의 개선폭이 시드 분산을 넘어선다고 단정할 수 없습니다. **최소 5-Run 반복 + AUROC 산출**이 1순위 후속 작업입니다.
2. **AI 리뷰 생성원의 편중 — 감사로 확인됨.** AI 라벨 10K는 Qwen3-1.7B와 Llama-3.2-1B-Instruct 두 모델로만 생성되었습니다.
   누수 감사의 생성기 교차 실험에서, TF-IDF 분류기는 같은 생성기 안에서 F1 0.9744였다가 **미학습 생성기에서 0.7228까지 떨어졌습니다.**
   98%대 성능에 "두 소형 모델의 고유 문체를 암기한" 부분이 섞여 있다는 것이 가설이 아니라 측정된 사실입니다.
   GPT-4급 등 더 다양한 생성기에 대한 일반화는 여전히 미검증입니다.
3. **frozen 백본.** 계산 비용과 비교 공정성(모든 모델 동일 임베딩) 때문에 RoBERTa를 동결했습니다. 파인튜닝 시 성능·해석성이 어떻게 변하는지는 미측정입니다.
4. **성능 포화로 인한 구조 비교의 한계 — 원인 규명됨.** 전 모델이 F1 0.98대에 몰려 있어 융합 구조 간 차이를 분해하기 어렵습니다.
   누수 감사 결과 그 원인은 **과제 자체가 쉽기 때문**이었습니다. TF-IDF bag-of-words가 이미 0.9744에 도달하므로 구조 비교에 남는 여유는 F1 0.009뿐입니다.
   **난이도를 올린 평가셋**(정형 도입부 제거, 표층 통계 정규화, 미학습 생성기, 사람이 부분 편집한 리뷰, 도메인 외 리뷰)을 따로 구성해야 구조의 우열이 드러납니다.
5. **어텐션 해석의 정량화 미완.** `src/attention.py`가 방향1/방향2 가중치를 집계하는 probe를 제공하지만, 저장소 내 노트북에는 해당 셀의 출력이 남아 있지 않습니다. 재실행 시 산출되는 피처별 attention 분포를 **permutation importance 등 독립 지표와 교차 검증**하는 작업이 남아 있습니다.
6. **감사에서 실행하지 못한 항목.** 적대적 재작성(표층 통계를 human 분포에 맞춘 생성물)에 대한 성능, 도입부 제거 후 재학습,
   도메인 이전 평가는 **측정하지 않았습니다.** 배포 환경 예상치를 서술할 때 이 세 항목은 근거가 없는 구간입니다.
7. **도메인 전이 미실험.** 서론에서 언급한 산업 신호(센서 시계열 + 정비 로그) 적용은 아직 가설 단계입니다. 텍스트 토큰 시퀀스를 시계열 윈도우로 치환했을 때 FeatureTokenizer·마스킹 설계가 그대로 유효한지 확인이 필요합니다.

---

## 더 읽을거리

| 문서 | 내용 |
|---|---|
| [docs/실험결과_상세.md](docs/실험결과_상세.md) | 융합 구조 비교 전체 표와 근거 파일, 텍스트 단독 백본 베이스라인, Greedy Search 결과 |
| [docs/누수감사_상세.md](docs/누수감사_상세.md) | 사소한 베이스라인, 길이 누수 기각, 단일 피처 AUC, permutation importance, bag-of-words, 정형 도입부, 생성기 교차 전이, 0.9832의 분해 |
| [docs/피처엔지니어링_상세.md](docs/피처엔지니어링_상세.md) | HC 23개 피처 정의, `perplexity`/`burstiness` 산출식, 모델별 스케일링과 누수 방지 장치 |
| [docs/실행재현_상세.md](docs/실행재현_상세.md) | 실행 절차, Makefile 타깃별 소요, Docker, 시드 고정, 기술 스택 |
| [DATA.md](DATA.md) | Yelp 데이터 확보와 parquet 구성 |

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
