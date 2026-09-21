# AI 생성 리뷰 탐지 — Handcrafted Feature × RoBERTa Cross-Attention

> **정형 수치 피처(23차원)와 비정형 텍스트 임베딩(RoBERTa)을 Cross-Attention으로 융합하는 이종(heterogeneous) 신호 융합 분류기.**
> Yelp 리뷰 20,000건(Human 10K / LLM 생성 10K)에서 **Test F1 0.9832 (5-Run 평균), AUROC 0.9989** 달성.
>
> **다만 이 수치는 모델 성능이 아니라 벤치마크 난이도를 반영합니다.** 같은 test set에서 TF-IDF bag-of-words만으로 F1 0.9744,
> 파이썬 문자열 통계 11개만으로 F1 0.8867이 나옵니다. 실측 근거는 [누수 감사](#누수-감사--f1-098은-어디에서-오는가) 절에 정리했습니다.

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

- **HC만으로도 F1 0.9220.** LM 기반 통계가 강한 판별력을 갖습니다. 즉 이 문제는 "정형 피처가 실제로 정보를 갖는" 융합 문제입니다.
  (다만 후속 [누수 감사](#누수-감사--f1-098은-어디에서-오는가)에서 이 판별력이 사실상 `perplexity` 한 개에 몰려 있고,
  `burstiness`의 기여는 permutation importance +0.0123으로 작다는 것을 확인했습니다.)
- **텍스트를 더하면 +0.061.** Simple Concat만으로 0.9831까지 올라가며, 두 신호가 상호보완적임이 확인됩니다.
  (단, 같은 split에서 TF-IDF bag-of-words + LR이 단독으로 F1 0.9744에 도달합니다. 이 +0.061 중 트랜스포머 임베딩에 고유하게 귀속되는 몫은 크지 않습니다.)
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

## 누수 감사 — F1 0.98은 어디에서 오는가

AI 생성 텍스트 탐지에서 F1 0.98대는 정상적인 수치가 아닙니다. 이런 값이 나오면 모델이 의미를 판별한 것이 아니라
**LLM 재작성이 데이터셋에 남긴 표층 단서**(길이 분포, 구두점 빈도, 어휘 다양성, 정형 도입부)를 잡고 있을 가능성을 먼저 배제해야 합니다.
그래서 학습 파이프라인과 완전히 독립된 감사 스크립트를 따로 두고, 주 실험과 **동일한 test set**(stratified 70/15/15, `seed=42`, N=3,000) 위에서 측정했습니다.

```bash
make audit     # = python analysis/leakage_audit.py --data data/data_yelp.parquet
```

전체 수치는 `analysis/leakage_results.json`, 그림은 `docs/images/audit_*.png` 에 있습니다. GPU 없이 CPU 수 분이면 끝납니다.
아래 수치는 전부 **실행해서 얻은 값**이며, 추정치는 없습니다.

### 1. 사소한 베이스라인 — 원문 문자열만으로 F1 0.8867

HC 피처도, GPT-2도, 트랜스포머도 쓰지 않고 **파이썬 문자열 연산만으로** 뽑은 표층 피처 11개
(문자 수, 단어 수, 구두점 비율, 쉼표·마침표 비율, 느낌표 수, 대문자 비율, 숫자 비율, 평균 단어 길이, type-token ratio, 줄바꿈 수)로 분류기를 돌린 결과입니다.

| 입력 | LogisticRegression F1 / AUC | HistGradientBoosting F1 / AUC |
|------|------|------|
| `char_len` 하나 | 0.5509 / 0.4968 | 0.6547 / 0.6276 |
| `word_count` 하나 | 0.5769 / 0.5321 | 0.6447 / 0.6152 |
| `punct_ratio` 하나 | 0.5556 / 0.5248 | 0.6659 / 0.6314 |
| 길이 + 단어수 + 구두점 (3개) | 0.7464 / 0.8296 | 0.7784 / 0.8631 |
| **표층 11개 전체** | 0.8619 / 0.9302 | **0.8867 / 0.9513** |

**표층 통계 11개만으로 F1 0.8867.** 같은 감사에서 재현한 HC 23개 전체 MLP가 F1 0.9114(AUC 0.9735)이므로,
README가 보고한 HC-only F1 0.9220의 **대부분은 GPT-2 perplexity가 없어도 도달 가능한 수준**입니다.

### 2. 길이는 범인이 아니었다 — 진짜 단서는 어휘·구두점 스타일

가장 흔한 의심인 길이 누수는 이 데이터셋에서는 **기각**됩니다.

- `char_len` 단독 AUC **0.5032** (사실상 무작위). 원 데이터 구축 시 길이를 맞춘 것으로 보입니다.
- `char_len` 5%-분위 20구간 안에서 human/ai 수를 동일하게 맞춘 length-matched 부분집합(N=15,644, 7,822 대 7,822)에서
  길이 단독 AUC는 0.5100으로 떨어지지만, **HC 전체 성능은 F1 0.9056 → 0.9003 으로 거의 그대로**입니다.
- 길이 계열 피처 8개(`char`·`letter`·`syllable`·`lexicon`·`sentence`·`polysyllab`·`monosyllab`·`reading_time`)를 아예 빼도 F1 0.8848 / AUC 0.9511.

대신 실제로 판별력을 만드는 표층 단서는 **문체 지문**입니다 (단일 피처 AUC, test):

| 표층 피처 | AUC | human 평균 | AI 평균 |
|---|---|---|---|
| 평균 단어 길이 | 0.8122 | 4.43 | 4.86 |
| 쉼표 비율(단어당) | 0.7914 | 0.03 | 0.06 |
| 대문자 비율 | 0.7568 | 0.03 | 0.02 |
| 느낌표 개수 | 0.6699 | 0.90 | 0.05 |

느낌표가 특히 노골적입니다 — 사람 리뷰는 평균 0.90개, LLM 재작성본은 0.05개입니다.

![분포 비교](docs/images/audit_distribution_top6.png)

### 3. 단일 피처 판별력 — 0.9를 넘는 단일 아티팩트는 없음

HC 23개를 **하나씩** 단독으로 썼을 때의 test AUC 상위 5개입니다. 0.9 이상은 없으므로 "피처 하나가 라벨을 그대로 들고 있는" 형태의 누수는 아닙니다.

| 순위 | 피처 | 단독 AUC | 단독 LR F1 |
|---|---|---|---|
| 1 | `perplexity` | 0.8635 | 0.7847 |
| 2 | `flesch_reading_ease` | 0.8154 | 0.7313 |
| 3 | `smog_index` | 0.8102 | 0.7400 |
| 4 | `fog_scale` | 0.7900 | 0.7085 |
| 5 | `flesch_kincaid_grade` | 0.7771 | 0.7053 |

![단일 피처 AUC](docs/images/audit_single_feature_auc.png)

### 4. HC-only 0.9220은 사실상 perplexity 하나

HistGradientBoosting permutation importance(AUC 감소, n_repeats=10)와 그룹 ablation(LogisticRegression) 결과입니다.

| permutation importance 상위 | AUC 감소 |
|---|---|
| `perplexity` | **+0.2109** ±0.0064 |
| `monosyllab` | +0.0487 ±0.0016 |
| `article` | +0.0435 ±0.0019 |
| `flesch_reading_ease` | +0.0383 ±0.0017 |
| `adverb` | +0.0127 ±0.0010 |
| `burstiness` | +0.0123 ±0.0009 |

| 그룹 ablation (F1 / AUC) | 그 그룹만 | 그 그룹 제외 |
|---|---|---|
| count(7) | 0.7589 / 0.8481 | 0.8851 / 0.9522 |
| readability(6) | 0.7441 / 0.8357 | 0.9013 / 0.9608 |
| sentiment(2) | 0.5322 / 0.4947 | 0.9009 / 0.9623 |
| **LM(2) — perplexity·burstiness** | 0.7929 / 0.8684 | **0.8532 / 0.9303** |
| POS(6) | 0.7191 / 0.7884 | 0.8684 / 0.9404 |

HC 23개 전체 LR이 0.9056인데 LM 2개만 빼면 0.8532로, 다른 어느 그룹을 빼는 것보다 낙폭이 큽니다.
다만 그 안에서도 기여는 `perplexity`에 몰려 있고 **`burstiness` 단독 기여는 +0.0123으로 작습니다.**
위 [실험 결과](#실험-결과)의 "`perplexity`·`burstiness` 같은 LM 기반 통계가 강한 판별력을 갖는다"는 서술은 `perplexity`에 한해 맞고, `burstiness`에 대해서는 과대평가였습니다.
한편 `sentiment`/`subjectivity` 2개는 단독 AUC 0.4947로 **정보가 사실상 없습니다.**

### 5. 가장 아픈 결과 — bag-of-words가 F1 0.9744

텍스트 쪽을 점검하면서 TF-IDF(word 1–2gram, `min_df=3`, 65,495차원) + LogisticRegression을 같은 split에서 돌렸습니다.

| 모델 | F1 | AUROC | test 3,000건 중 오분류 |
|---|---|---|---|
| **TF-IDF + LogisticRegression** | **0.9744** | 0.9964 | 약 76건 |
| ② HC + RoBERTa Cross-Attention (README) | 0.9832 | 0.9989 | 약 50건 |

**RoBERTa 임베딩과 이종 융합 구조 전체가 선형 bag-of-words 대비 추가로 얻은 것은 F1 +0.0088, 오분류 76건 → 50건입니다.**
즉 이 과제는 단어 빈도만으로 이미 97% 이상 풀리는 문제였고, 0.98이라는 숫자 자체는 모델의 우수성이 아니라 **과제의 쉬움**을 반영합니다.

### 6. 왜 bag-of-words로 풀리는가 — 정형 도입부

| 관측 | AI 10,000건 | human 10,000건 |
|---|---|---|
| 상위 10개 도입 5-gram이 덮는 비율 | **9.95%** | 0.74% |
| 고유 도입 5-gram 수 | 5,938 | 9,575 |
| 한쪽에만 쏠린 n-gram 수(한쪽 DF>1%, 반대쪽 <0.05%) | 116 | 39 |

AI 쪽 도입부는 특정 템플릿으로 수렴합니다 — `"I was really looking forward…"` 293건,
`"I stumbled upon this hidden…"` 159건, `"I stumbled upon this gem…"` 84건.
인간 리뷰의 최빈 도입부는 10건에 불과합니다.
어휘 수준에서도 `experience`(DF 0.094 → 0.470), `overall`(0.054 → 0.344), `atmosphere`(0.061 → 0.245)가 AI 쪽으로,
`very`(0.283 → 0.049), `will`(0.196 → 0.010), `so`(0.336 → 0.097)가 human 쪽으로 강하게 갈립니다.
**두 소형 모델의 프롬프트·디코딩 설정이 만든 정형성이 라벨을 거의 직접 알려주고 있습니다.**

### 7. 생성기가 바뀌면 무너진다

AI 라벨을 만든 두 모델 사이의 전이 성능을 측정했습니다. human 10K를 절반씩 나눠 한쪽 생성기로 학습하고 다른 생성기로 평가합니다.

| 조건 | HC 23개 LR (F1 / AUC) | TF-IDF LR (F1 / AUC) |
|---|---|---|
| in-generator Llama-3.2-1B | 0.9402 / 0.9856 | — |
| in-generator Qwen3-1.7B | 0.8820 / 0.9498 | — |
| Qwen3-1.7B 학습 → Llama-3.2-1B 평가 | 0.9105 / 0.9689 | 0.9352 / 0.9922 |
| **Llama-3.2-1B 학습 → Qwen3-1.7B 평가** | 0.8332 / 0.9274 | **0.7228 / 0.9679** |

TF-IDF는 같은 생성기 안에서 0.9744였다가 미학습 생성기에서 **F1 0.7228까지 떨어집니다.**
어휘 신호의 상당 부분이 일반적인 "AI 문체"가 아니라 **특정 생성기의 지문**이라는 직접 증거입니다.
HC 피처 쪽은 낙폭이 작아(0.9402 → 0.8332) 상대적으로 생성기에 덜 민감하지만, 절대 성능 자체가 낮습니다.

### 정리 — 0.9832의 분해

동일 test set 기준으로 쌓아 보면 이렇게 됩니다.

| 단계 | F1 | 이 단계가 새로 설명하는 몫 |
|---|---|---|
| 문자열 통계 11개 (GB) | 0.8867 | — |
| HC 23개 (본 감사 MLP 재현) | 0.9114 | +0.025 |
| TF-IDF bag-of-words + LR | 0.9744 | +0.063 |
| ② HC + RoBERTa Cross-Attention (README) | 0.9832 | **+0.009** |

**결론: F1 0.9832 중 실질적으로 융합 구조가 기여한 몫은 마지막 약 0.009입니다.**
나머지는 (a) LLM 재작성이 남긴 문체 지문, (b) 두 소형 생성기의 정형 도입부, (c) GPT-2 perplexity 한 개가 만들어냅니다.
**이 저장소의 가치는 탐지 성능 수치가 아니라, 이종 신호를 어느 지점에서 융합할지에 대한 통제된 대조 실험에 있습니다.**
표의 0.98대 숫자는 모델 품질이 아니라 **벤치마크의 난이도**를 읽는 값으로 취급해야 합니다.

### 실제 배포 환경에서 예상되는 성능

- **미학습 생성기**: 위 교차 실험 기준 F1 0.72~0.94. GPT-4급처럼 더 자연스럽고 프롬프트 다양성이 큰 생성기라면 이보다 낮을 가능성이 높습니다(미측정).
- **표층 단서를 통제한 재작성**: 길이·구두점·도입부를 human 분포에 맞추도록 지시하면 감사에서 확인된 신호 대부분이 사라집니다. 이 조건은 **아직 실측하지 않았습니다.**
- **도메인 이전**(레스토랑 리뷰 → 다른 도메인): `flesch_reading_ease` 같은 가독성 지표의 human 기준선 자체가 이동하므로, 재보정 없이는 성능 보장이 불가합니다. 역시 미측정입니다.
- 감사에서 측정한 범위 안에서는, **동일 생성기·동일 도메인이라는 조건이 깨지는 순간 F1이 0.9 아래로 내려가는 것이 기본 시나리오**로 보는 편이 안전합니다.

### 개선 경로

1. **표층 정규화 후 재평가** — 길이 매칭은 이미 적용했고(성능 유지), 다음은 구두점·대소문자·느낌표 빈도를 human 분포에 맞춰 정규화한 뒤 다시 측정.
2. **도입부 ablation** — 각 리뷰의 첫 1~2문장을 제거하고 재학습. 정형 도입부가 빠졌을 때 남는 성능이 "진짜 문체 탐지"의 상한선입니다.
3. **leave-one-generator-out 평가를 기본 지표로 승격** — in-generator 수치를 대표값으로 보고하지 않습니다.
4. **적대적 재작성 데이터 추가** — 표층 통계를 human 분포에 맞추도록 프롬프트한 생성물을 hard negative로 투입.
5. **bag-of-words를 상시 베이스라인으로 표에 포함** — 새 구조가 TF-IDF+LR을 유의미하게 넘지 못하면 그 구조는 기여가 없다고 판정.

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
├── analysis/
│   ├── leakage_audit.py                    #   피처 누수 · 데이터셋 아티팩트 감사 (학습 파이프라인과 독립)
│   └── leakage_results.json                #   감사 실행 결과 (전체 수치)
├── docs/images/                            # 감사 플롯 (audit_*.png)
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
make audit      # 누수 감사 (데이터 필요, GPU 불필요)
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
| `make audit` | 피처 누수·아티팩트 감사 — 사소한 베이스라인, 단일 피처 AUC, permutation importance, KS 검정, n-gram 편중, 생성기 교차 일반화 | **✓** | CPU 수 분 |
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

> 이 절의 1~6번은 구조·실험 설계상의 한계이고, 데이터셋 아티팩트에 관한 한계는
> [누수 감사](#누수-감사--f1-098은-어디에서-오는가) 절에 실측 수치와 함께 별도로 정리했습니다.
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
