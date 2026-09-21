# 데이터 준비

이 저장소에는 **데이터와 학습 산출물이 포함되어 있지 않습니다.**
아래 절차대로 데이터를 구성하면 README의 실험을 재현할 수 있습니다.

---

## 왜 포함하지 않았나

| 이유 | 내용 |
|---|---|
| **라이선스** | Yelp Open Dataset은 **Yelp Dataset License** 로 제공되며, 이용자가 직접 약관에 동의하고 내려받아야 합니다. 파생물을 그대로 재배포할 수 없어 저장소에 담지 않았습니다. |
| **용량** | 원본 Yelp JSON 약 5 GB, RoBERTa 임베딩 캐시 약 7.8 GB. git 저장소에 담기 부적절합니다. |
| **재생성 가능** | `data_yelp.parquet` 과 임베딩 캐시는 노트북 01→02→03 과 `make train` 으로 다시 만들 수 있는 중간 산출물입니다. |

`.gitignore` 에서 `data/`, `models/`, `*.parquet`, `*.npy`, `*.weights.h5` 를 제외하고 있습니다.

---

## 최종적으로 필요한 것: `data/data_yelp.parquet`

`src/` 파이프라인(③ 양방향 Co-Attention)이 요구하는 **단 하나의 입력 파일**입니다.

```
data/
└── data_yelp.parquet        # 20,000행 × (text + label + HC 23개 피처)
```

### 스키마

`src/config.py` 의 `ORIG_COLS` · `LABEL_MAP` 이 기대하는 형태입니다.

| 컬럼 | 타입 | 내용 |
|---|---|---|
| `pk` | int | 행 식별자 |
| `review_id` | str | Yelp 원본 리뷰 ID |
| `text` | str | 리뷰 본문 (human 원문 또는 LLM 재작성문) |
| `label` | str | `"human"` 또는 `"ai"` — 로드 시 `{human: 0, ai: 1}` 로 매핑 |
| `source` | str | 생성원 (human / 사용한 LLM 이름) |
| `review_stars` | float | 원본 평점 |
| `business_id` | str | Yelp 업소 ID |
| *(HC 23개)* | float | `perplexity`, `burstiness`, `flesch_reading_ease`, `nouns` … — README "피처 엔지니어링" 표 참고 |

**클래스 균형**: human 10,000 / ai 10,000 (총 20,000).
`src/dataset.py` 가 `SEED=42` stratified 70/15/15 로 Train 14,000 / Val 3,000 / Test 3,000 으로 나눕니다.

---

## 1단계 · Yelp Open Dataset 내려받기

- **출처**: Yelp Open Dataset — <https://www.yelp.com/dataset>
  (배포 경로가 이전된 적이 있어, 접속이 안 되면 Yelp 공식 사이트의 최신 데이터셋 페이지를 확인하세요. **(확인 필요)**)
- **라이선스**: Yelp Dataset License — 학술·비상업 목적, 약관 동의 후 개인이 직접 다운로드.
  **본 저장소는 원본도 파생 parquet 도 재배포하지 않습니다.**
- **용량**: 압축 해제 시 약 5 GB (JSON)

배치:

```
data/
└── yelp_academic_dataset_review.json     # 그 외 business.json 등도 필터링에 사용
```

---

## 2단계 · 노트북으로 데이터셋 구성

`notebooks/` 를 순서대로 실행하면 `data_yelp.parquet` 이 만들어집니다.

| 노트북 | 하는 일 | 산출물 |
|---|---|---|
| `01_preprocessing.ipynb` | Yelp 원본에서 레스토랑 리뷰 필터링·정제(이모지 제거 등) → human 10,000건 | 정제된 리뷰 |
| `02_augmentation.ipynb` | 같은 리뷰를 LLM으로 재작성해 AI 라벨 10,000건 생성 | 증강 리뷰 |
| `03_feature_extraction.ipynb` | HC 23개 피처 추출 (textstat / TextBlob / spaCy / GPT-2) | **`data/data_yelp.parquet`** |

### 2단계에서 쓰는 LLM (재작성용)

README 기준 AI 라벨 10,000건의 생성원입니다.

| 모델 | 건수 | 비고 |
|---|---|---|
| `Qwen/Qwen3-1.7B` | 5,000 | HuggingFace 공개 |
| `meta-llama/Llama-3.2-1B-Instruct` | 5,000 | **게이트 모델** — HF 계정으로 접근 승인 후 `huggingface-cli login` 필요 |

> 재작성 프롬프트와 디코딩 파라미터는 `notebooks/02_augmentation.ipynb` 안에 있습니다.
> 생성은 확률적이므로, 다시 돌리면 **문장 단위로 동일한 AI 리뷰가 나오지는 않습니다.**
> 통계적 성질은 유지되지만 README의 소수점 수치를 그대로 재현하려면
> 원 실험에서 만든 parquet 이 필요합니다. (→ [한계](#재현-범위에-대한-솔직한-한계) 참고)

### 3단계에서 필요한 부가 리소스

```bash
make setup           # requirements.txt + spaCy 모델 + TextBlob 코퍼스
```

- **spaCy `en_core_web_sm`** — 품사 6개 피처. PyPI 패키지가 아니므로 별도 설치(`make setup-spacy`).
- **GPT-2** — `perplexity` / `burstiness` 측정 도구. 최초 실행 시 HF 허브에서 자동 다운로드.
- **TextBlob 코퍼스** — `sentiment` / `subjectivity`.

---

## 피처 설계의 출처

HC 23개 피처의 정의식(readability·perplexity 계열)은 아래 논문의 속성 체계를 참조했습니다.

- **Gambetti, A. & Han, Q.**, *AiGen-FoodReview: A Multimodal Dataset of Machine-Generated
  Restaurant Reviews and Images on Social Media*, **ICWSM 2024**
- 원 논문의 `formulas_supplement.md` 가 수식 기준입니다.
  해당 보충자료·데이터셋의 **공개 저장소 경로는 확인이 필요합니다. (확인 필요)**

> **구분해 둘 점** — 본 저장소의 20,000건은 AiGen-FoodReview 데이터셋을 그대로 쓴 것이 아니라,
> **Yelp Open Dataset에서 직접 구성**한 것입니다. AiGen-FoodReview에서 가져온 것은
> **피처 정의 체계**이며 데이터가 아닙니다.

---

## 자동 생성되는 중간 산출물

`data_yelp.parquet` 만 있으면 나머지는 파이프라인이 만듭니다.

```
data/
└── roberta_emb/         # RoBERTa last_hidden_state float16 캐시 (약 7.8GB)
                         #   make train 첫 실행 시 자동 생성, 이후 memmap 부분 적재
models/
├── coattn_best.weights.h5      # 학습 가중치
├── coattn_grid_results.csv     # val 결과
└── coattn_best_test.csv        # test 결과
```

> 임베딩 캐시는 **한 번만** 만들어 모든 비교 실험이 동일한 텍스트 표현을 쓰도록 통제하는 장치입니다.
> `make clean` 은 캐시를 지우지 않습니다. 지우려면 `rm -rf data/roberta_emb`.

---

## 데이터 없이 확인할 수 있는 것

데이터를 구성하기 전에 설치와 모델 구조만 검증하려면:

```bash
make verify
```

③ 제안 모델(`FeatureTokenizer` + 양방향 Co-Attention)을 `build_model()` 로 빌드하고
난수 입력 `(text_emb, mask, hc)` 으로 forward 까지 통과하는지 확인합니다.
(다운로드·데이터 불필요)

---

## 재현 범위에 대한 솔직한 한계

| 항목 | 재현 가능 여부 |
|---|---|
| 파이프라인·모델 구조 | ✅ `make verify` 로 데이터 없이 확인 |
| 학습·평가 절차 | ✅ `data_yelp.parquet` 만 있으면 `make reproduce` |
| **README의 정확한 수치** | ⚠️ 2단계 LLM 재작성이 확률적이라, 데이터셋을 새로 만들면 수치가 달라질 수 있습니다. 원 실험의 parquet 을 확보해야 소수점까지 일치합니다. |
| 원 실험 parquet 배포 | ❌ Yelp 라이선스상 재배포 불가 |

---

## Docker 사용 시

데이터는 이미지에 굽지 않고 볼륨으로 마운트합니다.

```bash
docker run --rm -it \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  hc-roberta:cpu bash
```
