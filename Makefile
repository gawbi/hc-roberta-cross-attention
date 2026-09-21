# HC 피처 × RoBERTa Cross-Attention — 재현용 Makefile
#
#   make setup      의존성 설치 (+ spaCy 모델)
#   make lint       문법 검사
#   make verify     데이터 없이 ③ 제안 모델 빌드·forward 확인
#   make reproduce  ③ 양방향 Co-Attention 학습·평가 (data_yelp.parquet 필요)
#   make audit      피처 누수 / 데이터셋 아티팩트 감사 (data_yelp.parquet 필요)
#   make clean      캐시·산출물 정리
#
# Yelp 기반 데이터셋은 저장소에 포함되지 않는다. DATA.md 참고.
#
# ★ TensorFlow import 순서 — src/* 진입점은 TF 를 pandas/pyarrow 보다 먼저
#   import 한다(README 'macOS 주의' 참고). 아래 타깃도 그 순서를 지킨다.

PYTHON    ?= python3
PIP       ?= $(PYTHON) -m pip
PARQUET   ?= data/data_yelp.parquet
SPACY_URL ?= https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
IMAGE     ?= hc-roberta:cpu

export PYTHONPATH := $(CURDIR)
export MPLBACKEND := Agg
export TF_CPP_MIN_LOG_LEVEL := 2

.DEFAULT_GOAL := help
.PHONY: help setup setup-spacy lint verify check-data smoke train attention \
        reproduce audit notebooks clean docker-build docker-shell

help:
	@echo "타깃 목록:"
	@echo "  setup          requirements.txt + spaCy en_core_web_sm 설치"
	@echo "  lint           src/ 문법 검사 (+ruff 있으면 함께)"
	@echo "  verify         데이터 없이 build_model() + forward 확인 (수십 초)"
	@echo "  smoke          2K 서브샘플 · 2 epochs 빠른 학습 검증 (데이터 필요)"
	@echo "  train          ③ 양방향 Co-Attention 전체 학습 (데이터 필요)"
	@echo "  attention      학습된 가중치로 HC 피처별 attention 표 출력"
	@echo "  reproduce      train -> attention 순차 실행 (데이터 필요)"
	@echo "  audit          피처 누수·아티팩트 감사 (데이터 필요, GPU 불필요, 수 분)"
	@echo "  notebooks      ①② 비교 모델 재현 방법 안내"
	@echo "  clean          __pycache__ · 학습 산출물 정리"
	@echo "  docker-build   CPU 이미지 빌드"
	@echo "  docker-shell   data/models 볼륨을 마운트한 셸 진입"

setup:
	$(PIP) install -r requirements.txt \
		--index-url https://download.pytorch.org/whl/cpu \
		--extra-index-url https://pypi.org/simple
	$(MAKE) setup-spacy

# spaCy 모델은 PyPI 패키지가 아니라 릴리스 휠로 배포된다(버전 고정).
setup-spacy:
	$(PIP) install --no-cache-dir "$(SPACY_URL)"
	-$(PYTHON) -m textblob.download_corpora lite

# 문법 오류 · 정의되지 않은 이름만 잡는다(스타일 규칙은 기존 코드를 건드리지 않기 위해 제외).
lint:
	$(PYTHON) -m compileall -q src
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check --select E9,F63,F7,F82 src; \
	else \
		echo "(ruff 미설치 — compileall 문법 검사만 수행)"; \
	fi

# 데이터 없이 돌아가는 유일한 검증 경로.
# ③ 제안 모델(FeatureTokenizer + 양방향 Co-Attention)을 빌드해 forward 까지 확인한다.
# TF 를 가장 먼저 import 하는 순서를 지킨다.
verify:
	@$(PYTHON) -c "import tensorflow as tf; \
import numpy as np; \
from src.seed import set_seed; \
from src.model import build_model; \
from src import config as C; \
print('seeded:', set_seed()); \
m = build_model(d_model=C.D_MODEL, dropout=C.DROPOUT); \
print('params :', f'{m.count_params():,}'); \
B = 2; \
text = np.random.rand(B, C.MAX_LEN, C.HIDDEN).astype('float16'); \
mask = np.ones((B, C.MAX_LEN), dtype='float32'); \
mask[1, 200:] = 0.0; \
hc = np.random.rand(B, 23).astype('float32'); \
out = m.predict({'text_emb': text, 'mask': mask, 'hc': hc}, verbose=0); \
print('output :', out.shape, '(softmax 2-class)'); \
print('rowsum :', np.round(out.sum(axis=1), 5)); \
print('VERIFY OK')"

check-data:
	@test -f "$(PARQUET)" || { \
		echo "ERROR: 데이터가 없습니다 -> $(PARQUET)"; \
		echo "20,000행 (text + label + HC 23개) parquet 이 필요합니다."; \
		echo "확보 방법은 DATA.md 를 참고하세요."; \
		exit 1; }

smoke: check-data
	$(PYTHON) -m src.train --smoke

train: check-data
	$(PYTHON) -m src.train

attention:
	$(PYTHON) -m src.attention

# README '실험 결과' 표 ③행(test F1 0.9870)의 재현 경로.
# 첫 실행 시 RoBERTa 임베딩 캐시(약 7.8GB)를 data/roberta_emb/ 에 자동 생성한다.
reproduce: check-data train attention
	@echo "완료. models/ 에 coattn_best.weights.h5 · coattn_grid_results.csv · coattn_best_test.csv 가 생성됩니다."

# 성능이 표층 단서에서 오는지 검증한다 (README '누수 감사' 절).
# 학습 파이프라인과 독립이며 GPU 없이 CPU 수 분이면 끝난다.
audit: check-data
	$(PYTHON) analysis/leakage_audit.py --data "$(PARQUET)"
	@echo "결과: analysis/leakage_results.json · docs/images/audit_*.png"

notebooks:
	@echo "①② 비교 모델(Simple Concat / Cross-Attention)은 노트북으로 재현합니다."
	@echo "  notebooks/07_hc_roberta_simple_concat.ipynb"
	@echo "  notebooks/08_hc_roberta_cross_attention.ipynb"
	@echo "PyTorch/GPU 권장 — Colab T4 기준 RoBERTa CLS 임베딩 추출 30~40분 + Greedy Search 30~40분."

docker-build:
	docker build -t $(IMAGE) .

docker-shell:
	docker run --rm -it \
		-v "$(CURDIR)/data:/app/data" \
		-v "$(CURDIR)/models:/app/models" \
		$(IMAGE) bash

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	rm -rf .pytest_cache .ruff_cache .ipynb_checkpoints
	@echo "주의: data/roberta_emb/ (약 7.8GB) 와 models/ 는 보존했습니다."
	@echo "      임베딩 캐시까지 지우려면: rm -rf data/roberta_emb"
