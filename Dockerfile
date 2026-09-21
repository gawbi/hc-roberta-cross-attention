# HC 피처 × RoBERTa Cross-Attention — CPU 재현 환경
#
# 빌드:
#   docker build -t hc-roberta:cpu .
#
# 실행 (데이터·임베딩 캐시·가중치는 이미지에 넣지 않고 볼륨으로 마운트):
#   docker run --rm -it \
#     -v "$PWD/data:/app/data" \
#     -v "$PWD/models:/app/models" \
#     hc-roberta:cpu bash
#
# 데이터 확보·배치 방법은 DATA.md 참고.
#
# 주의 — 이미지 크기
#   TensorFlow(③ 제안 모델)와 PyTorch(①② 비교 모델)를 모두 담기 때문에
#   최종 이미지가 큽니다. 한쪽 프레임워크만 필요하면 requirements.txt 에서
#   해당 블록을 지우고 빌드하세요.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HF_HOME=/app/.cache/huggingface

# make : Makefile 타깃을 컨테이너 안에서도 쓰기 위함
RUN apt-get update \
 && apt-get install -y --no-install-recommends make \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 레이어를 소스보다 먼저 캐싱
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple

# spaCy 품사 태깅 모델 — PyPI 패키지가 아니므로 릴리스 휠을 버전 고정해 설치한다.
# (src/features.py 의 FeatureExtractor 가 en_core_web_sm 을 로드)
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# TextBlob 이 쓰는 NLTK 코퍼스 (sentiment/subjectivity)
RUN python -m textblob.download_corpora lite || true

# 소스만 복사. data/ · models/ 는 .gitignore 대상이며 런타임 볼륨으로 붙인다.
COPY src/ ./src/
COPY Makefile ./

# 데이터·산출물 마운트 지점
#   /app/data   : data_yelp.parquet + RoBERTa 임베딩 캐시(약 7.8GB)
#   /app/models : 학습 가중치·결과 CSV
VOLUME ["/app/data", "/app/models"]

# roberta-base · gpt2 등 사전학습 가중치는 이미지에 굽지 않고
# 최초 실행 시 HF 허브에서 내려받는다(HF_HOME 캐시). 실행 시 네트워크가 필요하다.

CMD ["bash"]
