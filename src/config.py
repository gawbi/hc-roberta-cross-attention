"""파이프라인 공통 설정 (경로·하이퍼파라미터).

⚠️ macOS 주의: TensorFlow는 pandas/pyarrow보다 *먼저* import해야 한다.
pandas 3.x가 import 시 로드하는 pyarrow와 TF가 각자 내장한 abseil의 심볼이
충돌하면, TF 첫 fit()의 absl::Mutex가 Arrow 쪽 구현에 바인딩되어 영구
데드락(CPU 0%)에 빠진다. src.train 등 진입점은 이 규칙을 따른다.
"""
from pathlib import Path

# --- 경로 (패키지 루트 기준) ---
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"          # 학습 산출물 (git 미포함, 자동 생성)
EMB_DIR   = DATA_DIR / "roberta_emb" # RoBERTa 임베딩 캐시 (자동 생성, ~7.8GB)

DATA_PARQUET = DATA_DIR / "data_yelp.parquet"   # 최종 20K (text+label+23 features)

GRID_CSV = MODEL_DIR / "coattn_grid_results.csv"
TEST_CSV = MODEL_DIR / "coattn_best_test.csv"
WEIGHTS  = MODEL_DIR / "coattn_best.weights.h5"

# --- 데이터 스키마 ---
ORIG_COLS = ["pk", "review_id", "text", "label", "source", "review_stars", "business_id"]
LABEL_MAP = {"human": 0, "ai": 1}

# --- RoBERTa 임베딩 ---
ROBERTA_NAME = "roberta-base"
# 토큰 길이 분포(balanced 4K 샘플): mean 113, p95 241, p99 329.
# MAX_LEN=128이면 31.7% 잘림 + human이 더 길어 라벨 편향 → 256(잘림 4%) 사용.
MAX_LEN = 256
HIDDEN  = 768   # roberta-base hidden size

# --- 모델 하이퍼파라미터 (단일 조합) ---
SEED       = 42
LR         = 1e-4
DROPOUT    = 0.3
D_MODEL    = 256
BATCH_SIZE = 32
NUM_HEADS  = 4
MAX_EPOCHS = 15
PATIENCE   = 3
