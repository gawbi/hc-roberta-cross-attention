"""데이터 로드·split·표준화·RoBERTa 임베딩 캐시 — notebooks/04 셀 2~6의 모듈화.

흐름:
    df = load_data()                          # parquet → label 매핑(y)
    splits = stratified_split(df)             # train/val/test 인덱스 70/15/15
    hc = scale_features(df, splits.train)     # StandardScaler (train에만 fit)
    ensure_embedding_cache(df)                # 캐시 없으면 RoBERTa로 추출 (1회, ~7.8GB)
    Xtr, ytr = gather(df, hc, splits.train)   # 모델 입력 dict 구성
"""
import os
import time
from dataclasses import dataclass

import numpy as np

from .config import (DATA_PARQUET, EMB_DIR, ORIG_COLS, LABEL_MAP,
                     ROBERTA_NAME, MAX_LEN, HIDDEN, SEED)

EMB_PATH  = EMB_DIR / f"last_hidden_{MAX_LEN}.npy"
MASK_PATH = EMB_DIR / f"mask_{MAX_LEN}.npy"


@dataclass
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def load_data():
    """최종 parquet 로드 + 라벨 매핑. 반환 df에 y(0=human, 1=ai) 컬럼 추가."""
    import pandas as pd
    df = pd.read_parquet(DATA_PARQUET)
    feature_cols = [c for c in df.columns if c not in ORIG_COLS]
    assert len(feature_cols) == 23, f"feature 수 이상: {len(feature_cols)}"
    df["y"] = df["label"].map(LABEL_MAP).astype("int64")
    assert df["y"].isin([0, 1]).all(), "라벨 매핑 실패"
    return df, feature_cols


def stratified_split(df, seed: int = SEED) -> Splits:
    """라벨 비율 유지 70/15/15 split (인덱스만 반환)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    y = df["y"].values
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed,
                                         stratify=y[temp_idx])
    return Splits(train_idx, val_idx, test_idx)


def scale_features(df, feature_cols, train_idx) -> np.ndarray:
    """HC feature 표준화 — scaler는 train에만 fit (정보 누수 방지)."""
    from sklearn.preprocessing import StandardScaler
    hc = df[feature_cols].astype("float32").values
    hc = np.nan_to_num(hc, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(hc[train_idx])
    return scaler.transform(hc).astype("float32")


def ensure_embedding_cache(df, batch_size: int = 32):
    """frozen roberta-base의 마지막 hidden layer를 float16으로 캐시 (없을 때만).
    추출 후 행 순서 정합성(df.iloc[i] ↔ MASK[i])을 검증한다."""
    if EMB_PATH.exists() and MASK_PATH.exists():
        return
    os.makedirs(EMB_DIR, exist_ok=True)
    import torch
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(ROBERTA_NAME)
    mdl = AutoModel.from_pretrained(ROBERTA_NAME).eval()
    device = ("cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu"))
    mdl = mdl.to(device)
    n = len(df)
    print(f"extracting RoBERTa embeddings → {EMB_PATH} (device={device}, N={n}, L={MAX_LEN})")

    emb_out = np.empty((n, MAX_LEN, HIDDEN), dtype=np.float16)
    mask_out = np.empty((n, MAX_LEN), dtype=np.int8)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, batch_size):
            batch = df["text"].iloc[s:s + batch_size].tolist()
            enc = tok(batch, max_length=MAX_LEN, padding="max_length",
                      truncation=True, return_tensors="pt").to(device)
            out = mdl(**enc).last_hidden_state.cpu().numpy().astype(np.float16)
            emb_out[s:s + batch_size] = out
            mask_out[s:s + batch_size] = enc["attention_mask"].cpu().numpy().astype(np.int8)
            if s % (batch_size * 50) == 0:
                print(f"  {s}/{n}  ({time.time() - t0:.0f}s)")
    np.save(EMB_PATH, emb_out)
    np.save(MASK_PATH, mask_out)
    print(f"cached in {time.time() - t0:.0f}s")

    # 행 순서 정합성 검증
    cached_mask = np.load(MASK_PATH, mmap_mode="r")
    for i in [0, n // 2, n - 1]:
        re_enc = tok(df["text"].iloc[i], max_length=MAX_LEN, padding="max_length",
                     truncation=True, return_tensors="pt")
        assert (re_enc["attention_mask"].numpy().astype(np.int8) == cached_mask[i]).all(), \
            f"행 순서 불일치 @ i={i}"
    print("row alignment OK")


def gather(df, hc_scaled, idx):
    """split 인덱스 → 모델 입력 dict + 라벨. 임베딩은 memmap에서 부분 적재."""
    emb = np.load(EMB_PATH, mmap_mode="r")
    mask = np.load(MASK_PATH, mmap_mode="r")
    y = df["y"].values.astype("int64")
    ix = np.asarray(idx)
    return ({"text_emb": np.asarray(emb[ix], dtype="float16"),
             "mask": np.asarray(mask[ix], dtype="float32"),
             "hc": hc_scaled[ix]},
            y[ix])
