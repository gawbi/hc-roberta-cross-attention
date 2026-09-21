"""학습 + 평가 진입점 — notebooks/04 셀 8~9의 모듈화.

실행:
    python -m src.train              # 전체 20K, 단일 조합 (lr=1e-4, dropout=0.3, d=256, bs=32)
    python -m src.train --smoke      # 2K 서브샘플·2 epochs로 파이프라인만 빠르게 검증

산출물(models/): coattn_best.weights.h5, coattn_grid_results.csv(val), coattn_best_test.csv
"""
# ⚠️ TensorFlow를 가장 먼저 import (pandas 3.x의 pyarrow ↔ TF abseil 데드락 회피)
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

import argparse
import os
import random
import time

import numpy as np

from . import config as C
from .dataset import (load_data, stratified_split, scale_features,
                      ensure_embedding_cache, gather)
from .model import build_model


def set_seed(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main(smoke: bool = False):
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split

    set_seed()
    os.makedirs(C.MODEL_DIR, exist_ok=True)

    df, feature_cols = load_data()
    sp = stratified_split(df)
    if smoke:  # 라벨 균형 유지 서브샘플
        y = df["y"].values
        def _strat(ix, n):
            _, s = train_test_split(ix, test_size=n, random_state=C.SEED, stratify=y[ix])
            return s
        sp.train, sp.val, sp.test = _strat(sp.train, 1400), _strat(sp.val, 300), _strat(sp.test, 300)

    hc = scale_features(df, feature_cols, sp.train)
    ensure_embedding_cache(df)
    Xtr, ytr = gather(df, hc, sp.train)
    Xva, yva = gather(df, hc, sp.val)
    Xte, yte = gather(df, hc, sp.test)
    print("train/val/test:", len(ytr), len(yva), len(yte))

    # --- 학습 (단일 조합) ---
    tf.keras.backend.clear_session()
    set_seed()
    model = build_model(C.D_MODEL, C.DROPOUT, n_features=len(feature_cols))
    model.compile(optimizers.Adam(C.LR), "sparse_categorical_crossentropy", metrics=["accuracy"])
    es = callbacks.EarlyStopping("val_loss", patience=C.PATIENCE, restore_best_weights=True)

    t0 = time.time()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                     epochs=2 if smoke else C.MAX_EPOCHS,
                     batch_size=C.BATCH_SIZE, callbacks=[es], verbose=1)
    sec = time.time() - t0
    model.save_weights(C.WEIGHTS)

    # --- val 평가 ---
    pred = model.predict(Xva, batch_size=256, verbose=0).argmax(1)
    val_row = dict(lr=C.LR, dropout=C.DROPOUT, d_model=C.D_MODEL, batch=C.BATCH_SIZE,
                   epochs_ran=len(hist.history["loss"]), train_sec=round(sec, 1),
                   val_accuracy=accuracy_score(yva, pred),
                   val_precision=precision_score(yva, pred, zero_division=0),
                   val_recall=recall_score(yva, pred, zero_division=0),
                   val_f1=f1_score(yva, pred, zero_division=0))
    pd.DataFrame([val_row]).to_csv(C.GRID_CSV, index=False)
    print("VAL :", {k: round(v, 4) if isinstance(v, float) else v for k, v in val_row.items()})

    # --- test 평가 ---
    pred = model.predict(Xte, batch_size=256, verbose=0).argmax(1)
    test_row = dict(test_accuracy=accuracy_score(yte, pred),
                    test_precision=precision_score(yte, pred, zero_division=0),
                    test_recall=recall_score(yte, pred, zero_division=0),
                    test_f1=f1_score(yte, pred, zero_division=0))
    pd.DataFrame([test_row]).to_csv(C.TEST_CSV, index=False)
    print("TEST:", {k: round(v, 4) for k, v in test_row.items()})
    print("saved:", C.WEIGHTS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2K 서브샘플·2 epochs 빠른 검증")
    main(smoke=ap.parse_args().smoke)
