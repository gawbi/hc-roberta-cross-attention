"""학습된 모델의 attention weight 분석 — notebooks/04 섹션 10~11의 모듈화.

- feature_attention():  방향2(Text→HC) score 집계 → "토큰들이 어떤 HC feature를 보는가"
- token_attention():    방향1(HC→Text) score 집계 → "feature들이 어떤 토큰을 보는가"

실행:
    python -m src.attention      # 가중치 로드 → test셋 feature attention 표 출력

주의: 학습된 레이어로 score를 노출하는 함수형 probe Model 재구성은 Keras 3에서
그래프 연결 오류가 나므로, 레이어를 eager로 직접 호출하는 방식을 사용한다.
"""
# ⚠️ TensorFlow를 가장 먼저 import (pyarrow ↔ TF abseil 데드락 회피)
import tensorflow as tf
from tensorflow.keras import layers

import numpy as np

from . import config as C
from .dataset import load_data, stratified_split, scale_features, ensure_embedding_cache, gather
from .model import build_model, FeatureTokenizer


def _get_layers(model):
    """학습된 모델에서 분석에 필요한 레이어 핸들 추출."""
    dense_T = next(l for l in model.layers
                   if isinstance(l, layers.Dense) and l.kernel.shape[0] == C.HIDDEN)
    ft = next(l for l in model.layers if isinstance(l, FeatureTokenizer))
    return dense_T, ft, model.get_layer("hc2text"), model.get_layer("text2hc")


def feature_attention(model, X, batch_size: int = 256) -> np.ndarray:
    """방향2(Text→HC) 가중치 집계 → [N, n_features] (행 합=1).
    head 평균 후 실제 토큰만 평균 (패딩 쿼리 제외)."""
    dense_T, ft, _, mha2 = _get_layers(model)
    n = len(X["mask"])
    n_feat = X["hc"].shape[1]
    out = np.empty((n, n_feat), dtype="float32")
    for s in range(0, n, batch_size):
        e = tf.cast(tf.convert_to_tensor(X["text_emb"][s:s + batch_size]), tf.float32)
        mk = tf.convert_to_tensor(X["mask"][s:s + batch_size])
        hc = tf.convert_to_tensor(X["hc"][s:s + batch_size])
        T, F = dense_T(e), ft(hc)
        _, s2 = mha2(query=T, value=F, key=F, return_attention_scores=True, training=False)
        w = tf.reduce_mean(s2, axis=1)                  # head 평균 [b, L, n]
        m = mk[..., None]
        out[s:s + batch_size] = (tf.reduce_sum(w * m, axis=1) / tf.reduce_sum(m, axis=1)).numpy()
    return out


def token_attention(model, X, i: int) -> np.ndarray:
    """샘플 i의 방향1(HC→Text) 가중치 → 토큰별 weight [L].
    heads × HC쿼리 평균. 리뷰 원문 하이라이트용 (notebooks/04 섹션 11 참고)."""
    dense_T, ft, mha1, _ = _get_layers(model)
    e = tf.cast(tf.convert_to_tensor(X["text_emb"][i:i + 1]), tf.float32)
    mk = tf.convert_to_tensor(X["mask"][i:i + 1])
    hc = tf.convert_to_tensor(X["hc"][i:i + 1])
    T, F = dense_T(e), ft(hc)
    key_mask = tf.cast(mk, tf.bool)[:, None, :]
    _, s1 = mha1(query=F, value=T, key=T, attention_mask=key_mask,
                 return_attention_scores=True, training=False)   # [1, heads, n, L]
    return tf.reduce_mean(s1, axis=(1, 2)).numpy()[0]


def main():
    import pandas as pd
    df, feature_cols = load_data()
    sp = stratified_split(df)
    hc = scale_features(df, feature_cols, sp.train)
    ensure_embedding_cache(df)
    Xte, yte = gather(df, hc, sp.test)

    model = build_model(C.D_MODEL, C.DROPOUT, n_features=len(feature_cols))
    model.load_weights(C.WEIGHTS)

    fa = feature_attention(model, Xte)
    imp = pd.Series(fa.mean(axis=0), index=feature_cols).sort_values(ascending=False)
    print(f"=== mean attention weight (Text→HC, uniform={1/len(feature_cols):.4f}) ===")
    print(imp.round(4).to_string())

    attn_df = pd.DataFrame(fa, columns=feature_cols)
    attn_df["y"] = yte
    by_label = attn_df.groupby("y").mean().T
    by_label.columns = ["human", "ai"]
    by_label["diff(ai-human)"] = by_label["ai"] - by_label["human"]
    print("\n=== by label ===")
    print(by_label.loc[imp.index].round(4).to_string())


if __name__ == "__main__":
    main()
