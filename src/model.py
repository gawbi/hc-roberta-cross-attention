"""양방향 Cross-Attention 분류기 — notebooks/04 셀 7의 모듈화.

구조 (시퀀스↔시퀀스 bidirectional co-attention, ViLBERT co-TRM 방식):
    text_emb [B,256,768] ─Dense(d)→ T [B,256,d]      (frozen RoBERTa 마지막 hidden layer)
    hc [B,23] ─FeatureTokenizer→ F [B,23,d]          (feature 1개 = 토큰 1개, FT-Transformer)
    방향1: MHA(Q=F, K=V=T, 패딩마스크) → 평균 → co_text
    방향2: MHA(Q=T, K=V=F) → MaskedMeanPool → co_hc
    Concat[co_text, co_hc] → Dense(d,ReLU) → Dropout → softmax(2)

참고: 사전 요약 벡터 없이 padding mask + FeatureTokenizer만으로 양방향을 구성.
mask 역할 2가지 — 방향1 attention의 패딩 K/V 차단, 방향2 풀링의 패딩 쿼리 제외.
(human 리뷰가 ai보다 길어, mask 없이는 패딩량이 라벨 누수가 됨)
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import keras

from .config import MAX_LEN, HIDDEN, NUM_HEADS


class MaskedMeanPool(layers.Layer):
    """[B,L,d], mask [B,L] → [B,d]. 패딩 토큰 제외 평균."""

    def call(self, x, mask):
        m = tf.cast(mask, x.dtype)[..., None]
        return tf.reduce_sum(x * m, axis=1) / (tf.reduce_sum(m, axis=1) + 1e-9)


class FeatureTokenizer(layers.Layer):
    """[B,n] → [B,n,d]. HC feature 하나하나를 토큰으로 임베딩 (FT-Transformer 방식).
    out[b,i,:] = x[b,i] * W[i,:] + b[i,:] — feature별 고유 임베딩이라 정체성 보존."""

    def __init__(self, n_feat, d_model, **kw):
        super().__init__(**kw)
        self.n_feat, self.d_model = n_feat, d_model

    def build(self, _):
        self.W = self.add_weight(name="W", shape=(self.n_feat, self.d_model),
                                 initializer="glorot_uniform")
        self.B = self.add_weight(name="B", shape=(self.n_feat, self.d_model),
                                 initializer="zeros")

    def call(self, x):                          # [B, n]
        return x[..., None] * self.W + self.B   # [B, n, d]


def build_model(d_model: int, dropout: float, n_features: int = 23) -> Model:
    text_in = layers.Input((MAX_LEN, HIDDEN), dtype="float16", name="text_emb")
    mask_in = layers.Input((MAX_LEN,), dtype="float32", name="mask")
    hc_in = layers.Input((n_features,), dtype="float32", name="hc")

    # text: RoBERTa 마지막 hidden layer → Dense(d) 토큰별 투영
    T = layers.Dense(d_model)(keras.ops.cast(text_in, "float32"))      # [B, L, d]

    # HC: feature 토큰 시퀀스 (MLP 요약 없음)
    F = FeatureTokenizer(n_features, d_model)(hc_in)                   # [B, n, d]

    # 방향 1: HC → text (패딩 K/V 차단)
    key_mask = keras.ops.cast(mask_in, "bool")[:, None, :]             # [B, 1, L]
    A1 = layers.MultiHeadAttention(NUM_HEADS, d_model // NUM_HEADS, name="hc2text")(
        query=F, value=T, key=T, attention_mask=key_mask)              # [B, n, d]
    co_text = layers.GlobalAveragePooling1D()(A1)                      # [B, d]

    # 방향 2: text → HC (패딩 쿼리 출력은 풀링에서 제외)
    A2 = layers.MultiHeadAttention(NUM_HEADS, d_model // NUM_HEADS, name="text2hc")(
        query=T, value=F, key=F)                                       # [B, L, d]
    co_hc = MaskedMeanPool()(A2, mask_in)                              # [B, d]

    # 융합 → 분류: 양방향 co-attention 출력만 사용
    x = layers.Concatenate()([co_text, co_hc])                         # [B, 2d]
    x = layers.Dense(d_model, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(2, activation="softmax")(x)
    return Model([text_in, mask_in, hc_in], out, name="hc_bixattn")
