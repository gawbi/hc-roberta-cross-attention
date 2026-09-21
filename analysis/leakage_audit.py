"""피처 누수 / 데이터셋 아티팩트 감사 — README 성능(F1 0.98대)의 출처 규명.

배경: AI 생성 텍스트 탐지에서 F1 0.98은 비정상적으로 높다. LLM 재작성 리뷰는
길이·구두점·어휘다양성 같은 표층 단서를 남기므로, 모델이 의미가 아니라
그 단서를 잡고 있을 가능성을 먼저 배제해야 한다. 이 스크립트는 그것을 측정한다.

수행 항목:
  1) 단일 HC 피처 AUC          — 하나만으로 0.9 이상이면 아티팩트 후보
  2) 사소한 베이스라인          — 원문에서 직접 뽑은 길이/구두점/단어수만으로 LR
  3) 전체 HC 모델 + permutation importance  — README의 HC-only F1 0.9220의 출처
  4) 분포 비교 (KS 검정 + Cohen's d) + 시각화
  5) 텍스트 n-gram 편중 + TF-IDF 단독 베이스라인
  6) 길이 매칭 평가            — 길이 단서를 제거하면 성능이 얼마나 남는가

실행:
    python analysis/leakage_audit.py --data /path/to/data_yelp.parquet

산출물:
    analysis/leakage_results.json
    docs/images/*.png

주의: 기존 학습 파이프라인(src/*)은 건드리지 않는다. 읽기 전용 감사 코드다.
split은 src/dataset.py의 stratified_split과 동일한 방식(seed=42, 70/15/15)을
재현해 README 실험과 같은 test set 위에서 평가한다.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "docs" / "images"
OUT_JSON = Path(__file__).resolve().parent / "leakage_results.json"

SEED = 42
ORIG_COLS = ["pk", "review_id", "text", "label", "source", "review_stars", "business_id"]
LABEL_MAP = {"human": 0, "ai": 1}
PUNCT = set(string.punctuation)


# ──────────────────────────────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────────────────────────────
def load(data_path: Path):
    df = pd.read_parquet(data_path)
    feature_cols = [c for c in df.columns if c not in ORIG_COLS]
    df = df.copy()
    df["y"] = df["label"].map(LABEL_MAP).astype("int64")
    return df, feature_cols


def stratified_split(y: np.ndarray, seed: int = SEED):
    """src/dataset.py와 동일한 70/15/15 stratified split."""
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=y[tmp])
    return tr, va, te


# ──────────────────────────────────────────────────────────────────────
# 사소한(trivial) 표층 피처 — 원문 문자열에서 직접 계산
# ──────────────────────────────────────────────────────────────────────
def trivial_features(texts: pd.Series) -> pd.DataFrame:
    rows = []
    for t in texts:
        t = t if isinstance(t, str) else ""
        n_char = len(t)
        words = t.split()
        n_word = len(words)
        n_punct = sum(1 for ch in t if ch in PUNCT)
        n_upper = sum(1 for ch in t if ch.isupper())
        n_digit = sum(1 for ch in t if ch.isdigit())
        alpha = [ch for ch in t if ch.isalpha()]
        lw = [w.lower().strip(string.punctuation) for w in words]
        lw = [w for w in lw if w]
        rows.append({
            "char_len":      n_char,
            "word_count":    n_word,
            "punct_ratio":   n_punct / n_char if n_char else 0.0,
            "comma_ratio":   t.count(",") / n_word if n_word else 0.0,
            "period_ratio":  t.count(".") / n_word if n_word else 0.0,
            "excl_count":    t.count("!"),
            "upper_ratio":   n_upper / len(alpha) if alpha else 0.0,
            "digit_ratio":   n_digit / n_char if n_char else 0.0,
            "avg_word_len":  float(np.mean([len(w) for w in words])) if words else 0.0,
            "type_token_ratio": len(set(lw)) / len(lw) if lw else 0.0,
            "newline_count": t.count("\n"),
        })
    return pd.DataFrame(rows, index=texts.index)


# ──────────────────────────────────────────────────────────────────────
# 평가 헬퍼
# ──────────────────────────────────────────────────────────────────────
def eval_lr(Xtr, ytr, Xte, yte, seed: int = SEED) -> dict:
    """표준화 + 로지스틱 회귀. test F1/AUC/Acc 반환."""
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=seed))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return {
        "f1": float(f1_score(yte, (p >= 0.5).astype(int))),
        "auc": float(roc_auc_score(yte, p)),
        "accuracy": float(accuracy_score(yte, (p >= 0.5).astype(int))),
    }


def single_feature_auc(x: np.ndarray, y: np.ndarray) -> float:
    """방향 무관 AUC — 피처값 자체를 점수로 썼을 때의 판별력."""
    x = np.nan_to_num(x.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    a = roc_auc_score(y, x)
    return float(max(a, 1.0 - a))


# ──────────────────────────────────────────────────────────────────────
# 1) 단일 피처 판별력
# ──────────────────────────────────────────────────────────────────────
def step1_single_feature(df, feature_cols, tr, te) -> dict:
    y = df["y"].values
    out = {}
    for c in feature_cols:
        x = df[c].values
        raw = single_feature_auc(x[te], y[te])
        lr = eval_lr(x[tr].reshape(-1, 1), y[tr], x[te].reshape(-1, 1), y[te])
        out[c] = {"raw_auc_test": raw, "lr_f1_test": lr["f1"], "lr_auc_test": lr["auc"]}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["raw_auc_test"]))


# ──────────────────────────────────────────────────────────────────────
# 2) 사소한 베이스라인
# ──────────────────────────────────────────────────────────────────────
def step2_trivial(triv, y, tr, te) -> dict:
    groups = {
        "char_len_only":        ["char_len"],
        "word_count_only":      ["word_count"],
        "punct_ratio_only":     ["punct_ratio"],
        "type_token_ratio_only": ["type_token_ratio"],
        "len_word_punct(3)":    ["char_len", "word_count", "punct_ratio"],
        "all_trivial(11)":      list(triv.columns),
    }
    res = {}
    for name, cols in groups.items():
        Xtr, Xte = triv.iloc[tr][cols].values, triv.iloc[te][cols].values
        r = eval_lr(Xtr, y[tr], Xte, y[te])
        # 비선형 모델도 함께 — 단조 관계가 아닌 경우를 놓치지 않기 위해
        gb = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, y[tr])
        p = gb.predict_proba(Xte)[:, 1]
        r["gb_f1"] = float(f1_score(y[te], (p >= 0.5).astype(int)))
        r["gb_auc"] = float(roc_auc_score(y[te], p))
        r["n_features"] = len(cols)
        res[name] = r
    # 단일 trivial 피처 raw AUC
    res["_single_raw_auc"] = {c: single_feature_auc(triv.iloc[te][c].values, y[te])
                              for c in triv.columns}
    res["_single_raw_auc"] = dict(sorted(res["_single_raw_auc"].items(),
                                         key=lambda kv: -kv[1]))
    return res


# ──────────────────────────────────────────────────────────────────────
# 3) 전체 HC 모델 + permutation importance
# ──────────────────────────────────────────────────────────────────────
def step3_hc_full(df, feature_cols, tr, te) -> dict:
    y = df["y"].values
    X = np.nan_to_num(df[feature_cols].astype("float64").values,
                      nan=0.0, posinf=0.0, neginf=0.0)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    res = {"logreg": eval_lr(Xtr, ytr, Xte, yte)}

    # README의 HC-only MLP(F1 0.9220) 재현 성격의 모델
    mlp = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                      early_stopping=True, random_state=SEED))
    mlp.fit(Xtr, ytr)
    p = mlp.predict_proba(Xte)[:, 1]
    res["mlp"] = {"f1": float(f1_score(yte, (p >= 0.5).astype(int))),
                  "auc": float(roc_auc_score(yte, p)),
                  "accuracy": float(accuracy_score(yte, (p >= 0.5).astype(int)))}

    gb = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
    pg = gb.predict_proba(Xte)[:, 1]
    res["hist_gb"] = {"f1": float(f1_score(yte, (pg >= 0.5).astype(int))),
                      "auc": float(roc_auc_score(yte, pg)),
                      "accuracy": float(accuracy_score(yte, (pg >= 0.5).astype(int)))}

    pi = permutation_importance(gb, Xte, yte, n_repeats=10,
                                random_state=SEED, scoring="roc_auc")
    imp = {feature_cols[i]: {"mean": float(pi.importances_mean[i]),
                             "std": float(pi.importances_std[i])}
           for i in range(len(feature_cols))}
    res["permutation_importance_gb_auc"] = dict(
        sorted(imp.items(), key=lambda kv: -kv[1]["mean"]))

    # 피처 그룹별 ablation — 어느 묶음이 성능을 만드는가
    groups = {
        "count(7)": ["syllable", "lexicon", "sentence", "char", "letter",
                     "polysyllab", "monosyllab"],
        "readability(6)": ["smog_index", "flesch_reading_ease", "flesch_kincaid_grade",
                           "fog_scale", "dale_chall", "reading_time"],
        "sentiment(2)": ["sentiment", "subjectivity"],
        "lm(2)": ["perplexity", "burstiness"],
        "pos(6)": ["nouns", "adj", "verbs", "pronoun", "adverb", "article"],
    }
    abl = {}
    for name, cols in groups.items():
        cols = [c for c in cols if c in feature_cols]
        if not cols:
            continue
        ix = [feature_cols.index(c) for c in cols]
        abl[f"only_{name}"] = eval_lr(Xtr[:, ix], ytr, Xte[:, ix], yte)
        rest = [i for i in range(len(feature_cols)) if i not in ix]
        abl[f"without_{name}"] = eval_lr(Xtr[:, rest], ytr, Xte[:, rest], yte)
    res["group_ablation_logreg"] = abl
    return res


# ──────────────────────────────────────────────────────────────────────
# 4) 분포 비교 (KS + Cohen's d) + 플롯
# ──────────────────────────────────────────────────────────────────────
def cohens_d(a, b) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((b.mean() - a.mean()) / sp) if sp > 0 else 0.0


def step4_distribution(df, feature_cols, triv) -> dict:
    y = df["y"].values
    res = {}
    all_cols = [(c, df[c].values.astype(float)) for c in feature_cols] + \
               [(f"trivial:{c}", triv[c].values.astype(float)) for c in triv.columns]
    for name, x in all_cols:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h, a = x[y == 0], x[y == 1]
        ks = stats.ks_2samp(h, a)
        res[name] = {
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "cohens_d": cohens_d(h, a),
            "human_mean": float(h.mean()), "ai_mean": float(a.mean()),
            "human_std": float(h.std()), "ai_std": float(a.std()),
        }
    return dict(sorted(res.items(), key=lambda kv: -kv[1]["ks_stat"]))


def plot_distributions(df, triv, dist, single_auc, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    y = df["y"].values
    paths = []

    def series(name):
        if name.startswith("trivial:"):
            return triv[name.split(":", 1)[1]].values.astype(float)
        return df[name].values.astype(float)

    # (a) KS 상위 6개 분포 비교
    top = [k for k in list(dist)[:6]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes.ravel(), top):
        x = np.nan_to_num(series(name), nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(x, [0.5, 99.5])
        bins = np.linspace(lo, hi, 50) if hi > lo else 50
        ax.hist(x[y == 0], bins=bins, alpha=0.6, label="human", density=True)
        ax.hist(x[y == 1], bins=bins, alpha=0.6, label="ai", density=True)
        ax.set_title(f"{name}\nKS={dist[name]['ks_stat']:.3f}  d={dist[name]['cohens_d']:+.2f}",
                     fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("human vs AI: top-6 features by KS distance")
    fig.tight_layout()
    p = out_dir / "audit_distribution_top6.png"
    fig.savefig(p, dpi=130); plt.close(fig); paths.append(str(p))

    # (b) 단일 피처 AUC 막대
    items = list(single_auc.items())
    names = [k for k, _ in items]
    vals = [v["raw_auc_test"] for _, v in items]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#c0392b" if v >= 0.9 else ("#e67e22" if v >= 0.8 else "#7f8c8d") for v in vals]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.axvline(0.9, color="#c0392b", lw=0.8, ls=":")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("single-feature AUC on test (direction-agnostic)")
    ax.set_title("Discriminative power of each HC feature alone\n(>= 0.90 would be an artifact candidate)")
    fig.tight_layout()
    p = out_dir / "audit_single_feature_auc.png"
    fig.savefig(p, dpi=130); plt.close(fig); paths.append(str(p))

    # (c) 길이 분포
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cl = triv["char_len"].values.astype(float)
    bins = np.linspace(0, np.percentile(cl, 99), 60)
    ax.hist(cl[y == 0], bins=bins, alpha=0.6, label="human", density=True)
    ax.hist(cl[y == 1], bins=bins, alpha=0.6, label="ai", density=True)
    ax.set_xlabel("review length (characters)"); ax.set_ylabel("density")
    ax.set_title("Review length distribution: human vs LLM-rewritten")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "audit_length_distribution.png"
    fig.savefig(p, dpi=130); plt.close(fig); paths.append(str(p))
    return paths


# ──────────────────────────────────────────────────────────────────────
# 5) 텍스트 n-gram 편중 + TF-IDF 단독 베이스라인
# ──────────────────────────────────────────────────────────────────────
def step5_ngram(df, tr, te) -> dict:
    y = df["y"].values
    txt = df["text"].fillna("").values
    res = {}

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000,
                            sublinear_tf=True)
    Xtr = tfidf.fit_transform(txt[tr])
    Xte = tfidf.transform(txt[te])
    clf = LogisticRegression(max_iter=3000, random_state=SEED).fit(Xtr, y[tr])
    p = clf.predict_proba(Xte)[:, 1]
    res["tfidf_word12_logreg"] = {
        "f1": float(f1_score(y[te], (p >= 0.5).astype(int))),
        "auc": float(roc_auc_score(y[te], p)),
        "accuracy": float(accuracy_score(y[te], (p >= 0.5).astype(int))),
        "n_features": int(Xtr.shape[1]),
    }

    # 클래스 편중 n-gram: 문서빈도 기반
    cv = CountVectorizer(ngram_range=(1, 3), min_df=20, binary=True,
                         max_features=200_000)
    C = cv.fit_transform(txt)
    vocab = np.array(cv.get_feature_names_out())
    Ch = np.asarray(C[y == 0].sum(axis=0)).ravel()
    Ca = np.asarray(C[y == 1].sum(axis=0)).ravel()
    nh, na = int((y == 0).sum()), int((y == 1).sum())
    dfh, dfa = Ch / nh, Ca / na
    tot = Ch + Ca
    keep = tot >= 100

    def topk(score, k=20):
        ix = np.argsort(-score)
        ix = [i for i in ix if keep[i]][:k]
        return [{"ngram": vocab[i], "df_human": round(float(dfh[i]), 4),
                 "df_ai": round(float(dfa[i]), 4),
                 "ratio": round(float((dfa[i] + 1e-4) / (dfh[i] + 1e-4)), 2)} for i in ix]

    res["ai_skewed_ngrams"] = topk(np.where(keep, dfa - dfh, -np.inf))
    res["human_skewed_ngrams"] = topk(np.where(keep, dfh - dfa, -np.inf))

    # 한쪽 클래스에만 사실상 배타적으로 나타나는 n-gram 수
    excl_ai = int(((dfa > 0.01) & (dfh < 0.0005)).sum())
    excl_h = int(((dfh > 0.01) & (dfa < 0.0005)).sum())
    res["near_exclusive_ngrams"] = {
        "ai_only(df>1% & human<0.05%)": excl_ai,
        "human_only(df>1% & ai<0.05%)": excl_h,
        "vocab_size": int(len(vocab)),
    }

    # 정형 문구(boilerplate) 탐지 — AI 쪽에 반복되는 접두/접미
    def lead(t, n=5):
        return " ".join(re.sub(r"\s+", " ", t).strip().split()[:n]).lower()
    leads_ai = pd.Series([lead(t) for t in txt[y == 1]]).value_counts().head(10)
    leads_h = pd.Series([lead(t) for t in txt[y == 0]]).value_counts().head(10)
    res["top_opening_5grams"] = {
        "ai": {k: int(v) for k, v in leads_ai.items()},
        "human": {k: int(v) for k, v in leads_h.items()},
    }
    res["opening_5gram_concentration"] = {
        "ai_top10_share": round(float(leads_ai.sum() / (y == 1).sum()), 4),
        "human_top10_share": round(float(leads_h.sum() / (y == 0).sum()), 4),
        "ai_unique_openings": int(pd.Series([lead(t) for t in txt[y == 1]]).nunique()),
        "human_unique_openings": int(pd.Series([lead(t) for t in txt[y == 0]]).nunique()),
    }
    return res


# ──────────────────────────────────────────────────────────────────────
# 7) 생성기 교차 일반화 — Qwen으로 학습해 Llama를 맞출 수 있는가
# ──────────────────────────────────────────────────────────────────────
def step7_cross_generator(df, feature_cols) -> dict:
    """AI 라벨을 만든 두 모델(Qwen3-1.7B / Llama-3.2-1B) 사이의 전이 성능.
    같은 생성기 안에서만 잘 맞는다면, 98%는 '생성기 지문 암기'에 가깝다."""
    y = df["y"].values
    src = df["source"].values
    txt = df["text"].fillna("").values
    X = np.nan_to_num(df[feature_cols].astype("float64").values,
                      nan=0.0, posinf=0.0, neginf=0.0)
    gens = [g for g in pd.unique(src) if g != "human"]
    if len(gens) != 2:
        return {"_skipped": f"생성기 수가 2가 아님: {list(gens)}"}
    rng = np.random.default_rng(SEED)
    hidx = np.where(src == "human")[0]
    rng.shuffle(hidx)
    h_a, h_b = hidx[:len(hidx) // 2], hidx[len(hidx) // 2:]

    out = {}
    for ga, gb, ha, hb in [(gens[0], gens[1], h_a, h_b), (gens[1], gens[0], h_b, h_a)]:
        tr_ix = np.concatenate([ha, np.where(src == ga)[0]])
        te_ix = np.concatenate([hb, np.where(src == gb)[0]])
        key = f"train={ga.split('/')[-1]} -> test={gb.split('/')[-1]}"
        r = {"hc_lr": eval_lr(X[tr_ix], y[tr_ix], X[te_ix], y[te_ix])}
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        A = tf.fit_transform(txt[tr_ix]); B = tf.transform(txt[te_ix])
        clf = LogisticRegression(max_iter=3000, random_state=SEED).fit(A, y[tr_ix])
        p = clf.predict_proba(B)[:, 1]
        r["tfidf_lr"] = {"f1": float(f1_score(y[te_ix], (p >= 0.5).astype(int))),
                         "auc": float(roc_auc_score(y[te_ix], p))}
        r["n_train"], r["n_test"] = int(len(tr_ix)), int(len(te_ix))
        out[key] = r

    # 대조군: 같은 생성기 안에서의 성능 (동일 규모)
    for g in gens:
        gi = np.where(src == g)[0]
        ix = np.concatenate([hidx[:len(gi)], gi])
        yy = y[ix]
        t, _, e = stratified_split(yy, SEED)
        out[f"in-generator {g.split('/')[-1]}"] = {
            "hc_lr": eval_lr(X[ix][t], yy[t], X[ix][e], yy[e]),
        }
    return out


# ──────────────────────────────────────────────────────────────────────
# 6) 길이 매칭 평가 — 길이 단서를 제거하면 무엇이 남는가
# ──────────────────────────────────────────────────────────────────────
def step6_length_matched(df, feature_cols, triv) -> dict:
    """char_len 구간별로 human/ai 수를 동일하게 맞춘 부분집합에서 재평가.
    길이 자체가 라벨을 알려주지 못하도록 통제한 상태의 성능."""
    y = df["y"].values
    cl = triv["char_len"].values
    edges = np.percentile(cl, np.linspace(0, 100, 21))
    edges[-1] += 1
    b = np.digitize(cl, edges[1:-1])
    rng = np.random.default_rng(SEED)
    keep = []
    for k in np.unique(b):
        ih = np.where((b == k) & (y == 0))[0]
        ia = np.where((b == k) & (y == 1))[0]
        m = min(len(ih), len(ia))
        if m == 0:
            continue
        keep.extend(rng.choice(ih, m, replace=False).tolist())
        keep.extend(rng.choice(ia, m, replace=False).tolist())
    keep = np.array(sorted(keep))
    ym = y[keep]
    tr, va, te = stratified_split(ym, SEED)

    X = np.nan_to_num(df[feature_cols].astype("float64").values,
                      nan=0.0, posinf=0.0, neginf=0.0)[keep]
    T = triv.values.astype("float64")[keep]
    ti = list(triv.columns)

    res = {
        "n_matched": int(len(keep)),
        "balance": {"human": int((ym == 0).sum()), "ai": int((ym == 1).sum())},
        "char_len_auc_after_matching": single_feature_auc(triv["char_len"].values[keep][te], ym[te]),
        "trivial_all_lr": eval_lr(T[tr], ym[tr], T[te], ym[te]),
        "hc_all_lr": eval_lr(X[tr], ym[tr], X[te], ym[te]),
    }
    gb = HistGradientBoostingClassifier(random_state=SEED).fit(X[tr], ym[tr])
    p = gb.predict_proba(X[te])[:, 1]
    res["hc_all_gb"] = {"f1": float(f1_score(ym[te], (p >= 0.5).astype(int))),
                        "auc": float(roc_auc_score(ym[te], p))}

    # 길이 계열 피처를 아예 뺀 HC (원본 split 기준)
    len_like = ["char", "letter", "syllable", "lexicon", "sentence",
                "polysyllab", "monosyllab", "reading_time"]
    rest = [i for i, c in enumerate(feature_cols) if c not in len_like]
    Xf = np.nan_to_num(df[feature_cols].astype("float64").values,
                       nan=0.0, posinf=0.0, neginf=0.0)
    ftr, fva, fte = stratified_split(y, SEED)
    res["hc_without_length_features_lr(full_data)"] = eval_lr(
        Xf[ftr][:, rest], y[ftr], Xf[fte][:, rest], y[fte])
    res["_note"] = ("length-matched: char_len 5%ile 구간 20개 안에서 human/ai 수를 "
                    "동일하게 맞춘 뒤 동일 방식으로 70/15/15 split")
    ti  # noqa: B018  (컬럼 순서 확인용)
    return res


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "data_yelp.parquet",
                    help="data_yelp.parquet 경로 (20,000행 × text+label+HC 23)")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--img-dir", type=Path, default=IMG_DIR)
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"데이터 없음: {args.data}  (DATA.md 참고)")

    df, feature_cols = load(args.data)
    y = df["y"].values
    tr, va, te = stratified_split(y, SEED)
    print(f"[data] {args.data}  rows={len(df)}  features={len(feature_cols)}")
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)}")

    triv = trivial_features(df["text"])

    print("[1/6] 단일 피처 판별력 …")
    s1 = step1_single_feature(df, feature_cols, tr, te)
    print("[2/6] 사소한 베이스라인 …")
    s2 = step2_trivial(triv, y, tr, te)
    print("[3/6] 전체 HC 모델 + permutation importance …")
    s3 = step3_hc_full(df, feature_cols, tr, te)
    print("[4/6] 분포 비교 (KS) …")
    s4 = step4_distribution(df, feature_cols, triv)
    print("[5/6] n-gram 편중 + TF-IDF 베이스라인 …")
    s5 = step5_ngram(df, tr, te)
    print("[6/7] 길이 매칭 평가 …")
    s6 = step6_length_matched(df, feature_cols, triv)
    print("[7/7] 생성기 교차 일반화 …")
    s7 = step7_cross_generator(df, feature_cols)

    print("[plot] 그림 저장 …")
    plots = plot_distributions(df, triv, s4, s1, args.img_dir)

    results = {
        "meta": {
            "data_file": str(args.data),
            "n_rows": int(len(df)),
            "n_hc_features": len(feature_cols),
            "hc_features": feature_cols,
            "label_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
            "source_counts": {k: int(v) for k, v in df["source"].value_counts().items()},
            "split": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                      "seed": SEED, "scheme": "stratified 70/15/15 (src/dataset.py와 동일)"},
        },
        "step1_single_feature_auc": s1,
        "step2_trivial_baselines": s2,
        "step3_hc_full_and_importance": s3,
        "step4_distribution_shift": s4,
        "step5_text_ngram": s5,
        "step6_length_matched": s6,
        "step7_cross_generator": s7,
        "plots": plots,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")

    # 요약 출력
    print("\n=== 단일 피처 AUC 상위 8 ===")
    for k, v in list(s1.items())[:8]:
        print(f"  {k:22s} raw_auc={v['raw_auc_test']:.4f}  lr_f1={v['lr_f1_test']:.4f}")
    print("\n=== 사소한 베이스라인 (test) ===")
    for k, v in s2.items():
        if k.startswith("_"):
            continue
        print(f"  {k:24s} LR f1={v['f1']:.4f} auc={v['auc']:.4f} | GB f1={v['gb_f1']:.4f} auc={v['gb_auc']:.4f}")
    print("\n=== HC 전체 ===")
    for k in ("logreg", "mlp", "hist_gb"):
        v = s3[k]
        print(f"  {k:10s} f1={v['f1']:.4f} auc={v['auc']:.4f} acc={v['accuracy']:.4f}")
    print("\n=== TF-IDF 텍스트 단독 ===")
    v = s5["tfidf_word12_logreg"]
    print(f"  f1={v['f1']:.4f} auc={v['auc']:.4f}")
    print("\n=== 길이 매칭 후 ===")
    print(f"  char_len AUC: {s6['char_len_auc_after_matching']:.4f}")
    print(f"  trivial all : f1={s6['trivial_all_lr']['f1']:.4f} auc={s6['trivial_all_lr']['auc']:.4f}")
    print(f"  HC all (LR) : f1={s6['hc_all_lr']['f1']:.4f} auc={s6['hc_all_lr']['auc']:.4f}")
    print("\n=== 생성기 교차 일반화 ===")
    for k, v in s7.items():
        if k.startswith("_"):
            continue
        line = f"  {k:46s} HC f1={v['hc_lr']['f1']:.4f} auc={v['hc_lr']['auc']:.4f}"
        if "tfidf_lr" in v:
            line += f" | TFIDF f1={v['tfidf_lr']['f1']:.4f} auc={v['tfidf_lr']['auc']:.4f}"
        print(line)


if __name__ == "__main__":
    main()
