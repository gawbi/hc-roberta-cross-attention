"""Hand-craft feature 추출 (23개) — notebooks/03_feature_extraction.ipynb의 모듈화.

카테고리:
- count (7):       syllable, lexicon, sentence, char, letter, polysyllab, monosyllab
- readability (6): smog_index, flesch_reading_ease, flesch_kincaid_grade, fog_scale,
                   dale_chall, reading_time
- sentiment (2):   sentiment(polarity), subjectivity
- LM 기반 (2):     perplexity, burstiness  (GPT-2 문장 단위)
- POS (6):         nouns, adj, verbs, pronoun, adverb, article

사용 예:
    from src.features import FeatureExtractor
    fx = FeatureExtractor()                  # spaCy/GPT-2 로드 (1회)
    feats = fx.extract_all("review text")    # dict[23]
    df_feats = fx.extract_dataframe(df["text"])   # DataFrame[N, 23]
"""
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ARTICLES = {"a", "an", "the"}   # LIWC article 근사

_COUNT_KEYS = ["syllable", "lexicon", "sentence", "char", "letter", "polysyllab", "monosyllab"]
_READ_KEYS = ["smog_index", "flesch_reading_ease", "flesch_kincaid_grade",
              "fog_scale", "dale_chall", "reading_time"]
_POS_KEYS = ["nouns", "adj", "verbs", "pronoun", "adverb", "article"]


def textstat_counts(text: str) -> dict:
    import textstat
    if not isinstance(text, str) or not text.strip():
        return dict.fromkeys(_COUNT_KEYS, 0)
    return {
        "syllable":   textstat.syllable_count(text),
        "lexicon":    textstat.lexicon_count(text),
        "sentence":   textstat.sentence_count(text),
        "char":       textstat.char_count(text),
        "letter":     textstat.letter_count(text),
        "polysyllab": textstat.polysyllabcount(text),
        "monosyllab": textstat.monosyllabcount(text),
    }


def readability(text: str) -> dict:
    import textstat
    try:
        return {
            "smog_index":           textstat.smog_index(text),
            "flesch_reading_ease":  textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "fog_scale":            textstat.gunning_fog(text),
            "dale_chall":           textstat.dale_chall_readability_score(text),
            "reading_time":         textstat.reading_time(text, ms_per_char=14.69),
        }
    except Exception:
        return dict.fromkeys(_READ_KEYS, 0.0)


def sentiment_feats(text: str) -> dict:
    from textblob import TextBlob
    if not isinstance(text, str) or not text.strip():
        return {"sentiment": 0.0, "subjectivity": 0.0}
    s = TextBlob(text).sentiment
    return {"sentiment": s.polarity, "subjectivity": s.subjectivity}


def pos_counts(doc) -> dict:
    """spaCy doc → POS 카운트. doc가 None이면 0."""
    if doc is None or len(doc) == 0:
        return dict.fromkeys(_POS_KEYS, 0)
    c = {"nouns": 0, "adj": 0, "verbs": 0, "pronoun": 0, "adverb": 0, "article": 0}
    for tok in doc:
        p = tok.pos_
        if p == "NOUN":   c["nouns"] += 1
        elif p == "ADJ":  c["adj"] += 1
        elif p == "VERB": c["verbs"] += 1
        elif p == "PRON": c["pronoun"] += 1
        elif p == "ADV":  c["adverb"] += 1
        if tok.lower_ in ARTICLES:
            c["article"] += 1
    return c


class FeatureExtractor:
    """spaCy(POS)·GPT-2(perplexity)를 1회 로드해 재사용하는 추출기."""

    def __init__(self, device: str | None = None):
        import spacy
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device).eval()
        self._torch = torch

    def perplexity_feats(self, text: str, max_length: int = 512) -> dict:
        """문장 단위 GPT-2 perplexity의 평균(perplexity)과 변동계수(burstiness).
        burstiness = std/mean — 문장 간 '의외성'이 얼마나 들쭉날쭉한지.
        유효 문장이 1개뿐이면 std=0 → burstiness=0 (한 문장 리뷰의 artifact)."""
        torch = self._torch
        if not isinstance(text, str) or not text.strip():
            return {"perplexity": 0.0, "burstiness": 0.0}
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.split()) >= 3]
        if not sents:
            sents = [text]
        ppls = []
        with torch.inference_mode():
            for s in sents:
                try:
                    ids = self.gpt2_tok(s, return_tensors="pt", truncation=True,
                                        max_length=max_length).input_ids.to(self.device)
                    if ids.shape[1] < 2:
                        continue
                    loss = self.gpt2(ids, labels=ids).loss
                    ppls.append(float(torch.exp(loss).item()))
                except Exception:
                    continue
        if not ppls:
            return {"perplexity": 0.0, "burstiness": 0.0}
        arr = np.array(ppls, dtype=float)
        mean = float(arr.mean())
        return {"perplexity": mean,
                "burstiness": float(arr.std() / mean) if mean > 0 else 0.0}

    def extract_all(self, text: str) -> dict:
        """텍스트 1건 → 23개 feature dict."""
        feats = {}
        feats.update(textstat_counts(text))
        feats.update(readability(text))
        feats.update(sentiment_feats(text))
        feats.update(self.perplexity_feats(text))
        doc = self.nlp(text[:100_000]) if isinstance(text, str) and text.strip() else None
        feats.update(pos_counts(doc))
        return feats

    def extract_dataframe(self, texts, checkpoint_path=None, checkpoint_every: int = 100) -> pd.DataFrame:
        """텍스트 시퀀스 → feature DataFrame. checkpoint_path를 주면 중간 저장."""
        from tqdm.auto import tqdm
        rows = []
        texts = list(texts)
        for i, txt in enumerate(tqdm(texts, total=len(texts))):
            rows.append(self.extract_all(txt))
            if checkpoint_path and (i + 1) % checkpoint_every == 0:
                pd.DataFrame(rows).to_parquet(checkpoint_path, index=False)
        return pd.DataFrame(rows)
