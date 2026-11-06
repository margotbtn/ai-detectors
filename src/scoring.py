"""Per-text scores for every detector, computed once and shared by experiments 2, 3, 5.

Reads cached features only (src/features.py), fits the trained detectors on
`train`, and writes:
  artifacts/scores/scores_<tokens>.parquet   one row per non-train corpus text
  artifacts/models/<detector>_<tokens>.joblib fitted trained detectors (for the app)
"""

import joblib
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.data import get_split, load_corpus
from src.detectors import EmbeddingLRDetector, ShallowCueDetector, TfidfLRDetector
from src.features import load

SCORES_DIR = ARTIFACTS_DIR / "scores"
MODELS_DIR = ARTIFACTS_DIR / "models"

# The three demo detectors (project-state decision 5), in plotting order.
DEMO_DETECTORS = ["perplexity-qwen", "tfidf-lr", "embedding-lr"]
# Table-only extras: GPT-2 (the "previous generation" scorer) and the shallow-cue baseline.
EXTRA_DETECTORS = {256: ["perplexity-gpt2", "shallow-lr"], 128: ["shallow-lr"]}


def scores_path(tokens: int):
    return SCORES_DIR / f"scores_{tokens}.parquet"


def compute_scores(tokens: int = 256) -> pd.DataFrame:
    col = f"text{tokens}"
    corpus = load_corpus()
    train = get_split(corpus, "train")
    out = corpus[corpus["group"] != "train"].drop(columns=["text256", "text128"]).reset_index(drop=True)
    out["text"] = corpus.loc[corpus["group"] != "train", col].to_numpy()
    texts, y_train = out["text"].tolist(), train["label"].to_numpy()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for key in ("qwen", "gpt2") if tokens == 256 else ("qwen",):
        out[f"perplexity-{key}"] = load(key, texts)  # zero-shot: score = feature

    tfidf = TfidfLRDetector().fit(train[col], y_train)
    out["tfidf-lr"] = tfidf.score(texts)

    emb = EmbeddingLRDetector()  # features come from the cache; the encoder is never loaded
    emb.fit_features(load("minilm", train[col].tolist()), y_train)
    out["embedding-lr"] = emb.score_features(load("minilm", texts))

    shallow = ShallowCueDetector().fit(train[col], y_train)
    out["shallow-lr"] = shallow.score(texts)

    joblib.dump(tfidf.pipeline, MODELS_DIR / f"tfidf-lr_{tokens}.joblib", compress=3)
    joblib.dump(emb.clf, MODELS_DIR / f"embedding-lr_{tokens}.joblib")
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(scores_path(tokens))
    return out


def load_scores(tokens: int = 256) -> pd.DataFrame:
    p = scores_path(tokens)
    return pd.read_parquet(p) if p.exists() else compute_scores(tokens)


def split_scores(scores: pd.DataFrame, name: str) -> pd.DataFrame:
    from src.data import SPLITS

    return scores[SPLITS[name](scores)].reset_index(drop=True)


def detectors_for(tokens: int, demo_only: bool = False) -> list[str]:
    return DEMO_DETECTORS + ([] if demo_only else EXTRA_DETECTORS[tokens])

