"""The one heavy job: compute and cache every expensive feature (`make features`).

Experiments only READ these caches (see `load`). Re-running is near-instant
once everything is cached, and an interrupted run resumes from the last
finished chunk (src/cache.py).

What is computed (decided 2026-11-05 after the speed benchmark):
  minilm  all corpus texts at 256 and 128 tokens, all attack-sweep texts
  qwen    non-train texts at 256 and 128 tokens, attack-sweep texts (bf16)
  gpt2    non-train texts at 256 tokens, attack-sweep texts (bf16)
Zero-shot perplexity detectors never see the train split, so it is skipped.
Stages run in the order that unblocks experiment 2 soonest.
"""

import time

import numpy as np
import pandas as pd

from src.attacks import ATTACK_PS, homoglyph_attack, normalize_defense
from src.cache import FeatureCache
from src.config import SEED
from src.data import attack_subset, get_split, load_corpus
from src.detectors import EmbeddingLRDetector, PerplexityDetector, set_threads

THREADS = 8  # = physical cores; 16 (SMT) was slower in the benchmark


def caches() -> dict[str, FeatureCache]:
    return {
        "qwen": FeatureCache("perplexity-qwen", PerplexityDetector.config_for("qwen")),
        "gpt2": FeatureCache("perplexity-gpt2", PerplexityDetector.config_for("gpt2")),
        "minilm": FeatureCache("embedding-minilm", EmbeddingLRDetector.config_for()),
    }


def attack_texts(corpus: pd.DataFrame) -> pd.DataFrame:
    """Long table of every text experiment 4 scores.

    Columns: id, p, defended, text. Covers the attack subset at every p, with
    and without the normalization defense, plus the val human texts with the
    defense applied (to recalibrate the threshold when the defense is on).
    """
    rows = []
    for r in attack_subset(corpus).itertuples():
        for p in ATTACK_PS:
            t = homoglyph_attack(r.text256, p, seed=SEED)
            rows.append((r.id, p, False, t))
            rows.append((r.id, p, True, normalize_defense(t)))
    for r in get_split(corpus, "val").query("label == 0").itertuples():
        rows.append((r.id, np.nan, True, normalize_defense(r.text256)))
    return pd.DataFrame(rows, columns=["id", "p", "defended", "text"])


def load(key: str, texts) -> np.ndarray:
    """Cached features for `texts`; fails loudly instead of recomputing."""
    cache = caches()[key]
    missing = cache.missing(texts)
    if missing:
        raise RuntimeError(f"{len(missing)} texts have no cached '{key}' features: run `make features`")
    return cache.get_or_compute(texts, compute_fn=None)


def main() -> None:
    t0 = time.perf_counter()
    set_threads(THREADS)
    corpus = load_corpus()
    lm = corpus[corpus["group"] != "train"]
    att = attack_texts(corpus)["text"].tolist()
    c = caches()
    print(f"corpus {len(corpus)} texts | LM-scored {len(lm)} | attack-sweep texts {len(att)} "
          f"({len(set(att))} unique)", flush=True)

    emb = EmbeddingLRDetector()
    for label, texts in (("minilm 256", corpus["text256"]), ("minilm 128", corpus["text128"]),
                         ("minilm attack", att)):
        c["minilm"].get_or_compute(texts, emb.features, chunk_size=1000, label=label)
    del emb

    qwen, gpt2 = PerplexityDetector("qwen"), PerplexityDetector("gpt2")
    stages = [
        ("qwen 256", c["qwen"], qwen, lm["text256"]),
        ("gpt2 256", c["gpt2"], gpt2, lm["text256"]),
        ("qwen attack", c["qwen"], qwen, att),
        ("gpt2 attack", c["gpt2"], gpt2, att),
        ("qwen 128", c["qwen"], qwen, lm["text128"]),
    ]
    for label, cache, det, texts in stages:
        cache.get_or_compute(texts, det.features, chunk_size=200, label=label)
        print(f"== done: {label} | total {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    print({k: len(v) for k, v in caches().items()})


if __name__ == "__main__":
    main()
