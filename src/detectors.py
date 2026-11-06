"""The three detectors, behind one interface.

Every detector exposes:
    name                     stable identifier
    fit(texts, labels)       no-op for zero-shot detectors
    score(texts) -> ndarray  HIGHER = MORE LIKELY AI

Each one splits into an expensive, cacheable `features(texts)` step and cheap
`fit_features` / `score_features` steps, so experiments can cache features
once and refit or rescore in milliseconds.
"""

import os

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from src.config import MODELS, SEED


def set_threads(n: int | None = None) -> None:
    torch.set_num_threads(n or os.cpu_count())


class Detector:
    name: str

    def features(self, texts) -> np.ndarray:
        raise NotImplementedError

    def fit_features(self, X, labels) -> "Detector":
        return self

    def score_features(self, X) -> np.ndarray:
        raise NotImplementedError

    def fit(self, texts, labels) -> "Detector":
        return self.fit_features(self.features(texts), np.asarray(labels))

    def score(self, texts) -> np.ndarray:
        return self.score_features(self.features(texts))


class PerplexityDetector(Detector):
    """Zero-shot: score = mean token log-likelihood under a small causal LM.

    LLM text tends to be low-perplexity, i.e. HIGH mean log-likelihood, so the
    raw mean log-likelihood already points the right way (higher = more AI).
    Do NOT negate it: negative mean log-likelihood is cross-entropy, which is
    high for human text and inverts the detector (AUROC ~0.2).
    """

    def __init__(self, model_key: str = "qwen", batch_size: int = 1, max_tokens: int = 1024,
                 dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo, rev = MODELS[model_key]
        self.model_key = model_key
        self.name = f"perplexity-{model_key}"
        # bf16 is ~2x faster than fp32 on CPUs with native bf16 (AVX512-BF16/AMX)
        # and matched fp32 scores with correlation 1.0000 in the speed benchmark.
        # Batch 1 was fastest: padding waste outweighs batching gains on CPU.
        self.dtype = dtype
        self.batch_size = batch_size
        # Texts are truncated to 256 tokens upstream; this cap only bounds the
        # cost of homoglyph-attacked texts, which tokenize into many more tokens.
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=rev)
        self.model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=dtype).eval()
        # Neither GPT-2 nor Qwen2.5 base adds a BOS token. Prepend EOS as a
        # document-start marker so the first real token is scored too.
        self.start_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.start_id

    @staticmethod
    def config_for(model_key: str, dtype: torch.dtype = torch.bfloat16, max_tokens: int = 1024) -> dict:
        """Everything that changes the features: the cache key. Static, so
        experiments can look features up without loading the model."""
        repo, rev = MODELS[model_key]
        return {"detector": "perplexity", "repo": repo, "rev": rev, "dtype": str(dtype),
                "max_tokens": max_tokens, "start_token": "eos", "upcast_logits": True}

    def cache_config(self) -> dict:
        return self.config_for(self.model_key, self.dtype, self.max_tokens)

    def token_counts(self, texts) -> np.ndarray:
        enc = self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return np.array([len(ids) for ids in enc])

    @torch.inference_mode()
    def features(self, texts) -> np.ndarray:
        texts = list(texts)
        ids = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        ids = [[self.start_id] + x[: self.max_tokens - 1] for x in ids]
        out = np.empty(len(texts), dtype=np.float64)
        order = np.argsort([len(x) for x in ids])  # length-sorted batches = little padding
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            width = max(len(ids[i]) for i in idx)
            input_ids = torch.full((len(idx), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for row, i in enumerate(idx):
                input_ids[row, : len(ids[i])] = torch.tensor(ids[i])
                mask[row, : len(ids[i])] = 1
            # Upcast before log-softmax: bf16 logits lose precision over a 150k vocab.
            logits = self.model(input_ids=input_ids, attention_mask=mask).logits[:, :-1].float()
            targets = input_ids[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
            ).view_as(targets)
            m = mask[:, 1:].float()
            out[idx] = (-(nll * m).sum(1) / m.sum(1).clamp(min=1)).numpy()  # mean log-likelihood
        return out

    def score_features(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)


class TfidfLRDetector(Detector):
    """Word 1-2-gram + char 3-5-gram TF-IDF, logistic regression."""

    name = "tfidf-lr"

    def __init__(self, C: float = 1.0):
        self.C = C
        self.pipeline = None

    def features(self, texts) -> np.ndarray:
        return np.asarray(list(texts), dtype=object)  # TF-IDF is fit on the texts themselves

    def fit_features(self, X, labels) -> "TfidfLRDetector":
        vec = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ])
        clf = LogisticRegression(C=self.C, solver="liblinear", random_state=SEED)
        self.pipeline = make_pipeline(vec, clf).fit(list(X), labels)
        return self

    def score_features(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(list(X))[:, 1]


class EmbeddingLRDetector(Detector):
    """Frozen all-MiniLM-L6-v2 sentence embeddings, logistic regression."""

    name = "embedding-lr"
    MAX_SEQ_LENGTH = 256  # word pieces; the model's own default

    def __init__(self, batch_size: int = 32, C: float = 1.0):
        self.batch_size = batch_size
        self.C = C
        self.clf = None
        self._encoder = None  # loaded on first use: fitting on cached features needs no model

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            repo, rev = MODELS["minilm"]
            self._encoder = SentenceTransformer(repo, revision=rev, device="cpu")
            if self._encoder.max_seq_length != self.MAX_SEQ_LENGTH:
                raise ValueError(f"expected max_seq_length {self.MAX_SEQ_LENGTH}, "
                                 f"got {self._encoder.max_seq_length}")
        return self._encoder

    @classmethod
    def config_for(cls) -> dict:
        repo, rev = MODELS["minilm"]
        return {"detector": "embedding", "repo": repo, "rev": rev,
                "max_seq_length": cls.MAX_SEQ_LENGTH, "normalize": True}

    def cache_config(self) -> dict:
        return self.config_for()

    def features(self, texts) -> np.ndarray:
        return self.encoder.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)

    def fit_features(self, X, labels) -> "EmbeddingLRDetector":
        self.clf = LogisticRegression(C=self.C, max_iter=2000, random_state=SEED).fit(X, labels)
        return self

    def score_features(self, X) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]


class ShallowCueDetector(Detector):
    """~14 surface statistics + logistic regression. No language model.

    Not a real detector: it measures how much of the AI/human gap in this
    data is explained by length, punctuation and register alone (project-state
    section 5). Computed once in experiment 2, never plotted or shown in the app.
    """

    name = "shallow-lr"
    FEATURES = ("n_chars", "n_words", "mean_sentence_words", "comma_rate", "period_rate", "qe_rate",
                "quote_rate", "paren_rate", "colon_semi_rate", "type_token_ratio", "stopword_rate",
                "digit_rate", "upper_rate", "newline_rate")

    def __init__(self, C: float = 1.0):
        self.C = C
        self.pipeline = None

    @staticmethod
    def _one(text: str) -> list[float]:
        import re

        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        words = re.findall(r"[A-Za-z']+", text)
        lower = [w.lower() for w in words]
        nw, nc = max(len(words), 1), max(len(text), 1)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        return [
            len(text), len(words), len(words) / max(len(sentences), 1),
            text.count(",") / nw, text.count(".") / nw, (text.count("?") + text.count("!")) / nw,
            sum(text.count(q) for q in "\"'“”‘’") / nw,
            sum(text.count(b) for b in "()[]") / nw, (text.count(":") + text.count(";")) / nw,
            len(set(lower)) / nw, sum(w in ENGLISH_STOP_WORDS for w in lower) / nw,
            sum(ch.isdigit() for ch in text) / nc, sum(ch.isupper() for ch in text) / nc,
            text.count("\n") / nw,
        ]

    def features(self, texts) -> np.ndarray:
        return np.array([self._one(t) for t in texts], dtype=float)

    def fit_features(self, X, labels) -> "ShallowCueDetector":
        from sklearn.preprocessing import StandardScaler

        self.pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000,
                                                                           random_state=SEED))
        self.pipeline.fit(X, labels)
        return self

    def score_features(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]
