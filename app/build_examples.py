"""Pick the app's prefilled example texts from RAID, deterministically.

Each example makes one point of the talk, chosen by its cached scores at the
val-calibrated 5% FPR thresholds (experiment 2). Writes app/examples.json,
which is committed: the Space has no corpus.

    python app/build_examples.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import SEED  # noqa: E402
from src.metrics import calibrate_threshold  # noqa: E402
from src.scoring import DEMO_DETECTORS, load_scores, split_scores  # noqa: E402

OUT = Path(__file__).resolve().parent / "examples.json"
FPR = 0.05


def sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def head_words(text: str, n_words: int) -> str:
    """Whole sentences from the start until about n_words words."""
    out, n = [], 0
    for s in sentences(text):
        if n and n + len(s.split()) > n_words:
            break
        out.append(s)
        n += len(s.split())
    return " ".join(out)


# Neutral topics for a live demo, hand-picked among the texts matching each
# example's criteria (seeded samples of the candidates). Changing an id is fine
# as long as `pick` still accepts it.
PINNED = {"chatgpt": "95a1468d-8f97-47e8-b359-d328bdbdf92d", "mixed_human": "4f822bc7-37ae-4a08-a68c-4c619dde37a9"}


def pick(d: pd.DataFrame, flags: pd.DataFrame, pattern: tuple[bool, bool, bool], pinned: str | None = None,
         **filters) -> pd.Series:
    """A text with the given flag pattern (Qwen, TF-IDF, embeddings) at 5% FPR."""
    m = np.ones(len(d), bool)
    for det, want in zip(DEMO_DETECTORS, pattern):
        m &= (flags.loc[d.index, det] == want).to_numpy()
    for col, val in filters.items():
        m &= (d[col] == val).to_numpy()
    m &= (d["n_tokens_full"] >= 256).to_numpy()  # full 256-token texts, like the experiments
    ok = d[m]
    if pinned is None:
        return ok.sample(1, random_state=SEED).iloc[0]
    if pinned not in set(ok.id):
        raise ValueError(f"pinned text {pinned} no longer matches its criteria")
    return ok[ok.id == pinned].iloc[0]


def main() -> None:
    scores = load_scores(256)
    val = split_scores(scores, "val")
    th = {k: calibrate_threshold(val.loc[val.label == 0, k], FPR) for k in DEMO_DETECTORS}
    flags = pd.DataFrame({k: scores[k] > th[k] for k in DEMO_DETECTORS})
    test_in = scores[scores.group == "test"]
    ood = scores[scores.group == "ood"]

    human = pick(test_in, flags, (False, False, False), model="human", domain="news")
    ai = pick(test_in, flags, (True, True, True), PINNED["chatgpt"], model="chatgpt", domain="news")
    gpt4 = pick(test_in, flags, (False, True, True), model="gpt4", domain="news")
    abstract = pick(ood, flags, (False, True, True), model="human", domain="abstracts")
    # Mixed: RAID generates each AI text from a human article's title, so a human
    # article and its chatgpt counterpart tell the same story.
    pair_src = test_in[(test_in.model == "chatgpt")].source_id
    mixed_h = pick(test_in[test_in.source_id.isin(pair_src)], flags, (False, False, False), PINNED["mixed_human"],
                   model="human", domain="news")
    mixed_ai = test_in[(test_in.source_id == mixed_h.source_id) & (test_in.model == "chatgpt")].iloc[0]

    examples = [
        {"name": "Human news article", "label": "human", "source": f"RAID {human.id}",
         "point": "Real journalism. No detector flags it.", "text": human.text},
        {"name": "ChatGPT news article", "label": "AI", "source": f"RAID {ai.id}",
         "point": "The training generator. Every detector flags it.", "text": ai.text},
        {"name": "GPT-4 news article", "label": "AI", "source": f"RAID {gpt4.id}",
         "point": "A generator none of them trained on. Perplexity misses it.", "text": gpt4.text},
        {"name": "Human arXiv abstract", "label": "human", "source": f"RAID {abstract.id}",
         "point": "A new domain. The trained detectors call a scientist's writing AI.", "text": abstract.text},
        {"name": "Mixed: human opening + ChatGPT continuation", "label": "mixed",
         "source": f"RAID {mixed_h.id} + {mixed_ai.id}",
         "point": "Same story, half and half. What should the answer even be?",
         "text": head_words(mixed_h.text, 95) + " " + head_words(mixed_ai.text, 95)},
        {"name": "Short ChatGPT text", "label": "AI", "source": f"RAID {ai.id} (first sentences)",
         "point": "Under 100 tokens: scores are unreliable.", "text": head_words(ai.text, 50)},
    ]
    OUT.write_text(json.dumps(examples, indent=1, ensure_ascii=False))
    for e in examples:
        print(f"== {e['name']} ({e['source']})\n{e['text'][:400]}...\n")


if __name__ == "__main__":
    main()
