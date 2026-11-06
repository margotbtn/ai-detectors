"""Experiment 3: detectors calibrated to the same FPR still flag different texts.

For each pair of demo detectors, on test_in and test_ood_both, at thresholds
calibrated on val humans (5% headline, 1% alongside):
  - Jaccard overlap of the flagged sets |A & B| / |A | B|, separately on AI
    texts, human texts and all texts (project-state correction 3: not kappa)
  - McNemar exact test on the discordant pairs: is one detector systematically
    stricter than the other?
Plus: a scatter of two detectors' scores on a shared "percentile among val
humans" scale, a table of disagreement examples, and a breakdown of the
3-way disagreement rate by length, domain and generator.

Writes artifacts/results/03_*.csv and artifacts/figures/03_*.png.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import plotting  # noqa: E402
from src.config import ARTIFACTS_DIR, SEED  # noqa: E402
from src.metrics import FPR_TARGETS, calibrate_threshold  # noqa: E402
from src.scoring import DEMO_DETECTORS, load_scores, split_scores  # noqa: E402

SPLITS = ["test_in", "test_ood_both"]
# Per split: on test_ood_both TF-IDF flags ~99% of everything, so pairing it says
# nothing about disagreement there; embeddings still make real decisions.
SCATTER_PAIRS = {"test_in": ("perplexity-qwen", "tfidf-lr"), "test_ood_both": ("perplexity-qwen", "embedding-lr")}
RESULTS_DIR = ARTIFACTS_DIR / "results"
N_EXAMPLES = 10


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else np.nan


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar: among texts flagged by exactly one detector, is it a coin flip?"""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a": only_a, "only_b": only_b, "mcnemar_p": float(p)}


def flags(scores: pd.DataFrame, fpr: float) -> tuple[dict, dict]:
    """Val-calibrated thresholds, and the flag of every text in `scores` (all splits)."""
    val = split_scores(scores, "val")
    th = {d: calibrate_threshold(val.loc[val.label == 0, d], fpr) for d in DEMO_DETECTORS}
    return th, {d: scores[d].to_numpy() > th[d] for d in DEMO_DETECTORS}


def pairwise(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fpr in FPR_TARGETS:
        th, _ = flags(scores, fpr)
        for split in SPLITS:
            d = split_scores(scores, split)
            f = {k: d[k].to_numpy() > th[k] for k in DEMO_DETECTORS}
            y = d["label"].to_numpy()
            for a, b in itertools.combinations(DEMO_DETECTORS, 2):
                row = {"target_fpr": fpr, "split": split, "a": a, "b": b,
                       "flagged_a": int(f[a].sum()), "flagged_b": int(f[b].sum())}
                for name, mask in (("ai", y == 1), ("human", y == 0), ("all", np.ones_like(y, bool))):
                    row[f"jaccard_{name}"] = jaccard(f[a][mask], f[b][mask])
                    row[f"n_{name}"] = int(mask.sum())
                row.update(mcnemar(f[a], f[b]))
                rows.append(row)
    return pd.DataFrame(rows)


def breakdown(scores: pd.DataFrame, fpr: float = 0.05) -> pd.DataFrame:
    """Share of texts on which the three detectors do NOT all agree."""
    th, _ = flags(scores, fpr)
    d = pd.concat([split_scores(scores, s).assign(split=s) for s in SPLITS]).drop_duplicates("id")
    f = np.column_stack([d[k].to_numpy() > th[k] for k in DEMO_DETECTORS])
    d = d.assign(disagree=f.any(1) & ~f.all(1))
    d["length"] = pd.cut(d["n_tokens_full"], [0, 128, 256, 512, 10**6],
                         labels=["<128", "128-255", "256-511", "512+"], right=False)
    out = []
    for col in ("domain", "model", "length"):
        g = d.groupby(col, observed=True)["disagree"].agg(["mean", "size"]).reset_index()
        out.append(g.rename(columns={col: "value", "mean": "disagreement_rate", "size": "n"}).assign(by=col))
    return pd.concat(out)[["by", "value", "n", "disagreement_rate"]]


def examples(scores: pd.DataFrame, fpr: float = 0.05) -> pd.DataFrame:
    """Disagreements on test_ood_both, mixed over who-flags-what, both labels."""
    th, _ = flags(scores, fpr)
    d = split_scores(scores, "test_ood_both")
    f = pd.DataFrame({k: d[k] > th[k] for k in DEMO_DETECTORS})
    d = d[(f.any(axis=1) & ~f.all(axis=1)).to_numpy()].copy()
    d["pattern"] = f.loc[d.index].apply(lambda r: "".join("F" if v else "." for v in r), axis=1)
    # One per (pattern, label) first, so the table shows the variety, then fill up.
    picked = d.groupby(["pattern", "label"]).sample(1, random_state=SEED)
    if len(picked) < N_EXAMPLES:
        picked = pd.concat([picked, d.drop(picked.index).sample(N_EXAMPLES - len(picked), random_state=SEED)])
    picked = picked.head(N_EXAMPLES)
    out = picked[["id", "label", "model", "domain"]].copy()
    out["snippet"] = picked["text"].str.slice(0, 220) + "..."
    for k in DEMO_DETECTORS:
        out[f"{k}_score"] = picked[k].round(4)
        out[f"{k}_flag"] = picked[k] > th[k]
    return out


def rank_normalize(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Share of `reference` scores strictly below each value (0..1)."""
    ref = np.sort(reference)
    return np.searchsorted(ref, values, side="left") / len(ref)


def scatter(scores: pd.DataFrame, split: str, fpr: float = 0.05) -> None:
    import matplotlib.pyplot as plt

    plotting.setup()
    a, b = SCATTER_PAIRS[split]
    name_a, name_b = (plotting.DETECTOR_STYLE[k][0] for k in (a, b))
    th, _ = flags(scores, fpr)
    d = split_scores(scores, split)
    # Rank within this split (spec: "rank-normalized"): spreads the cloud out.
    # Threshold lines are the val-calibrated thresholds mapped onto the same ranks.
    d = d.assign(xa=rank_normalize(d[a].to_numpy(), d[a].to_numpy()),
                 xb=rank_normalize(d[b].to_numpy(), d[b].to_numpy()))
    cut_a = float(rank_normalize(np.array([th[a]]), d[a].to_numpy())[0])
    cut_b = float(rank_normalize(np.array([th[b]]), d[b].to_numpy())[0])
    fa, fb = d[a] > th[a], d[b] > th[b]  # counts on the FULL split, before sampling
    counts = {"both": (fa & fb), f"only {name_a.split(' (')[0]}": (fa & ~fb), f"only {name_b}": (~fa & fb),
              "neither": (~fa & ~fb)}
    sample = d.groupby("label").sample(n=min(800, d.label.value_counts().min()), random_state=SEED)

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.subplots_adjust(right=0.60)
    styles = {0: ("Human", plotting.MUTED, "o"), 1: ("AI", "#6250d6", "^")}
    for lab, (name, color, marker) in styles.items():
        s = sample[sample.label == lab]
        ax.scatter(s.xa, s.xb, s=22, c=color, marker=marker, alpha=0.55, linewidths=0, label=name, zorder=2)
    ax.axvline(cut_a, color=plotting.INK_2, lw=1.5, zorder=1)
    ax.axhline(cut_b, color=plotting.INK_2, lw=1.5, zorder=1)
    ax.set_xlim(0, 1.01), ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    ax.xaxis.set_major_formatter(pct), ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel(f"{name_a}: score rank in this split")
    ax.set_ylabel(f"{name_b}: score rank in this split")
    ax.grid(True, axis="both")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, markerscale=1.8)
    y = d["label"].to_numpy()
    lines = [f"Flagged at {fpr:.0%} FPR ({split}, all {len(d)} texts)", "", f"{'':22s}{'AI':>7s}{'Human':>8s}"]
    lines += [f"{k:22s}{int(m[y == 1].sum()):7d}{int(m[y == 0].sum()):8d}" for k, m in counts.items()]
    fig.text(0.63, 0.5, "\n".join(lines), family="monospace", fontsize=12.5, va="center", color=plotting.INK)
    fig.text(0.125, 0.0, f"Points: balanced sample of {len(sample)}. Lines: each detector's {fpr:.0%}-FPR "
             "threshold, calibrated on val humans. Same false-positive budget, different texts flagged.",
             fontsize=11.5, color=plotting.INK_2)
    plotting.save(fig, f"03_scatter_{a}_vs_{b}_{split}")


def main() -> None:
    scores = load_scores(256)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pw = pairwise(scores)
    pw.to_csv(RESULTS_DIR / "03_pairwise.csv", index=False)
    br = breakdown(scores)
    br.to_csv(RESULTS_DIR / "03_breakdown.csv", index=False)
    ex = examples(scores)
    ex.to_csv(RESULTS_DIR / "03_examples.csv", index=False)
    for split in SPLITS:
        scatter(scores, split)

    pd.set_option("display.width", 220, "display.max_colwidth", 80)
    cols = ["target_fpr", "split", "a", "b", "flagged_a", "flagged_b", "jaccard_ai", "jaccard_human",
            "jaccard_all", "only_a", "only_b", "mcnemar_p"]
    print("== Pairwise agreement of flagged sets ==")
    print(pw[cols].round(3).to_string(index=False))
    print("\n== 3-way disagreement rate at 5% FPR (test_in + test_ood_both) ==")
    print(br.round(3).to_string(index=False))
    print(f"\n== {len(ex)} disagreement examples (test_ood_both; F = flagged, order {DEMO_DETECTORS}) ==")
    print(ex[["label", "model", "domain"] + [f"{k}_flag" for k in DEMO_DETECTORS] + ["snippet"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
