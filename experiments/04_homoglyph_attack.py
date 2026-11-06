"""Experiment 4: a homoglyph attack flips the perplexity detector.

1. Validation against RAID: our attack at p=100% vs RAID's own `homoglyph`
   rows for the same documents (RAID replaces every eligible character).
2. Sweep: 500 chatgpt texts from test_in, p in {0, 5, 10, 15, 20, 100}%.
   Thresholds stay calibrated on UNATTACKED val humans (the attacker only
   touches AI text). TPR with bootstrap 95% CI per detector and p.
3. Mechanism: tokens per text and Qwen mean log-likelihood vs p.
4. Defense: NFKC + confusables map, applied to attacked texts and to the val
   humans used for calibration.

Needs `make features` and experiment 2 (fitted models + scores_256).
Writes artifacts/results/04_*.csv and artifacts/figures/04_homoglyph_attack.png.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import plotting  # noqa: E402
from src.attacks import _GREEK_CAPITALS, ATTACK_PS, HOMOGLYPHS, homoglyph_attack  # noqa: E402
from src.config import ARTIFACTS_DIR, MODELS, SEED  # noqa: E402
from src.data import DECODING, DOMAIN_A, GEN_TRAIN, REP_PENALTY, download_raid_shards, load_corpus  # noqa: E402
from src.features import attack_texts, load  # noqa: E402
from src.metrics import calibrate_threshold, ci  # noqa: E402
from src.scoring import DEMO_DETECTORS, MODELS_DIR, load_scores, split_scores  # noqa: E402

RESULTS_DIR = ARTIFACTS_DIR / "results"
DETECTORS = DEMO_DETECTORS + ["perplexity-gpt2"]
FPR = 0.05
N_BOOT = 200


# --- 1. validation against RAID ---------------------------------------------

def validate_against_raid(n: int = 500) -> dict:
    f = ds.field
    expr = ((f("domain") == DOMAIN_A) & (f("model") == GEN_TRAIN) & (f("decoding") == DECODING)
            & (f("repetition_penalty") == REP_PENALTY) & pc.is_in(f("attack"), pa.array(["none", "homoglyph"])))
    t = ds.dataset(download_raid_shards(), format="parquet").to_table(
        columns=["id", "adv_source_id", "attack", "generation"], filter=expr).to_pandas()
    clean = t[t.attack == "none"].set_index("id")["generation"]
    raid = t[t.attack == "homoglyph"].head(n)
    # RAID picks Greek or Cyrillic at random for capitals: fold Greek to ours before comparing.
    fold = str.maketrans({g: HOMOGLYPHS[latin] for latin, g in _GREEK_CAPITALS.items()})
    same_positions, identical, ours_extra, raid_extra = [], 0, 0, 0
    for r in raid.itertuples():
        orig = clean[r.adv_source_id]
        ours = homoglyph_attack(orig, 1.0)
        theirs = r.generation.translate(fold)
        a = {i for i, (x, y) in enumerate(zip(orig, ours)) if x != y}
        b = {i for i, (x, y) in enumerate(zip(orig, theirs)) if x != y}
        same_positions.append(len(a & b) / max(len(a | b), 1))
        identical += ours == theirs
        ours_extra += len(a - b)
        raid_extra += len(b - a)
    return {"n": len(raid), "identical_texts": identical / len(raid),
            "mean_jaccard_changed_positions": float(np.mean(same_positions)),
            "chars_only_we_change": ours_extra, "chars_only_raid_changes": raid_extra}


# --- 2-4. sweep ---------------------------------------------------------------

def score_texts(det: str, texts: list[str]) -> np.ndarray:
    if det.startswith("perplexity-"):
        return load(det.split("-", 1)[1], texts)
    if det == "tfidf-lr":
        return joblib.load(MODELS_DIR / "tfidf-lr_256.joblib").predict_proba(texts)[:, 1]
    if det == "embedding-lr":
        return joblib.load(MODELS_DIR / "embedding-lr_256.joblib").predict_proba(load("minilm", texts))[:, 1]
    raise ValueError(det)


def bootstrap_rate(flags: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    return ci([flags[rng.integers(0, len(flags), len(flags))].mean() for _ in range(N_BOOT)])


def sweep(corpus: pd.DataFrame) -> pd.DataFrame:
    att = attack_texts(corpus)
    scores = load_scores(256)
    val = split_scores(scores, "val")
    val_h = val[val.label == 0]
    defended_val = att[att.p.isna()]  # val humans, defense applied
    rows = []
    for det in DETECTORS:
        th_plain = calibrate_threshold(val_h[det], FPR)
        th_def = calibrate_threshold(score_texts(det, defended_val.text.tolist()), FPR)
        for defended, th in ((False, th_plain), (True, th_def)):
            for p in ATTACK_PS:
                sub = att[(att.p == p) & (att.defended == defended)]
                s = score_texts(det, sub.text.tolist())
                flags = s > th
                lo, hi = bootstrap_rate(flags)
                rows.append({"detector": det, "defended": defended, "p": p, "n": len(sub), "threshold": th,
                             "tpr": flags.mean(), "tpr_lo": lo, "tpr_hi": hi, "mean_score": s.mean()})
    return pd.DataFrame(rows)


def token_counts(corpus: pd.DataFrame) -> pd.DataFrame:
    from transformers import AutoTokenizer

    att = attack_texts(corpus)
    att = att[(att.defended == False) & att.p.notna()]  # noqa: E712
    rows = []
    for key in ("qwen", "gpt2"):
        repo, rev = MODELS[key]
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        for p, g in att.groupby("p"):
            n = [len(x) for x in tok(g.text.tolist(), add_special_tokens=False)["input_ids"]]
            rows.append({"tokenizer": key, "p": p, "mean_tokens": float(np.mean(n)), "n": len(n)})
    return pd.DataFrame(rows)


def _gapped_line(ax, x, y, color, label=None) -> None:
    ax.plot(x[:-1], y[:-1], color=color, lw=2, marker="o", ms=8, mec=plotting.SURFACE, mew=2, label=label)
    ax.plot(x[-2:], y[-2:], color=color, lw=1.5, ls=(0, (1, 2.5)))
    ax.plot(x[-1:], y[-1:], color=color, marker="o", ms=8, mec=plotting.SURFACE, mew=2, ls="none")


def figure(res: pd.DataFrame, tok: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plotting.setup()
    # 0-20% are evenly spaced; 100% sits after a gap, joined by a dotted segment
    # (nothing was measured in between).
    x = np.array([0, 1, 2, 3, 4, 5.6])
    xlabels = [f"{p:.0%}" for p in ATTACK_PS]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), gridspec_kw={"width_ratios": [1.6, 1, 1], "wspace": 0.35})

    ax = axes[0]
    for det in DEMO_DETECTORS:
        label, color = plotting.DETECTOR_STYLE[det]
        r = res[(res.detector == det) & ~res.defended].set_index("p").loc[ATTACK_PS]
        ax.fill_between(x[:-1], r.tpr_lo.iloc[:-1], r.tpr_hi.iloc[:-1], color=color, alpha=0.10, linewidth=0)
        ax.vlines(x[-1], r.tpr_lo.iloc[-1], r.tpr_hi.iloc[-1], color=color, lw=1.5, alpha=0.6)
        _gapped_line(ax, x, r.tpr.to_numpy(), color, label)
        ax.text(x[-1] + 0.12, r.tpr.iloc[-1], f"{r.tpr.iloc[-1]:.0%}", va="center", fontsize=13)
    ax.set_xticks(x, xlabels)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of eligible letters swapped for look-alikes")
    ax.set_title(f"AI texts still caught (TPR at {FPR:.0%} FPR)", loc="left", color=plotting.INK, pad=12)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3, handlelength=1.4)

    t = tok[tok.tokenizer == "qwen"].set_index("p").loc[ATTACK_PS]
    axes[1].bar(x, t.mean_tokens, width=0.55, color=plotting.DETECTOR_STYLE["perplexity-qwen"][1])
    for xi, v in zip(x, t.mean_tokens):
        axes[1].text(xi, v + 12, f"{v:.0f}", ha="center", fontsize=12.5)
    axes[1].set_xticks(x, xlabels)
    axes[1].set_title("Tokens per text (Qwen tokenizer)", loc="left", color=plotting.INK, pad=12)
    axes[1].set_xlabel("Letters swapped")

    q = res[(res.detector == "perplexity-qwen") & ~res.defended].set_index("p").loc[ATTACK_PS]
    _gapped_line(axes[2], x, q.mean_score.to_numpy(), plotting.DETECTOR_STYLE["perplexity-qwen"][1])
    axes[2].axhline(q.threshold.iloc[0], color=plotting.INK_2, lw=1.5)
    axes[2].text(x[-1], q.threshold.iloc[0], f"threshold\n({FPR:.0%} FPR)", ha="right", va="bottom",
                 fontsize=12, color=plotting.INK_2)
    axes[2].set_xticks(x, xlabels)
    axes[2].set_title("Mean log-likelihood per token", loc="left", color=plotting.INK, pad=12)
    axes[2].set_xlabel("Letters swapped")
    for ax in axes:
        ax.tick_params(axis="x", length=0)

    d = res[res.defended & res.detector.isin(DEMO_DETECTORS) & (res.p == 1.0)]
    note = ", ".join(f"{plotting.DETECTOR_STYLE[r.detector][0].split(' (')[0]} {r.tpr:.0%}" for r in d.itertuples())
    n = int(res.n.iloc[0])
    fig.text(0.125, -0.06, f"n = {n} chatgpt news texts, thresholds calibrated on unattacked val humans. Bands: 95% "
             f"bootstrap CI.\nDefense (Unicode NFKC + confusables map) at 100% swapped: {note}.",
             fontsize=12, color=plotting.INK_2)
    plotting.save(fig, "04_homoglyph_attack")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    v = validate_against_raid()
    pd.DataFrame([v]).to_csv(RESULTS_DIR / "04_raid_validation.csv", index=False)
    print("== Our attack (p=100%) vs RAID's homoglyph rows ==")
    for k, val in v.items():
        print(f"  {k}: {val:.3f}" if isinstance(val, float) else f"  {k}: {val}")

    corpus = load_corpus()
    res = sweep(corpus)
    res.to_csv(RESULTS_DIR / "04_sweep.csv", index=False)
    tok = token_counts(corpus)
    tok.to_csv(RESULTS_DIR / "04_tokens.csv", index=False)
    figure(res, tok)

    pd.set_option("display.width", 200)
    print(f"\n== TPR at {FPR:.0%} FPR vs share of letters swapped ==")
    piv = res.assign(cell=[f"{a:.0%} [{b:.0%},{c:.0%}]" for a, b, c in zip(res.tpr, res.tpr_lo, res.tpr_hi)])
    print(piv.pivot_table(index=["detector", "defended"], columns="p", values="cell", aggfunc="first").to_string())
    print("\n== Mean tokens per text ==")
    print(tok.pivot_table(index="tokenizer", columns="p", values="mean_tokens").round(0).to_string())


if __name__ == "__main__":
    main()
