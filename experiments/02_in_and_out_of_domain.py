"""Experiment 2: in-domain vs out-of-distribution.

Train on `train` (news, chatgpt), calibrate thresholds on `val` human texts
(FPR 5% headline, 1% alongside), apply those FIXED thresholds to the four test
splits. Reports AUROC, TPR and realized FPR, all with bootstrap 95% CIs that
include threshold uncertainty (val humans are resampled too).

Usage: python experiments/02_in_and_out_of_domain.py [--tokens 256|128]

Writes:
  artifacts/results/02_metrics_<tokens>.csv
  artifacts/calibration/calibration_<tokens>.json   (read by the app)
  artifacts/figures/02_tpr_fpr_by_split_<tokens>.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import plotting  # noqa: E402
from src.config import ARTIFACTS_DIR, CALIBRATION_DIR  # noqa: E402
from src.metrics import FPR_TARGETS, bootstrap_auroc, bootstrap_calibrated, roc_points  # noqa: E402
from src.scoring import DEMO_DETECTORS, detectors_for, load_scores, split_scores  # noqa: E402

TEST_SPLITS = ["test_in", "test_ood_gen", "test_ood_domain", "test_ood_both"]
N_BOOT = 200
RESULTS_DIR = ARTIFACTS_DIR / "results"


def evaluate(scores: pd.DataFrame, detectors: list[str]) -> pd.DataFrame:
    val = split_scores(scores, "val")
    rows = []
    for det in detectors:
        val_h = val.loc[val["label"] == 0, det].to_numpy()
        for split in TEST_SPLITS:
            d = split_scores(scores, split)
            y, s = d["label"].to_numpy(), d[det].to_numpy()
            a = bootstrap_auroc(y, s, n_boot=N_BOOT)
            for fpr in FPR_TARGETS:
                rows.append({"detector": det, "split": split, **a,
                             **bootstrap_calibrated(val_h, y, s, fpr, n_boot=N_BOOT)})
    return pd.DataFrame(rows)


def calibration_for_app(scores: pd.DataFrame, res: pd.DataFrame, tokens: int) -> dict:
    """Everything the app needs, nothing it doesn't: thresholds, ROC points, rates,
    and the sorted val human scores (so the app can say "scores higher than X% of
    human news texts", a scale shared by all three detectors)."""
    out = {"tokens": tokens, "n_boot": N_BOOT, "detectors": {}}
    val = split_scores(scores, "val")
    for det in DEMO_DETECTORS:
        entry = {"thresholds": {}, "roc": {}, "rates": {},
                 "val_human_scores": np.sort(val.loc[val.label == 0, det].to_numpy()).tolist()}
        for fpr in FPR_TARGETS:
            r = res[(res.detector == det) & (res.target_fpr == fpr) & (res.split == "test_in")].iloc[0]
            entry["thresholds"][str(fpr)] = {"value": r.threshold, "lo": r.threshold_lo, "hi": r.threshold_hi,
                                             "n_val_human": int(r.n_val_human)}
        for split in ("test_in", "test_ood_both"):
            d = split_scores(scores, split)
            entry["roc"][split] = roc_points(d["label"], d[det])
            entry["rates"][split] = {
                str(fpr): {k: float(v) for k, v in res[(res.detector == det) & (res.split == split)
                                                       & (res.target_fpr == fpr)].iloc[0][
                    ["tpr", "tpr_lo", "tpr_hi", "fpr", "fpr_lo", "fpr_hi", "n_ai", "n_human"]].items()}
                for fpr in FPR_TARGETS}
        out["detectors"][det] = entry
    return out


def figure(res: pd.DataFrame, tokens: int, fpr: float = 0.05) -> None:
    """Two stacked panels on one shared x: TPR (top) and realized FPR (bottom).

    Both are shares of texts flagged, so they share units; stacking keeps each
    panel full-width so labels never collide on a projector.
    """
    import matplotlib.pyplot as plt

    plotting.setup()
    r = res[(res.target_fpr == fpr) & res.detector.isin(DEMO_DETECTORS)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True, gridspec_kw={"hspace": 0.32})
    x = np.arange(len(TEST_SPLITS))
    width = 0.22
    pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    for ax, metric, title in ((axes[0], "tpr", "AI texts caught (TPR)"),
                              (axes[1], "fpr", "Human texts wrongly flagged (FPR)")):
        # y-limit first: value labels are offset relative to it.
        ax.set_ylim(0, 1.12 if metric == "tpr" else max(0.2, float(r["fpr_hi"].max()) * 1.18))
        for k, det in enumerate(DEMO_DETECTORS):
            label, color = plotting.DETECTOR_STYLE[det]
            d = r[r.detector == det].set_index("split").loc[TEST_SPLITS]
            xs = x + (k - 1) * (width + 0.03)
            ax.bar(xs, d[metric], width=width, color=color, label=label, zorder=2)
            ax.vlines(xs, d[f"{metric}_lo"], d[f"{metric}_hi"], color=plotting.INK_2, lw=1.5, zorder=3)
            for xi, v, hi in zip(xs, d[metric], d[f"{metric}_hi"]):
                ax.text(xi, hi + 0.02 * ax.get_ylim()[1], f"{v:.0%}", ha="center", va="bottom",
                        fontsize=13, color=plotting.INK)
        ax.set_title(title, loc="left", color=plotting.INK, pad=14)
        ax.yaxis.set_major_formatter(pct)
        ax.tick_params(axis="x", length=0)
    axes[1].axhline(fpr, color=plotting.INK_2, lw=1.5, zorder=1)
    axes[1].text(len(TEST_SPLITS) - 0.55, fpr, f" target {fpr:.0%}", va="bottom", ha="left",
                 color=plotting.INK_2, fontsize=13)
    axes[1].set_xticks(x, [plotting.SPLIT_LABELS[s] for s in TEST_SPLITS])
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=3, handlelength=1.2)
    fig.text(0.125, 0.035, f"Threshold set once on news validation humans (n={int(r.iloc[0].n_val_human)}) at "
             f"{fpr:.0%} FPR, then applied unchanged. {tokens} tokens. Whiskers: 95% bootstrap CI, "
             "including threshold uncertainty.", fontsize=12, color=plotting.INK_2)
    plotting.save(fig, f"02_tpr_fpr_by_split_{tokens}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256, choices=[256, 128])
    tokens = ap.parse_args().tokens

    scores = load_scores(tokens)
    res = evaluate(scores, detectors_for(tokens))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / f"02_metrics_{tokens}.csv", index=False)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    (CALIBRATION_DIR / f"calibration_{tokens}.json").write_text(
        json.dumps(calibration_for_app(scores, res, tokens), indent=1))
    figure(res, tokens)

    pd.set_option("display.width", 220)
    fmt = lambda v, lo, hi, pct=True: (f"{v:.1%} [{lo:.1%}, {hi:.1%}]" if pct else f"{v:.3f} [{lo:.3f}, {hi:.3f}]")
    for fpr in FPR_TARGETS:
        r = res[res.target_fpr == fpr]
        print(f"\n== {tokens} tokens | threshold calibrated on val humans at FPR = {fpr:.0%} ==")
        tab = r.assign(
            AUROC=[fmt(a, b, c, False) for a, b, c in zip(r.auroc, r.auroc_lo, r.auroc_hi)],
            TPR=[fmt(a, b, c) for a, b, c in zip(r.tpr, r.tpr_lo, r.tpr_hi)],
            FPR_realized=[fmt(a, b, c) for a, b, c in zip(r.fpr, r.fpr_lo, r.fpr_hi)],
            n=[f"{a}+{h}" for a, h in zip(r.n_ai, r.n_human)],
        )[["detector", "split", "n", "AUROC", "TPR", "FPR_realized"]]
        print(tab.to_string(index=False))
    th = res[res.split == "test_in"][["detector", "target_fpr", "threshold", "threshold_lo", "threshold_hi"]]
    print("\n== Calibrated thresholds (val humans, bootstrap 95% CI) ==")
    print(th.to_string(index=False))


if __name__ == "__main__":
    main()
