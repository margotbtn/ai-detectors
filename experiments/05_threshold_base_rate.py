"""Experiment 5: threshold and base rate. Why "99% accurate" can still mean many wrong flags.

PPV (share of flagged texts that are really AI) as a function of AI prevalence,
using each detector's MEASURED TPR and realized FPR from experiment 2, at the
val-calibrated 5% and 1% FPR thresholds, in-domain and on test_ood_both.

One extra curve is NOT measured: a typical vendor claim (TPR 99%, FPR 0.1%),
labeled as such. 0.1% FPR cannot be calibrated or measured here: with 515 val
human texts it would be set by half a text (decision 2026-11-05).

Needs experiment 2's results (artifacts/results/02_metrics_256.csv).
Writes artifacts/results/05_ppv_table.csv and artifacts/figures/05_ppv_vs_prevalence.png.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import plotting  # noqa: E402
from src.config import ARTIFACTS_DIR  # noqa: E402
from src.metrics import FPR_TARGETS, ppv  # noqa: E402
from src.scoring import DEMO_DETECTORS  # noqa: E402

RESULTS_DIR = ARTIFACTS_DIR / "results"
SPLITS = {"test_in": "In-domain (news, chatgpt)", "test_ood_both": "New domain + new generators"}
ADVERTISED = {"tpr": 0.99, "fpr": 0.001, "label": "Advertised claim: TPR 99%, FPR 0.1% (not measured)"}
PREVALENCES = np.geomspace(0.001, 0.5, 200)
TABLE_PREVALENCES = [0.01, 0.05, 0.10, 0.50]


def ppv_band(r: pd.Series, prevalence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PPV curve and a conservative band from the TPR/FPR CI extremes."""
    return (ppv(r.tpr, r.fpr, prevalence), ppv(r.tpr_lo, r.fpr_hi, prevalence),
            ppv(r.tpr_hi, r.fpr_lo, prevalence))


def table(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in res.iterrows():
        rows.append({"detector": r.detector, "split": r.split, "target_fpr": r.target_fpr,
                     "tpr": r.tpr, "fpr_realized": r.fpr,
                     **{f"ppv@{p:.0%}": float(ppv(r.tpr, r.fpr, p)) for p in TABLE_PREVALENCES}})
    rows.append({"detector": "advertised", "split": "-", "target_fpr": ADVERTISED["fpr"],
                 "tpr": ADVERTISED["tpr"], "fpr_realized": ADVERTISED["fpr"],
                 **{f"ppv@{p:.0%}": float(ppv(ADVERTISED["tpr"], ADVERTISED["fpr"], p))
                    for p in TABLE_PREVALENCES}})
    return pd.DataFrame(rows)


def figure(res: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plotting.setup()
    fig, axes = plt.subplots(len(FPR_TARGETS), len(SPLITS), figsize=(15, 11), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.3, "wspace": 0.08})
    adv = ppv(ADVERTISED["tpr"], ADVERTISED["fpr"], PREVALENCES)
    for i, fpr in enumerate(FPR_TARGETS):
        for j, (split, split_label) in enumerate(SPLITS.items()):
            ax = axes[i, j]
            ax.plot(PREVALENCES, adv, color=plotting.MUTED, lw=2, zorder=2,
                    label=ADVERTISED["label"] if (i, j) == (0, 0) else None)
            for det in DEMO_DETECTORS:
                label, color = plotting.DETECTOR_STYLE[det]
                r = res[(res.detector == det) & (res.split == split) & (res.target_fpr == fpr)].iloc[0]
                mid, lo, hi = ppv_band(r, PREVALENCES)
                ax.fill_between(PREVALENCES, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=1)
                ax.plot(PREVALENCES, mid, color=color, lw=2, zorder=3, label=label if (i, j) == (0, 0) else None)
            ax.axvline(0.05, color=plotting.BASELINE, lw=1, zorder=0)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            ax.set_xticks([0.001, 0.01, 0.05, 0.1, 0.5], ["0.1%", "1%", "5%", "10%", "50%"])
            ax.minorticks_off()
            ax.grid(True, axis="both")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.set_title(f"{split_label} · threshold for {fpr:.0%} FPR", loc="left", fontsize=15,
                         color=plotting.INK)
    for ax in axes[-1]:
        ax.set_xlabel("Share of texts that are AI (prevalence)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Flags that are really AI (PPV)")
    fig.legend(loc="lower left", bbox_to_anchor=(0.08, 0.93), ncol=2, handlelength=1.6)
    fig.text(0.125, 0.015, "Curves: PPV from TPR and realized FPR measured in experiment 2. Bands: PPV at the "
             "TPR/FPR 95% CI extremes (conservative). Gray line: a vendor-style claim, not measured here.",
             fontsize=11.5, color=plotting.INK_2)
    plotting.save(fig, "05_ppv_vs_prevalence")


def main() -> None:
    res = pd.read_csv(RESULTS_DIR / "02_metrics_256.csv")
    res = res[res.detector.isin(DEMO_DETECTORS) & res.split.isin(SPLITS)]
    tab = table(res)
    tab.to_csv(RESULTS_DIR / "05_ppv_table.csv", index=False)
    figure(res)
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
