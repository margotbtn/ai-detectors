"""AUROC, threshold calibration, TPR/FPR at a fixed threshold, PPV, bootstrap CIs.

Convention everywhere: higher score = more likely AI; a text is FLAGGED when
score > threshold.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import SEED

FPR_TARGETS = (0.05, 0.01)  # 5% is the headline (project-state decision 4)


def auroc(y, s) -> float:
    return float(roc_auc_score(y, s))


def calibrate_threshold(human_scores, fpr: float) -> float:
    """Threshold t such that at most `fpr` of these human texts score > t.

    t is the ceil((1 - fpr) * n)-th smallest human score, so the realized FPR
    on the calibration set is the largest achievable value <= fpr. With
    n = 516 val humans, fpr=5% is set by the top ~26 scores, fpr=1% by ~5:
    hence the bootstrap below.
    """
    h = np.sort(np.asarray(human_scores, dtype=float))
    if not 0 < fpr < 1:
        raise ValueError(f"fpr must be in (0, 1), got {fpr}")
    k = int(np.ceil((1 - fpr) * len(h))) - 1
    return float(h[min(max(k, 0), len(h) - 1)])


def rates_at(y, s, threshold: float) -> tuple[float, float]:
    """(TPR, FPR) when flagging score > threshold."""
    y, flagged = np.asarray(y), np.asarray(s) > threshold
    return float(flagged[y == 1].mean()), float(flagged[y == 0].mean())


def tpr_at_fpr(y, s, fpr: float) -> float:
    """Oracle TPR@FPR: threshold calibrated on this same set's human texts."""
    y, s = np.asarray(y), np.asarray(s)
    return rates_at(y, s, calibrate_threshold(s[y == 0], fpr))[0]


def ppv(tpr, fpr, prevalence):
    """Share of flagged texts that are really AI, at a given AI prevalence."""
    tpr, fpr, prevalence = np.asarray(tpr), np.asarray(fpr), np.asarray(prevalence)
    num = tpr * prevalence
    den = num + fpr * (1 - prevalence)
    return np.divide(num, den, out=np.full(np.broadcast(num, den).shape, np.nan), where=den > 0)


def roc_points(y, s, max_points: int = 200) -> dict:
    """Down-sampled ROC curve (for the app's threshold/base-rate tab)."""
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).round().astype(int))
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}


def _stratified(rng, y):
    """Bootstrap indices that resample each class separately (fixed class counts)."""
    return np.concatenate([rng.choice(np.flatnonzero(y == c), size=(y == c).sum(), replace=True)
                           for c in np.unique(y)])


def ci(values, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.nanquantile(values, a)), float(np.nanquantile(values, 1 - a))


def bootstrap_auroc(y, s, n_boot: int = 200, seed: int = SEED) -> dict:
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = [auroc(y[i], s[i]) for i in (_stratified(rng, y) for _ in range(n_boot))]
    lo, hi = ci(vals)
    return {"auroc": auroc(y, s), "auroc_lo": lo, "auroc_hi": hi}


def bootstrap_calibrated(val_human, y, s, fpr: float, n_boot: int = 200, seed: int = SEED) -> dict:
    """Calibrate on val humans, apply to a test set; bootstrap BOTH.

    Each resample redraws the val human texts (so the threshold itself
    varies, project-state correction 4) and the test set, then recomputes the
    threshold, TPR and realized FPR. The CIs therefore include calibration
    noise, not just test-set noise.
    """
    val_human, y, s = np.asarray(val_human), np.asarray(y), np.asarray(s)
    t = calibrate_threshold(val_human, fpr)
    tpr, real_fpr = rates_at(y, s, t)
    rng = np.random.default_rng(seed)
    ts, tprs, fprs = [], [], []
    for _ in range(n_boot):
        tb = calibrate_threshold(rng.choice(val_human, size=len(val_human), replace=True), fpr)
        i = _stratified(rng, y)
        a, b = rates_at(y[i], s[i], tb)
        ts.append(tb), tprs.append(a), fprs.append(b)
    out = {"target_fpr": fpr, "threshold": t, "tpr": tpr, "fpr": real_fpr,
           "n_ai": int((y == 1).sum()), "n_human": int((y == 0).sum()), "n_val_human": len(val_human)}
    for name, vals in (("threshold", ts), ("tpr", tprs), ("fpr", fprs)):
        out[f"{name}_lo"], out[f"{name}_hi"] = ci(vals)
    return out
