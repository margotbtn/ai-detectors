import numpy as np
import pytest

from src.metrics import (auroc, bootstrap_auroc, bootstrap_calibrated, calibrate_threshold, ppv,
                         rates_at, roc_points, tpr_at_fpr)


def test_ppv_spec_sanity_checks():
    # Spec section 5: tpr=0.99, fpr=0.01 -> PPV ~0.84 at 5% prevalence, 0.50 at 1%.
    assert ppv(0.99, 0.01, 0.05) == pytest.approx(0.839, abs=1e-3)
    assert ppv(0.99, 0.01, 0.01) == pytest.approx(0.50, abs=1e-9)


def test_ppv_vectorized_and_edge_cases():
    out = ppv(0.9, 0.05, np.array([0.0, 0.5, 1.0]))
    assert out[0] == 0.0 and out[2] == 1.0
    assert np.isnan(ppv(0.0, 0.0, 0.5))


@pytest.mark.parametrize("fpr", [0.01, 0.05, 0.2])
@pytest.mark.parametrize("n", [100, 516, 1000])
def test_calibration_realized_fpr_is_largest_achievable_below_target(fpr, n):
    h = np.random.default_rng(0).normal(size=n)  # continuous: no ties
    t = calibrate_threshold(h, fpr)
    realized = (h > t).mean()
    assert realized <= fpr + 1e-12
    assert realized > fpr - 1 / n - 1e-12


def test_calibration_on_known_values():
    h = np.arange(1, 101)  # 1..100
    assert calibrate_threshold(h, 0.05) == 95  # flags 96..100 = 5 texts = 5%
    assert calibrate_threshold(h, 0.01) == 99  # flags 100 only


def test_rates_at_and_tpr_at_fpr():
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([np.arange(100), np.arange(50, 150)])  # humans 0..99, AI 50..149
    assert rates_at(y, s, 94.5) == (0.55, 0.05)
    # Oracle: threshold at the 95th human score (94) -> AI > 94 are 95..149 = 55
    assert tpr_at_fpr(y, s, 0.05) == pytest.approx(0.55)


def test_auroc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, [0, 1, 2, 3]) == 1.0
    assert auroc(y, [3, 2, 1, 0]) == 0.0


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 300)
    s = rng.normal(size=600) + y
    a = bootstrap_auroc(y, s, n_boot=100)
    assert a["auroc_lo"] <= a["auroc"] <= a["auroc_hi"]
    assert a == bootstrap_auroc(y, s, n_boot=100)
    val_h = rng.normal(size=500)
    b = bootstrap_calibrated(val_h, y, s, 0.05, n_boot=100)
    for k in ("threshold", "tpr", "fpr"):
        assert b[f"{k}_lo"] <= b[k] <= b[f"{k}_hi"]
    assert b["threshold_hi"] > b["threshold_lo"]  # the threshold itself is uncertain


def test_roc_points_monotone_and_bounded():
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1], 500)
    r = roc_points(y, rng.normal(size=1000) + y, max_points=50)
    assert len(r["fpr"]) <= 50
    assert r["fpr"][0] == 0 and r["fpr"][-1] == 1 and r["tpr"][-1] == 1
    assert np.all(np.diff(r["fpr"]) >= 0) and np.all(np.diff(r["tpr"]) >= 0)
