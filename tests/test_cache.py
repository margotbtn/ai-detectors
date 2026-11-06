import numpy as np
import pytest

from src.cache import FeatureCache


def fake_features(texts):
    return np.array([[len(t), t.count("a")] for t in texts], dtype=float)


def test_roundtrip_order_and_dedup(tmp_path):
    cache = FeatureCache("x", {"a": 1}, root=tmp_path)
    texts = ["banana", "apple", "banana", "kiwi"]
    X = cache.get_or_compute(texts, fake_features, chunk_size=2)
    np.testing.assert_array_equal(X, fake_features(texts))
    assert len(cache) == 3  # "banana" computed once


def test_reload_computes_nothing(tmp_path):
    FeatureCache("x", {"a": 1}, root=tmp_path).get_or_compute(["a", "b"], fake_features)
    again = FeatureCache("x", {"a": 1}, root=tmp_path)
    X = again.get_or_compute(["b", "a"], lambda t: pytest.fail("should not recompute"))
    np.testing.assert_array_equal(X, fake_features(["b", "a"]))


def test_resume_after_interruption(tmp_path):
    calls = []

    def flaky(texts):
        calls.append(list(texts))
        if len(calls) == 2:
            raise KeyboardInterrupt
        return fake_features(texts)

    texts = [f"t{i}" for i in range(6)]
    with pytest.raises(KeyboardInterrupt):
        FeatureCache("x", {"a": 1}, root=tmp_path).get_or_compute(texts, flaky, chunk_size=2)
    resumed = FeatureCache("x", {"a": 1}, root=tmp_path)
    assert len(resumed) == 2  # first chunk survived, no half-written second chunk
    seen = []
    X = resumed.get_or_compute(texts, lambda t: seen.extend(t) or fake_features(t), chunk_size=2)
    assert seen == texts[2:]
    np.testing.assert_array_equal(X, fake_features(texts))


def test_config_change_is_a_different_cache(tmp_path):
    FeatureCache("x", {"a": 1}, root=tmp_path).get_or_compute(["a"], fake_features)
    assert len(FeatureCache("x", {"a": 2}, root=tmp_path)) == 0
