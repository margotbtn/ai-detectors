"""Disk cache for expensive per-text features, keyed by config hash + text hash.

Layout: artifacts/features/<detector-name>-<config hash>/
            config.json         the config the hash was computed from
            chunk_00000.npz     keys (text sha1 hex) + X (features), one per chunk

Why per-text keys: the same text appears in several splits (shared human test
texts) and in several runs (attacked texts at p=0 are the originals). Any text
already scored under the same config is reused, whatever list it arrives in.

Why chunks: feature jobs run for hours. Each finished chunk is on disk at
once, so an interrupted run resumes where it stopped.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS_DIR

FEATURES_DIR = ARTIFACTS_DIR / "features"


def config_hash(config: dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class FeatureCache:
    def __init__(self, name: str, config: dict, root: Path = FEATURES_DIR):
        self.dir = root / f"{name}-{config_hash(config)}"
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.dir / "config.json"
        if not cfg_path.exists():
            cfg_path.write_text(json.dumps(config, indent=2, sort_keys=True))
        self._store: dict[str, np.ndarray] = {}
        for f in sorted(self.dir.glob("chunk_*.npz")):
            with np.load(f, allow_pickle=False) as z:
                self._store.update(zip(z["keys"].tolist(), z["X"]))

    def __len__(self) -> int:
        return len(self._store)

    def missing(self, texts) -> list[str]:
        seen, out = set(), []
        for t in texts:
            k = text_key(t)
            if k not in self._store and k not in seen:
                seen.add(k)
                out.append(t)
        return out

    def get_or_compute(self, texts, compute_fn, chunk_size: int = 200, label: str = "") -> np.ndarray:
        """Features for `texts` (in order), computing only the uncached ones."""
        texts = list(texts)
        todo = self.missing(texts)
        if todo:
            t0, done = time.perf_counter(), 0
            n_chunks = len(list(self.dir.glob("chunk_*.npz")))
            for start in range(0, len(todo), chunk_size):
                batch = todo[start : start + chunk_size]
                X = np.asarray(compute_fn(batch))
                keys = np.array([text_key(t) for t in batch])
                tmp = self.dir / f"chunk_{n_chunks:05d}.tmp.npz"
                np.savez(tmp, keys=keys, X=X)
                tmp.rename(self.dir / f"chunk_{n_chunks:05d}.npz")  # atomic: no half-written chunks
                n_chunks += 1
                self._store.update(zip(keys.tolist(), X))
                done += len(batch)
                el = time.perf_counter() - t0
                eta = el / done * (len(todo) - done)
                print(f"  [{label or self.dir.name}] {done}/{len(todo)} new texts "
                      f"| {el / 60:.1f} min elapsed | ETA {eta / 60:.1f} min", flush=True)
        return np.stack([self._store[text_key(t)] for t in texts])
