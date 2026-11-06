"""Speed benchmark: run FIRST, before any heavy job (spec section 9.2).

Times each scorer on 100 real news texts truncated to 256 tokens, across
thread counts and batch sizes, then projects the cost of the full workload:
  - corpus at 256 tokens and at 128 tokens (~14.2k texts each)
  - homoglyph sweep: 500 AI texts x p in {5, 10, 15, 20, 100}%
Also times ONE 256-token text at 2 threads, a stand-in for the free HF CPU Space
(local cores are faster than a Space's shared vCPUs: treat it as a lower bound).

Writes artifacts/logs/01_benchmark_speed.json.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.attacks import homoglyph_attack  # noqa: E402
from src.config import ARTIFACTS_DIR, SEED  # noqa: E402
from src.data import DOMAIN_A, load_clean, sample_rows  # noqa: E402
from src.text import truncate  # noqa: E402
from src.detectors import EmbeddingLRDetector, PerplexityDetector, TfidfLRDetector, set_threads  # noqa: E402

N_CORPUS = 7120 + 7064  # news + abstracts, human + 3 generators each
N_ATTACK = 500
ATTACK_PS = [0.05, 0.10, 0.15, 0.20, 1.00]
THREADS = [8, 16]


def timed(fn, *args):
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def bench(label, fn, texts, warmup=8):
    fn(texts[:warmup])
    secs = timed(fn, texts)
    print(f"  {label:42s} {secs:7.2f} s / {len(texts)} texts = {1000 * secs / len(texts):7.1f} ms/text", flush=True)
    return secs / len(texts)


def main():
    df = sample_rows(load_clean(domains=(DOMAIN_A,)), 100, seed=SEED)
    print("sample:", df["model"].value_counts().to_dict())

    qwen = PerplexityDetector("qwen")
    gpt2 = PerplexityDetector("gpt2")
    emb = EmbeddingLRDetector()
    tfidf = TfidfLRDetector()

    t256 = truncate(df["generation"], qwen.tokenizer, 256)
    t128 = truncate(df["generation"], qwen.tokenizer, 128)
    attacked = {p: [homoglyph_attack(t, p, seed=SEED) for t in t256] for p in ATTACK_PS}

    tokens = {"qwen": {}, "gpt2": {}}
    for key, det in (("qwen", qwen), ("gpt2", gpt2)):
        tokens[key]["256"] = float(det.token_counts(t256).mean())
        for p in ATTACK_PS:
            tokens[key][f"p={p:.2f}"] = float(det.token_counts(attacked[p]).mean())
    print("\nmean tokens per text (truncated at 256 Qwen tokens):")
    print(json.dumps(tokens, indent=2))

    res = {"cpu_count": os.cpu_count(), "tokens": tokens, "per_text_s": {}}
    print("\n== Throughput on 100 texts (256 tokens) ==")
    for th in THREADS:
        set_threads(th)
        for key, det, batches in (("qwen", qwen, [1, 8, 16]), ("gpt2", gpt2, [1, 8, 16])):
            for bs in batches:
                det.batch_size = bs
                res["per_text_s"][f"{key}|t{th}|b{bs}"] = bench(f"{key} threads={th} batch={bs}", det.features, t256)
        res["per_text_s"][f"minilm|t{th}|b32"] = bench(f"minilm threads={th} batch=32", emb.features, t256)

    best = {}
    for key in ("qwen", "gpt2", "minilm"):
        cfg = min((k for k in res["per_text_s"] if k.startswith(key + "|")), key=res["per_text_s"].get)
        best[key] = cfg
    print("\nbest configs:", best)
    res["best"] = best

    print("\n== At best config: 128 tokens and fully attacked (p=100%) ==")
    for key, det in (("qwen", qwen), ("gpt2", gpt2), ("minilm", emb)):
        _, th, bs = best[key].split("|")
        set_threads(int(th[1:]))
        if key != "minilm":
            det.batch_size = int(bs[1:])
        f = det.features
        res["per_text_s"][f"{key}|128"] = bench(f"{key} 128 tokens", f, t128)
        res["per_text_s"][f"{key}|p100"] = bench(f"{key} attacked p=100%", f, attacked[1.0])

    set_threads(max(THREADS))
    labels = (df["label"].to_numpy())
    t_fit = timed(tfidf.fit, t256, labels)
    t_score = timed(tfidf.score, t256)
    print(f"  tfidf fit 100 texts {t_fit:.2f} s, score 100 texts {t_score:.3f} s")
    res["tfidf_fit_100_s"], res["tfidf_score_100_s"] = t_fit, t_score

    # --- projection --------------------------------------------------------
    print("\n== Projected one-off feature cost (best config) ==")
    proj = {}
    for key in ("qwen", "gpt2", "minilm"):
        s256 = res["per_text_s"][best[key]]
        s128 = res["per_text_s"][f"{key}|128"]
        s100 = res["per_text_s"][f"{key}|p100"]
        tok = tokens.get(key, tokens["qwen"])  # minilm: scale by Qwen counts, rough
        # Partial attacks: interpolate per-text cost by token count between p=0 and p=100.
        partial = sum(
            s256 + (s100 - s256) * (tok[f"p={p:.2f}"] - tok["256"]) / (tok["p=1.00"] - tok["256"])
            for p in ATTACK_PS[:-1]
        )
        attack_s = N_ATTACK * (partial + s100)
        proj[key] = {"corpus_256_min": N_CORPUS * s256 / 60, "corpus_128_min": N_CORPUS * s128 / 60,
                     "attack_sweep_min": attack_s / 60}
        proj[key]["total_min"] = sum(proj[key].values())
        print(f"  {key:7s} " + "  ".join(f"{k}={v:6.1f}" for k, v in proj[key].items()))
    total = sum(p["total_min"] for p in proj.values())
    print(f"  ALL     total={total:.1f} min (TF-IDF fits are seconds: not counted)")
    res["projection_min"] = proj

    # --- app latency at 2 threads (HF free Space stand-in) -----------------
    print("\n== App latency: ONE 256-token text, 2 threads (median of 5) ==")
    set_threads(2)
    emb.fit(t256, labels)
    one = [t256[int(np.argmax([len(t) for t in t256]))]]
    app = {}
    for key, det in (("qwen", qwen), ("gpt2", gpt2), ("minilm+lr", emb), ("tfidf+lr", tfidf)):
        det.score(one)
        app[key] = float(np.median([timed(det.score, one) for _ in range(5)]))
        print(f"  {key:10s} {1000 * app[key]:7.0f} ms")
    app["all_3_demo_detectors_s"] = app["qwen"] + app["minilm+lr"] + app["tfidf+lr"]
    app["all_4_incl_gpt2_s"] = app["all_3_demo_detectors_s"] + app["gpt2"]
    print(f"  Score tab (qwen + tfidf + embeddings): {app['all_3_demo_detectors_s']:.2f} s"
          f"   | Attack tab (original + attacked): ~{2 * app['all_3_demo_detectors_s']:.2f} s")
    res["app_latency_2threads_s"] = app

    out = ARTIFACTS_DIR / "logs" / "01_benchmark_speed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
