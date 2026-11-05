# AI-text detector demo: build spec for a coding agent

## 0. Context

I'm giving a ~40 min talk to data scientists (ML/LLM-fluent) on AI-generated text detectors: how they work, why advertised accuracy differs from independent benchmarks, and why detectors disagree with each other. This project is the **~10 minute live demo** inside that talk. It must be reliable, fast, and visually clear.

The demo trains a few small, cheap detectors from scratch and uses them to show four things:

1. **Out-of-distribution drop**: great in-domain, worse on another domain or generator.
2. **Disagreement**: detectors calibrated to the same false-positive rate still flag different texts.
3. **Attack**: a homoglyph attack flips a perplexity-based detector.
4. **Threshold and base rate**: why a "99% accurate" detector can still be wrong on many flags.

Audience language: **English** (all UI text, plots, comments, README in English).

## 1. Hard constraints

- **Hardware:** laptop CPU only, **no GPU, 32 GB RAM**. Everything must run locally.
- **Time budget for me to review/adapt:** a few hours. Prefer simple, readable code over clever code.
- **Reliability over ambition:** heavy computation is done once and **cached to disk**. The demo notebook and app only read caches, except lightweight live inference (one paragraph of text).
- **Reproducibility:** fixed seeds, pinned `requirements.txt`, all caches regenerable by one command.
- **No secrets, no paid APIs** required. Network may or may not be available in your environment: see section 3 for fallbacks.
- Do not use any commercial detector API in this project. (A separate pre-recorded table of commercial detectors is handled outside this repo.)

## 2. Deliverables

```
ai-detector-demo/
├── README.md                  # how to run everything, timings, known limitations
├── requirements.txt
├── Makefile                   # make data | make features | make train | make experiments | make app
├── src/
│   ├── data.py                # loading, splitting, subsampling
│   ├── detectors.py           # the 3 detectors, common interface
│   ├── attacks.py             # homoglyph attack (+ optional light perturbations)
│   ├── metrics.py             # AUROC, TPR@FPR, threshold calibration, PPV vs prevalence
│   └── cache.py               # disk cache helpers (parquet/npz), keyed by config hash
├── experiments/
│   ├── 01_benchmark_speed.py  # run FIRST: timing of each scorer on 100 texts
│   ├── 02_in_and_out_of_domain.py
│   ├── 03_disagreement.py
│   ├── 04_homoglyph_attack.py
│   └── 05_threshold_base_rate.py
├── notebooks/
│   └── demo.ipynb             # the 10-minute narrative, reads caches only
├── app/
│   └── app.py                 # Gradio app (see section 7)
└── artifacts/                 # cached scores, models, figures (gitignored except small examples)
```

## 3. Data

**Preferred source: a subsample of the RAID benchmark** (Dugan et al., ACL 2024, https://github.com/liamdugan/raid). It labels each text with generator model, domain, decoding strategy, repetition penalty, and adversarial attack, which makes the OOD experiment a simple split. Verify the current download method (HF dataset / `raid-bench` package / GitHub) and that the labeled training split is accessible. Only use rows with `attack == "none"` for training and OOD evaluation.

Subsample plan (adjust to what the data offers):

- 2 domains (e.g., news and one other, such as reviews or abstracts), 3 generators (a mix of closed and open models if available), decoding = default, no attack.
- ~1,000 AI texts per (domain, generator) cell, plus a matched number of human texts per domain. Target total: roughly 6,000-10,000 texts.
- **Truncate every text to 256 tokens** (also keep a 128-token variant if cheap): short-text behavior is part of the story.
- Splits: `train` (domain A, generator X), `val` (same distribution, for threshold calibration), `test_in`, `test_ood_domain` (domain B, generator X), `test_ood_gen` (domain A, other generators), `test_ood_both`.

**Fallback if RAID is not downloadable:** build a small corpus from any public human text dataset (two different genres), and generate AI counterparts with 2-3 open LLMs locally or via an API if the user supplies a key. Document the exact fallback taken in the README. Check dataset licenses before committing any text to the repo; if redistribution is unclear, commit only scores/features and a handful of short example snippets.

## 4. Detectors (common interface)

All implement `fit(train_texts, labels)` (no-op for zero-shot), `score(texts) -> np.ndarray` where **higher = more likely AI**, and a stable `name`.

1. **`PerplexityDetector` (zero-shot).** Score = **mean token log-likelihood** under a small causal LM (equivalently, negative perplexity). Sign matters: LLM text is typically *low*-perplexity, so a *higher* mean log-likelihood means more likely AI, consistent with the shared interface. Do not use negative mean log-likelihood (i.e. cross-entropy) here — that is high for surprising, human-ish text and would invert the detector. Start with `gpt2` (124M). Add an option for a small modern model (e.g., `Qwen/Qwen2.5-0.5B`) and compare: if GPT-2 is nearly useless on the modern generators, say so in the README and prefer the modern scorer for the demo (a strawman detector hurts the talk). Batch inference, `torch.set_num_threads(os.cpu_count())`, `torch.inference_mode()`. Cache per-text scores.
2. **`TfidfLRDetector`.** Word 1-2-grams + char 3-5-grams TF-IDF, logistic regression (`liblinear` or `saga`). Strong in-domain, expected to degrade OOD.
3. **`EmbeddingLRDetector`.** Frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings + logistic regression. Cache embeddings.

Optional extras, **only if the above are done and time remains**: a Binoculars-style score with a small model pair (observer/performer sharing a tokenizer); a DistilBERT fine-tune (CPU, small data, expect long runtimes). Do not attempt 7B models.

## 5. Metrics and calibration (important)

- Report **AUROC** and **TPR at FPR = 1% and 5%** (never accuracy alone).
- For comparability, **calibrate each detector's threshold on the `val` human texts** so that FPR = 1% (and 5%), then **apply that fixed threshold** to all test sets. Also report the realized FPR on each test set: OOD shifts often break the calibrated FPR, and that is itself a finding worth plotting.
- Bootstrap 95% CIs for AUROC and TPR@FPR (e.g., 200 resamples). Show CIs in plots/tables.
- `metrics.py` must include `ppv(tpr, fpr, prevalence)`; sanity check: tpr=0.99, fpr=0.01, prevalence=0.05 gives PPV ≈ 0.84, and prevalence=0.01 gives PPV = 0.50.

## 6. Experiments (each writes a cached table + a figure)

1. **Speed benchmark** (`01_...`): time 100 texts per scorer, print projected total time for the full corpus. Run this first and **stop to report the numbers** before running heavy jobs.
2. **In-domain vs OOD** (`02_...`): train on `train`, evaluate on `test_in`, `test_ood_domain`, `test_ood_gen`, `test_ood_both`. Output: grouped bar chart of TPR@FPR=1% (and realized FPR) per detector per split, plus an AUROC table.
3. **Disagreement** (`03_...`): using FPR=1%-calibrated thresholds on `test_ood_both` (and `test_in`), compute pairwise agreement/Cohen's kappa between detectors on AI texts and on human texts, a 2D scatter of rank-normalized scores for two detectors colored by true label, and a table of ~10 example texts where detectors disagree (store text snippet, each score, each decision, true label, and generator/domain). Add a short automatic breakdown of disagreement by text length, domain and generator.
4. **Homoglyph attack** (`04_...`): `attacks.py` replaces a fraction p of eligible Latin characters with visually identical Cyrillic/Greek ones (small explicit map, e.g., a, e, o, p, c, x, y). Sweep p in {0, 5, 10, 15, 20, 100}% on ~500 AI texts. Output: TPR@FPR=1% vs p for each detector (the perplexity detector should drop sharply because tokenization changes; check whether TF-IDF and embeddings react differently, and report honestly whatever happens). Also output: mean number of tokens per text vs p, to show the mechanism (more tokens, lower per-token likelihood). Keep the same human texts unattacked for FPR calibration. Add a one-line note that normalizing Unicode (NFKC plus a confusables map) is a cheap defense and, if time permits, show the effect of that defense.
5. **Threshold and base rate** (`05_...`): interactive-ready function and a static figure: PPV as a function of prevalence for the measured TPR/FPR of each detector at FPR in {0.1%, 1%, 5%}.

## 7. App (Gradio, runs locally; must also be deployable to a Hugging Face Space)

Keep dependencies light so a CPU Space can host it. Models load once at startup; use `gr.Blocks` with the queue enabled. Tabs:

1. **Score a text**: textbox, and for each detector a score plus decision at the FPR=1% threshold (show the threshold), with a short note when text is under ~100 tokens ("unreliable on short texts"). Include 4-6 prefilled examples (human, AI, mixed, non-native-style, short).
2. **Attack it**: textbox plus a slider for homoglyph percentage; shows the original and attacked text side by side (note it looks identical), token counts, and updated scores per detector.
3. **Threshold and base rate**: sliders for target FPR and prevalence; shows PPV for each detector using the measured ROC from `test_in` or `test_ood_both` (selector).
4. **Results**: static figures from experiments 2 and 3 (read from `artifacts/`).

Privacy note displayed in the UI: text is processed in memory and not stored. Make sure nothing logs user text.

## 8. Notebook (`notebooks/demo.ipynb`)

Narrative cells matching the 10-minute flow, each reading cached results only:

1. Setup: the three detectors in one slide-sized table, and in-domain results (all look good).
2. Out-of-domain drop (figure from experiment 2).
3. Disagreement scatter plus 3-4 live-displayed disagreement examples.
4. Homoglyph attack curve; then one live cell calling the attack on a pasted text.
5. Threshold/base-rate figure.

Use clean, large-font, high-contrast plots suitable for a projector (font size 14+, colorblind-safe palette, no chartjunk). Every figure saved also as PNG in `artifacts/figures/` for the slides.

## 9. Order of work

1. Environment + `requirements.txt`; check what can be downloaded (data, models). **Report findings before continuing if RAID or the models are unavailable.**
2. `01_benchmark_speed.py` and report timings.
3. Data loading and splits; print a table of counts per split/domain/generator/label.
4. Detectors, features cached; experiment 2.
5. Experiment 3, then 4, then 5.
6. Notebook.
7. App.
8. README (how to run, timings, caveats).

## 10. Acceptance criteria

- `make all` reproduces every cache and figure from scratch on a CPU laptop in a reasonable time (state the measured time in the README); re-running with caches present takes under a minute.
- Notebook runs top to bottom without recomputation in under 2 minutes.
- App: scoring one 256-token paragraph with all three detectors takes under ~3 seconds on CPU after startup.
- Every number shown has an n and a CI (or is clearly labeled as a single run).
- Unit tests for `metrics.py` (calibration, TPR@FPR, PPV) and `attacks.py` (output differs from input but is visually identical; p=0 is identity).
- README lists **limitations honestly**: small models, small data, one dataset, English only, results will not match commercial detectors.

## 11. Non-goals

No commercial-detector integration, no multilingual support, no paraphrase attacks beyond an optional light one, no large-model fine-tuning, no deployment automation (just keep it deployable).

## 12. Decisions to ask me before proceeding if blocked

- RAID unavailable: OK with the fallback corpus, and who supplies LLM API keys if needed?
- GPT-2 vs a modern small model as the demo's zero-shot scorer, if results differ a lot.
- Whether to include the optional Binoculars-style detector or the DistilBERT fine-tune after benchmark timings are known.
