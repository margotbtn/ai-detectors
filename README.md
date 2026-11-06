# ai-detectors

Three AI-text detectors, evaluated the way you would evaluate any classifier you
had to deploy: on data that does not look like the training set, against a cheap
attack, and at the base rate you will actually meet. Companion code for a talk to
data scientists, plus a live demo app.

**Live demo:** <!-- TODO: Hugging Face Space URL --> (source in [app/](app/))

| Detector | What it is | Trained on |
|---|---|---|
| **Perplexity** | Mean token log-likelihood under Qwen2.5-0.5B. Zero-shot | Nothing. Only the threshold is calibrated |
| **TF-IDF + LR** | Word 1-2-grams + character 3-5-grams, logistic regression | RAID news, ChatGPT vs human |
| **Embeddings + LR** | Frozen all-MiniLM-L6-v2 sentence embeddings, logistic regression | Same |

Every threshold is set on **human** texts only, so that 5% of them are flagged
(the "5% FPR" threshold; 1% reported alongside). 5% is the headline: a 1% threshold
on 515 validation texts is set by about 5 texts and is far too noisy to argue from.

## Results

All numbers: RAID, texts truncated to 256 Qwen tokens, threshold calibrated on 515
human news texts (validation split), 95% bootstrap CIs (200 resamples of both the
validation humans and the test set, so the CI includes threshold uncertainty).

### 1. Every detector breaks, each on a different shift (experiment 2)

TPR = share of AI texts caught. FPR = share of human texts flagged, on this data,
with the threshold set for 5% on news.

| | In-domain<br>news · ChatGPT<br>516 AI + 516 human | New generators<br>news · GPT-4, Llama<br>1,032 + 516 | New domain<br>abstracts · ChatGPT<br>1,766 + 1,766 | Both new<br>abstracts · GPT-4, Llama<br>3,532 + 1,766 |
|---|---|---|---|---|
| **Perplexity** AUROC | 0.988 [0.984, 0.993] | 0.851 [0.833, 0.869] | 0.976 [0.971, 0.981] | 0.812 [0.800, 0.822] |
| TPR | 95.5% [91.3, 97.5] | 56.4% [52.2, 60.3] | 98.3% [97.1, 99.0] | 58.1% [55.6, 61.4] |
| FPR | 4.8% [2.9, 7.8] | 4.8% [2.5, 7.2] | 11.1% [9.1, 13.6] | 11.1% [9.2, 14.1] |
| **TF-IDF + LR** AUROC | 0.998 [0.996, 1.000] | 0.995 [0.993, 0.997] | 0.840 [0.826, 0.851] | 0.877 [0.867, 0.887] |
| TPR | 99.4% [98.4, 100] | 97.7% [95.6, 98.7] | 100% [100, 100] | 100% [99.8, 100] |
| FPR | 6.8% [2.5, 9.7] | 6.8% [2.7, 9.9] | **98.8%** [97.3, 99.4] | **98.8%** [97.3, 99.4] |
| **Embeddings + LR** AUROC | 0.960 [0.948, 0.969] | 0.910 [0.898, 0.925] | 0.605 [0.586, 0.623] | 0.656 [0.641, 0.672] |
| TPR | 84.5% [81.2, 88.4] | 63.5% [59.5, 67.8] | 59.3% [52.7, 66.1] | 66.3% [61.6, 71.3] |
| FPR | 5.8% [4.3, 9.1] | 5.8% [4.1, 8.5] | 45.6% [39.0, 50.7] | 45.6% [39.9, 50.0] |

- **TF-IDF** survives new generators and collapses on a new domain: the news
  threshold flags 99% of human arXiv abstracts.
- **Perplexity** survives a new domain and fails on new generators. That is all
  GPT-4: TPR 15% on GPT-4 news against 98% on Llama. To a small LM, GPT-4 reads
  almost like a person (mean log-likelihood −2.61 vs −2.82 for human news).
- **AUROC hides calibration.** Perplexity's AUROC on abstracts is still 0.976,
  but its 5% threshold now flags 11% of humans. GPT-2 perplexity (in the CSV, not
  the figure) keeps AUROC 0.962 there while its FPR drifts the other way, to 1.2%,
  and its TPR falls to 35%.
- **A shallow baseline** (14 surface features such as sentence length, comma rate
  and type-token ratio, + LR) reaches AUROC 0.960 in-domain: as good as embeddings.
- **Short texts:** at 128 tokens, in-domain perplexity TPR falls from 95.5% to
  71.9% [60.4, 79.5] (`artifacts/results/02_metrics_128.csv`).

### 2. Same false-positive budget, different texts flagged (experiment 3)

At 5% FPR in-domain, the detectors agree on which AI texts to flag (Jaccard
0.82 to 0.95), but **the human texts that perplexity wrongly flags overlap with
neither other detector's: Jaccard 0.00** (516 human texts). Across the in-domain
and both-new test sets (6,330 texts), the three detectors do not all agree on 90%
of GPT-4 texts (19% of ChatGPT texts), and on 93% of texts under 128 tokens (15%
of texts over 512).

### 3. A homoglyph attack defeats perplexity (experiment 4)

500 ChatGPT news texts; a share of eligible Latin letters replaced by identical-looking
Cyrillic or Greek letters. Our attack at 100% reproduces RAID's own `homoglyph` rows
exactly (500/500 identical texts).

| Letters swapped | 0% | 5% | 10% | 20% | 100% | With the defense |
|---|---|---|---|---|---|---|
| Perplexity TPR | 95% | **0%** | 0% | 0% | 0% | 95% |
| TF-IDF + LR | 99% | 99% | 99% | 99% | 58% | 99% |
| Embeddings + LR | 84% | 48% | 34% | 18% | 96% | 84% |

Swapping 5% of letters takes a text from 253 to 304 Qwen tokens, and mean
log-likelihood from −1.99 to −2.98, far below the threshold (−2.34). The defense
(Unicode NFKC + a look-alike → Latin map) is a few lines and restores every
detector exactly. CIs: `artifacts/results/04_sweep.csv`.

### 4. "99% accurate" at a 5% base rate (experiment 5)

Share of flagged texts that are really AI (PPV) when 5% of texts are AI, from the
measured TPR and realized FPR at the 5% threshold:

| | In-domain | Both new |
|---|---|---|
| Perplexity | 51% | 22% |
| TF-IDF + LR | 44% | 5% |
| Embeddings + LR | 43% | 7% |
| *Vendor-style claim, TPR 99% / FPR 0.1% (not measured)* | *98%* | *98%* |

0.1% FPR cannot be calibrated or verified here: on 515 human texts it would be set
by half a text.

Figures for the slides: [artifacts/figures/](artifacts/figures/). Full tables:
`artifacts/results/` and the printouts in `artifacts/logs/` (both regenerated, not committed).

## Data

[RAID](https://github.com/liamdugan/raid) (Dugan et al., 2024), pinned revision in
[src/config.py](src/config.py). Subset: no adversarial attack, sampling decoding,
no repetition penalty.

| Split | Domain | Generators | AI | Human |
|---|---|---|---|---|
| train | news | ChatGPT | 687 | 687 |
| val (threshold calibration) | news | ChatGPT | 515 | 515 |
| test_in | news | ChatGPT | 516 | 516 |
| test_ood_gen | news | GPT-4, Llama-chat | 1,032 | 516 |
| test_ood_domain | abstracts | ChatGPT | 1,766 | 1,766 |
| test_ood_both | abstracts | GPT-4, Llama-chat | 3,532 | 1,766 |

News is split 40/30/30 **by source document**: a human article and every AI text
generated from its title land in the same split. Abstracts are test-only.
11,532 unique texts.

## Running it

Everything runs inside the dev container (`.devcontainer/`, not committed). Nothing
is installed into the host Python. CPU only, 16 GB RAM is enough.

| Step | Command | Time measured (8 cores) |
|---|---|---|
| Model weights (~1.5 GB) | `make models` | network-bound |
| RAID download (~2.4 GB parquet) + corpus | `make data` | 32 s once downloaded |
| LM and embedding features, cached, resumable | `make features` | **2 h 53 min** |
| Fit classifiers, score all texts | `make train` | 27 s |
| Experiments 2-5 + app examples | `make experiments` | 60 s |
| Tests | `make test` | seconds |
| App, locally | `make app` | 17 s startup |

`make all` runs data, features, train and experiments in order. With caches present,
everything after `make features` re-runs in under 2 minutes, with identical numbers.

Feature caches are keyed by detector config and a hash of each text, written in
atomic chunks: an interrupted `make features` continues where it stopped.

### Deploying the app to a Hugging Face Space

```bash
make space          # assembles build/space/ (1.8 MB): app, needed src/, models, calibration
```

Create a Gradio Space (free CPU hardware), then push the contents of
`build/space/` to it. The first start downloads Qwen2.5-0.5B and MiniLM (~1 GB) at
their pinned revisions. At startup the app times bf16 against fp32 for Qwen and keeps
the faster (`DETECTOR_DTYPE=bf16|fp32` to force one), and sizes torch threads to the
container's real CPU quota. Measured on 2 threads: about 1.5 s to score a text with
all three detectors, 2 to 4 s per attack. Free Spaces sleep when idle: open it a few
minutes before presenting.

The app applies exactly the experiments' pipeline (whitespace normalization, first
256 Qwen tokens) and the same thresholds. Text is processed in memory only; nothing
is logged or stored.

## Choices and deviations

- **Partial RAID.** The Hub's parquet conversion of RAID's train set is partial
  (about 2.3 M of 7.4 M rows); the full CSV is 11.8 GB. All subsets used here are
  covered (`experiments/00_raid_coverage.py`), but this is not all of RAID.
- **Whitespace is normalized** in every text. In RAID the classes are formatted
  differently (human news uses single line breaks, AI news blank lines; arXiv abstracts
  are hard-wrapped). A newline-rate feature alone scored AUROC 0.92 on news and 0.00
  on abstracts: without normalization, part of every result would be measuring layout.
- **Duplicate human articles removed.** 62 human news articles appear under two
  source ids, which put identical text in train and test. Their whole source groups
  are dropped.
- **The attack matches RAID's homoglyph map**, including the Greek capitals RAID uses.
- **bf16** for both LMs: 2x faster than fp32 on this CPU, scores identical to fp32
  at correlation 1.0000. int8 quantization was rejected: it scrambled the scores.
- **GPT-2 perplexity and the shallow-cue baseline** are computed in experiment 2 only
  (tables, not figures or the app).
- **Disagreement is measured with Jaccard + McNemar**, not Cohen's kappa: kappa on
  single-class subsets has degenerate marginals.
- **Not done:** a Binoculars-style detector and a fine-tuned DistilBERT (deferred,
  both slow on CPU), and the demo notebook (the app and the slides replace it).

## Limitations

- One training generator (ChatGPT) and one training domain (news). The detectors
  are deliberately simple; stronger ones would shift the numbers, not the lessons.
- Test sets are hundreds to a few thousand texts; the 1% FPR results in particular
  have wide CIs.
- Perplexity depends on the scoring LM: Qwen2.5-0.5B and GPT-2 already disagree on
  where the threshold drifts.
- PPV curves assume the measured TPR and FPR hold at every prevalence.

## Layout

```
src/           data, detectors, features cache, metrics, attacks, plotting
experiments/   00 coverage, 01 speed, 02-05 the experiments
app/           Gradio app, example builder, Space requirements and README
tests/         metrics, attacks, cache, text normalization
artifacts/     calibration/ figures/ models/ committed; everything else regenerated
```
