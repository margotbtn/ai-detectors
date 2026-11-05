# Project state and decisions

Living handoff note: what has been decided, what was found, what to do next.
Update it as things change. The build spec is [ai-detector-demo-spec.md](ai-detector-demo-spec.md);
this file records deviations from it and findings that postdate it.

Last updated: 2026-11-04.

## 1. Decisions made (supersede the spec where they conflict)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Drop `notebooks/demo.ipynb`.** Static figures go in the slide deck; the app carries the live beats. | A live notebook on a projector is fragile and overlaps the app. Spec section 2/8 deliverable is cancelled. |
| 2 | **The Gradio app is the primary demo vehicle, deployed to a Hugging Face Space.** | Margot may not be able to present from her own laptop; a URL is checkable from any machine. Treat "runs on a free CPU Space" as a hard constraint, not a nice-to-have. |
| 3 | **App has 3 tabs, not 4:** Score a text / Attack it / Threshold and base rate. The "Results" tab is dropped. | Those figures live in the slides. |
| 4 | **FPR = 5% is the headline metric**, with FPR = 1% reported alongside. | A 1% threshold is the 99th percentile of val human scores; at n=500 it is set by ~5 observations and is far too noisy to carry the argument. |
| 5 | **Three detectors in the app and in all figures**: perplexity, TF-IDF+LR, embeddings+LR. | Visual clarity on a projector. |
| 6 | **The shallow-cue baseline is computed once in experiment 2 only.** Not in the app, not in any plot — it exists to produce one number to say out loud. | See section 4 below for why it is worth having at all. |
| 7 | **Isolated dev environment via dev container.** No installing into the host Python, ever. | Margot's explicit instruction. `.devcontainer/` is committed; `uv`/`.venv` experiments were removed from the host. |

## 2. Corrections to the spec

1. ~~**Section 4.1 had a sign error**~~ — **FIXED IN THE SPEC on 2026-11-04.** It had defined the
   perplexity score as "negative mean token log-likelihood" while declaring *higher = more likely
   AI*, which is inverted: negative mean log-likelihood is cross-entropy, **high** for
   surprising/human text and **low** for the flat, high-probability text LLMs produce. Section 4.1
   now specifies **mean token log-likelihood** (equivalently negative perplexity) and warns against
   the inverted form. Left uncorrected this produces AUROC around 0.2.
2. **Section 1 says 32 GB RAM; the machine has 16 GB** (16 cores, 946 GB free disk).
   **Deliberately NOT corrected in the spec** — Margot chose to fix the sign error only. Treat
   16 GB as the real budget regardless of what section 1 says.
3. **Section 6.3 should not use Cohen's kappa.** It is computed there on single-class subsets
   (AI-only, human-only), where the marginals are degenerate and kappa is unstable and
   hard to narrate. Use **Jaccard overlap of the flagged sets** (|A∩B| / |A∪B|) plus
   **McNemar** for whether one detector is systematically stricter. "At the same 5% FPR these
   two detectors agree on only 40% of what they flag" is the sentence the audience keeps.
4. **Section 5 should bootstrap the threshold itself**, not only AUROC and TPR@FPR. The
   instability of the calibration point is on-thesis for this talk.

## 3. Environment findings (2026-11-04)

- Host: WSL2, Ubuntu, Python 3.12.3 system-wide with **no pip, no ensurepip, no venv module**;
  `uv` is present at `~/.local/bin/uv`. Do not use any of these — build in the container.
- **Docker blocker:** Docker Desktop's binary is on PATH via `/mnt/c/...` but WSL integration is
  **not enabled for this distro**, so `docker` refuses to run. Fix is a GUI toggle:
  Docker Desktop → Settings → Resources → WSL Integration → enable for this distro → restart.
- Inside the container the workspace is `/workspaces/ai-detectors`, not `/home/margot/ai-detectors`.
- Network is open: PyPI and huggingface.co both reachable.
- All three models exist and are fetchable: `Qwen/Qwen2.5-0.5B`, `gpt2`,
  `sentence-transformers/all-MiniLM-L6-v2`. **Pin exact revisions** before the talk so a Space
  cold start is predictable.
- First container build pulls ~2.5 GB. Model weights persist in the named volume
  `ai-detectors-hf-cache` across rebuilds.

## 4. RAID findings (2026-11-04)

Dataset is `liamdugan/raid` on the Hub.

- **Ungated, MIT-licensed.** This resolves the spec's section-3 licensing hedge: example
  snippets may be committed to the repo and shipped in the Space.
- **`train.csv` is 11.8 GB.** Do not download it whole. (`test.csv` 1.22 GB is unlabeled —
  RAID withholds test labels for its leaderboard, so all splits must be carved from `train`.)
- **The Hub parquet conversion is PARTIAL: 2.27 M of ~7.4 M train rows** (the Hub converts only
  the first ~5 GB). 10 shards of ~240 MB at
  `https://huggingface.co/api/datasets/liamdugan/raid/parquet/raid/train/{0..9}.parquet`.
  **First job in the container:** read only the `model`/`domain`/`attack` columns from those
  shards and print the coverage grid. If the CSV is ordered by generator or domain, a cell we
  need may be absent — fall back to a ranged read of the CSV.
- **The dataset-viewer `/filter` API is unusable.** It returns
  `"Authentication check on the Hugging Face Hub failed or timed out"` on repeated calls, even
  unauthenticated. Do not build the pipeline on it.
- **There is no label column.** Human rows are `model == "human"` (verified on a real row:
  `model=human, decoding=None, attack=none, domain=abstracts`).
- Columns: `id, adv_source_id, source_id, model, decoding, repetition_penalty, attack, domain,
  title, prompt, generation`.
- Generators: `chatgpt, gpt4, gpt3, gpt2, llama-chat, mistral, mistral-chat, mpt, mpt-chat,
  cohere, cohere-chat`. Train domains: `abstracts, books, news, poetry, recipes, reddit,
  reviews, wiki`. (`code, czech, german` are in `extra.csv`, out of scope — English only.)
- **RAID already ships `attack == "homoglyph"` rows.** Use them to validate our own
  `attacks.py` against the benchmark's implementation instead of only asserting ours is correct.
- **Unverified, check early:** how many human rows exist per domain. The spec wants ~1,000 AI
  texts per (domain, generator) cell with a *matched* human count — with 3 generators that is
  ~3,000 AI per domain, and RAID may not have that many human docs per domain. If not, reduce
  AI texts per cell rather than shipping an imbalanced split.

## 5. Methodological risk to handle in experiment 2

RAID's AI texts are generated from prompts derived from the human documents, so the two classes
differ in length, formatting and register — not only in being machine-written. Truncating to 256
tokens mitigates this. The **shallow-cue baseline** (decision 6) quantifies what is left: a
logistic regression on ~12 surface features only (char/token length, mean sentence length,
punctuation rates, type-token ratio, stopword rate, digit rate), no language model, nothing
semantic.

- If it reaches AUROC ~0.85 in-domain, the TF-IDF detector's ~0.95 is much less impressive than
  it looks, and it sets up the OOD collapse: shallow cues are exactly what fails to transfer.
- If it scores ~0.6, the real detectors are picking up something beyond surface statistics,
  which makes the rest of the numbers more credible.

Either outcome is usable. Ten lines of code, sub-second to fit.

## 6. Next steps, in order

1. **(Margot)** Enable Docker Desktop WSL integration; reopen in container.
2. Verify the container: `torch.__version__`, thread count, models downloadable.
3. `make lock` → freeze exact versions into `requirements.lock.txt`.
4. RAID coverage grid from the partial parquet shards (section 4). **Report before downloading
   bulk data.**
5. `experiments/01_benchmark_speed.py` — time 100 texts per scorer, project the full-corpus
   cost. **Stop and report the numbers before running heavy jobs** (spec section 9.2).
6. Data loading and splits; print counts per split/domain/generator/label.
7. Detectors + cached features; experiment 2 (incl. shallow-cue baseline).
8. Experiments 3, 4, 5.
9. App (3 tabs), then Space deployment, then README.

## 7. Still open

- Whether to commit the current scaffolding (`.devcontainer/`, `requirements.txt`,
  `app/requirements.txt`, `.gitignore`, this file). **Nothing is committed yet: `README.md` is
  the only file in git history.** The spec and this doc are untracked and have no history.
- Spec section 12 questions still unanswered: Binoculars-style detector and the DistilBERT
  fine-tune. **Both deferred** until the speed benchmark exists; default is to skip both.
- GPT-2 vs Qwen2.5-0.5B as the zero-shot scorer: **keep both.** Qwen is the demo's primary; if
  GPT-2 is near-useless against 2024 generators, that *is* a finding and gives the "detector
  built for a previous generation does not transfer" beat for free.
- Measure the app's latency **on the Space**, not locally — the spec's "<3 s per paragraph"
  criterion must hold on ~2 vCPU. Plan to wake the Space a few minutes before the talk
  (free Spaces sleep after inactivity).
