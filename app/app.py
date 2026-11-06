"""Live demo: three AI-text detectors, one homoglyph attack, one base-rate calculator.

Tabs
  1. Score a text      all three detectors, decisions at a val-calibrated threshold (5% FPR, or 1%)
  2. Attack it         swap a share of letters for Cyrillic/Greek look-alikes, rescore
  3. Base rate         what share of flagged texts is really AI, given how common AI text is

Every text goes through exactly the experiments' pipeline: whitespace
normalization, then the first 256 Qwen tokens. Thresholds, ROC rates and the
val human scores come from artifacts/calibration/calibration_256.json
(experiment 2); the two trained classifiers from artifacts/models/.

Privacy: text is processed in memory only. Nothing logs, stores or caches it.

Run locally:  make app   (or: python app/app.py)
Env:          DETECTOR_DTYPE=auto|bf16|fp32  (default auto: time both at startup, keep the faster)
"""

import html
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import plotting  # noqa: E402
from src.attacks import HOMOGLYPHS, homoglyph_attack, normalize_defense  # noqa: E402
from src.config import CALIBRATION_DIR, MODELS  # noqa: E402
from src.detectors import EmbeddingLRDetector, PerplexityDetector  # noqa: E402
from src.metrics import ppv  # noqa: E402
from src.text import normalize_whitespace, truncate  # noqa: E402

TOKENS = 256
SHORT_TOKENS = 100
DETECTORS = ["perplexity-qwen", "tfidf-lr", "embedding-lr"]
FPR_CHOICES = {"5% (headline)": "0.05", "1%": "0.01"}
SPLIT_CHOICES = {"In-domain: news, ChatGPT": "test_in", "New domain + new generators: abstracts, GPT-4 & Llama": "test_ood_both"}
PREVALENCE_CHOICES = ["0.1%", "1%", "5%", "10%", "20%", "50%"]
ADVERTISED = {"tpr": 0.99, "fpr": 0.001}
N_POP = 10_000
SCORE_KIND = {"perplexity-qwen": "mean log-likelihood per token", "tfidf-lr": "P(AI) from logistic regression",
              "embedding-lr": "P(AI) from logistic regression"}

CAL = json.loads((CALIBRATION_DIR / f"calibration_{TOKENS}.json").read_text())["detectors"]
CAL_128 = json.loads((CALIBRATION_DIR / "calibration_128.json").read_text())["detectors"]
EXAMPLES = json.loads((Path(__file__).resolve().parent / "examples.json").read_text())
VAL_HUMANS = {d: np.asarray(CAL[d]["val_human_scores"]) for d in DETECTORS}


# --- models ---------------------------------------------------------------------

def _time_features(det: PerplexityDetector, text: str, runs: int = 2) -> float:
    det.features([text])  # warm-up
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        det.features([text])
        best = min(best, time.perf_counter() - t0)
    return best


def available_cpus() -> int:
    """CPUs this process may really use. os.cpu_count() reports the host's cores,
    but a Space container is capped by a cgroup quota (free tier: 2 vCPU), and
    too many torch threads on 2 CPUs is far slower than 2 threads."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def load_models() -> tuple[PerplexityDetector, object, EmbeddingLRDetector, str]:
    # 8 threads was the fastest setting in the speed benchmark (8 physical cores);
    # more only adds contention. APP_THREADS overrides.
    threads = int(os.environ.get("APP_THREADS", min(available_cpus(), 8)))
    torch.set_num_threads(threads)
    print(f"[startup] torch threads: {threads}", flush=True)
    qwen = PerplexityDetector("qwen", dtype=torch.bfloat16)  # the dtype every threshold was calibrated in
    choice = os.environ.get("DETECTOR_DTYPE", "auto").lower()
    probe = EXAMPLES[0]["text"]
    if choice == "auto":
        # Qwen2.5 weights ship in bf16, so bf16 -> fp32 -> bf16 casts are exact.
        # bf16 is ~2x faster on CPUs with native bf16, much slower on CPUs without.
        t_bf16 = _time_features(qwen, probe)
        qwen.model.to(torch.float32)
        t_fp32 = _time_features(qwen, probe)
        choice = "fp32" if t_fp32 < t_bf16 else "bf16"
        print(f"[startup] Qwen 256 tokens: bf16 {t_bf16:.2f}s, fp32 {t_fp32:.2f}s -> {choice}", flush=True)
    qwen.model.to(torch.float32 if choice == "fp32" else torch.bfloat16)
    tfidf = joblib.load(ROOT / "artifacts" / "models" / f"tfidf-lr_{TOKENS}.joblib")
    emb = EmbeddingLRDetector()
    emb.clf = joblib.load(ROOT / "artifacts" / "models" / f"embedding-lr_{TOKENS}.joblib")
    emb.features(["warm-up"])
    return qwen, tfidf, emb, choice


QWEN, TFIDF, EMB, DTYPE = load_models()


def prepare(raw: str) -> tuple[str, int]:
    """Exactly the corpus pipeline: normalize whitespace, keep the first 256 Qwen tokens."""
    text = normalize_whitespace(raw or "")
    n_full = int(QWEN.token_counts([text])[0]) if text else 0
    return truncate([text], QWEN.tokenizer, TOKENS)[0] if text else "", n_full


def score_all(text: str) -> dict[str, float]:
    return {"perplexity-qwen": float(QWEN.features([text])[0]),
            "tfidf-lr": float(TFIDF.predict_proba([text])[0, 1]),
            "embedding-lr": float(EMB.score_features(EMB.features([text]))[0])}


def threshold(det: str, fpr_key: str) -> float:
    return CAL[det]["thresholds"][fpr_key]["value"]


def human_percentile(det: str, s: float) -> float:
    """Share of val human news texts scoring strictly lower: one scale for all three detectors."""
    v = VAL_HUMANS[det]
    return np.searchsorted(v, s, side="left") / len(v)


# --- HTML helpers -----------------------------------------------------------------

def dot(det: str) -> str:
    return f'<span class="dot" style="background:{plotting.DETECTOR_STYLE[det][1]}"></span>'


def name(det: str) -> str:
    return plotting.DETECTOR_STYLE[det][0]


def decision(flag: bool) -> str:
    return '<span class="flag">Flagged as AI</span>' if flag else '<span class="noflag">Not flagged</span>'


def fmt_score(det: str, s: float) -> str:
    return f"{s:.2f}" if det.startswith("perplexity") else f"{s:.3f}"


def note(text: str) -> str:
    return f'<p class="note">{text}</p>'


def short_warning(n_tokens: int) -> str:
    if n_tokens >= SHORT_TOKENS:
        return ""
    tpr256 = CAL["perplexity-qwen"]["rates"]["test_in"]["0.05"]["tpr"]
    tpr128 = CAL_128["perplexity-qwen"]["rates"]["test_in"]["0.05"]["tpr"]
    return (f'<p class="warn">Short text ({n_tokens} tokens): scores are unreliable under ~{SHORT_TOKENS} tokens. '
            f"Even at 128 tokens, perplexity caught {tpr128:.0%} of ChatGPT news texts, against "
            f"{tpr256:.0%} at 256 (experiment 2).</p>")


# --- tab 1: score -------------------------------------------------------------------

def tab_score(raw: str, fpr_label: str) -> str:
    text, n_full = prepare(raw)
    if not text:
        return note("Paste a text, or pick an example below.")
    fpr_key = FPR_CHOICES[fpr_label]
    s = score_all(text)
    rows, n_flags = [], 0
    for det in DETECTORS:
        th = threshold(det, fpr_key)
        flag = s[det] > th
        n_flags += flag
        rows.append(f"<tr><td>{dot(det)}{name(det)}</td><td class='num'>{fmt_score(det, s[det])}</td>"
                    f"<td class='num'>{fmt_score(det, th)}</td>"
                    f"<td class='num'>{human_percentile(det, s[det]):.0%}</td><td>{decision(flag)}</td></tr>")
    n_val = CAL[DETECTORS[0]]["thresholds"][fpr_key]["n_val_human"]
    scored = f"first {TOKENS} of {n_full} tokens" if n_full > TOKENS else f"all {n_full} tokens"
    return (
        short_warning(n_full)
        + f'<p class="headline">{n_flags} of 3 detectors flag this text.</p>'
        + "<table class='scores'><thead><tr><th>Detector</th><th>Score</th>"
        + f"<th>Threshold<br><small>({float(fpr_key):.0%} FPR)</small></th>"
        + "<th>Higher than …<br><small>of human news texts</small></th><th>Decision</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
        + note(f"Higher score = more AI-like. Perplexity score = {SCORE_KIND['perplexity-qwen']}; the two "
               f"classifiers output {SCORE_KIND['tfidf-lr']}. Each threshold flags {float(fpr_key):.0%} of "
               f"{n_val} human news texts (RAID validation split). Scored: {scored}.")
    )


# --- tab 2: attack --------------------------------------------------------------------

def reveal(original: str, attacked: str) -> str:
    """The attacked text with every swapped character highlighted."""
    out = []
    for a, b in zip(original, attacked):
        out.append(f"<mark>{html.escape(b)}</mark>" if a != b else html.escape(b))
    return f'<div class="reveal">{"".join(out)}</div>'


def tab_attack(raw: str, pct: float, defend: bool, fpr_label: str, last: dict | None):
    """`last` = this session's previous original text and its scores (gr.State, in
    memory, dropped with the session), so moving the slider rescores only the attack."""
    text, _ = prepare(raw)
    if not text:
        return "", "", note("Paste a text to attack."), "", last
    fpr_key = FPR_CHOICES[fpr_label]
    attacked = homoglyph_attack(text, pct / 100)
    scored_attacked = normalize_defense(attacked) if defend else attacked
    if not last or last["text"] != text:
        last = {"text": text, "scores": score_all(text)}
    s0, s1 = last["scores"], score_all(scored_attacked)
    n0, n1 = (int(x) for x in QWEN.token_counts([text, scored_attacked]))
    eligible = sum(c in HOMOGLYPHS for c in text)
    swapped = sum(a != b for a, b in zip(text, attacked))
    rows = []
    for det in DETECTORS:
        th = threshold(det, fpr_key)
        rows.append(f"<tr><td>{dot(det)}{name(det)}</td>"
                    f"<td class='num'>{fmt_score(det, s0[det])}</td><td>{decision(s0[det] > th)}</td>"
                    f"<td class='num'>{fmt_score(det, s1[det])}</td><td>{decision(s1[det] > th)}</td></tr>")
    after = "Attacked, then defended" if defend else "Attacked"
    table = (
        f'<p class="headline">{swapped} of {eligible} eligible letters swapped. '
        f"Qwen tokens: {n0} → {n1}.</p>"
        + "<table class='scores'><thead><tr><th>Detector</th><th>Original</th><th></th>"
        + f"<th>{after}</th><th></th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        + note(f"Thresholds: {float(fpr_key):.0%} FPR, calibrated on unattacked human news texts. "
               + ("Defense: Unicode NFKC + a look-alike → Latin map, applied before scoring." if defend else
                  "The attacker only touches the AI text; the threshold does not move."))
    )
    return text, attacked, table, reveal(text, attacked), last


# --- tab 3: base rate -------------------------------------------------------------------

def ppv_figure(split: str, fpr_key: str, prevalence: float):
    import matplotlib.pyplot as plt

    plotting.setup()
    prev = np.geomspace(0.001, 0.5, 200)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.plot(prev, ppv(ADVERTISED["tpr"], ADVERTISED["fpr"], prev), color=plotting.MUTED, lw=2, ls=(0, (4, 3)),
            label="Advertised claim: TPR 99%, FPR 0.1% (not measured)")
    for det in DETECTORS:
        r = CAL[det]["rates"][split][fpr_key]
        label, color = plotting.DETECTOR_STYLE[det]
        ax.fill_between(prev, ppv(r["tpr_lo"], r["fpr_hi"], prev), ppv(r["tpr_hi"], r["fpr_lo"], prev),
                        color=color, alpha=0.10, linewidth=0)
        ax.plot(prev, ppv(r["tpr"], r["fpr"], prev), color=color, lw=2, label=label)
    ax.axvline(prevalence, color=plotting.INK_2, lw=1.5)
    ax.set_xscale("log")
    ax.set_xticks([0.001, 0.01, 0.05, 0.1, 0.2, 0.5], ["0.1%", "1%", "5%", "10%", "20%", "50%"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="both")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of texts that are AI (prevalence)")
    ax.set_ylabel("Flags that are really AI (PPV)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=12.5, handlelength=1.8)
    fig.tight_layout()
    return fig


def tab_base_rate(split_label: str, fpr_label: str, prev_label: str):
    import matplotlib.pyplot as plt

    split, fpr_key = SPLIT_CHOICES[split_label], FPR_CHOICES[fpr_label]
    prevalence = float(prev_label.rstrip("%")) / 100
    n_ai = round(N_POP * prevalence)
    n_h = N_POP - n_ai
    rows = []
    entries = [(det, CAL[det]["rates"][split][fpr_key]) for det in DETECTORS]
    entries.append(("advertised", ADVERTISED))
    for det, r in entries:
        tp, fp = round(r["tpr"] * n_ai), round(r["fpr"] * n_h)
        label = (f"{dot(det)}{name(det)}" if det != "advertised"
                 else '<span class="dot" style="background:#898781"></span><i>Advertised claim (not measured)</i>')
        p = tp / (tp + fp) if tp + fp else float("nan")
        rows.append(f"<tr><td>{label}</td><td class='num'>{r['tpr']:.0%}</td><td class='num'>{r['fpr']:.1%}</td>"
                    f"<td class='num'>{tp + fp:,}</td><td class='num'>{tp:,}</td><td class='num'>{fp:,}</td>"
                    f"<td class='num'><b>{p:.0%}</b></td></tr>")
    n = CAL[DETECTORS[0]]["rates"][split][fpr_key]
    table = (
        f'<p class="headline">Out of {N_POP:,} texts, {n_ai:,} are AI ({prev_label}).</p>'
        + "<table class='scores'><thead><tr><th>Detector</th><th>Catches<br><small>(TPR)</small></th>"
        + "<th>Wrongly flags<br><small>(FPR on this data)</small></th><th>Flagged</th>"
        + "<th>…really AI</th><th>…human</th><th>Flags that are AI<br><small>(PPV)</small></th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
        + note(f"TPR and FPR measured in experiment 2 on {int(n['n_ai']):,} AI + {int(n['n_human']):,} human "
               f"texts, with the threshold set for {float(fpr_key):.0%} FPR on human news. Off its home domain "
               "the realized FPR drifts away from that target. Bands on the plot: 95% bootstrap CI (conservative).")
    )
    fig = ppv_figure(split, fpr_key, prevalence)
    plt.close(fig)
    return table + overlap_note(split, fpr_key), fig


def overlap_note(split: str, fpr_key: str) -> str:
    """Curves that coincide hide each other on the plot: say so in words."""
    prev = np.geomspace(0.001, 0.5, 200)
    curves = {d: ppv(CAL[d]["rates"][split][fpr_key]["tpr"], CAL[d]["rates"][split][fpr_key]["fpr"], prev)
              for d in DETECTORS}
    pairs = [(a, b) for i, a in enumerate(DETECTORS) for b in DETECTORS[i + 1:]
             if np.nanmax(np.abs(curves[a] - curves[b])) < 0.02]
    return "".join(note(f"On the plot, the {name(a)} and {name(b)} curves overlap (PPV within 2 points "
                        "everywhere): one hides the other.") for a, b in pairs)


# --- UI -----------------------------------------------------------------------------

CSS = """
.gradio-container {max-width: 1250px !important; font-size: 17px;}
textarea {font-size: 17px !important; line-height: 1.5 !important;}
table.scores {border-collapse: collapse; width: 100%; font-size: 18px; margin: 6px 0 10px;}
table.scores th {text-align: left; font-weight: 600; color: #52514e; border-bottom: 1.5px solid #c3c2b7;
                 padding: 8px 12px; vertical-align: bottom;}
table.scores th small {font-weight: 400;}
table.scores td {padding: 10px 12px; border-bottom: 1px solid #e1e0d9; color: #0b0b0b;}
table.scores td.num {font-variant-numeric: tabular-nums; text-align: right;}
.dot {display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 10px;}
.flag {font-weight: 700; color: #0b0b0b;}
.noflag {color: #898781;}
.headline {font-size: 22px; font-weight: 600; color: #0b0b0b; margin: 4px 0 8px;}
.note {font-size: 14.5px; color: #52514e; margin: 4px 0;}
.warn {font-size: 16px; color: #0b0b0b; border-left: 4px solid #52514e; padding: 6px 12px; background: #f3f2ee;}
.reveal {font-size: 17px; line-height: 1.6; padding: 10px; border: 1px solid #e1e0d9; border-radius: 6px;}
.reveal mark {background: #e0dcf7; color: #0b0b0b; border-radius: 2px; padding: 0 1px;}
"""

PRIVACY = ("Text is processed in memory only: nothing is logged, stored or sent anywhere. "
           "Detectors: Qwen2.5-0.5B perplexity, TF-IDF + logistic regression, MiniLM embeddings + logistic "
           "regression, all trained or calibrated on RAID news (ChatGPT vs human).")


def build() -> gr.Blocks:
    ex_texts = [[e["text"]] for e in EXAMPLES]
    ex_labels = [e["name"] for e in EXAMPLES]
    ai_example = next(e["text"] for e in EXAMPLES if e["name"] == "ChatGPT news article")
    with gr.Blocks(title="Can you detect AI text?", css=CSS, analytics_enabled=False,
                   theme=gr.themes.Base(font=[gr.themes.GoogleFont("Inter"), "sans-serif"])) as demo:
        gr.Markdown("# Can you detect AI-written text?\nThree detectors, the same text, very different answers.")
        with gr.Tabs():
            with gr.Tab("1 · Score a text"):
                with gr.Row():
                    txt = gr.Textbox(label="Text", lines=9, max_lines=16, scale=4,
                                     placeholder="Paste a paragraph (about 200 words works best).")
                    with gr.Column(scale=1, min_width=200):
                        fpr1 = gr.Radio(list(FPR_CHOICES), value="5% (headline)", label="Threshold: false-positive rate")
                        go1 = gr.Button("Score", variant="primary")
                gr.Examples(ex_texts, inputs=[txt], example_labels=ex_labels, label="Examples (real RAID texts)")
                out1 = gr.HTML()
                go1.click(tab_score, [txt, fpr1], out1)
                fpr1.change(tab_score, [txt, fpr1], out1)

            with gr.Tab("2 · Attack it"):
                with gr.Row():
                    txt2 = gr.Textbox(label="Text to attack", value=ai_example, lines=6, max_lines=10, scale=4)
                    with gr.Column(scale=1, min_width=220):
                        pct = gr.Slider(0, 100, value=15, step=5, label="Letters swapped for look-alikes (%)")
                        defend = gr.Checkbox(value=False, label="Apply the defense before scoring")
                        fpr2 = gr.Radio(list(FPR_CHOICES), value="5% (headline)", label="Threshold")
                        go2 = gr.Button("Attack", variant="primary")
                with gr.Row():
                    orig = gr.Textbox(label="Original (as scored)", lines=7, max_lines=10, interactive=False)
                    att = gr.Textbox(label="Attacked (copy it, paste it anywhere: it looks the same)", lines=7,
                                     max_lines=10, interactive=False, show_copy_button=True)
                out2 = gr.HTML()
                with gr.Accordion("Show which letters were swapped", open=False):
                    rev = gr.HTML()
                last = gr.State(None)
                ins2 = [txt2, pct, defend, fpr2, last]
                outs2 = [orig, att, out2, rev, last]
                go2.click(tab_attack, ins2, outs2)
                pct.release(tab_attack, ins2, outs2)
                defend.change(tab_attack, ins2, outs2)
                fpr2.change(tab_attack, ins2, outs2)

            with gr.Tab("3 · Base rate"):
                with gr.Row():
                    split3 = gr.Radio(list(SPLIT_CHOICES), value=list(SPLIT_CHOICES)[0], label="Measured on")
                    fpr3 = gr.Radio(list(FPR_CHOICES), value="5% (headline)", label="Threshold")
                    prev3 = gr.Radio(PREVALENCE_CHOICES, value="5%", label="Share of texts that are AI")
                out3 = gr.HTML()
                plot3 = gr.Plot(show_label=False)
                ins3 = [split3, fpr3, prev3]
                for c in ins3:
                    c.change(tab_base_rate, ins3, [out3, plot3])
                demo.load(tab_base_rate, ins3, [out3, plot3])

        gr.Markdown(f"<p class='note'>{PRIVACY} Qwen runs in {DTYPE}. "
                    f"Model revisions pinned: {', '.join(r for r, _ in MODELS.values() if 'gpt2' not in r)}.</p>")
    return demo


if __name__ == "__main__":
    build().queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), show_api=False)
