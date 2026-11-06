"""Shared figure style: projector-first (large type, high contrast, light surface).

One fixed color per detector across every figure and the app, so the audience
learns the mapping once. Colors are categorical slots 1-3 of the dataviz
reference palette, the only three validated all-pairs (colorblind-safe).
Slot 3 (aqua) is below 3:1 contrast on the surface, so bars always carry
direct value labels.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import FIGURES_DIR  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

DETECTOR_STYLE = {
    "perplexity-qwen": ("Perplexity (Qwen2.5-0.5B)", "#2a78d6"),
    "tfidf-lr": ("TF-IDF + LR", "#eb6834"),
    "embedding-lr": ("Embeddings + LR", "#1baf7a"),
}
SPLIT_LABELS = {
    "test_in": "In-domain\nnews · chatgpt",
    "test_ood_gen": "New generators\nnews · gpt4, llama",
    "test_ood_domain": "New domain\nabstracts · chatgpt",
    "test_ood_both": "Both new\nabstracts · gpt4, llama",
}


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 15,
        "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 14, "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": BASELINE,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "legend.frameon": False, "lines.linewidth": 2,
    })


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
