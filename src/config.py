"""Single source of truth for pinned external resources and global settings.

Revisions are pinned commit hashes so a Hugging Face Space cold start (and a
re-run on the day of the talk) fetches exactly the weights we measured with.
"""

from pathlib import Path

SEED = 0

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
CALIBRATION_DIR = ARTIFACTS_DIR / "calibration"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# name -> (hub repo id, pinned commit), revisions resolved on 2026-11-05.
MODELS = {
    "qwen": ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
    "gpt2": ("gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
    "minilm": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
}

RAID_REPO = "liamdugan/raid"
RAID_REVISION = "865cac74188466cb0c3b7574a10204007b57a459"
# Hub's automatic parquet conversion: PARTIAL, first ~2.27M of ~7.4M train rows.
RAID_PARQUET_GLOB = "datasets/liamdugan/raid@refs%2Fconvert%2Fparquet/raid/partial-train/*.parquet"
