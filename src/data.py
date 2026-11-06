"""RAID loading, subsampling and splitting.

Source: the Hub's automatic parquet conversion of RAID train, which is PARTIAL
(~2.27M of ~7.4M rows; only abstracts, books, news and part of poetry).
See docs/project-state.md section 4 and artifacts/logs/00_raid_coverage.txt.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from huggingface_hub import HfFileSystem, hf_hub_download

from src.config import DATA_DIR, RAID_PARQUET_GLOB, RAID_REPO
from src.text import normalize_whitespace, truncate

PARQUET_DIR = DATA_DIR / "raid_parquet"
# The conversion lives on a ref of its own, not on RAID_REVISION.
PARQUET_REVISION = "refs/convert/parquet"

# Subset agreed on 2026-11-05 from the coverage grid.
DOMAIN_A, DOMAIN_B = "news", "abstracts"
GEN_TRAIN = "chatgpt"
GENS_OOD = ["gpt4", "llama-chat"]
DECODING, REP_PENALTY = "sampling", "no"
COLUMNS = ["id", "source_id", "model", "domain", "generation"]


def download_raid_shards() -> list[Path]:
    """Fetch the 10 parquet shards (~2.4 GB) whole; no-op if already present.

    Whole-file download beats column-projected remote reads here: the shards
    use 1,000-row row groups, so a remote column read is thousands of tiny
    HTTP range requests per shard.
    """
    local = sorted((PARQUET_DIR / "raid" / "partial-train").glob("*.parquet"))
    if len(local) == 10:
        return local
    names = sorted(p.split("@refs%2Fconvert%2Fparquet/", 1)[1] for p in HfFileSystem().glob(RAID_PARQUET_GLOB))
    paths = []
    for name in names:  # e.g. raid/partial-train/0000.parquet
        paths.append(Path(hf_hub_download(
            RAID_REPO, name, repo_type="dataset", revision=PARQUET_REVISION, local_dir=PARQUET_DIR,
        )))
        print(f"  {name}: {paths[-1].stat().st_size / 2**20:.0f} MB", flush=True)
    return paths


def load_clean(domains=(DOMAIN_A, DOMAIN_B), models=(GEN_TRAIN, *GENS_OOD), attack="none") -> pd.DataFrame:
    """Human rows plus AI rows (fixed decoding / rep penalty) for the given cells.

    Adds `label` (1 = AI, 0 = human): RAID has no label column, human rows are
    model == "human". For AI rows, `source_id` is the id of the human document
    the generation was prompted from.
    """
    f = ds.field
    is_human = f("model") == "human"
    is_ai = (pc.is_in(f("model"), pa.array(list(models)))
             & (f("decoding") == DECODING) & (f("repetition_penalty") == REP_PENALTY))
    expr = (f("attack") == attack) & pc.is_in(f("domain"), pa.array(list(domains))) & (is_human | is_ai)
    table = ds.dataset(download_raid_shards(), format="parquet").to_table(columns=COLUMNS, filter=expr)
    df = table.to_pandas()
    df["label"] = (df["model"] != "human").astype(int)
    return df.sort_values(["domain", "source_id", "model"], ignore_index=True)


def sample_rows(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=np.random.RandomState(seed))


# --- corpus and splits -------------------------------------------------------

SPLIT_FRACS = {"train": 0.4, "val": 0.3, "test": 0.3}  # over news source documents
CORPUS_PATH = DATA_DIR / "corpus.parquet"
N_ATTACK = 500

# Each split is a view of the corpus. Human test texts are deliberately shared:
# test_in and test_ood_gen use the same human news documents, so the only thing
# that changes between them is the generator (same for the two abstracts splits).
SPLITS = {
    "train": lambda d: d["group"] == "train",
    "val": lambda d: d["group"] == "val",
    "test_in": lambda d: (d["group"] == "test") & d["model"].isin(["human", GEN_TRAIN]),
    "test_ood_gen": lambda d: (d["group"] == "test") & d["model"].isin(["human", *GENS_OOD]),
    "test_ood_domain": lambda d: (d["group"] == "ood") & d["model"].isin(["human", GEN_TRAIN]),
    "test_ood_both": lambda d: (d["group"] == "ood") & d["model"].isin(["human", *GENS_OOD]),
}


def build_corpus() -> pd.DataFrame:
    """One row per text used anywhere, whitespace-normalized, then truncated to
    256 and 128 Qwen tokens.

    News is split BY SOURCE DOCUMENT: a human article and every AI text
    generated from it land in the same group, so no split ever sees a
    generation of a document whose original (or sibling) is in another split.
    Abstracts are never trained or calibrated on: all of them are group "ood".
    """
    from transformers import AutoTokenizer

    from src.config import MODELS, SEED

    df = load_clean()
    # Formatting differs by class in RAID (see src/text.py): normalize first.
    df["generation"] = df["generation"].map(normalize_whitespace)
    repo, rev = MODELS["qwen"]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    df["n_tokens_full"] = [len(x) for x in tok(df["generation"].tolist(), add_special_tokens=False)["input_ids"]]
    df["text256"] = truncate(df["generation"], tok, 256)
    df["text128"] = truncate(df["generation"], tok, 128)
    df = df.drop(columns="generation")

    # RAID news contains 60 human articles twice (same first 256 tokens,
    # different full versions). Left in, 40 of the pairs straddle splits:
    # identical human text in train and test. Keep the first copy (by
    # source_id) and drop the other copy's whole source: human + generations.
    human = df[df["model"] == "human"].sort_values("source_id")
    dup_sources = set(human.loc[human.duplicated(["domain", "text256"]), "source_id"])
    df = df[~df["source_id"].isin(dup_sources)].reset_index(drop=True)
    print(f"dropped {len(dup_sources)} duplicate human source documents (and their generations)")

    sources = np.sort(df.loc[(df["domain"] == DOMAIN_A) & (df["model"] == "human"), "source_id"].unique())
    perm = np.random.RandomState(SEED).permutation(sources)
    n_train = round(SPLIT_FRACS["train"] * len(perm))
    n_val = round(SPLIT_FRACS["val"] * len(perm))
    group_of = {s: "train" for s in perm[:n_train]}
    group_of.update({s: "val" for s in perm[n_train : n_train + n_val]})
    group_of.update({s: "test" for s in perm[n_train + n_val :]})
    df["group"] = np.where(df["domain"] == DOMAIN_A, df["source_id"].map(group_of), "ood")
    if df["group"].isna().any():
        raise ValueError("AI rows whose human source document is missing")

    # Train/val only need human + the training generator.
    keep = (df["group"].isin(["test", "ood"])) | df["model"].isin(["human", GEN_TRAIN])
    df = df[keep].reset_index(drop=True)
    if df["text256"].duplicated().any():
        raise ValueError("duplicate texts remain after de-duplication")
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(CORPUS_PATH)
    return df


def load_corpus() -> pd.DataFrame:
    return pd.read_parquet(CORPUS_PATH) if CORPUS_PATH.exists() else build_corpus()


def get_split(corpus: pd.DataFrame, name: str) -> pd.DataFrame:
    return corpus[SPLITS[name](corpus)].reset_index(drop=True)


def attack_subset(corpus: pd.DataFrame, n: int = N_ATTACK) -> pd.DataFrame:
    """The AI texts the homoglyph sweep runs on: n chatgpt texts from test_in."""
    from src.config import SEED

    ai = get_split(corpus, "test_in").query("label == 1")
    return sample_rows(ai, n, seed=SEED).sort_values("id", ignore_index=True)
