"""Coverage grid of the partial RAID parquet conversion (`make coverage`).

Downloads the shards (once), reads only metadata columns and caches them to
data/raid_meta.parquet. Prints:
  - AI rows per (domain, generator) for attack == none, by decoding/rep-penalty
  - human rows per domain, and unique source_ids per domain
  - which shards each generator/domain lives in (is the conversion ordered?)
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR  # noqa: E402
from src.data import download_raid_shards  # noqa: E402

COLS = ["id", "source_id", "model", "decoding", "repetition_penalty", "attack", "domain"]
CACHE = DATA_DIR / "raid_meta.parquet"


def load_meta() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    frames = []
    for i, path in enumerate(download_raid_shards()):
        t = pq.read_table(path, columns=COLS).to_pandas()
        t["shard"] = i
        print(f"shard {i}: {len(t):,} rows", flush=True)
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(CACHE)
    return df


def main() -> None:
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)
    df = load_meta()
    print(f"\nTotal rows: {len(df):,}")
    print("\nattack values:", df["attack"].value_counts().to_dict())

    clean = df[df["attack"] == "none"]
    human = clean[clean["model"] == "human"]
    ai = clean[clean["model"] != "human"]

    print("\n== Human rows per domain (attack=none): rows / unique source_id ==")
    print(human.groupby("domain").agg(rows=("id", "size"), sources=("source_id", "nunique")))

    print("\n== decoding x repetition_penalty (AI, attack=none) ==")
    print(pd.crosstab(ai["decoding"], ai["repetition_penalty"]))

    for (dec, rep), g in ai.groupby(["decoding", "repetition_penalty"]):
        print(f"\n== AI rows, attack=none, decoding={dec}, repetition_penalty={rep} ==")
        print(pd.crosstab(g["domain"], g["model"]))

    print("\n== Shards containing each domain (is the conversion ordered?) ==")
    print(df.groupby("domain")["shard"].agg(lambda s: sorted(s.unique())))
    print("\n== Shards containing each generator ==")
    print(df.groupby("model")["shard"].agg(lambda s: sorted(s.unique())))


if __name__ == "__main__":
    main()
