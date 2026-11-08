"""
build_dataset.py  (Stage 4)
Turns raw minor league stats + career labels into the training file:
one row per player per level per season, same columns as the fake data.
Run from the project root:  python src/build_dataset.py
Output: data/processed/real_player_seasons.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "milb_hitting"
LABELS_FILE = ROOT / "data" / "processed" / "labels.csv"
OUT_FILE = ROOT / "data" / "processed" / "real_player_seasons.csv"

MIN_PA = 100
# sport ID -> level number, lowest to highest
LEVEL_MAP = {16: 0, 15: 1, 14: 2, 13: 3, 12: 4, 11: 5}
LEVEL_NAMES = ["Rookie", "Short-season A", "Single-A", "High-A", "Double-A", "Triple-A"]
COUNT_COLS = ["pa", "ab", "h", "doubles", "triples", "hr", "bb", "so", "hbp", "sf", "tb", "sb"]


def load_raw():
    """All raw minor league rows, minus pitchers and the Mexican League."""
    raw = pd.concat([pd.read_csv(f) for f in RAW_DIR.glob("*.csv") if f.stat().st_size > 1])
    return raw[(raw["position_type"] != "Pitcher") & (raw["league"] != "MEX")]


def combine_stints(raw):
    """
    A player traded within the same level and season has one row per team.
    Combine them into ONE row: counting stats added up, and the other details
    (name, age, league, position) taken from the stint with the most plate appearances.
    """
    keys = ["player_id", "season", "sport_id"]
    counts = raw.groupby(keys, as_index=False)[COUNT_COLS].sum()
    details = (raw.sort_values("pa", ascending=False)
                  .groupby(keys, as_index=False).first()[keys + ["name", "age", "league", "position"]])
    return counts.merge(details, on=keys)


def build_features(df):
    """
    Adds the model's features to combined player-season rows.
    Reusable: the player lookup will call this same function later.
    """
    df = df.copy()

    df["k_pct"] = df["so"] / df["pa"]
    df["bb_pct"] = df["bb"] / df["pa"]
    df["iso"] = (df["tb"] - df["h"]) / df["ab"]
    df["babip"] = (df["h"] - df["hr"]) / (df["ab"] - df["so"] - df["hr"] + df["sf"])

    df["level"] = df["sport_id"].map(LEVEL_MAP)

    level_avg_age = df.groupby(["season", "sport_id"])["age"].transform("mean")
    df["age_rel"] = df["age"] - level_avg_age

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=["k_pct", "bb_pct", "iso", "babip", "level", "age"])


def apply_filters(df, labels):
    """
    Keep only real prospect seasons: pre-graduation, with enough plate appearances.
    """
    grad = df["player_id"].map(labels.set_index("player_id")["graduation_season"])
    is_prospect_season = grad.isna() | (df["season"] <= grad)
    enough_pa = df["pa"] >= MIN_PA
    return df[is_prospect_season & enough_pa]

    
def main():
    labels = pd.read_csv(LABELS_FILE)

    df = combine_stints(load_raw())
    df = build_features(df)
    df = apply_filters(df, labels)

    df = df.merge(labels[["player_id", "tier", "career_complete"]], on="player_id", how="inner")

    keep = ["player_id", "name", "season", "age", "age_rel", "level", "league", "position", "pa",
            "k_pct", "iso", "bb_pct", "babip", "tier", "career_complete"]
    df = df[keep].sort_values(["player_id", "season", "level"])

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"Saved {len(df)} player-seasons from {df['player_id'].nunique()} players to {OUT_FILE}\n")
    print("Player-seasons per level:")
    print(df["level"].map(lambda i: LEVEL_NAMES[int(i)]).value_counts())
    print("\nPlayers per tier (complete careers only):")
    print(df[df["career_complete"]].groupby("tier")["player_id"].nunique())
    print("\nAverage stats by tier:")
    print(df.groupby("tier")[["k_pct", "iso", "bb_pct", "babip", "age_rel", "level"]].mean().round(3))


if __name__ == "__main__":
    main()
