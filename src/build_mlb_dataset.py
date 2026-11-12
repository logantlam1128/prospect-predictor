"""
build_mlb_dataset.py  (MLB model, Step 1)
One row per player per MLB season: his career SO FAR at that point,
labeled with his final career tier. Only early-career seasons are kept.
Run from the project root:  python src/build_mlb_dataset.py
Output: data/processed/mlb_player_seasons.csv
"""

from pathlib import Path
import pandas as pd
from pybaseball import bwar_bat

ROOT = Path(__file__).resolve().parent.parent
LABELS_FILE = ROOT / "data" / "processed" / "labels.csv"
PROSPECT_FILE = ROOT / "data" / "processed" / "real_player_seasons.csv"
OUT_FILE = ROOT / "data" / "processed" / "mlb_player_seasons.csv"

MAX_MLB_SEASONS = 6    # only snapshot a player's first N MLB seasons


def load_mlb_seasons():
    """
    One row per hitter per MLB season, with mid-season trades combined.
    OPS+ can't simply be added across teams, so it's averaged, weighted by plate appearances.
    """
    war = bwar_bat(return_all=True)
    war = war[war["pitcher"] == "N"].dropna(subset=["mlb_ID"]).copy()
    war["player_id"] = war["mlb_ID"].astype(int)
    for col in ["PA", "WAR", "runs_bat", "runs_defense", "OPS_plus"]:
        war[col] = pd.to_numeric(war[col], errors="coerce").fillna(0)
    war["ops_plus_x_pa"] = war["OPS_plus"] * war["PA"]     # weighted, so it can be summed

    seasons = war.groupby(["player_id", "year_ID"], as_index=False).agg(
        name=("name_common", "first"),
        age=("age", "first"),
        pa=("PA", "sum"),
        war=("WAR", "sum"),
        bat_runs=("runs_bat", "sum"),
        def_runs=("runs_defense", "sum"),
        ops_plus_x_pa=("ops_plus_x_pa", "sum"),
    )
    return seasons[seasons["pa"] > 0]


def add_career_so_far(seasons):
    """
    Adds career-to-date features: each row includes ONLY that season and the ones before it.
    """

    seasons = seasons.sort_values(["player_id", "year_ID"]).copy()
    g = seasons.groupby("player_id")

    seasons["mlb_season_num"]  = g.cumcount() + 1     
    seasons["pa_so_far"] = g["pa"].cumsum()       
    seasons["war_so_far"] = g["war"].cumsum()
    seasons["war_this_season"] = seasons["war"]
    seasons["bat_runs_so_far"] = g["bat_runs"].cumsum()
    seasons["def_runs_so_far"] = g["def_runs"].cumsum()
    seasons["ops_plus_so_far"] = g["ops_plus_x_pa"].cumsum() / seasons["pa_so_far"]

    return seasons


def main():
    labels = pd.read_csv(LABELS_FILE)
    prospect_ids = pd.read_csv(PROSPECT_FILE)["player_id"].unique()

    seasons = load_mlb_seasons()
    seasons = seasons[seasons["player_id"].isin(prospect_ids)]    # players with a prospect profile
    seasons = add_career_so_far(seasons)

    df = seasons.merge(labels[["player_id", "tier", "career_complete"]], on="player_id", how="inner")
    keep = ["player_id", "name", "year_ID", "age", "mlb_season_num", "pa", "pa_so_far",
            "war_this_season", "war_so_far", "ops_plus_so_far", "bat_runs_so_far", "def_runs_so_far",
            "tier", "career_complete"]
    df = df[keep].rename(columns={"year_ID": "season"})

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"Saved {len(df)} MLB snapshots from {df['player_id'].nunique()} players to {OUT_FILE}\n")
    print("Players per tier (complete careers only):")
    print(df[df["career_complete"]].groupby("tier")["player_id"].nunique())
    print("\nAverage after first MLB season, by final tier:")
    first = df[df["mlb_season_num"] == 1]
    print(first.groupby("tier")[["age", "pa", "war_this_season", "ops_plus_so_far"]].mean().round(2))


if __name__ == "__main__":
    main()
