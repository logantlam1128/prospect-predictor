"""
build_labels.py  (Stage 3)
One row per minor league hitter: his career tier, whether his career is complete,
and the season he graduated from rookie status.
Run from the project root:  python src/build_labels.py
Output: data/processed/labels.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from pybaseball import bwar_bat

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "milb_hitting"
OUT_FILE = ROOT / "data" / "processed" / "labels.csv"

# ---------------- SETTINGS ----------------
MLB_ACTIVE_SINCE = 2025     # played in MLB in 2025 or later -> career still active
MILB_ACTIVE_SINCE = 2023    # never reached MLB, but played in the minors 2023+ -> still active
GRADUATION_PA = 150         # ~130 at-bats, MLB's rookie limit

MIN_SEASONS = 2                              # qualifying seasons a tier requires
SEASON_CUTOFFS = {2: 2.0, 3: 4.0, 4: 6.0}    # single-season WAR for starter, All-Star, All-MLB
CATCHER_SCALE = 0.8                          # catchers accumulate WAR more slowly
HOF_CAREER_WAR = {"C": 54.3, "1B": 68.0, "2B": 69.4, "3B": 68.0, "SS": 67.0,
                  "LF": 68.0, "CF": 68.0, "RF": 68.0, "DH": 68.0}
DEFAULT_HOF = 68.0                           # for any position not listed (like "OF")
# ------------------------------------------


def load_milb():
    """All minor league hitter rows, without pitchers or the Mexican League."""
    milb = pd.concat([pd.read_csv(f) for f in RAW_DIR.glob("*.csv") if f.stat().st_size > 1])
    return milb[(milb["position_type"] != "Pitcher") & (milb["league"] != "MEX")]


def player_info(milb, graduation):
    """Most-played position from PRE-graduation seasons, plus last minor league season."""
    grad = milb["player_id"].map(graduation)                      # each row's player's graduation season
    dev = milb[(grad.isna() | (milb["season"] <= grad)) & (milb["position"] != "X")]
    pos = (dev.groupby(["player_id", "position"])["pa"].sum().reset_index()
              .sort_values("pa").groupby("player_id").tail(1)[["player_id", "position"]])
    last = milb.groupby("player_id")["season"].max().rename("last_milb_season").reset_index()
    return last.merge(pos, on="player_id", how="left")


def load_war():
    """One row per hitter per MLB season: WAR and plate appearances, with mid-season trades combined."""
    war = bwar_bat(return_all=True)
    war = war[war["pitcher"] == "N"].dropna(subset=["mlb_ID"])
    war["player_id"] = war["mlb_ID"].astype(int)
    war["WAR"] = pd.to_numeric(war["WAR"], errors="coerce").fillna(0)
    war["PA"] = pd.to_numeric(war["PA"], errors="coerce").fillna(0)
    return war.groupby(["player_id", "year_ID"], as_index=False)[["WAR", "PA"]].sum()


def graduation_season(seasons):
    """
    seasons: ONE player's MLB rows, sorted by year (columns year_ID, PA).
    returns: the first year his running total of MLB plate appearances passes GRADUATION_PA,
             or np.nan if he never passes it.
    """
    running = seasons["PA"].cumsum()
    passed = seasons[running > GRADUATION_PA]
    if passed.empty:
        return np.nan
    return passed["year_ID"].iloc[0]
    

def summarize_mlb(war, positions):
    """One row per MLB hitter: career WAR, last season, graduation season, and qualifying-season counts."""
    rows = []
    for player_id, seasons in war.groupby("player_id"):
        seasons = seasons.sort_values("year_ID")
        scale = CATCHER_SCALE if positions.get(player_id) == "C" else 1.0
        row = {
            "player_id": player_id,
            "career_war": seasons["WAR"].sum(),
            "last_mlb_season": seasons["year_ID"].max(),
            "graduation_season": graduation_season(seasons),
        }
        for tier, cutoff in SEASON_CUTOFFS.items():
            row[f"q{tier}"] = int((seasons["WAR"] >= cutoff * scale).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def assign_tier(row):
    """
    Highest tier the player qualifies for.
    """
    if not row["reached_mlb"]:
        return 0

    tier = 1
    for t in (2, 3, 4):
        if row[f"q{t}"] >= MIN_SEASONS:
            tier = t

    hof_line = HOF_CAREER_WAR.get(row["position"], DEFAULT_HOF)
    if row["career_war"] >= hof_line:
        return 5
    return tier

    
def is_complete(row):
    """
    Is this career finished, so his tier is final?
    """
    if row["tier"] == 5:
        return True
    if row["reached_mlb"]:
        return row["last_mlb_season"] < MLB_ACTIVE_SINCE
    return row["last_milb_season"] < MILB_ACTIVE_SINCE


def main():
    milb = load_milb()
    war = load_war()
    war = war[war["player_id"].isin(milb["player_id"].unique())]    # only players in our minor league data

    mlb = summarize_mlb(war, positions={})
    graduation = mlb.set_index("player_id")["graduation_season"]

    info = player_info(milb, graduation)
    positions = info.set_index("player_id")["position"].to_dict()
    mlb = summarize_mlb(war, positions)

    labels = info.merge(mlb, on="player_id", how="left")
    labels["reached_mlb"] = labels["last_mlb_season"].notna()
    labels["career_war"] = labels["career_war"].fillna(0)
    for t in SEASON_CUTOFFS:
        labels[f"q{t}"] = labels[f"q{t}"].fillna(0)

    labels["tier"] = labels.apply(assign_tier, axis=1)
    labels["career_complete"] = labels.apply(is_complete, axis=1)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUT_FILE, index=False)

    print(f"Saved {len(labels)} players to {OUT_FILE}\n")
    print("Players per tier (all / completed careers):")
    print(pd.DataFrame({
        "all": labels["tier"].value_counts().sort_index(),
        "complete": labels[labels["career_complete"]]["tier"].value_counts().sort_index(),
    }))


if __name__ == "__main__":
    main()
