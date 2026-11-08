"""
fetch_milb.py  (Stage 2)
Downloads minor league hitting stats from the MLB Stats API,
one CSV per season and level, saved to data/raw/milb_hitting/.
Run from the project root:  python src/fetch_milb.py
"""

from pathlib import Path
import time
import pandas as pd
import requests

BASE = "https://statsapi.mlb.com/api/v1"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "milb_hitting"

SEASONS = range(1995, 2026)              # 1995 through 2025
SPORT_IDS = [11, 12, 13, 14, 15, 16]     # AAA, AA, High-A, Single-A, (old short-season A?), Rookie
PAUSE = 0.5                              # seconds between requests, to be polite to the API


def fetch_splits(season, sport_id):
    """Ask the API for one level's hitting stats in one season. Returns a list of rows."""
    params = {
        "stats": "season",
        "group": "hitting",
        "playerPool": "All",
        "sportId": sport_id,
        "season": season,
        "limit": 10000,
    }
    response = requests.get(f"{BASE}/stats", params=params, timeout=60)
    response.raise_for_status()          # stop with a clear error if the request failed
    stats = response.json().get("stats", [])
    return stats[0]["splits"] if stats else []


def extract_row(split):
    """Turn one API row (a nested dictionary) into one flat dictionary of the fields we keep."""
    stat = split["stat"]
    position = split.get("position", {})

    return {
        # who, when, where
        "season": int(split["season"]),
        "player_id": split["player"]["id"],
        "name": split["player"]["fullName"],
        "age": stat.get("age"),
        "sport_id": split["sport"]["id"],
        "team": split.get("team", {}).get("name"),
        "league_id": split.get("league", {}).get("id"),
        "league": split.get("league", {}).get("name"),
        "position": position.get("abbreviation"),
        "position_type": position.get("type"),

        # counting stats
        "pa": stat.get("plateAppearances", 0),
        "ab": stat.get("atBats", 0),
        "h": stat.get("hits", 0),
        "doubles": stat.get("doubles", 0),
        "triples": stat.get("triples", 0),
        "hr": stat.get("homeRuns", 0),
        "bb": stat.get("baseOnBalls", 0),
        "so": stat.get("strikeOuts", 0),
        "hbp": stat.get("hitByPitch", 0),
        "sf": stat.get("sacFlies", 0),
        "tb": stat.get("totalBases", 0),
        "sb": stat.get("stolenBases", 0),
    }


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        for sport_id in SPORT_IDS:
            out_file = RAW_DIR / f"{season}_{sport_id}.csv"
            if out_file.exists():
                continue                     # already downloaded: skip it, so reruns pick up where they left off

            splits = fetch_splits(season, sport_id)
            rows = [extract_row(s) for s in splits]
            pd.DataFrame(rows).to_csv(out_file, index=False)
            print(f"{season} sport {sport_id}: {len(rows)} rows")

            time.sleep(PAUSE)


if __name__ == "__main__":
    main()
