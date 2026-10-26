"""
fake_data.py  (Step 1)
Generates a fake player-season dataset that mimics the real one, so you can
build and debug the neural network before the real data pipeline is done.

Your job: fill in the three TODO functions. check_dataset() is already written
and will tell you if something's off.

Output columns (one row per player per minor league season):
    player_id, season, age, level, position, k_pct, iso, bb_pct, babip,
    tier, career_complete
"""

import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
N_PLAYERS = 1000
SEED = 42

# Probability of each eventual tier (0 = never MLB ... 5 = Hall of Fame caliber).
TIER_PROBS = [0.70, 0.15, 0.09, 0.04, 0.015, 0.005]

LEVELS = ["Rookie", "Low-A", "High-A", "Double-A", "Triple-A"]   # encode as 0-4
POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]

ACTIVE_FRACTION = 0.2   # share of players whose careers aren't finished yet
# ----------------------------------------


def generate_players(n_players, rng):
    """
    Return a DataFrame with ONE row per player:
        player_id, position, tier, career_complete, start_age

    TODO
    - player_id: something unique, like 0..n-1
    - position: random choice from POSITIONS
    - tier: random choice from 0-5 using TIER_PROBS  (hint: rng.choice has a p= argument)
    - career_complete: True for most players, False for about ACTIVE_FRACTION of them
    - start_age: age of first minor league season, roughly 18-23
      (idea: better players often start younger)
    """
    player_ids = np.arange(n_players)
    positions = rng.choice(POSITIONS, size = n_players)
    tier = rng.choice([0, 1, 2, 3, 4, 5], size = n_players,
                       p = TIER_PROBS)
    inactive_player = rng.random(n_players) > ACTIVE_FRACTION

    start_age = 21 - 0.3 * tier + rng.normal(0, 1, size = n_players)
    start_age = np.clip(np.round(start_age), 18, 24).astype(int)

    return pd.DataFrame({
        "player_id": player_ids,
        "position": positions,
        "tier": tier,
        "career_complete": inactive_player,
        "start_age": start_age,
    })
    
def generate_seasons_for_player(player, rng):
    """
    Given one player (a row from generate_players), return a list of dicts,
    one per minor league season.

    TODO
    - Number of seasons: 1-6. Think about whether better players should spend
      fewer seasons in the minors.
    - age goes up by 1 each season, starting at player["start_age"]
    - level usually goes up over time, faster for better tiers, capped at 4
    - Stats should depend on tier + random noise, with lots of overlap between tiers:
        k_pct  (strikeout rate, ~0.12-0.35): LOWER for better tiers
        iso    (power, ~0.05-0.30):          HIGHER for better tiers
        bb_pct (walk rate, ~0.04-0.15):      slightly higher for better tiers
        babip  (~0.25-0.38):                 mostly noise
      hint: something like  base + slope * tier + rng.normal(0, noise)
      then np.clip to keep values realistic
    - Copy player_id, position, tier, career_complete onto every season dict
    - season: a year; pick a starting year and add 1 each season
    """
    tier = player["tier"]
    n_seasons = rng.integers(1, 7)
    first_year = rng.integers(2005, 2016)

    seasons = []
    for i in range(n_seasons):
        age = player["start_age"] + i
        level = min(4, i + tier // 2)

        k_pct = 0.27 - 0.02 * tier + rng.normal(0, 0.04)
        k_pct = np.clip(k_pct, 0.10, 0.40)
        iso = 0.10 + 0.025 * tier + rng.normal(0, 0.04)
        iso = np.clip(iso, 0.02, 0.35)
        bb_pct = 0.07 + 0.007 * tier + rng.normal(0, 0.02)
        bb_pct = np.clip(bb_pct, 0.02, 0.18)
        babip = 0.300 + 0 * tier + rng.normal(0, 0.03)
        babip = np.clip(babip, 0.22, 0.40)

        seasons.append({
            "player_id": player["player_id"],
            "season": first_year + i,
            "age": age,
            "level": level,
            "position": player["position"],
            "k_pct": float(k_pct),
            "iso": float(iso),
            "bb_pct": float(bb_pct),
            "babip": float(babip),
            "tier": tier,
            "career_complete": player["career_complete"],
        })

    return seasons

def generate_dataset(n_players=N_PLAYERS, seed=SEED):
    """
    TODO
    - create rng = np.random.default_rng(seed)  (so results are reproducible)
    - generate players, loop over them, collect all season dicts
    - return one DataFrame of player-seasons
    """
    rng = np.random.default_rng(seed)
    players = generate_players(n_players, rng)
    all_seasons = []
    for _, player in players.iterrows():
        new_seasons = generate_seasons_for_player(player, rng)
        all_seasons.extend(new_seasons)

    return pd.DataFrame(all_seasons)


# ---------- already written: checks your output ----------
def check_dataset(df):
    required = ["player_id", "season", "age", "level", "position", "k_pct",
                "iso", "bb_pct", "babip", "tier", "career_complete"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

    # A player's tier and completion flag must be identical on all his rows.
    per_player = df.groupby("player_id")[["tier", "career_complete"]].nunique()
    assert (per_player <= 1).all().all(), "Some player has different labels across seasons"

    # Ages should increase by exactly 1 within each player.
    age_steps = df.sort_values("season").groupby("player_id")["age"].diff().dropna()
    assert (age_steps == 1).all(), "Ages don't increase by 1 each season"

    assert df["level"].between(0, 4).all(), "Levels must be 0-4"
    assert df["tier"].between(0, 5).all(), "Tiers must be 0-5"

    n_players = df["player_id"].nunique()
    print(f"OK: {len(df)} rows, {n_players} players, "
          f"{len(df) / n_players:.1f} seasons per player on average\n")
    print("Players per tier:")
    print(df.groupby("tier")["player_id"].nunique(), "\n")
    print("Average stats by tier (better tiers should have lower k_pct, higher iso):")
    print(df.groupby("tier")[["k_pct", "iso", "bb_pct", "age", "level"]].mean().round(3))




if __name__ == "__main__":
    from pathlib import Path
    df = generate_dataset()
    check_dataset(df)   
    out_dir = Path(__file__).resolve().parent.parent/ "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "fake_player_seasons.csv", index=False)

