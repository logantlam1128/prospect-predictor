"""
lookup.py
Look up any minor league hitter by name and see the model's projection:
tier probabilities for each of his minor league seasons, plus a plain-English
explanation of his most recent one.

Run from the project root:
    python src/lookup.py "Konnor Griffin"
or just  python src/lookup.py  and type a name when asked.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from dataset import load_data, split_by_player, FEATURE_COLS, NUMERIC_COLS, POSITIONS
from model import ProspectNet, CoralNet
from build_dataset import load_raw, combine_stints, build_features, MIN_PA, LEVEL_NAMES
from predict import predict_table, explain_player, DISPLAY_COLS, PROB_COLS

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "coral.pt"
MODEL_PATH = ROOT / "models" / MODEL_NAME
LABELS_FILE = ROOT / "data" / "processed" / "labels.csv"
SEED = 42    # must match training, so the standardization numbers are identical


def training_stats():
    """The mean and std of each numeric feature in the TRAINING set, exactly as used in training."""
    df = load_data()
    train_df, _, _ = split_by_player(df, seed=SEED)
    return train_df[NUMERIC_COLS].mean(), train_df[NUMERIC_COLS].std()


def find_player(rows, query):
    """Search by name. If several players match, let the user pick one. Returns a player ID or None."""
    matches = rows[rows["name"].str.contains(query, case=False, na=False)]
    if matches.empty:
        print(f"No minor league hitter found matching '{query}'.")
        return None

    options = (matches.groupby("player_id")
                      .agg(name=("name", "first"), first=("season", "min"), last=("season", "max"))
                      .reset_index())
    if len(options) == 1:
        return options.loc[0, "player_id"]

    print(f"{len(options)} players match '{query}':")
    for i, opt in options.iterrows():
        print(f"  [{i}] {opt['name']}  ({opt['first']}-{opt['last']})")
    choice = int(input("Enter a number: "))
    return options.loc[choice, "player_id"]


def main():
    query = " ".join(sys.argv[1:]) or input("Player name: ")

    # 1. Build features for every minor league hitter, with the same recipe as training
    rows = build_features(combine_stints(load_raw()))

    # 2. Find the player
    player_id = find_player(rows, query)
    if player_id is None:
        return
    seasons = rows[(rows["player_id"] == player_id) & (rows["pa"] >= MIN_PA)]
    seasons = seasons.sort_values(["season", "level"]).copy()

    # 3. If he's already graduated to the majors, only use his prospect seasons
    labels = pd.read_csv(LABELS_FILE)
    grad = labels.loc[labels["player_id"] == player_id, "graduation_season"]
    if not grad.empty and not np.isnan(grad.iloc[0]):
        grad_year = int(grad.iloc[0])
        print(f"\nNote: he graduated to the majors in {grad_year}; "
              f"using only his minor league seasons through {grad_year}.")
        seasons = seasons[seasons["season"] <= grad_year]

    if seasons.empty:
        print(f"No minor league seasons with at least {MIN_PA} plate appearances to project from.")
        return

    # 4. Same position columns and standardization as training
    for p in POSITIONS:
        seasons[f"pos_{p}"] = (seasons["position"] == p).astype(float)
    for col in DISPLAY_COLS:                     # e.g. tier: unknown for a lookup, so leave it blank
        if col not in seasons.columns:
            seasons[col] = np.nan

    mean, std = training_stats()
    std_rows = seasons.copy()
    std_rows[NUMERIC_COLS] = (std_rows[NUMERIC_COLS] - mean) / std

    # 5. Load the model and predict every season
    n_features = len(FEATURE_COLS)
    model = CoralNet(n_features=n_features) if MODEL_NAME.startswith("coral") else ProspectNet(n_features=n_features)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    table = predict_table(model, seasons, std_rows)
    table["level"] = table["level"].map(lambda i: LEVEL_NAMES[int(i)])

    name = table["name"].iloc[0]
    print(f"\n{name}: projection by minor league season ({MODEL_NAME})\n")
    show = ["season", "age", "level", "position", "k_pct", "iso", "bb_pct"] + PROB_COLS + ["pred_tier"]
    pd.set_option("display.width", 200)
    print(table[show].round(3).to_string(index=False))

    # 6. Explain his most recent season
    latest = table.index[-1]
    x_row = torch.tensor(std_rows.loc[latest, FEATURE_COLS].values.astype(float), dtype=torch.float32)
    print(f"\nMost recent season ({table.loc[latest, 'season']}, {table.loc[latest, 'level']}):")
    print(explain_player(model, x_row, table.loc[latest]))


if __name__ == "__main__":
    main()
