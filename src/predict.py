"""
predict.py
Shows players with their real (unscaled) stats, predicted tier,
and the network's probability of reaching each tier.
"""

from pathlib import Path
import pandas as pd
import torch

from dataset import load_data, split_by_player, standardize, FEATURE_COLS, NUMERIC_COLS
from model import ProspectNet, CoralNet

MODEL_NAME = "coral.pt"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / MODEL_NAME
SEED = 42    # same split as training, so these are players the model never trained on

LEVEL_NAMES = ["Rookie", "Low-A", "High-A", "Double-A", "Triple-A"]
PROB_COLS = ["P(MLB)", "P(Starter+)", "P(All-Star+)", "P(All-MLB+)", "P(HOF)"]
DISPLAY_COLS = ["name", "season", "age", "level", "position", "k_pct", "iso", "bb_pct", "babip"]

EXPLAIN_Q = 2
TIER_NAMES = ["Never reaches MLB", "Fringe major leaguer", "MLB starter",
              "All-Star caliber", "All-MLB caliber", "Hall of Fame caliber"]
STAT_LABELS = {"age": "age", "level": "level", "k_pct": "strikeout rate",
               "iso": "power (ISO)", "bb_pct": "walk rate", "babip": "BABIP"}


def predict_table(model, raw_df, std_df):
    """
    raw_df: real stats, for display
    std_df: the same rows, standardized, for the model
    returns: raw stats + predicted tier + one probability column per tier question
    """

    x = torch.tensor(std_df[FEATURE_COLS].values, dtype = torch.float32)

    with torch.no_grad():
        probs = torch.sigmoid(model(x))

    table = raw_df[DISPLAY_COLS].copy()

    for k, col in enumerate(PROB_COLS):
        table[col] = probs[:, k].numpy()

    table["pred_tier"] = (probs > 0.5).sum(dim = 1).numpy()

    return table


def latest_season(table):
    """Keep only each player's most recent season."""
    return table.sort_values("season").groupby("name").tail(1)

def stat_effects(model, x_row):
    """
    x_row: one player-season's 14 standardized features, shape [14]
    returns: {stat: effect in percentage points on his All-Star+ chance}
    """
    rows = [x_row]
    for col in NUMERIC_COLS:
        modified = x_row.clone()
        modified[FEATURE_COLS.index(col)] = 0.0
        rows.append(modified)

    batch = torch.stack(rows)                      # shape [7, 14]
    with torch.no_grad():
        probs = torch.sigmoid(model(batch))[:, EXPLAIN_Q]   # 7 All-Star+ chances

    base = probs[0]                                # his real chance
    effects = {}
    for i, col in enumerate(NUMERIC_COLS):
        effects[col] = (base - probs[i + 1]).item() * 100                   # (row 0 is the original, so stat i's modified row is i + 1)

    return effects


def describe(col, raw_row):
    """Readable value for a stat, like '12%' or '.199' or 'Triple-A'."""
    value = raw_row[col]
    if col in ("k_pct", "bb_pct"):
        return f"{value:.0%}"
    if col in ("iso", "babip"):
        return f"{value:.3f}".lstrip("0")
    return str(value)


def explain_player(model, x_row, raw_row, threshold=3.0):
    """One or two plain-English sentences explaining a player's projection."""
    effects = stat_effects(model, x_row)

    strengths = sorted([c for c, e in effects.items() if e >= threshold], 
                       key=lambda c: effects[c], reverse=True)
    concerns = sorted([c for c, e in effects.items() if e <= -threshold], 
                      key=lambda c: effects[c])


    tier = TIER_NAMES[int(raw_row["pred_tier"])]
    text = f"Projected: {tier} ({raw_row['P(All-Star+)']:.0%} All-Star+). "
    if strengths:
        parts = [f"{STAT_LABELS[c]} ({describe(c, raw_row)}, +{effects[c]:.0f} pts)" for c in strengths[:2]]
        text += "Biggest strengths: " + " and ".join(parts) + ". "
    if concerns:
        parts = [f"{STAT_LABELS[c]} ({describe(c, raw_row)}, {effects[c]:.0f} pts)" for c in concerns[:2]]
        text += "Working against him: " + " and ".join(parts) + "."
    return text


def main():
    df = load_data()
    df["name"] = "Player " + df["player_id"].astype(str)    # real names replace this later

    train_df, val_df, test_df = split_by_player(df, seed=SEED)
    _, val_std, _ = standardize(train_df, val_df, test_df)  # val_df stays raw

    model = CoralNet() if MODEL_NAME.startswith("coral") else ProspectNet()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    table = predict_table(model, val_df, val_std)
    table["level"] = table["level"].map(lambda i: LEVEL_NAMES[i])   # 3 -> "Double-A"

    table = latest_season(table)
    table = table.sort_values("P(All-Star+)", ascending=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"Top 15 prospects by All-Star+ probability ({MODEL_PATH.name}):\n")
    print(table.head(15).round(3).to_string(index=False))

    print("\nExplanations for the top 5:\n")
    for idx, row in table.head(5).iterrows():
        x_row = torch.tensor(val_std.loc[idx, FEATURE_COLS].values.astype(float), dtype=torch.float32)
        print(f"{row['name']}: {explain_player(model, x_row, row)}\n")


if __name__ == "__main__":
    main()
