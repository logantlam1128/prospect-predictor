"""
predict.py
Shows players with their real (unscaled) stats, predicted tier,
and the network's probability of reaching each tier.
"""

from pathlib import Path
import pandas as pd
import torch

from dataset import load_data, split_by_player, standardize, FEATURE_COLS
from model import ProspectNet

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "prospect_net.pt"
SEED = 42    # same split as training, so these are players the model never trained on

LEVEL_NAMES = ["Rookie", "Low-A", "High-A", "Double-A", "Triple-A"]
PROB_COLS = ["P(MLB)", "P(Starter+)", "P(All-Star+)", "P(All-MLB+)", "P(HOF)"]
DISPLAY_COLS = ["name", "season", "age", "level", "position", "k_pct", "iso", "bb_pct"]


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


def main():
    df = load_data()
    df["name"] = "Player " + df["player_id"].astype(str)    # real names replace this later

    train_df, val_df, test_df = split_by_player(df, seed=SEED)
    _, val_std, _ = standardize(train_df, val_df, test_df)  # val_df stays raw

    model = ProspectNet()
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


if __name__ == "__main__":
    main()
