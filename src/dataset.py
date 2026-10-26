from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DATA_PATH = Path(__file__).resolve().parent.parent/ "data" / "processed" / "fake_player_seasons.csv"

POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
NUMERIC_COLS = ["age", "level", "k_pct", "iso", "bb_pct", "babip"]
POSITION_COLS = [f"pos_{p}" for p in POSITIONS]
FEATURE_COLS = NUMERIC_COLS + POSITION_COLS

def load_data(path = DATA_PATH):
    df = pd.read_csv(path)

    for p in POSITIONS:
        df[f"pos_{p}"] = (df["position"] == p).astype(float)

    return df

def split_by_player(df, val_frac = 0.15, test_frac = 0.15, seed =42):
    """Split rows into trian/val/test so each player appears in only one set"""
    rng = np.random.default_rng(seed)
    player_ids = df["player_id"].unique()

    player_ids = rng.permutation(player_ids)
    n_test = int(len(player_ids) * test_frac)
    n_val = int(len(player_ids) * val_frac)

    test_ids = player_ids[:n_test]
    val_ids = player_ids[n_test:n_test + n_val]
    train_ids = player_ids[n_test + n_val:]

    test_df = df[df["player_id"].isin(test_ids)]
    val_df = df[df["player_id"].isin(val_ids)]
    train_df = df[df["player_id"].isin(train_ids)]

    return train_df, val_df, test_df

def standardize(train_df, val_df, test_df):
    mean = train_df[NUMERIC_COLS].mean()
    standard = train_df[NUMERIC_COLS].std()

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[NUMERIC_COLS] = (train_df[NUMERIC_COLS] - mean) / standard
    val_df[NUMERIC_COLS] = (val_df[NUMERIC_COLS] - mean) / standard
    test_df[NUMERIC_COLS] = (test_df[NUMERIC_COLS] - mean) / standard

    return train_df, val_df, test_df

class ProspectDataset(Dataset):
    def __init__(self, df):
        self.x = torch.tensor(df[FEATURE_COLS].values, dtype = torch.float32)
        self.tier = torch.tensor(df["tier"].values, dtype = torch.long)
        self.complete = torch.tensor(df["career_complete"].values.astype(float), dtype = torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.tier[idx], self.complete[idx]

def make_loaders(batch_size=64, seed=42):
    df = load_data()
    train_df, val_df, test_df = split_by_player(df, seed = seed)
    train_df, val_df, test_df = standardize(train_df, val_df, test_df)

    train_ds = ProspectDataset(train_df)
    val_ds = ProspectDataset(val_df)
    test_ds = ProspectDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    df = load_data()
    print(df[FEATURE_COLS].head())
    print("Number of features:", len(FEATURE_COLS))
    print("Position columns sum to 1 on every row:", (df[POSITION_COLS].sum(axis=1) == 1).all())

    train_df, val_df, test_df = split_by_player(df)
    print("Players:", train_df["player_id"].nunique(), val_df["player_id"].nunique(),
          test_df["player_id"].nunique())
    overlap = set(train_df["player_id"]) & (set(val_df["player_id"]) | set(test_df["player_id"]))
    print("Players in more than one set:", len(overlap))
 
    train_df, val_df, test_df = standardize(train_df, val_df, test_df)
    print("\nTraining set after standardizing (should be mean 0, std 1):")
    print(train_df[NUMERIC_COLS].describe().loc[["mean", "std"]].round(2))

    train_loader, val_loader, test_loader = make_loaders()
    x, tier, complete = next(iter(train_loader))
    print("\nOne batch:")
    print("x:", x.shape, x.dtype)                 # torch.Size([64, 14]) torch.float32
    print("tier:", tier.shape, tier.dtype)        # torch.Size([64]) torch.int64
    print("complete:", complete.shape, complete.dtype)
    print("Batches per epoch:", len(train_loader))
    