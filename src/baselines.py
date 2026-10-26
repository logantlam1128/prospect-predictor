import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from dataset import load_data, split_by_player, standardize, FEATURE_COLS

SEED = 42

def report(name, pred, true):
    accuracy = (pred == true).mean()
    tier_error = np.abs(pred - true).mean()
    print(f"{name:22s} accuracy {accuracy:.3f} | mean tier error {tier_error:.3f}")

def main():
    df = load_data()
    train_df, val_df, test_df = split_by_player(df, seed=SEED)
    train_df, val_df, test_df = standardize(train_df, val_df, test_df)

    train_done = train_df[train_df["career_complete"]]
    val_done = val_df[val_df["career_complete"]]
    true = val_done["tier"].values

    zero_pred = np.zeros(len(true), dtype = int)
    report("Always tier 0", zero_pred, true)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(train_done[FEATURE_COLS], train_done["tier"])
    logreg_pred = logreg.predict(val_done[FEATURE_COLS])
    report("Logistic regression", logreg_pred, true)

    print("\nYour neural network: accuracy 0.754 | mean tier error 0.264")
    print("\nLogistic regression confusion matrix(rows = true, columns = predicted):")
    print(pd.crosstab(true, logreg_pred, rownames=["true"], colnames=["pred"]))

if __name__ == "__main__":
    main()