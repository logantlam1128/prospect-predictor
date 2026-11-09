from pathlib import Path
import pandas as pd
import torch

from dataset import make_loaders, FEATURE_COLS
from model import ProspectNet, CoralNet
from sklearn.metrics import roc_auc_score, average_precision_score
from targets import make_targets_and_mask

MODEL_NAME = "coral.pt"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / MODEL_NAME
SEED = 42
STAR_TIER = 3

def predict_tiers(logits):
    probs = torch.sigmoid(logits)
    return(probs > 0.5).sum(dim = 1)

def collect_predictions(model, loader):
    preds, trues, completes = [], [], []
    model.eval()
    with torch.no_grad():
        for x, tier, complete in loader:
            logits = model(x)
            pred = predict_tiers(logits)

            preds.append(pred)
            trues.append(tier)
            completes.append(complete)
    return torch.cat(preds), torch.cat(trues), torch.cat(completes)


def star_metics(pred, true):
    true_star = true >= STAR_TIER
    pred_star = pred >= STAR_TIER
    n_hits = (true_star & pred_star).sum().item()
    n_true_stars = true_star.sum().item()
    n_pred_stars = pred_star.sum().item()

    recall = n_hits / max(n_true_stars, 1)
    precision = n_hits / max(n_pred_stars, 1)

    return recall, precision


def ranking_report(model, loader, k=2):
    """How well does the model RANK players on tier question k (default 2 = All-Star+)?"""
    probs, targets, masks = [], [], []
    model.eval()
    with torch.no_grad():
        for x, tier, complete in loader:
            probs.append(torch.sigmoid(model(x))[:, k])
            target, mask = make_targets_and_mask(tier, complete)
            targets.append(target[:, k])
            masks.append(mask[:, k])
    p, y, m = torch.cat(probs), torch.cat(targets), torch.cat(masks)
    known = m == 1
    p, y = p[known].numpy(), y[known].numpy()

    print(f"\nRanking quality (All-Star+, {int(y.sum())} known stars out of {len(y)} rows):")
    print(f"  ROC-AUC:           {roc_auc_score(y, p):.3f}")
    print(f"  Average precision: {average_precision_score(y, p):.3f}   (random guessing = {y.mean():.3f})")

    print("\n  Cutoff | Recall | Precision | Flagged")
    for cutoff in [0.5, 0.3, 0.2, 0.1, 0.05]:
        flagged = p >= cutoff
        hits = (flagged & (y == 1)).sum()
        recall = hits / max(y.sum(), 1)
        precision = hits / max(flagged.sum(), 1)
        print(f"  {cutoff:>6} | {recall:6.1%} | {precision:9.1%} | {flagged.sum():7d}")


def main():
    _, val_loader, _ = make_loaders(seed=SEED)

    n_features = len(FEATURE_COLS)
    model = CoralNet(n_features=n_features) if MODEL_NAME.startswith("coral") else ProspectNet(n_features=n_features)
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Evaluating:", MODEL_NAME)

    pred, true, complete = collect_predictions(model, val_loader)

    # Only grade finished careers: active players' true tier isn't final yet
    keep = complete == 1
    pred, true = pred[keep], true[keep]

    accuracy = (pred == true).float().mean().item()
    mean_tier_error = (pred - true).abs().float().mean().item()
    print(f"Exact tier accuracy: {accuracy:.3f}")
    print(f"Mean tier error:     {mean_tier_error:.3f}")

    recall, precision = star_metics(pred, true)
    print(f"Star recall (tier {STAR_TIER}+):  {recall:.3f}")
    print(f"Star precision (tier {STAR_TIER}+): {precision:.3f}")

    print("\nConfusion matrix (rows = true tier, columns = predicted tier):")
    print(pd.crosstab(true.numpy(), pred.numpy(), rownames=["true"], colnames=["pred"]))
    ranking_report(model, val_loader)


if __name__ == "__main__":
    main()