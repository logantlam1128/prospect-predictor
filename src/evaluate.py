from pathlib import Path
import pandas as pd
import torch

from dataset import make_loaders
from model import ProspectNet

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "prospect_net_weighted.pt"
SEED = 42

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

def main():
    _, val_loader, _ = make_loaders(seed=SEED)

    model = ProspectNet()
    model.load_state_dict(torch.load(MODEL_PATH))

    pred, true, complete = collect_predictions(model, val_loader)

    # Only grade finished careers: active players' true tier isn't final yet
    keep = complete == 1
    pred, true = pred[keep], true[keep]

    accuracy = (pred == true).float().mean().item()
    mean_tier_error = (pred - true).abs().float().mean().item()
    print(f"Exact tier accuracy: {accuracy:.3f}")
    print(f"Mean tier error:     {mean_tier_error:.3f}")

    print("\nConfusion matrix (rows = true tier, columns = predicted tier):")
    print(pd.crosstab(true.numpy(), pred.numpy(), rownames=["true"], colnames=["pred"]))


if __name__ == "__main__":
    main()