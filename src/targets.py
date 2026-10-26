import torch
from torch import nn
import torch.nn.functional as F

N_THRESHOLDS = 5 # question: above tier 0, 1, 2, 3, 4?
bce = nn.BCEWithLogitsLoss(reduction="none")

def make_targets_and_mask(tier, complete):
    k = torch.arange(N_THRESHOLDS)

    target = (tier.unsqueeze(1) > k).float()
    mask = torch.maximum(complete.unsqueeze(1), target)

    return target, mask

def masked_loss(logits, target, mask, pos_weight=None):
    per_cell = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    masked = per_cell * mask

    return masked.sum() / mask.sum().clamp(min = 1)

def compute_pos_weight(tier, complete, soften=True):
    target, mask = make_targets_and_mask(tier, complete)
    n_yes = (target * mask).sum(dim=0)
    n_no = ((1 - target) * mask).sum(dim=0)

    weight = n_no / n_yes.clamp(min=1)
    if soften:
        weight = weight.sqrt()

    return weight.float()

if __name__ == "__main__":
    tier = torch.tensor([0, 3, 5, 2, 0])
    complete = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    target, mask = make_targets_and_mask(tier, complete)
    print("target:\n", target)
    print("mask:\n", mask)

    logits = torch.zeros(5, 5)
    print("loss:", masked_loss(logits, target, mask).item())
    
    from dataset import load_data, split_by_player
    df = load_data()
    train_df, _, _ = split_by_player(df, seed=42)
    tier_all = torch.tensor(train_df["tier"].values)
    complete_all = torch.tensor(train_df["career_complete"].values.astype(float))
    print("pos_weight:", compute_pos_weight(tier_all, complete_all))

