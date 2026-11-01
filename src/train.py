from pathlib import Path
import torch
import copy

from dataset import make_loaders
from model import ProspectNet
from targets import make_targets_and_mask, masked_loss, compute_pos_weight

EPOCHS = 100
LEARNING_RATE = 0.001
SEED = 42
USE_CLASS_WEIGHTS = True
PATIENCE = 10

def run_epoch(model, loader, optimizer=None, pos_weights=None):
    total_loss = 0.0

    for x, tier, complete in loader:
        logits = model(x)
        target, mask = make_targets_and_mask(tier, complete)
        loss = masked_loss(logits, target, mask, pos_weights)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    torch.manual_seed(SEED)
    train_loader, val_loader, test_loader = make_loaders(seed = SEED)
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    train_ds = train_loader.dataset
    pos_weights = compute_pos_weight(train_ds.tier, train_ds.complete) if USE_CLASS_WEIGHTS else None
    print("pos_weights:", pos_weights)

    model = ProspectNet()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = run_epoch(model, train_loader, optimizer, pos_weights)

        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, pos_weights=pos_weights)
            
        print(f"Epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs, stopping early at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"Best epoch: {best_epoch} (val loss {best_val_loss:.4f})")

    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    name = "prospect_net_weighted.pt" if USE_CLASS_WEIGHTS else "prospect_net.pt"
    torch.save(model.state_dict(), model_dir / name)
    print("Saved model to", model_dir / name)

if __name__ == "__main__":
    main()

