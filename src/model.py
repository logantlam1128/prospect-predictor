import torch
from torch import nn
import torch.nn.functional as F

class ProspectNet(nn.Module):
    def __init__(self, n_features = 14, hidden1 = 64, hidden2 = 32, n_outputs = 5):
        super().__init__()

        self.layer1 = nn.Linear(n_features, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.output = nn.Linear(hidden2, n_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output(x)

        return x

class CoralNet(nn.Module):
    def __init__(self, n_features=14, hidden1=64, hidden2=32, n_outputs=5):
        super().__init__()

        self.layer1 = nn.Linear(n_features, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.relu = nn.ReLU()

        self.score = nn.Linear(hidden2, 1, bias=False)

        self.first_cutoff = nn.Parameter(torch.zeros(1))
        self.gaps = nn.Parameter(torch.zeros(n_outputs - 1))

    def cutoffs(self):
        positive_gaps = F.softplus(self.gaps)
        return torch.cat([self.first_cutoff, self.first_cutoff - torch.cumsum(positive_gaps, dim=0)])

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)

        s = self.score(x)
        return s + self.cutoffs()


if __name__ == "__main__":
    model = ProspectNet()
    print(model)

    fake_batch = torch.randn(64, 14)          # 64 rows of random "features"
    out = model(fake_batch)
    print("Output shape:", out.shape)         # should be torch.Size([64, 5])
    n_params = sum(p.numel() for p in model.parameters())
    print("Learnable parameters:", n_params)  # should be 3205

    coral = CoralNet()
    out = coral(fake_batch)
    print("\nCORAL output shape:", out.shape)
    print("CORAL parameters:", sum(p.numel() for p in coral.parameters()))
    print("Cutoffs:", coral.cutoffs())
    print("Always decreasing:", (out[:, :-1] >= out[:, 1:]).all().item())

