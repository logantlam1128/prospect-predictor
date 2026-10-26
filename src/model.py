import torch
from torch import nn

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

if __name__ == "__main__":
    model = ProspectNet()
    print(model)

    fake_batch = torch.randn(64, 14)          # 64 rows of random "features"
    out = model(fake_batch)
    print("Output shape:", out.shape)         # should be torch.Size([64, 5])

    n_params = sum(p.numel() for p in model.parameters())
    print("Learnable parameters:", n_params)  # should be 3205

