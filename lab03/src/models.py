"""Prescribed Fashion-MNIST architectures for Lab 03."""
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x):
        return self.network(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10))
    def forward(self, x):
        return self.network(x)

class PatchAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(49, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4,
                    dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.classifier = nn.Linear(64, 10)
    def forward(self, x):
        batch = x.shape[0]
        patches = x.reshape(batch, 1, 4, 7, 4, 7).permute(0, 2, 4, 1, 3, 5)
        tokens = patches.reshape(batch, 16, 49)
        tokens = self.encoder(self.projection(tokens))
        return self.classifier(tokens.mean(dim=1))

FACTORIES = {"MLP": MLP, "CNN": CNN, "Attention": PatchAttention}
