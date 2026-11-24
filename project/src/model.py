"""Small CNN; train-only normalization is saved inside the model."""
import torch
from torch import nn


class GenreCNN(nn.Module):
    def __init__(self, mean=0.0, std=1.0, classes=8):
        super().__init__()
        self.register_buffer('mean', torch.tensor(float(mean)))
        self.register_buffer('std', torch.tensor(float(std)))
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(64, classes)

    def forward(self, x):
        return self.classifier(self.features((x - self.mean) / self.std.clamp_min(1e-6)))
