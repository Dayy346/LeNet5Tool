import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # C1 layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                # S2 layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)            # C3 layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                # S4 layer
        self.fc1 = nn.Linear(16 * 6 * 6, 120)                            # C5 fully connected layer
        self.fc2 = nn.Linear(120, 84)                                    # F6 fully connected layer
        self.num_classes = 10
        self.centers = nn.Parameter(torch.randn(self.num_classes, 84))   # Class centroids
        self.beta = nn.Parameter(torch.randn(self.num_classes) * 0.1 + 1.0)  # Scaling factors

    def forward(self, x):
        # Normalize 
        self.centers.data = nn.functional.normalize(self.centers.data, dim=1)

        # Clamp beta to ensure positivity
        self.beta.data = torch.clamp(self.beta.data, min=1e-3, max=10.0)

        x = 1.7159 * torch.tanh(self.conv1(x) * 2 / 3)
        x = self.pool1(x)
        x = 1.7159 * torch.tanh(self.conv2(x) * 2 / 3)
        x = self.pool2(x)
        x = x.view(-1, 16 * 6 * 6)  # Flatten
        x = 1.7159 * torch.tanh(self.fc1(x) * 2 / 3)
        x = 1.7159 * torch.tanh(self.fc2(x) * 2 / 3)

        # Compute distances and RBF outputs
        dists = torch.cdist(x, self.centers)
        rbf_output = torch.exp(-self.beta * (dists ** 2))
        probabilities = F.softmax(rbf_output, dim=1)
        return probabilities, rbf_output, dists
