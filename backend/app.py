from flask import Flask, jsonify, request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# Flask app
app = Flask(__name__)

# This was how u defined it so I kept it
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

        # Use clamp to make sure beta is positive
        self.beta.data = torch.clamp(self.beta.data, min=1e-3, max=10.0)  # Keeps values in a range

        x = 1.7159 * torch.tanh(self.conv1(x) * 2 / 3)
        x = self.pool1(x)
        x = 1.7159 * torch.tanh(self.conv2(x) * 2 / 3)
        x = self.pool2(x)
        x = x.view(-1, 16 * 6 * 6)  # Flatten
        x = 1.7159 * torch.tanh(self.fc1(x) * 2 / 3)
        x = 1.7159 * torch.tanh(self.fc2(x) * 2 / 3)

        # Compute Euclidean distance
        dists = torch.cdist(x, self.centers)

        # RBF output aka similarity scores
        rbf_output = torch.exp(-self.beta * (dists ** 2))  # RBF
        probabilities = F.softmax(rbf_output, dim=1)       # Convert to probabilities
        return probabilities, rbf_output, dists

# This is the equation from the paper
def confidence_loss(probabilities, labels):
    true_probs = probabilities[range(probabilities.shape[0]), labels]
    loss = -torch.mean(torch.log(true_probs))
    return loss

# Model and data loaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Automatically utilizes GPU when available
model = SimpleCNN().to(device)
train_loader = None
test_loader = None
model_path = "./LeNet5_1.pth"

# Weight initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        fan_in = m.weight.size(1)  # Number of input connections
        bound = 2.4 / fan_in
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)

# Endpoint: Load Dataset
@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    global train_loader, test_loader
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to match model input
        transforms.Pad(2),           # Padding for consistent dimensions
        transforms.ToTensor()
    ])
    # Default to MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    return jsonify({"message": "No dataset uploaded. Defaulted to MNIST dataset."})

# Endpoint: Train Model
@app.route('/train', methods=['POST'])
def train_model():
    global train_loader, test_loader
    if train_loader is None or test_loader is None:
        return jsonify({"error": "Dataset not uploaded or processed yet!"}), 400

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    train_errors = []  # List to store training errors
    test_errors = []   # List to store test errors
    epochs = 20

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero gradients
            probabilities, _, _ = model(images)
            loss = confidence_loss(probabilities, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = torch.argmax(probabilities, dim=1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_accuracy = 100 * correct_train / total_train
        train_error = 1 - (correct_train / total_train)
        train_errors.append(train_error)
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Evaluation
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                probabilities, _, _ = model(images)
                predicted = torch.argmax(probabilities, dim=1)
                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)

        test_accuracy = 100 * correct_test / total_test
        test_error = 1 - (correct_test / total_test)
        test_errors.append(test_error)
        print(f"Epoch {epoch + 1}, Test Accuracy: {test_accuracy:.2f}%")

    torch.save(model.state_dict(), model_path)  # Save the model
    return jsonify({"message": "Training completed!", "train_errors": train_errors, "test_errors": test_errors})

# Endpoint: Test Model
@app.route('/test', methods=['POST'])
def test_model():
    global test_loader
    if test_loader is None:
        return jsonify({"error": "Dataset not uploaded or processed yet!"}), 400
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            probabilities, _, _ = model(images)
            predicted = torch.argmax(probabilities, dim=1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)
    test_accuracy = 100 * correct_test / total_test
    return jsonify({"message": f"Test completed! Accuracy: {test_accuracy:.2f}%"})

if __name__ == '__main__':
    app.run(debug=True)
