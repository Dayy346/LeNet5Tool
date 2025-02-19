from flask import Flask, jsonify, request, Response
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from flask_cors import CORS # was having an issue where the local host counldnt connect to the server, this was the solution
from flask import send_file
from pathlib import Path
from flask_cors import CORS

is_mnist = False #for default dataset


# Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

CORS(app)  # was having an issue where the local host counldnt connect to the server, this was the solution
# Define the model
class SimpleCNN(nn.Module):
    def __init__(self ):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,6, kernel_size=5, stride=1, padding=2)  # C1 layer #Changed input channels to 3 for RGB and 1 for grayscale
        self.bn1 = nn.BatchNorm2d(6)  # BatchNorm after conv1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                # S2 layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)            # C3 layer
        self.bn2 = nn.BatchNorm2d(16)  # BatchNorm after conv2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                # S4 layer
        self.fc1 = nn.Linear(16 * 6 * 6, 120)                            # C5 fully connected layer
        self.fc2 = nn.Linear(120, 84)                                    # F6 fully connected layer
        self.num_classes = 10 # Number of classes in the dataset
        self.dropout = nn.Dropout(0.1)  # Add dropout, this was added because i noticed that the model was overfitting, the train accuracy imporved significantly but the test accuracy wasnt as good
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
model_path = "./CustomLeNetModel.pth"

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
import shutil
import zipfile

@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    global train_loader, test_loader

    use_grayscale = request.form.get("useGrayscale", "false").lower() == "true"

    # Build transformation pipeline, had it at the end right before trainging but changed due to grayscaling option being added
    # Data augmentation
    if use_grayscale:
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Pad(2),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Pad(2),
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images
            transforms.RandomRotation(15),          # Randomly rotate
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust colors
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for RGB
        ])

    if 'file' in request.files:  # Check if a custom dataset is uploaded
        # Save the uploaded ZIP file
         
        uploaded_file = request.files['file']
        dataset_zip_path = './data/dataset.zip'
        dataset_extract_path = './data/dataset'
        print("Saving uploaded file...")
        uploaded_file.save(dataset_zip_path)

        # Extract the ZIP file
        if os.path.exists(dataset_extract_path):
            shutil.rmtree(dataset_extract_path)  # Remove existing dataset folder
        os.makedirs(dataset_extract_path, exist_ok=True)
        with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_extract_path)

        # Dynamically find train/test folders
        try:
            train_path = os.path.join(dataset_extract_path, 'train')
            test_path = os.path.join(dataset_extract_path, 'test')

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                # Look deeper if 'train' and 'test' aren't directly in root
                for root, dirs, files in os.walk(dataset_extract_path):
                    if 'train' in dirs and 'test' in dirs:
                        train_path = os.path.join(root, 'train')
                        test_path = os.path.join(root, 'test')
                        break

            # Load the dataset using ImageFolder
            
          
            train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
            test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

            # Create DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            # Adjust model for grayscale input
            if use_grayscale:
                model.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # Adjust input channels for grayscale
            else:
                model.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)  # Adjust input channels for RGB

            model.apply(init_weights)  # Reinitialize weights for updated model
            return jsonify({"message": "Custom dataset uploaded and processed successfully!"})
        
            # Start training after successful dataset preparation
            
            
        except Exception as e:
            return jsonify({"error": f"Failed to process dataset: {str(e)}"}), 400

    else:
        global is_mnist
        is_mnist = True  # Set the flag for MNIST
        # Default to MNIST dataset
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Pad(2),
            transforms.Grayscale(num_output_channels=1),  # Default to grayscale for MNIST
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Adjust model for MNIST grayscale
        model.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        model.apply(init_weights)  # Reinitialize weights for updated model
        return jsonify({"message": "No dataset uploaded. Defaulted to MNIST dataset."})

import time  # Import for timing

@app.route('/epoch-time', methods=['POST'])
def epoch_time():
    global train_loader
    if train_loader is None:
        return jsonify({"error": "Dataset not uploaded or processed yet!"}), 400

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    num_epochs = 1  # Only train for one epoch to measure time
    data = request.get_json()
    num_classes = int(data.get("num_classes", 10))  # Default to 10 classes
    print(f"Received num_classes: {num_classes}")  # Log num_classes for debugging

    # Update model's number of classe
    model.num_classes = num_classes
    model.centers = nn.Parameter(torch.randn(model.num_classes, 84).to(device))  # Update centroids
    model.beta = nn.Parameter(torch.randn(model.num_classes).to(device) * 0.1 + 1.0)  # Update scaling factors
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) #L2 regularization

    # Record the start time
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            probabilities, _, _ = model(images)
            loss = confidence_loss(probabilities, labels)
            loss.backward()
            optimizer.step()
        # Record end time after the first epoch
    end_time = time.time()

    epoch_duration = end_time - start_time
    return jsonify({"epoch_time": epoch_duration})


# Endpoint: Train Model
@app.route('/train', methods=['POST'])
def train_model():
    global train_loader, test_loader
    if train_loader is None or test_loader is None:
        return jsonify({"error": "Dataset not uploaded or processed yet!"}), 400

    # Get number of epochs from the request
    data = request.get_json()
    num_epochs = int(data.get("num_epochs", 1))  # Default to 1 epoch
    num_classes = int(data.get("num_classes", 10))  # Default to 10 classes

    # Update model's number of classe
    model.num_classes = num_classes
    model.centers = nn.Parameter(torch.randn(model.num_classes, 84).to(device))  # Update centroids
    model.beta = nn.Parameter(torch.randn(model.num_classes).to(device) * 0.1 + 1.0)  # Update scaling factors

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) #L2 regularization
    train_errors = []  # List to store training errors
    test_errors = []   # List to store test errors
    train_accuracies = []  # Store train accuracies for each epoch
    test_accuracies = []   # Store test accuracies for each epoch
    # Training loop
    for epoch in range(num_epochs):
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
        train_accuracies.append(train_accuracy)
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
        test_accuracies.append(test_accuracy)
        test_error = 1 - (correct_test / total_test)
        test_errors.append(test_error)
        print(f"Epoch {epoch + 1}, Test Accuracy: {test_accuracy:.2f}%")
        # Send progress to frontend
        progress = int((epoch + 1) / num_epochs * 100)
        print(f"Progress: {progress}%")

    torch.save(model.state_dict(), model_path)  # Save the model
    return jsonify({"message": "Training completed!", "train_errors": train_errors, "test_errors": test_errors,
    "train_accuracy": train_accuracies[-1],  # Return the final train accuracy
    "test_accuracy": test_accuracies[-1]    # Return the final test accuracy
    })

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

@app.route('/download-model', methods=['GET']) # Endpoint to download the trained model
def download_model():
    try:
        return send_file(model_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


#|Images test on recent model route
from PIL import Image
# Endpoint: Test a single image
@app.route('/test-image', methods=['POST'])
def test_image():
    global is_mnist

    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No image file selected"}), 400

    # Save the image temporarily
    image_path = "./temp_image.jpg"
    file.save(image_path)

    try:
        # Determine the transformation pipeline
        if is_mnist:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # Ensure grayscale for MNIST
                transforms.Resize((28, 28)),
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            class_names = [str(i) for i in range(10)]  # MNIST class names
        else:
            use_grayscale = request.form.get("useGrayscale", "false").lower() == "true"
            if use_grayscale:
                transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((28, 28)),
                    transforms.Pad(2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.Pad(2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            # Dynamically determine class names
            dataset_root = './data/dataset'
            train_folder = os.path.join(dataset_root, 'train')
            if os.path.exists(train_folder):
                class_names = sorted([d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))])
            else:
                return jsonify({"error": "Training folder not found in dataset."}), 400

        # Preprocess the image
        image = Image.open(image_path)
        image = image.convert("L") if is_mnist or use_grayscale else image.convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Predict
        model.eval()
        with torch.no_grad():
            probabilities, _, _ = model(image)
            predicted_index = torch.argmax(probabilities, dim=1).item()

        # Remove the temporary file
        os.remove(image_path)

        # Validate predicted index
        if predicted_index >= len(class_names):
            return jsonify({"error": "Predicted class index out of bounds."}), 400

        predicted_class_name = class_names[predicted_index]
        return jsonify({"predicted_class": predicted_class_name})

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Ensure it works inside Docker
