import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F


# Define the SimpleCNN model (same as in app.py)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)  # C1 layer
        self.bn1 = nn.BatchNorm2d(6)  # BatchNorm after conv1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                # S2 layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)            # C3 layer
        self.bn2 = nn.BatchNorm2d(16)  # BatchNorm after conv2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                # S4 layer
        self.fc1 = nn.Linear(16 * 6 * 6, 120)                            # C5 fully connected layer
        self.fc2 = nn.Linear(120, 84)                                    # F6 fully connected layer
        self.num_classes = 5
        self.dropout = nn.Dropout(0.1)  # Add dropout to reduce overfitting
        self.centers = nn.Parameter(torch.randn(self.num_classes, 84))   # Class centroids
        self.beta = nn.Parameter(torch.randn(self.num_classes) * 0.1 + 1.0)  # Scaling factors

    def forward(self, x):
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


# Function to preprocess the image
def preprocess_image(image_path):
 # Data augmentation
    transform = transforms.Compose([
                transforms.Resize((32, 32)),
                # transforms.Grayscale(num_output_channels=1), # Convert to grayscale
                transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images
                transforms.RandomRotation(15),          # Randomly rotate
                # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust colors
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
                # transforms.Normalize(mean=[0.5], std=[0.5])  # Adjusted for grayscale images
            ])
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels (RGB)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Function to load the model and test on an image
def predict(image_path, model_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess the image
    image = preprocess_image(image_path).to(device)

    # Predict the class
    with torch.no_grad():
        probabilities, _, _ = model(image)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()

    # Get the class name from the folder structure
    predicted_class_name = class_names[predicted_class_index]
    print(predicted_class_name)
    return predicted_class_name


# Main test function
if __name__ == "__main__":
    # Path to the test image
    image_path = "../test_images/test.jpg"  # Replace with the actual path to your test image

    # Path to the trained model
    model_path = "./CustomLeNetModel.pth"  # Ensure the model file exists in the backend directory

    # Classes (folder names) - ensure these match the training data structure
    class_names = os.listdir("data/dataset/train")  # Adjust to your training folder path

    # Predict the class
    if os.path.exists(image_path) and os.path.exists(model_path):
        predict(image_path, model_path, class_names)
    else:
        print("Error: Ensure the image and model paths are correct.")
