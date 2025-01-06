import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# Define the updated SimpleCNN model
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.num_classes = 10
        self.centers = torch.nn.Parameter(torch.randn(self.num_classes, 84))
        self.beta = torch.nn.Parameter(torch.randn(self.num_classes) * 0.1 + 1.0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Compute Euclidean distance
        dists = torch.cdist(x, self.centers)

        # RBF output aka similarity scores
        rbf_output = torch.exp(-self.beta * (dists ** 2))  # RBF
        probabilities = F.softmax(rbf_output, dim=1)  # Convert to probabilities
        return probabilities

# Function to preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
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
        probabilities = model(image)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()

    # Get the class name from the folder structure
    predicted_class_name = class_names[predicted_class_index]
    print(predicted_class_name)
    return predicted_class_name

# Main test function
if __name__ == "__main__":
    # Path to the test image
    image_path = "../test_images/test.JPEG"  # Replace with the actual path to your test image

    # Path to the trained model
    model_path = "./CustomLeNetModel.pth"  # Ensure the model file exists in the backend directory

    # Classes (folder names) - ensure these match the training data structure
    class_names = os.listdir("../CustomTest/train")  # Adjust to your training folder path

    # Predict the class
    if os.path.exists(image_path) and os.path.exists(model_path):
        predict(image_path, model_path, class_names)
    else:
        print("Error: Ensure the image and model paths are correct.")
