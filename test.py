import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import LungClassifier  # Assuming your model is saved as 'model.py'
import os

# Define the device for GPU or CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = LungClassifier(num_classes=3).to(DEVICE)
model.load_state_dict(torch.load("lung_model.pth"))  # Load the trained weights

# Define the transforms (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Check if the LIDC_Y_Net folder exists
test_folder = "LIDC_Y-Net"  # Main folder containing 'benign', 'malignant', and 'normal'

if not os.path.exists(test_folder):
    print(f"Dataset folder not found at {test_folder}")
else:
    # Load the dataset from 'LIDC_Y_Net' directory
    test_dataset = datasets.ImageFolder(test_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Set model to evaluation mode
    model.eval()

    # Initialize variables for accuracy calculation
    correct = 0
    total = 0

    # Testing loop
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Predicted class
            _, predicted = torch.max(outputs, 1)

            # Calculate the number of correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
