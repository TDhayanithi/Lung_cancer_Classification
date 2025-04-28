import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import LungClassifier  # Import your model class

# Load the trained model
model = LungClassifier(num_classes=3)
model.load_state_dict(torch.load("lung_model.pth"))  # Load the saved model
model.eval()  # Set the model to evaluation mode

# Class names
classes = ['Benign', 'Malignant', 'Normal']

# Streamlit app title
st.title("Lung Cancer Classification")

# Instructions
st.write("Upload a lung image to predict whether it's Benign, Malignant, or Normal.")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction function
def predict(image):
    # Define the transformations to match model input size and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]

# Check if user uploaded an image
if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    
    # Make prediction
    result = predict(image)
    
    # Display the result
    st.write(f"Prediction: {result}")
