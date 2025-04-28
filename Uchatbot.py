import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st
from model import LungClassifier  # Import the LungClassifier from model.py
import os

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model_path = "C:/Users/Dhayanithi/Desktop/Computer Vision/lung_model.pth"  # Adjust path to your model
model = LungClassifier(num_classes=3).to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Set custom CSS for a more creative and visually appealing UI
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
        color: #333;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .header {
        text-align: center;
        font-size: 3em;
        color: #6c757d;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 1.7em;
        color: #495057;
        margin-top: 10px;
        text-align: center;
    }
    .info-text {
        font-size: 1.2em;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .input-field {
        margin-bottom: 30px;
    }
    .upload-btn {
        background-color: #007bff;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        text-align: center;
    }
    .upload-btn:hover {
        background-color: #0056b3;
    }
    .result-text {
        font-size: 1.5em;
        font-weight: bold;
        color: #28a745;
        text-align: center;
        margin-top: 20px;
    }
    .error-text {
        color: #dc3545;
        font-size: 1.2em;
        text-align: center;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #777;
        margin-top: 40px;
    }
    .container {
        display: flex;
        justify-content: center;
        flex-direction: column;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# Welcome message with styled header
st.markdown('<div class="header">Lung Cancer Detection Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Welcome to the Lung Cancer Detection Assistant</div>', unsafe_allow_html=True)
st.write("Hello! I will guide you through the process to assess your condition and provide you with a suggestion.")

# Step 1: Collect personal details (name, age)
with st.form(key='personal_details_form'):
    name = st.text_input("What's your name?", key="name", placeholder="Enter your name...", label_visibility="visible", help="Enter your name here.")
    age = st.number_input("How old are you?", min_value=1, key="age", label_visibility="visible")
    submit_button = st.form_submit_button(label="Proceed")

if submit_button and name and age:
    st.write(f"Hello {name}, age {age}! Let's get started.")

    # Step 2: Ask about symptoms
    symptoms = st.text_area("Please describe your symptoms (e.g., coughing, chest pain, shortness of breath, etc.)", key="symptoms", placeholder="Describe your symptoms here...", label_visibility="visible")

    if symptoms:
        st.write("Thank you for sharing your symptoms. Now, let's proceed to the next step.")

        # Step 3: Upload an image
        uploaded_file = st.file_uploader("Upload an X-ray or CT scan image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Open image for prediction
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Step 4: Predict using the lung classifier
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                class_names = ["Benign", "Malignant", "Normal"]  # Class names for prediction
                prediction = class_names[predicted.item()]

            # Step 5: Display prediction result
            st.markdown(f'<div class="result-text">Prediction: {prediction}</div>', unsafe_allow_html=True)

            # Step 6: Suggestions based on prediction
            if prediction == "Malignant":
                st.write("Based on the prediction, it seems you may have a malignant condition. We recommend that you meet a doctor immediately for further consultation and possibly a biopsy.")
            elif prediction == "Benign":
                st.write("It seems like the condition might be benign. However, we still suggest you follow up with a healthcare professional for further guidance.")
            else:
                st.write("The condition seems normal. However, if you're experiencing persistent symptoms, we recommend checking with a healthcare provider.")

        else:
            st.markdown('<div class="error-text">Please upload an image to continue.</div>', unsafe_allow_html=True)

# Footer with contact or disclaimer message
st.markdown('<div class="footer">Disclaimer: This tool provides predictions based on machine learning models and should not replace professional medical consultation. Always consult a healthcare provider for medical advice.</div>', unsafe_allow_html=True)
