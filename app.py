import streamlit as st
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import requests
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Load the saved models
checkpoint = torch.load('vit_model.pth', map_location='cpu')
model_config = checkpoint['config']

# Initialize the models
model = ViTForImageClassification(model_config)
feature_extractor = ViTFeatureExtractor(model_config)

# Load the state dictionaries
model.load_state_dict(checkpoint['model_state_dict'])
feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])

# Set the models to evaluation mode
model.eval()
feature_extractor.eval()

# Define class labels
class_labels = ['Normal', 'Pneumonia']

# Function to make predictions
def predict(image):
    # Ensure image has three dimensions
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Make prediction
    outputs = model(**inputs)
    predicted_class = class_labels[outputs.logits.argmax().item()]
    return predicted_class

# Main function
def main():
    st.title("Synthetic Image Generation  using GAN's and Chest X-ray Pneumonia Detection")

    # Upload image
    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        result = predict(image)
        st.write('Prediction:', result)

if __name__ == "__main__":
    main()
