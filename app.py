import streamlit as st
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import requests

# Load the pre-trained model and feature extractor
model = ViTForImageClassification.from_pretrained("nickmuchi/vit-finetuned-chest-xray-pneumonia")
feature_extractor = ViTFeatureExtractor.from_pretrained("nickmuchi/vit-finetuned-chest-xray-pneumonia")

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
