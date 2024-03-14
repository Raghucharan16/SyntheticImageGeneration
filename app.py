import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import functional as F
from transformers import ViTForImageClassification

# Load the model from the local .pth file
model_path = "chest_xray_pneumonia_detection.pth"

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("dima806/chest_xray_pneumonia_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/chest_xray_pneumonia_detection")

model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # Load the model state dictionary
# model = ViTForImageClassification(num_labels=2)  # Initialize the model
model.load_state_dict(model_state_dict)  # Load the state dictionary into the model

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define label names
class_names = ["NORMAL", "PNEUMONIA"]

# Define prediction function
def predict(image):
    # Preprocess image
    image = F.resize(image, (224, 224))
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = image.unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return class_names[predicted_class]

# Streamlit app
def main():
    st.title("Chest X-ray Pneumonia Detection")
    st.sidebar.title("About")

    st.sidebar.info(
        "This app detects pneumonia in chest X-ray images using a pre-trained deep learning model."
    )

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform prediction
        prediction = predict(image)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
