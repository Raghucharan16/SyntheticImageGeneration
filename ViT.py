import torch
from transformers import AutoModelForImageClassification, AutoTokenizer

# Load the pre-trained model
model = AutoModelForImageClassification.from_pretrained("dima806/chest_xray_pneumonia_detection")

# Save the model's state dictionary
torch.save(model.state_dict(), "chest_xray_pneumonia_detection.pth")
