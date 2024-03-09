from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch

# Load the pre-trained models
model = ViTForImageClassification.from_pretrained("nickmuchi/vit-finetuned-chest-xray-pneumonia")
feature_extractor = ViTFeatureExtractor.from_pretrained("nickmuchi/vit-finetuned-chest-xray-pneumonia")

# Convert the models to PyTorch format
pytorch_model = model.to(torch.device('cpu'))
pytorch_feature_extractor = feature_extractor.to(torch.device('cpu'))

# Save the models as a .pth file
torch.save({
    'model_state_dict': pytorch_model.state_dict(),
    'feature_extractor_state_dict': pytorch_feature_extractor.state_dict(),
    'config': model.config
}, 'vit_model.pth')
