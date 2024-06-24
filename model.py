import os
import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
from torch.utils.data import Dataset
from cjm_pytorch_utils.core import get_torch_device

# Image preprocessing transformation
def get_image_transform(size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

# Custom Dataset class for loading images
class RiceLeafDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.file_list = []
        self.root_dir = root_dir
        self.transform = transform
        self.classes = None
        self._load_file_list()

    def _load_file_list(self):
        self.classes = os.listdir(self.root_dir)
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                self.file_list.append((file_path, self.classes.index(class_name)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_path, label = self.file_list[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Function to load pretrained models and their weights
def load_pretrained_model(model_name, weight_path, nb_classes=4, device=None, dtype=torch.float32):
    model = create_model(model_name, pretrained=True, num_classes=nb_classes).to(device=device, dtype=dtype)
    map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load(weight_path, map_location=map_location))
    model.name = model_name
    return model

# Ensemble model class
class RiceLeafClassifier(nn.Module):
    def __init__(self, model1, model2):
        super(RiceLeafClassifier, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.name = f"{model1.name}_{model2.name}"

    def forward(self, x, weights):
        output1 = self.model1(x)
        output2 = self.model2(x)
        output = output1 * weights[0] + output2 * weights[1]
        return output

# Function to perform inference on a single image
def infer_image(image_path, model, weights, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image, weights)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Main function to setup and run inference
def load_model_and_predict(image_path):
    device = get_torch_device()
    dtype = torch.float32

    # Model and weight paths
    deit_model = 'deit_base_patch16_224.fb_in1k'
    davit_model = 'davit_base.msft_in1k'
    deit_weight_path = 'deit_base_16.pt'
    davit_weight_path = 'davit_base_16.pt'

    # Load models
    deit_base = load_pretrained_model(deit_model, deit_weight_path, device=device, dtype=dtype)
    davit_base = load_pretrained_model(davit_model, davit_weight_path, device=device, dtype=dtype)

    # Initialize ensemble model
    ensemble_model = RiceLeafClassifier(deit_base, davit_base).to(device)

    # Image transformation
    test_transform = get_image_transform()

    # Inference
    weights = [0.5, 0.5]  # Example weights for the ensemble
    predicted_label = infer_image(image_path, ensemble_model, weights, test_transform, device)
    labels = ["Brown Spot","Healthy","Hispa","Leaf Blast"]  
    return labels[predicted_label]
def main():
    parser = argparse.ArgumentParser(description='Rice Leaf Disease Classification')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()
    print(f'Predicted label: {load_model_and_predict(args.image_path)}')

if __name__ == "__main__":
    main()
