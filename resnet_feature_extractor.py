# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 00:37:28 2023

@author: saina
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(images_path, batch_size=64):
    """
    Extract features from the images using a pretrained ResNet18 model.
    :param images_path: Path to the folder containing the images.
    :param batch_size: Batch size for processing images.
    :return: A dictionary containing the extracted features with image filenames as keys.
    """
    # Set up image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the images using ImageFolder
    image_dataset = ImageFolder(images_path, transform=preprocess)

    # Create a DataLoader for the images
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load the pretrained ResNet18 model and remove the last fully connected layer
    model = resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Extract features from the images
    features = {}
    with torch.no_grad():
        for inputs, _, filenames in image_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1).cpu().numpy()

            for output, filename in zip(outputs, filenames):
                features[filename] = output

    return features
