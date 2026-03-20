import os
import gc
import tempfile
import shutil
import time
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from transformers import BlipProcessor, BlipForQuestionAnswering 
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
import numpy as np
from pypdf import PdfReader, PdfWriter

# --- Configuration ---
BATCH_SIZE = 4 
CLASS_NAMES = ['drawings_0', 'drawings_180', 'drawings_270', 'drawings_90', 'non_drawings']
FOLDER_MAPPING = {k: ('drawings' if 'drawings' in k else 'non_drawings') for k in CLASS_NAMES}
ROTATION_FIXES = {'drawings_0': 0, 'drawings_90': 270, 'drawings_180': 180, 'drawings_270': 90, 'non_drawings': 0}

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Model Loading ---
@st.cache_resource
def load_models(classifier_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load ResNet Classifier
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    model.to(device).eval()
    
    # 2. Load BLIP VQA Model
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device).eval()
    
    return model, vqa_model, vqa_processor, device

def get_vqa_description(image, question, model, processor, device):
    """Uses BLIP to answer a question about the page image."""
    try:
        inputs = processor(image.convert("RGB"), question, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"Description Error: {e}"

def classify_batch(raw_images, model, device):
    tensor_list = []
    for raw_img in raw_images:
        img_array = np.array(raw_img.convert('RGB'))
        darkened_img = Image.fromarray(np.min(img_array, axis=2).astype(np.uint8))
        tensor_list.append(IMAGE_TRANSFORMS(darkened_img))

    batch_tensor = torch.stack(tensor_list).to(device)
