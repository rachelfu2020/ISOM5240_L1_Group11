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
BATCH_SIZE = 2 # Lowered to ensure stability during VQA generation
CLASS_NAMES = ['drawings_0', 'drawings_180', 'drawings_270', 'drawings_90', 'non_drawings']
FOLDER_MAPPING = {k: ('drawings' if 'drawings' in k else 'non_drawings') for k in CLASS_NAMES}
ROTATION_FIXES = {'drawings_0': 0, 'drawings_90': 270, 'drawings_180': 180, 'drawings_270': 90, 'non_drawings': 0}

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_models(classifier_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    model.to(device).eval()
    
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device).eval()
    return model, vqa_model, vqa_processor, device

def get_vqa_description(image, question, model, processor, device):
    try:
        inputs = processor(image.convert("RGB"), question, return_tensors="pt").to(device)
        with torch.no_grad():
            # Adjusting generation config for better descriptions
            out = model.generate(**inputs, max_new_tokens=80, min_length=20, num_beams=3)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"Description Error: {e}"

def classify_batch(raw_images, model, device):
    tensor_list = [IMAGE_TRANSFORMS(Image.fromarray(np.min(np.array(img.convert('RGB')), axis=2).astype(np.uint8))) for img in raw_images]
    batch_tensor = torch.stack(tensor_list).to(device)
    with torch.no_grad():
        outputs = model(batch_tensor)
        probs = F.softmax(outputs, dim=1)
        confs, indices = torch.max(probs, 1)
    return [CLASS_NAMES[i.item()] for i in indices], [c.item() * 100 for c in confs]

# --- UI Setup ---
st.set_page_config(page_title="AI PDF Reader", layout="wide")
st.title("🏗️ AI Construction Document Reader")

MODEL_PATH = "drawing_classifier.pth"
if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ Model file '{MODEL_PATH}' not found.")
    st.stop()

classifier, vqa_model, vqa_proc, device = load_models(MODEL_PATH)

# --- 1. Upload ---
uploaded_pdfs = st.file_uploader("Upload PDF Files (e.g., your 22-page set)", type=["pdf"], accept_multiple_files=True)

if uploaded_pdfs:
    st.divider()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Config & Preview")
        user_question = st.text_area("What should the AI describe or look for?", "Describe the main architectural elements and notes on this page.", height=100)
        
        with st.expander("Preview Page 1"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_prev:
                tmp_prev.write(uploaded_pdfs[0].getbuffer())
                preview_img = convert_from_path(tmp_prev.name, dpi=60, first_page=1, last_page=1)[0]
                st.image(preview_img, use_container_width=True)

    # --- 2. Processing ---
    if st.button("🔍 Analyze & Describe All Pages", type="primary", use_container_width=True):
        with col2:
            st.subheader("AI Live Feed")
            log_container = st.container(border=True)
            progress_bar = st.progress(0)
            
        all_results = []
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(os.path.join(output_dir, 'drawings'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'non_drawings'), exist_ok=True)

            for f_idx, uploaded_pdf in enumerate(uploaded_pdfs):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.getbuffer())
                    pdf_path = tmp.name

                reader = PdfReader(pdf_path)
                total_pages = len(reader.pages)
                
                for start in range(1, total_pages + 1, BATCH_SIZE):
                    end = min(
