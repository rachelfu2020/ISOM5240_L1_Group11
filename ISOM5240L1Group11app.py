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
from transformers import BlipProcessor, BlipForQuestionAnswering # New Imports
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
import numpy as np
from pypdf import PdfReader, PdfWriter

# --- Configuration ---
BATCH_SIZE = 8  
CLASS_NAMES = ['drawings_0', 'drawings_180', 'drawings_270', 'drawings_90', 'non_drawings']
FOLDER_MAPPING = {k: ('drawings' if 'drawings' in k else 'non_drawings') for k in CLASS_NAMES}
ROTATION_FIXES = {'drawings_0': 0, 'drawings_90': 270, 'drawings_180': 180, 'drawings_270': 90, 'non_drawings': 0}

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

FUNNY_QUOTES = ["🛌 喺邊度跌倒，就喺邊度攤唞下！", "😌 努力唔一定會成功，但唔努力一定會好舒服。", "🛡️ 只要我夠廢，就冇人可以利用到我。"]

# --- New Caching for BLIP & ResNet ---
@st.cache_resource
def load_models(classifier_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load ResNet Classifier
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    model.to(device).eval()
    
    # 2. Load BLIP VQA Model
    vqa_processor = BlipProcessor.from_remote_code=True.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device).eval()
    
    return model, vqa_model, vqa_processor, device

def get_vqa_description(image, question, model, processor, device):
    """Uses BLIP to answer a question about the page image."""
    try:
        inputs = processor(image, question, return_tensors="pt").to(device)
        out = model.generate(**inputs)
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
    with torch.no_grad():
        outputs = model(batch_tensor)
        probs = F.softmax(outputs, dim=1)
        confidences, indices = torch.max(probs, 1)

    return [CLASS_NAMES[i.item()] for i in indices], [c.item() * 100 for c in confidences]

# --- Main App Logic ---
st.set_page_config(page_title="AI PDF Drawing Describer", layout="centered")

st.title("🏗️ Smart PDF Classifier & Describer")
st.info("This version uses **BLIP-VQA** to read your drawings and describe them.")

# VQA Settings
user_question = st.text_input("What should the AI look for in the drawings?", "Describe this architectural drawing in detail")

MODEL_PATH = "drawing_classifier.pth"
if not os.path.exists(MODEL_PATH):
    st.error("⚠️ drawing_classifier.pth not found!")
    st.stop()

with st.spinner("Initializing AI Models (this may take a minute)..."):
    classifier, vqa_model, vqa_proc, device = load_models(MODEL_PATH)

uploaded_pdfs = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

if uploaded_pdfs and st.button("Analyze, Sort & Describe"):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(os.path.join(output_dir, 'drawings'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'non_drawings'), exist_ok=True)
        
        all_results = []
        log_placeholder = st.empty()
        progress_bar = st.progress(0)

        for f_idx, uploaded_pdf in enumerate(uploaded_pdfs):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.getbuffer())
                pdf_path = tmp.name

            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            for start in range(1, total_pages + 1, BATCH_SIZE):
                log_placeholder.info(f"Processing {uploaded_pdf.name}: Page {start}/{total_pages}...")
                end = min(start + BATCH_SIZE - 1, total_pages)
                
                # Convert PDF to Image for AI
                imgs = convert_from_path(pdf_path, dpi=72, first_page=start, last_page=end)
                preds, confs = classify_batch(imgs, classifier, device)

                for i, (pred, conf) in enumerate(zip(preds, confs)):
                    curr_pg = start + i
                    # 1. Classification & Rotation
                    writer = PdfWriter()
                    page_obj = reader.pages[curr_pg - 1]
                    if ROTATION_FIXES[pred] != 0:
                        page_obj.rotate(ROTATION_FIXES[pred])
                    writer.add_page(page_obj)

                    # 2. VQA Description (Optional: Only describe drawings to save time)
                    description = "N/A"
                    if "drawings" in FOLDER_MAPPING[pred]:
                        description = get_vqa_description(imgs[i], user_question, vqa_model, vqa_proc, device)

                    # 3. Save individual page
                    folder = FOLDER_MAPPING[pred]
                    out_name = f"{uploaded_pdf.name}_p{curr_pg}.pdf"
                    with open(os.path.join(output_dir, folder, out_name), "wb") as f_out:
                        writer.write(f_out)

                    all_results.append({
                        "Folder": folder, "File": uploaded_pdf.name, "Page": curr_pg, 
                        "AI Description": description, "Confidence (%)": conf
                    })

            progress_bar.progress((f_idx + 1) / len(uploaded_pdfs))

        # --- Report & Download ---
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_excel(os.path.join(output_dir, 'report.xlsx'), index=False)
            
            zip_path = os.path.join(tempfile.gettempdir(), "ai_results")
            shutil.make_archive(zip_path, 'zip', output_dir)
            
            with open(f"{zip_path}.zip", "rb") as f:
                st.download_button("⬇️ Download Sorted PDFs & AI Descriptions", f, "ai_processed_drawings.zip")
            st.success("Complete!")
