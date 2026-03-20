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
BATCH_SIZE = 4  # Reduced slightly to prevent memory issues with two models
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
    
    # 2. Load BLIP VQA Model (Fixed Syntax Here)
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device).eval()
    
    return model, vqa_model, vqa_processor, device

def get_vqa_description(image, question, model, processor, device):
    """Uses BLIP to answer a question about the page image."""
    try:
        # Convert PIL to RGB for BLIP
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
    with torch.no_grad():
        outputs = model(batch_tensor)
        probs = F.softmax(outputs, dim=1)
        _, indices = torch.max(probs, 1)
        confs, _ = torch.max(probs, 1)

    return [CLASS_NAMES[i.item()] for i in indices], [c.item() * 100 for c in confs]

# --- Main App UI ---
st.set_page_config(page_title="AI Drawing Classifier", layout="centered")
st.title("🏗️ PDF Drawing Classifier & AI Describer")

MODEL_PATH = "drawing_classifier.pth"
if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ Model file '{MODEL_PATH}' not found in directory.")
    st.stop()

with st.spinner("Loading AI Models..."):
    classifier, vqa_model, vqa_proc, device = load_models(MODEL_PATH)

user_question = st.text_input("VQA Prompt (Ask the AI about the drawings):", "What kind of architectural plan is this?")
uploaded_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_pdfs and st.button("Process PDFs"):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(os.path.join(output_dir, 'drawings'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'non_drawings'), exist_ok=True)
        
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for f_idx, uploaded_pdf in enumerate(uploaded_pdfs):
            # Save upload to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.getbuffer())
                pdf_path = tmp.name

            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            for start in range(1, total_pages + 1, BATCH_SIZE):
                end = min(start + BATCH_SIZE - 1, total_pages)
                status_text.info(f"Processing {uploaded_pdf.name}: Pages {start}-{end}")
                
                # 1. Convert PDF to Image
                imgs = convert_from_path(pdf_path, dpi=72, first_page=start, last_page=end)
                
                # 2. Classify (ResNet)
                preds, confs = classify_batch(imgs, classifier, device)

                # 3. Handle results
                for i, (pred, conf) in enumerate(zip(preds, confs)):
                    curr_pg = start + i
                    writer = PdfWriter()
                    page_obj = reader.pages[curr_pg - 1]
                    
                    if ROTATION_FIXES[pred] != 0:
                        page_obj.rotate(ROTATION_FIXES[pred])
                    writer.add_page(page_obj)

                    # 4. Describe (BLIP VQA) - only if it's a drawing
                    description = "N/A"
                    if "drawings" in FOLDER_MAPPING[pred]:
                        description = get_vqa_description(imgs[i], user_question, vqa_model, vqa_proc, device)

                    # Save separated page
                    folder = FOLDER_MAPPING[pred]
                    out_name = f"{uploaded_pdf.name}_page_{curr_pg}.pdf"
                    with open(os.path.join(output_dir, folder, out_name), "wb") as f_out:
                        writer.write(f_out)

                    all_results.append({
                        "File": uploaded_pdf.name, 
                        "Page": curr_pg, 
                        "Classification": pred,
                        "AI Description": description,
                        "Confidence (%)": f"{conf:.2f}"
                    })

            progress_bar.progress((f_idx + 1) / len(uploaded_pdfs))

        # --- Generate Report & Zip ---
        if all_results:
            df = pd.DataFrame(all_results)
            report_path = os.path.join(output_dir, 'summary_report.xlsx')
            df.to_excel(report_path, index=False)
            
            zip_base = os.path.join(tempfile.gettempdir(), "processed_files")
            shutil.make_archive(zip_base, 'zip', output_dir)
            
            with open(f"{zip_base}.zip", "rb") as f:
                st.download_button("📦 Download Results (ZIP)", f, "processed_drawings.zip", "application/zip")
            st.success("Analysis Complete!")
