import os
import gc
import tempfile
import shutil
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
import numpy as np
from pypdf import PdfReader, PdfWriter

# --- Configuration & Mappings ---
BATCH_SIZE = 8  # Safe for Streamlit Free Tier

CLASS_NAMES = ['drawings_0', 'drawings_180', 'drawings_270', 'drawings_90', 'non_drawings']

FOLDER_MAPPING = {
    'drawings_0': 'test_drawings',
    'drawings_90': 'test_drawings',
    'drawings_180': 'test_drawings',
    'drawings_270': 'test_drawings',
    'non_drawings': 'test_non_drawings'
}

ROTATION_FIXES = {
    'drawings_0': 0,
    'drawings_90': 270,
    'drawings_180': 180,
    'drawings_270': 90,
    'non_drawings': 0
}

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- Streamlit Caching ---
@st.cache_resource
def load_cached_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device


def setup_directories(base_dir):
    """Creates output directories inside the designated output folder."""
    drawings_dir = os.path.join(base_dir, 'test_drawings')
    non_drawings_dir = os.path.join(base_dir, 'test_non_drawings')
    os.makedirs(drawings_dir, exist_ok=True)
    os.makedirs(non_drawings_dir, exist_ok=True)


def classify_batch(raw_images, model, device):
    tensor_list = []
    for raw_img in raw_images:
        img_array = np.array(raw_img.convert('RGB'))
        darkened_array = np.min(img_array, axis=2)
        darkened_img = Image.fromarray(darkened_array.astype(np.uint8))

        input_tensor = IMAGE_TRANSFORMS(darkened_img)
        tensor_list.append(input_tensor)

    batch_tensor = torch.stack(tensor_list).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predicted_indices = torch.max(probabilities, 1)

    pred_classes = [CLASS_NAMES[idx.item()] for idx in predicted_indices]
    conf_percents = [conf.item() * 100 for conf in confidences]

    return pred_classes, conf_percents


def process_and_save_page(pdf_page, pred_class, base_dir, base_name, page_num):
    degrees_to_fix = ROTATION_FIXES[pred_class]
    if degrees_to_fix != 0:
        pdf_page.rotate(degrees_to_fix)

    pdf_writer = PdfWriter()
    pdf_writer.add_page(pdf_page)

    target_folder_name = FOLDER_MAPPING[pred_class]
    out_folder = os.path.join(base_dir, target_folder_name)
    out_pdf_name = f"{base_name}_page_{page_num}.pdf"
    out_pdf_path = os.path.join(out_folder, out_pdf_name)

    with open(out_pdf_path, "wb") as f_out:
        pdf_writer.write(f_out)

    return target_folder_name, out_pdf_name


def process_single_pdf(pdf_path, filename, base_dir, model, device, log_placeholder):
    base_name = os.path.splitext(filename)[0]
    results = []

    try:
        pdf_reader = PdfReader(pdf_path)
        total_pages = len(pdf_reader.pages)
        log_placeholder.info(f"Processing '{filename}' ({total_pages} pages)...")

        for batch_start_page in range(1, total_pages + 1, BATCH_SIZE):
            batch_end_page = min(batch_start_page + BATCH_SIZE - 1, total_pages)
            
            batch_images = convert_from_path(
                pdf_path, 
                dpi=72, 
                first_page=batch_start_page, 
                last_page=batch_end_page
            )
            
            pred_classes, conf_percents = classify_batch(batch_images, model, device)

            for i, (pred_class, conf_percent) in enumerate(zip(pred_classes, conf_percents)):
                current_page_num = batch_start_page + i
                pdf_page = pdf_reader.pages[current_page_num - 1]

                target_folder, saved_name = process_and_save_page(
                    pdf_page, pred_class, base_dir, base_name, current_page_num
                )

                # Reordered to make "Folder Name" the first column
                results.append({
                    "Folder Name": target_folder,
                    "File Name": filename,
                    "Page Number": current_page_num,
                    "Prediction": pred_class,
                    "Confidence (%)": conf_percent
                })

            del batch_images
            gc.collect()

    except Exception as e:
        log_placeholder.error(f"Error processing {filename}: {e}")

    return results


# --- Streamlit UI ---
st.set_page_config(page_title="Drawing Classifier App", layout="centered")

st.title("🏗️ PDF Construction Drawing Classifier")
st.write("Upload your PDFs, and the AI will split, rotate, and sort them into drawings and non-drawings.")

MODEL_PATH = "drawing_classifier.pth" 

if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ Model file not found! Please ensure '{MODEL_PATH}' is in the same directory as this script.")
    st.stop()

with st.spinner("Loading AI Model..."):
    model, device = load_cached_model(MODEL_PATH)

# File Uploader
uploaded_pdfs = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

# --- NEW: User Upload Status UI ---
if uploaded_pdfs:
    st.success(f"✅ Successfully uploaded {len(uploaded_pdfs)} file(s) ready for processing.")
    with st.expander("Review uploaded files"):
        for pdf in uploaded_pdfs:
            st.write(f"📄 {pdf.name}")

if st.button("Classify and Sort PDFs"):
    if not uploaded_pdfs:
        st.warning("Please upload at least one PDF first.")
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            # --- NEW: Isolate Input and Output folders ---
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            setup_directories(output_dir)
            all_results = []
            
            log_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            # 1. Process each uploaded file
            for idx, uploaded_pdf in enumerate(uploaded_pdfs):
                # Save raw upload to the INPUT folder
                temp_pdf_path = os.path.join(input_dir, uploaded_pdf.name)
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                
                # Run model and save results to the OUTPUT folder
                file_results = process_single_pdf(
                    temp_pdf_path, uploaded_pdf.name, output_dir, model, device, log_placeholder
                )
                all_results.extend(file_results)
                
                progress_bar.progress((idx + 1) / len(uploaded_pdfs))

            # 2. Export Excel to the OUTPUT folder
            if all_results:
                log_placeholder.success("Classification complete! Generating report...")
                
                # Ensure the dataframe columns are strictly in this order
                df = pd.DataFrame(all_results)
                columns_order = ["Folder Name", "File Name", "Page Number", "Prediction", "Confidence (%)"]
                df = df[columns_order]
                
                excel_path = os.path.join(output_dir, 'pdf_classification_results.xlsx')
                
                with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Results')
                    workbook = writer.book
                    worksheet = writer.sheets['Results']
                    decimal_format = workbook.add_format({'num_format': '0.00'})
                    
                    for i, col_name in enumerate(df.columns):
                        max_len = max(df[col_name].astype(str).map(len).max(), len(str(col_name)))
                        adjusted_width = max_len + 2 
                        if col_name == "Confidence (%)":
                            worksheet.set_column(i, i, adjusted_width, decimal_format)
                        else:
                            worksheet.set_column(i, i, adjusted_width)

                # 3. Zip ONLY the OUTPUT folder for download
                log_placeholder.info("Zipping files for download...")
                zip_base_path = os.path.join(tempfile.gettempdir(), "sorted_drawings")
                shutil.make_archive(zip_base_path, 'zip', output_dir)
                zip_file_path = f"{zip_base_path}.zip"

                # 4. Provide Download Button
                st.success("🎉 All files processed and sorted successfully!")
                with open(zip_file_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Sorted PDFs and Excel Report",
                        data=f,
                        file_name="sorted_drawings.zip",
                        mime="application/zip"
                    )
            else:
                log_placeholder.error("No pages were successfully processed.")