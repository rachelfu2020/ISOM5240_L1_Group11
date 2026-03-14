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
from pdf2image import convert_from_path
from PIL import Image
import pandas as pd
import numpy as np
from pypdf import PdfReader, PdfWriter

# --- Configuration & Mappings ---
BATCH_SIZE = 8  

CLASS_NAMES = ['drawings_0', 'drawings_180', 'drawings_270', 'drawings_90', 'non_drawings']

FOLDER_MAPPING = {
    'drawings_0': 'drawings',
    'drawings_90': 'drawings',
    'drawings_180': 'drawings',
    'drawings_270': 'drawings',
    'non_drawings': 'non_drawings'
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

# --- Funny Quotes List ---
FUNNY_QUOTES = [
    "🛌 喺邊度跌倒，就喺邊度攤唞下！",
    "😌 努力唔一定會成功，但唔努力一定會好舒服。",
    "🛡️ 只要我夠廢，就冇人可以利用到我。",
    "🐟 做人如果冇夢想，就可以無憂無慮。",
    "🤦‍♂️ 今日解決唔到嘅事唔緊要，因為聽日你都一樣係解決唔到。",
    "🐢 搏一搏，單車變摩托；縮一縮，舒舒服服。",
    "🧘‍♂️ 與其提升自己，不如接受自己。",
    "🥲 趁後生捱多啲苦，咁你大個嘅時候就可以再捱多啲苦。",
    "🦥 聽日嘅事聽日先算，聽日搞唔掂都仲有後日。"
]


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
    drawings_dir = os.path.join(base_dir, 'drawings')
    non_drawings_dir = os.path.join(base_dir, 'non_drawings')
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


def process_single_pdf(pdf_path, filename, base_dir, model, device, log_placeholder, state_tracker):
    base_name = os.path.splitext(filename)[0]
    results = []

    try:
        pdf_reader = PdfReader(pdf_path)
        total_pages = len(pdf_reader.pages)

        for batch_start_page in range(1, total_pages + 1, BATCH_SIZE):
            
            # --- Sequential Quote Logic using original placeholder ---
            current_time = time.time()
            if current_time - state_tracker['last_quote_time'] > 5.0:
                current_quote_idx = state_tracker['quote_index']
                log_placeholder.info(FUNNY_QUOTES[current_quote_idx])
                
                state_tracker['last_quote_time'] = current_time
                state_tracker['quote_index'] = (current_quote_idx + 1) % len(FUNNY_QUOTES)

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
        st.error(f"Error processing {filename}: {e}")

    return results


# --- Streamlit UI & Session State ---
st.set_page_config(page_title="Drawing Classifier App", layout="centered")

if "ignored_files" not in st.session_state:
    st.session_state.ignored_files = []

st.title("🏗️ PDF Construction Drawing Classifier")
st.write("Upload your PDFs, and the AI will split, rotate, and sort them into drawings and non-drawings.")

MODEL_PATH = "drawing_classifier.pth" 

if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ Model file not found! Please ensure '{MODEL_PATH}' is in the same directory as this script.")
    st.stop()

with st.spinner("Loading AI Model..."):
    model, device = load_cached_model(MODEL_PATH)

uploaded_pdfs = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

valid_files = []

if uploaded_pdfs:
    current_upload_names = [f.name for f in uploaded_pdfs]
    st.session_state.ignored_files = [f for f in st.session_state.ignored_files if f in current_upload_names]
    
    valid_files = [f for f in uploaded_pdfs if f.name not in st.session_state.ignored_files]

    if valid_files:
        st.success(f"✅ {len(valid_files)} file(s) ready for processing.")
        
        with st.expander("Review and Manage Uploaded Files", expanded=False):
            for pdf in valid_files:
                col1, col2 = st.columns([0.8, 0.2])
                col1.write(f"📄 {pdf.name}")
                if col2.button("❌ Remove", key=f"remove_{pdf.name}"):
                    st.session_state.ignored_files.append(pdf.name)
                    st.rerun() 
    else:
        st.info("All uploaded files have been removed from the queue. Upload more to continue.")

if st.button("Classify and Sort PDFs"):
    if not valid_files:
        st.warning("Please upload and keep at least one PDF first.")
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            setup_directories(output_dir)
            all_results = []
            
            # --- Clean UI Elements for Processing ---
            log_placeholder = st.empty()
            
            # Show the first quote immediately and set the tracker
            log_placeholder.info(FUNNY_QUOTES[0])
            state_tracker = {
                'last_quote_time': time.time(),
                'quote_index': 1 
            }
            
            progress_bar = st.progress(0)
            
            for idx, uploaded_pdf in enumerate(valid_files):
                temp_pdf_path = os.path.join(input_dir, uploaded_pdf.name)
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                
                file_results = process_single_pdf(
                    temp_pdf_path, uploaded_pdf.name, output_dir, model, device, log_placeholder, state_tracker
                )
                all_results.extend(file_results)
                
                progress_bar.progress((idx + 1) / len(valid_files))

            if all_results:
                log_placeholder.success("🎉 Classification complete! Generating report...")
                
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

                zip_base_path = os.path.join(tempfile.gettempdir(), "sorted_drawings")
                shutil.make_archive(zip_base_path, 'zip', output_dir)
                zip_file_path = f"{zip_base_path}.zip"

                with open(zip_file_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Sorted PDFs and Excel Report",
                        data=f,
                        file_name="sorted_drawings.zip",
                        mime="application/zip"
                    )
            else:
                log_placeholder.error("No pages were successfully processed.")