# import streamlit as st

# st.write("ISOM5240_L1_Group11 Application")

# import streamlit as st
# import fitz  # PyMuPDF
# import pandas as pd
# import zipfile
# import io
# import os
# import requests
# from PIL import Image

# # -----------------------------------------
# # 1. PAGE CONFIGURATION & CUSTOM CSS (Smallpdf Style)
# # -----------------------------------------
# st.set_page_config(page_title="PDF Drawing Separator", page_icon="📄", layout="centered")

# # Custom CSS to mimic a clean, centralized web-tool layout
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: 700;
#         text-align: center;
#         color: #1E1E1E;
#         margin-bottom: 0px;
#     }
#     .sub-header {
#         font-size: 1.2rem;
#         text-align: center;
#         color: #555555;
#         margin-bottom: 30px;
#     }
#     /* Hide default Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     </style>
# """, unsafe_allow_html=True)

# # -----------------------------------------
# # 2. HUGGING FACE MODEL INTEGRATION (Local)
# # -----------------------------------------
# from transformers import pipeline
# import io
# from PIL import Image

# # Cache the model so it only loads once into memory
# @st.cache_resource
# def load_classifier():
#     # Using your specific Hugging Face model
#     return pipeline("image-classification", model="hanslab37/architectural-classifier-resnet-50")

# def classify_page(image_bytes, pipe):
#     """
#     Step (3): Connect to Hugging Face Model.
#     This function converts the page to an image and runs it through the local pipeline.
#     """
#     try:
#         # The pipeline expects a PIL Image, so we convert the bytes
#         image = Image.open(io.BytesIO(image_bytes))
        
#         # Run inference using the pipeline
#         results = pipe(image)
        
#         # 'results' is a list of dicts sorted by score, e.g., [{'label': 'drawing', 'score': 0.99}, ...]
#         top_prediction_label = str(results[0]['label']).lower()
        
#         # --- IMPORTANT LABEL MAPPING ---
#         # Update "drawing" or "architectural" to match the exact labels 
#         # your specific model outputs when it sees a drawing.
#         if "drawing" in top_prediction_label or "architectural" in top_prediction_label or "plan" in top_prediction_label:
#             return "Drawing"
#         else:
#             return "Non-Drawing"
            
#     except Exception as e:
#         st.error(f"Error querying model: {e}")
#         return "Non-Drawing" # Default fallback on error
        

# # -----------------------------------------
# # 3. HELPER FUNCTIONS
# # -----------------------------------------
# def format_size(size_in_bytes):
#     """Converts bytes to a human-readable format."""
#     for unit in ['B', 'KB', 'MB']:
#         if size_in_bytes < 1024.0:
#             return f"{size_in_bytes:.2f} {unit}"
#         size_in_bytes /= 1024.0
#     return f"{size_in_bytes:.2f} GB"


# # -----------------------------------------
# # 4. MAIN UI & PROCESS FLOW
# # -----------------------------------------
# def main():
#     # --- Initialize Session State Memory ---
#     if 'processed' not in st.session_state:
#         st.session_state.processed = False
#     if 'zip_buffer' not in st.session_state:
#         st.session_state.zip_buffer = None
#     if 'excel_buffer' not in st.session_state:
#         st.session_state.excel_buffer = None
#     if 'current_file' not in st.session_state:
#         st.session_state.current_file = ""

#     # Step (1): Instruction and description
#     st.markdown('<div class="main-header">Extract Drawings from PDF</div>', unsafe_allow_html=True)
#     st.markdown('<div class="sub-header">Instantly identify and separate drawing pages from standard text pages using AI.</div>', unsafe_allow_html=True)

#     # Step (2): File Upload (Removed the custom dashed box!)
#     uploaded_file = st.file_uploader(
#         "Please upload PDF in order to identify drawing vs non-drawing", 
#         type=["pdf"]
#     )

#     if uploaded_file is not None:
#         file_name = uploaded_file.name
#         original_size_bytes = uploaded_file.size
        
#         # Reset memory if the user uploads a brand new file
#         if st.session_state.current_file != file_name:
#             st.session_state.processed = False
#             st.session_state.current_file = file_name
        
#         if st.button("Analyze & Separate PDF", type="primary", use_container_width=True):
#             with st.spinner("Analyzing pages with Hugging Face AI... This may take a moment."):
                
#                 hf_pipeline = load_classifier()
                
#                 pdf_bytes = uploaded_file.read()
#                 doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#                 total_pages = len(doc)
                
#                 drawing_doc = fitz.open()
#                 non_drawing_doc = fitz.open()
                
#                 drawing_count = 0
#                 non_drawing_count = 0
                
#                 # Step (3) & (4): Process each page
#                 for page_num in range(total_pages):
#                     page = doc.load_page(page_num)
#                     pix = page.get_pixmap(dpi=150) 
#                     img_bytes = pix.tobytes("jpeg")
                    
#                     classification = classify_page(img_bytes, hf_pipeline)
                    
#                     if classification == "Drawing":
#                         drawing_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
#                         drawing_count += 1
#                     else:
#                         non_drawing_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
#                         non_drawing_count += 1

#                 # Save the new PDFs to memory (with safety checks)
#                 if drawing_count > 0:
#                     drawing_pdf_bytes = drawing_doc.write()
#                 else:
#                     drawing_pdf_bytes = None
                    
#                 if non_drawing_count > 0:
#                     non_drawing_pdf_bytes = non_drawing_doc.write()
#                 else:
#                     non_drawing_pdf_bytes = None
                
#                 drawing_doc.close()
#                 non_drawing_doc.close()
#                 doc.close()

#                 # Step (5): Output in .zip file
#                 zip_buffer = io.BytesIO()
#                 with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
#                     if drawing_count > 0:
#                         zip_file.writestr("Drawing/Drawing_Pages.pdf", drawing_pdf_bytes)
#                     if non_drawing_count > 0:
#                         zip_file.writestr("Non-Drawing/Non-Drawing_Pages.pdf", non_drawing_pdf_bytes)
                
#                 zip_buffer.seek(0)
#                 zip_size_bytes = zip_buffer.getbuffer().nbytes

#                 # Step (5): Create .xlsx summary
#                 summary_data = {
#                     "Metric": [
#                         "Uploaded File Name", "Total Pages", "Total Drawing Pages", 
#                         "Total Non-Drawing Pages", "Original PDF Size", "Zip File Size"
#                     ],
#                     "Value": [
#                         file_name, total_pages, drawing_count, non_drawing_count,
#                         format_size(original_size_bytes), format_size(zip_size_bytes)
#                     ]
#                 }
                
#                 df_summary = pd.DataFrame(summary_data)
#                 excel_buffer = io.BytesIO()
#                 with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
#                     df_summary.to_excel(writer, index=False, sheet_name="Summary")
#                 excel_buffer.seek(0)

#                 # --- Save to Session State Memory ---
#                 st.session_state.zip_buffer = zip_buffer
#                 st.session_state.excel_buffer = excel_buffer
#                 st.session_state.processed = True

#         # --- Display Results Outside the Button Block ---
#         # This ensures the buttons stay on screen even after clicking one!
#         if st.session_state.processed:
#             st.success("✅ Analysis Complete!")
#             st.markdown("### Download Your Results")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.download_button(
#                     label="📦 Download ZIP (Separated PDFs)",
#                     data=st.session_state.zip_buffer,
#                     file_name=f"Separated_{file_name.replace('.pdf', '')}.zip",
#                     mime="application/zip",
#                     use_container_width=True
#                 )
                
#             with col2:
#                 st.download_button(
#                     label="📊 Download Excel Summary",
#                     data=st.session_state.excel_buffer,
#                     file_name=f"Summary_{file_name.replace('.pdf', '')}.xlsx",
#                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#                     use_container_width=True
#                 )

# if __name__ == "__main__":
#     main()

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

# UPDATED: Changed folder names
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
    # UPDATED: Changed folder names
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


# --- Streamlit UI & Session State ---
st.set_page_config(page_title="Drawing Classifier App", layout="centered")

# Initialize an ignore list in session state for our custom remove buttons
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

# File Uploader
uploaded_pdfs = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

valid_files = []

if uploaded_pdfs:
    # Cleanup session state: if a user clears the uploader completely, clear the ignore list
    current_upload_names = [f.name for f in uploaded_pdfs]
    st.session_state.ignored_files = [f for f in st.session_state.ignored_files if f in current_upload_names]
    
    # Filter the uploaded files to exclude anything the user clicked "Remove" on
    valid_files = [f for f in uploaded_pdfs if f.name not in st.session_state.ignored_files]

    if valid_files:
        st.success(f"✅ {len(valid_files)} file(s) ready for processing.")
        with st.expander("Review and Manage Uploaded Files", expanded=False):
            for pdf in valid_files:
                col1, col2 = st.columns([0.8, 0.2])
                col1.write(f"📄 {pdf.name}")
                # NEW: A custom Remove button for each file
                if col2.button("❌ Remove", key=f"remove_{pdf.name}"):
                    st.session_state.ignored_files.append(pdf.name)
                    st.rerun() # Instantly refreshes the app to hide the removed file
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
            
            log_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            # Use valid_files instead of the raw uploaded_pdfs
            for idx, uploaded_pdf in enumerate(valid_files):
                temp_pdf_path = os.path.join(input_dir, uploaded_pdf.name)
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                
                file_results = process_single_pdf(
                    temp_pdf_path, uploaded_pdf.name, output_dir, model, device, log_placeholder
                )
                all_results.extend(file_results)
                
                progress_bar.progress((idx + 1) / len(valid_files))

            if all_results:
                log_placeholder.success("Classification complete! Generating report...")
                
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

                log_placeholder.info("Zipping files for download...")
                zip_base_path = os.path.join(tempfile.gettempdir(), "sorted_drawings")
                shutil.make_archive(zip_base_path, 'zip', output_dir)
                zip_file_path = f"{zip_base_path}.zip"

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