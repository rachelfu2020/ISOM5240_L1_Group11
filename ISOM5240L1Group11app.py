import streamlit as st

st.write("ISOM5240_L1_Group11 Application")

import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import zipfile
import io
import os
import requests
from PIL import Image

# -----------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS (Smallpdf Style)
# -----------------------------------------
st.set_page_config(page_title="PDF Drawing Separator", page_icon="📄", layout="centered")

# Custom CSS to mimic a clean, centralized web-tool layout
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #1E1E1E;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #555555;
        margin-bottom: 30px;
    }
    .upload-container {
        background-color: #F8F9FA;
        border: 2px dashed #007BFF;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# 2. HUGGING FACE MODEL INTEGRATION (Local)
# -----------------------------------------
from transformers import pipeline
import io
from PIL import Image

# Cache the model so it only loads once into memory
@st.cache_resource
def load_classifier():
    st.toast("Loading AI Model into memory... this might take a second.")
    # Using your specific Hugging Face model
    return pipeline("image-classification", model="hanslab37/architectural-classifier-resnet-50")

def classify_page(image_bytes, pipe):
    """
    Step (3): Connect to Hugging Face Model.
    This function converts the page to an image and runs it through the local pipeline.
    """
    try:
        # The pipeline expects a PIL Image, so we convert the bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run inference using the pipeline
        results = pipe(image)
        
        # 'results' is a list of dicts sorted by score, e.g., [{'label': 'drawing', 'score': 0.99}, ...]
        top_prediction_label = str(results[0]['label']).lower()
        
        # --- IMPORTANT LABEL MAPPING ---
        # Update "drawing" or "architectural" to match the exact labels 
        # your specific model outputs when it sees a drawing.
        if "drawing" in top_prediction_label or "architectural" in top_prediction_label or "plan" in top_prediction_label:
            return "Drawing"
        else:
            return "Non-Drawing"
            
    except Exception as e:
        st.error(f"Error querying model: {e}")
        return "Non-Drawing" # Default fallback on error

# -----------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------
def format_size(size_in_bytes):
    """Converts bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} GB"

# -----------------------------------------
# 4. MAIN UI & PROCESS FLOW
# -----------------------------------------
def main():
    # Step (1): Instruction and description
    st.markdown('<div class="main-header">Extract Drawings from PDF</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Instantly identify and separate drawing pages from standard text pages using AI.</div>', unsafe_allow_html=True)

    # Step (2): File Upload
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Please upload PDF in order to identify drawing vs non-drawing", 
        type=["pdf"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        file_name = uploaded_file.name
        original_size_bytes = uploaded_file.size
      
#Scroll down to your main() function. Right after the with st.spinner(...) line, you need to load the pipeline and pass it to your new classification function. Change the inside of your if st.button(...): block to look like this:        

    if st.button("Analyze & Separate PDF", type="primary", use_container_width=True):
                with st.spinner("Analyzing pages with Hugging Face AI... This may take a moment."):
                    
                    # Load the model pipeline (this uses the cached version instantly)
                    hf_pipeline = load_classifier()
                    
                    # Load PDF using PyMuPDF
                    pdf_bytes = uploaded_file.read()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    total_pages = len(doc)
                    
                    # Create empty PDFs for separation
                    drawing_doc = fitz.open()
                    non_drawing_doc = fitz.open()
                    
                    drawing_count = 0
                    non_drawing_count = 0
                    
                    # Step (3) & (4): Process each page
                    for page_num in range(total_pages):
                        page = doc.load_page(page_num)
                        
                        # Convert page to an image
                        pix = page.get_pixmap(dpi=150) 
                        img_bytes = pix.tobytes("jpeg")
                        
                        # Call our new local Hugging Face pipeline function
                        classification = classify_page(img_bytes, hf_pipeline)
                        
                        if classification == "Drawing":
                            drawing_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                            drawing_count += 1
                        else:
                            non_drawing_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                            non_drawing_count += 1
                    
                    # ... (The rest of your code for zipping and Excel stays exactly the same!) ...

                # Save the new PDFs to memory
                drawing_pdf_bytes = drawing_doc.write()
                non_drawing_pdf_bytes = non_drawing_doc.write()
                
                drawing_doc.close()
                non_drawing_doc.close()
                doc.close()

                # Step (5): Output in .zip file
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    if drawing_count > 0:
                        # Placing it in a "Drawing" folder within the zip
                        zip_file.writestr("Drawing/Drawing_Pages.pdf", drawing_pdf_bytes)
                    if non_drawing_count > 0:
                        # Placing it in a "Non-Drawing" folder within the zip
                        zip_file.writestr("Non-Drawing/Non-Drawing_Pages.pdf", non_drawing_pdf_bytes)
                
                zip_buffer.seek(0)
                zip_size_bytes = zip_buffer.getbuffer().nbytes

                # Step (5): Create .xlsx summary
                summary_data = {
                    "Metric": [
                        "Uploaded File Name", 
                        "Total Pages", 
                        "Total Drawing Pages", 
                        "Total Non-Drawing Pages", 
                        "Original PDF Size", 
                        "Zip File Size"
                    ],
                    "Value": [
                        file_name,
                        total_pages,
                        drawing_count,
                        non_drawing_count,
                        format_size(original_size_bytes),
                        format_size(zip_size_bytes)
                    ]
                }
                
                df_summary = pd.DataFrame(summary_data)
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df_summary.to_excel(writer, index=False, sheet_name="Summary")
                excel_buffer.seek(0)

                st.success("✅ Analysis Complete!")

                # Step (6): Download Buttons
                st.markdown("### Download Your Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📦 Download ZIP (Separated PDFs)",
                        data=zip_buffer,
                        file_name=f"Separated_{file_name.replace('.pdf', '')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    
                with col2:
                    st.download_button(
                        label="📊 Download Excel Summary",
                        data=excel_buffer,
                        file_name=f"Summary_{file_name.replace('.pdf', '')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()
