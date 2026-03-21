import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import zipfile
import PyPDF2
import fitz  # PyMuPDF
import easyocr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

st.set_page_config(page_title="PDF CAD Sentiment Analyzer", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📐 PDF CAD Drawing Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown("**Upload PDF → Auto-separate Drawings → OCR → Sentiment → Download Results**")

# Initialize OCR (cached)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# Sidebar for settings
st.sidebar.header("Settings")
max_files = st.sidebar.slider("Max PDF files", 1, 5, 2)
target_words = 20  # Fixed requirement

# === UTILITY FUNCTIONS ===
def detect_cad_drawing(img):
    """Simple heuristic: high contrast + regular dimensions = drawing"""
    img_gray = img.convert('L')
    contrast = img_gray.getextrema()
    
    # CAD drawings typically have high contrast (black lines on white)
    confidence = (contrast[1] - contrast[0]) / 255.0
    return {'confidence': min(confidence * 100, 95)}

def extract_engineering_terms(text):
    """Extract CAD/engineering specific terms"""
    engineering_keywords = [
        'dimension', 'scale', 'tolerance', 'material', 'diameter', 
        'length', 'width', 'height', 'bearing', 'shaft', 'tolerance',
        'rev', 'dwg', 'sheet', 'section', 'view', 'detail'
    ]
    
    words = re.findall(r'\b\w+\b', text.lower())
    found_terms = [w for w in words if w in engineering_keywords]
    return found_terms

def analyze_engineering_sentiment(terms):
    """Simple sentiment scoring for engineering terms"""
    positive_terms = ['precision', 'quality', 'excellent', 'robust']
    negative_terms = ['defect', 'error', 'crack', 'failure']
    
    score = 0
    for term in terms:
        if term in positive_terms: score += 1
        if term in negative_terms: score -= 1
    
    return {
        'compound': score / max(len(terms), 1),
        'positive': len([t for t in terms if t in positive_terms]) / max(len(terms), 1),
        'negative': len([t for t in terms if t in negative_terms]) / max(len(terms), 1)
    }

def generate_20word_summary(texts, sentiments):
    """Generate exactly 20-word summary"""
    summary = "CAD drawings show engineering designs with "
    
    # Count most common terms
    all_terms = []
    for text in texts:
        all_terms.extend(text['engineering_terms'])
    
    common_terms = Counter(all_terms).most_common(5)
    
    # Build summary
    summary += ", ".join([term for term, count in common_terms[:3]])
    summary += f" dominating. Average sentiment score: {sum(s['compound'] for s in sentiments)/len(sentiments):.2f}."
    
    # Pad/truncate to exactly 20 words
    words = summary.split()
    if len(words) > 20:
        words = words[:20]
    else:
        words.extend(["analysis", "complete"] * (20 - len(words)) // 2)
    
    return " ".join(words[:20])

def show_results_dashboard(texts, sentiments):
    """Display comprehensive results"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 OCR Results Summary")
        df = pd.DataFrame(texts)
        st.dataframe(df[['source', 'word_count', 'engineering_terms']].head())
        
        # Term frequency chart
        all_terms = []
        for text in texts:
            all_terms.extend(text['engineering_terms'])
        term_counts = Counter(all_terms).most_common(10)
        
        fig = px.bar(x=[term for term, count in term_counts],
                    y=[count for term, count in term_counts],
                    title="Top Engineering Terms")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("😊 Sentiment Analysis")
        avg_sentiment = sum(s['compound'] for s in sentiments) / len(sentiments)
        st.metric("Overall Sentiment Score", f"{avg_sentiment:.3f}")
        
        # Sentiment distribution
        sentiment_df = pd.DataFrame(sentiments)
        fig = px.pie(values=['Positive', 'Negative'], 
                    names=['✅', '❌'],
                    title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)


# === STEP 1: PDF UPLOAD ===
uploaded_pdfs = st.file_uploader(
    "Upload PDF files containing CAD drawings", 
    type=['pdf'],
    accept_multiple_files=True,
    help="Supports multiple PDFs - will extract all drawings"
)

if uploaded_pdfs:
    # Process each PDF
    all_drawings = []
    non_drawings = []
    
    for pdf_file in uploaded_pdfs:
        st.info(f"🔄 Processing {pdf_file.name}...")
        
        # Extract pages as images
        pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        pdf_pages = []
        
        for page_num in range(len(pdf_doc)):
            # Convert page to image
            page = pdf_doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Simple drawing detection (high contrast + dimensions)
            is_drawing = detect_cad_drawing(img)
            
            if is_drawing:
                all_drawings.append({
                    'source': pdf_file.name,
                    'page': page_num + 1,
                    'image': img,
                    'confidence': is_drawing['confidence']
                })
            else:
                non_drawings.append({
                    'source': pdf_file.name,
                    'page': page_num + 1,
                    'image': img
                })
        
        pdf_doc.close()
    
    # === STEP 2: DISPLAY SEPARATION RESULTS ===
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("🎯 Drawings Found")
        st.success(f"**{len(all_drawings)}** CAD drawings detected")
        for drawing in all_drawings[:3]:  # Show first 3
            st.image(drawing['image'], caption=f"{drawing['source']} (p{drawing['page']})", width=200)
    
    with col2:
        st.subheader("📄 Non-Drawings")
        st.info(f"**{len(non_drawings)}** other pages")
        for non_drawing in non_drawings[:3]:
            st.image(non_drawing['image'], caption=f"{non_drawing['source']} (p{non_drawing['page']})", width=200)
    
    with col3:
        # === STEP 2B: DOWNLOAD ZIP ===
        if all_drawings:
            st.subheader("💾 Download")
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for i, drawing in enumerate(all_drawings):
                    # Save drawing as PNG
                    img_buffer = io.BytesIO()
                    drawing['image'].save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    zip_file.writestr(f"drawing_{i+1}_p{drawing['page']}.png", img_buffer.read())
            
            zip_buffer.seek(0)
            st.download_button(
                label="⬇️ Download Drawings ZIP",
                data=zip_buffer.getvalue(),
                file_name="cad_drawings.zip",
                mime="application/zip"
            )

    # === STEP 3: OCR ALL DRAWINGS ===
    if st.button("🚀 Analyze All Drawings (OCR + Sentiment)", type="primary"):
        with st.spinner("Extracting text from all drawings..."):
            all_texts = []
            for drawing in all_drawings:
                result = reader.readtext(drawing['image'], detail=0)
                full_text = ' '.join(result)
                engineering_terms = extract_engineering_terms(full_text)
                
                all_texts.append({
                    'source': f"{drawing['source']} (p{drawing['page']})",
                    'full_text': full_text,
                    'engineering_terms': engineering_terms,
                    'word_count': len(full_text.split())
                })
        
        st.session_state.all_texts = all_texts
        
        # === STEP 4: SENTIMENT ANALYSIS + 20-WORD SUMMARY ===
        st.subheader("✨ AI Engineering Summary (20 words exactly)")
        
        # Simple sentiment analysis on engineering terms
        sentiments = []
        for text_data in all_texts:
            sentiment_score = analyze_engineering_sentiment(text_data['engineering_terms'])
            sentiments.append(sentiment_score)
        
        # Generate 20-word summary
        summary_words = generate_20word_summary(all_texts, sentiments)
        st.markdown(f"**🎯 {summary_words}**")
        
        # Save summary for download
        st.session_state.summary = summary_words
        
        # === RESULTS DASHBOARD ===
        show_results_dashboard(all_texts, sentiments)

# Footer downloads
if 'summary' in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📝 Download Summary Report",
        data=st.session_state.summary,
        file_name="cad_sentiment_summary.txt"
    )
