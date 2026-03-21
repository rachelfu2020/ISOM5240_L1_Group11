import streamlit as st
from PIL import Image
import easyocr
import pandas as pd
import plotly.express as px
from ocr_processor import analyze_cad_drawing
from sentiment_analyzer import analyze_engineering_sentiment

st.set_page_config(page_title="CAD Sentiment Analyzer", layout="wide")

st.title("🔍 CAD Drawing Sentiment Analyzer")
st.markdown("Extract text from engineering drawings + analyze engineer sentiment")

# Initialize OCR
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# File upload
uploaded_file = st.file_uploader("Upload CAD Drawing (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Drawing", use_column_width=True)
    
    # OCR Analysis
    with st.spinner("Extracting engineering text..."):
        ocr_result = analyze_cad_drawing(uploaded_file)
    
    # Sentiment Analysis  
    sentiment_result = analyze_engineering_sentiment(ocr_result['full_text'])
    
    # Results Dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Extracted Engineering Data")
        st.json({
            "Texts Found": ocr_result['texts_found'],
            "Title": ocr_result['title'],
            "Scale": ocr_result['scale'],
            "Dimensions": ocr_result['dimensions'],
            "Sample Text": ocr_result['sample_text'][:100]
        })
    
    with col2:
        st.subheader("😊 Sentiment Analysis")
        st.metric("Overall Sentiment", f"{sentiment_result['compound']:.3f}")
        st.metric("Positive %", f"{sentiment_result['pos']:.1%}")
        st.metric("Negative %", f"{sentiment_result['neg']:.1%}")
        
        # Sentiment gauge
        fig = px.line_polar(
            pd.DataFrame([sentiment_result]), 
            r=['neg', 'neu', 'pos'], 
            theta=['Negative', 'Neutral', 'Positive'],
            title="Sentiment Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
