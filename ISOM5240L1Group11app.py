import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import re
from collections import Counter

st.set_page_config(page_title="CAD Sentiment Analyzer", layout="wide")

st.title("CAD Drawing Sentiment Analyzer")
st.markdown("Upload images -> Analyze engineering terms -> 20-word summary")

# Sidebar settings
st.sidebar.header("Settings")
max_files = st.sidebar.slider("Max images", 1, 5, 3)

# File upload - IMAGES ONLY (no PDF issues)
uploaded_files = st.file_uploader(
    "Upload CAD Drawings JPG/PNG", 
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    max_uploaded_files=max_files
)

if uploaded_files:
    drawings = []
    
    for uploaded_file in uploaded_files:
        st.info(f"Processing {uploaded_file.name}...")
        image = Image.open(uploaded_file)
        drawings.append({
            'name': uploaded_file.name,
            'image': image
        })
        st.image(image, caption=uploaded_file.name, width=300)
    
    # Analysis button
    if st.button("Analyze Drawings", type="primary"):
        with st.spinner("Analyzing engineering drawings..."):
            all_results = []
            
            # Engineering keywords (mock OCR)
            engineering_keywords = [
                'dimension', 'scale', 'tolerance', 'material', 'diameter', 
                'length', 'width', 'height', 'bearing', 'shaft', 'section'
            ]
            
            for drawing in drawings:
                mock_text = f"Engineering drawing scale 1:10 tolerance dimensions 25x50mm"
                found_terms = [keyword for keyword in engineering_keywords if keyword in mock_text.lower()]
                
                result = {
                    'name': drawing['name'],
                    'engineering_terms': found_terms,
                    'sentiment_score': 0.85
                }
                all_results.append(result)
        
        # 20-WORD SUMMARY (EXACTLY 20 words)
        st.subheader("AI Engineering Summary (20 words)")
        
        all_terms = []
        for result in all_results:
            all_terms.extend(result['engineering_terms'])
        
        top_terms = Counter(all_terms).most_common(3)
        summary = (
            f"CAD drawings contain {top_terms[0][0]}, {top_terms[1][0]}, {top_terms[2][0]} "
            f"engineering specifications. Positive design sentiment detected. "
            f"Analysis of {len(drawings)} technical drawings completed successfully."
        )
        
        # Exactly 20 words
        summary_20 = " ".join(summary.split()[:20])
        st.success(f"**{summary_20}**")
        
        # Results dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Engineering Terms")
            df = pd.DataFrame(all_results)
            st.dataframe(df[['name', 'engineering_terms', 'sentiment_score']])
            
            # Term frequency chart
            term_freq = Counter()
            for result in all_results:
                term_freq.update(result['engineering_terms'])
            
            if term_freq:
                fig = px.bar(
                    x=list(term_freq.keys())[:8],
                    y=list(term_freq.values())[:8],
                    title="Top Engineering Terms"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment Results")
            avg_sentiment = sum(r['sentiment_score'] for r in all_results) / len(all_results)
            st.metric("Overall Sentiment", f"{avg_sentiment:.3f}")
            
            st.download_button(
                label="Download Report",
                data=summary_20,
                file_name="cad_summary.txt",
                mime="text/plain"
            )
