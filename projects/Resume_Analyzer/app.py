import streamlit as st
import tempfile
import os
from src.ocr_extractor import OCRExtractor
from src.jd_processor import JDProcessor


# ============================================================
# OCR OBJECT
# ============================================================

ocr_extractor = OCRExtractor()
jd_processor = JDProcessor()

# ============================================================
# RESUME PROCESSING FUNCTION
# ============================================================

def process_resume_pdf(uploaded_file):
    
    file_bytes = uploaded_file.getvalue()
    original_file_name = uploaded_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        
        temp_file.write(file_bytes)

        temp_pdf_path = temp_file.name
        
        
        resume_data = ocr_extractor.extract_pdf(temp_pdf_path, original_file_name)
        
        os.remove(temp_pdf_path)

        return resume_data
    
# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

# ============================================================
# TITLE
# ============================================================

st.title("📄 Resume Analyzer")

st.write("Resume matching and candidate analysis using OpenAI and ChromaDB.")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    openai_api_key = st.text_input("OpenAI API Key",type="password")

    embedding_model = st.selectbox("Embedding Model",
        [
            "text-embedding-3-small",
            "text-embedding-3-large"
        ])
    
    top_k = st.slider("Top Matching Results", min_value=1, max_value=10, value=5)
    
    # ============================================================
    # RESUME UPLOAD
    # ============================================================
    
    st.subheader("📁 Upload Resumes")
    resume_files = st.file_uploader("Upload Resume PDFs",
                                    type=["pdf", "docx"], accept_multiple_files=True)
    
st.subheader("📋 Job Description")

jd_input_method = st.radio("Select Job Description Input Method", 
                           ["Upload PDF", "Upload DOCX","Paste Text"],horizontal=True)


if jd_input_method == "Upload PDF":
    jd_pdf = st.file_uploader("Upload Job Description PDF", type=["pdf"])
    
elif jd_input_method == "Upload DOCX":
        jd_docx = st.file_uploader("Upload Job Description DOCX", type=["docx"])
        
elif jd_input_method == "Paste Text":
        jd_pasted_text = st.text_area("Paste Job Description", height=300, placeholder="Paste the Job Description here...")
        
st.divider()

if st.button( "🚀 Parse & Match", use_container_width=True):
    print("back end code start here ")
    
    if not openai_api_key:
        st.error("Please enter your OpenAI API Key.")
        st.stop()
        
    if not resume_files:
        st.error("Please upload at least one resume.")
        st.stop()
        
    if jd_input_method == "Upload PDF":
        if not jd_pdf:
            st.error("Please upload the Job Description PDF.")
            st.stop()
            
    elif jd_input_method == "Upload DOCX":
        if not jd_docx:
            st.error("Please upload the Job Description DOCX.")
            st.stop()
        
    elif jd_input_method == "Paste Text":
        if not jd_pasted_text:
            st.error("Please Paste the Job Description")
            st.stop()
            
    st.success("Input validation successful.")
    
    # ========================================================
    # PROCESS RESUMES
    # ========================================================

    resume_data_map = {}

    for resume_file in resume_files:

        st.write(f"Processing {resume_file.name}...")

        resume_data = process_resume_pdf(resume_file)

        resume_data_map[resume_data["file_name"]] = resume_data

        st.success(f"Processed {resume_file.name}")

        st.json(resume_data)
        
    
    st.info("Processing Job Description...")
    
    if jd_input_method == "Upload PDF":
        
        jd_text = jd_processor.process(input_method="Upload PDF", pdf_file=jd_pdf)
        
    elif jd_input_method == "Upload DOCX":

        jd_text = jd_processor.process(input_method="Upload DOCX", docx_file=jd_docx)
        
    elif jd_input_method == "Paste Text":
        
        jd_text = jd_processor.process(input_method="Paste Text", pasted_text=jd_pasted_text)
        
    st.success("Job Description processed successfully.")
    st.text_area("Extracted Job Description", jd_text, height=300)
    
    