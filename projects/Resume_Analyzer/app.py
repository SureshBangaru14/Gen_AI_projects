import streamlit as st

from document_process import DocumentProcess


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📄 Resume Analyzer")

st.write(
    "Resume matching and candidate analysis using OpenAI and ChromaDB."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    # --------------------------------------------------------
    # OPENAI API KEY
    # --------------------------------------------------------

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password"
    )

    # --------------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------------

    embedding_model = st.selectbox(
        "Embedding Model",
        [
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]
    )

    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    top_k = st.slider(
        "Top Matching Results",
        min_value=1,
        max_value=10,
        value=5
    )

    # --------------------------------------------------------
    # RESUME UPLOAD
    # --------------------------------------------------------

    st.subheader("📁 Upload Resumes")

    resume_files = st.file_uploader(
        "Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    # --------------------------------------------------------
    # DISPLAY SELECTED RESUMES
    # --------------------------------------------------------

    if resume_files:

        st.write("### Selected Resumes")

        for resume_file in resume_files:

            st.write(
                f"✓ {resume_file.name}"
            )


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.subheader("📋 Job Description")


# ============================================================
# JD INPUT METHOD
# ============================================================

jd_input_method = st.radio(
    "Select Job Description Input Method",
    [
        "Upload PDF",
        "Upload DOCX",
        "Paste Text"
    ],
    horizontal=True
)


# ============================================================
# INITIALIZE JD VARIABLES
# ============================================================

jd_pdf = None

jd_docx = None

jd_pasted_text = ""


# ============================================================
# JD PDF
# ============================================================

if jd_input_method == "Upload PDF":

    jd_pdf = st.file_uploader(
        "Upload Job Description PDF",
        type=["pdf"]
    )


# ============================================================
# JD DOCX
# ============================================================

elif jd_input_method == "Upload DOCX":

    jd_docx = st.file_uploader(
        "Upload Job Description DOCX",
        type=["docx"]
    )


# ============================================================
# JD PASTE TEXT
# ============================================================

elif jd_input_method == "Paste Text":

    jd_pasted_text = st.text_area(
        "Paste Job Description",
        height=300,
        placeholder="Paste the Job Description here..."
    )


# ============================================================
# DIVIDER
# ============================================================

st.divider()


# ============================================================
# PARSE & MATCH
# ============================================================

if st.button(
    "🚀 Parse & Match",
    use_container_width=True
):

    print(
        "Backend processing started..."
    )


    # ========================================================
    # VALIDATE OPENAI API KEY
    # ========================================================

    if not openai_api_key:

        st.error(
            "Please enter your OpenAI API Key."
        )

        st.stop()


    # ========================================================
    # VALIDATE RESUMES
    # ========================================================

    if not resume_files:

        st.error(
            "Please upload at least one resume PDF."
        )

        st.stop()


    # ========================================================
    # VALIDATE JD
    # ========================================================

    if jd_input_method == "Upload PDF":

        if not jd_pdf:

            st.error(
                "Please upload the Job Description PDF."
            )

            st.stop()

        jd_file = jd_pdf


    elif jd_input_method == "Upload DOCX":

        if not jd_docx:

            st.error(
                "Please upload the Job Description DOCX."
            )

            st.stop()

        jd_file = jd_docx


    elif jd_input_method == "Paste Text":

        if not jd_pasted_text.strip():

            st.error(
                "Please paste the Job Description."
            )

            st.stop()

        jd_file = jd_pasted_text


    # ========================================================
    # VALIDATION SUCCESS
    # ========================================================

    st.success(
        "Input validation successful."
    )


    # ========================================================
    # CREATE DOCUMENT PROCESS
    # ========================================================

    document_process = DocumentProcess(
        resume_files=resume_files,
        jd_input_method=jd_input_method,
        jd_file_name=jd_file
    )


    # ========================================================
    # PROCESS RESUMES
    # ========================================================

    st.info(
        "Processing Resume PDFs with OCR..."
    )


    try:

        resume_data = (
            document_process.process_resume()
        )

    except Exception as e:

        st.error(
            f"Failed to process resumes: {e}"
        )

        st.stop()


    # ========================================================
    # RESUME SUCCESS
    # ========================================================

    st.success(
        f"{len(resume_files)} resume(s) processed successfully."
    )


    # ========================================================
    # DISPLAY OCR DATA
    # ========================================================

    st.subheader(
        "📄 Extracted Resume OCR Data"
    )

    st.json(
        resume_data
    )


    # ========================================================
    # PROCESS JOB DESCRIPTION
    # ========================================================

    st.info(
        "Processing Job Description..."
    )


    try:

        jd_text = (
            document_process.process_jd()
        )

    except Exception as e:

        st.error(
            f"Failed to process Job Description: {e}"
        )

        st.stop()


    # ========================================================
    # JD SUCCESS
    # ========================================================

    st.success(
        "Job Description processed successfully."
    )


    # ========================================================
    # DISPLAY JD
    # ========================================================

    st.subheader(
        "📋 Extracted Job Description"
    )


    st.text_area(
        "Job Description Text",
        jd_text,
        height=300
    )
