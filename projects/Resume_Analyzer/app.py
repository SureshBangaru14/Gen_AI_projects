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
    "Resume matching and candidate analysis using "
    "OpenAI and ChromaDB."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")


    # ========================================================
    # OPENAI API KEY
    # ========================================================

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-..."
    )


    # ========================================================
    # GENERATION MODEL
    # ========================================================

    generation_model = st.selectbox(
        "Generation Model",
        [
            "gpt-4o",
            "gpt-4o-mini"
        ]
    )


    # ========================================================
    # EMBEDDING MODEL
    # ========================================================

    embedding_model = st.selectbox(
        "Embedding Model",
        [
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]
    )


    # ========================================================
    # TOP K
    # ========================================================

    top_k = st.slider(
        "Top Matching Results",
        min_value=1,
        max_value=10,
        value=5
    )


    # ========================================================
    # RESUME UPLOAD
    # ========================================================

    st.subheader(
        "📁 Upload Resumes"
    )


    resume_files = st.file_uploader(
        "Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )


    # ========================================================
    # DISPLAY RESUMES
    # ========================================================

    if resume_files:

        st.write(
            "### Selected Resumes"
        )


        for resume_file in resume_files:

            st.write(
                f"✓ {resume_file.name}"
            )


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.subheader(
    "📋 Job Description"
)


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
# INITIALIZE VARIABLES
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
# JD TEXT
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

    # ========================================================
    # VALIDATE API KEY
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


    else:

        st.error(
            "Invalid Job Description input method."
        )

        st.stop()


    # ========================================================
    # CREATE DOCUMENT PROCESS
    # ========================================================

    document_process = DocumentProcess(

        resume_files=resume_files,

        jd_input_method=jd_input_method,

        jd_file_name=jd_file,

        openai_api_key=openai_api_key,

        generation_model=generation_model,

        embedding_model=embedding_model,

        top_k=top_k

    )


    # ========================================================
    # PROCESS
    # ========================================================

    with st.spinner(
        "Processing resumes..."
    ):

        try:

            result = (
                document_process.process()
            )

        except Exception as e:

            st.error(
                f"Processing failed: {e}"
            )

            st.stop()


    # ========================================================
    # SUCCESS
    # ========================================================

    st.success(
        "Resume processing completed successfully."
    )


    # ========================================================
    # TOKEN INFORMATION
    # ========================================================

    if "token_data" in result:

        st.subheader(
            "🧮 Token Information"
        )


        token_data = (
            result["token_data"]
        )


        # ----------------------------------------------------
        # SCHEMA
        # ----------------------------------------------------

        st.write(
            f"**Schema:** "
            f"{token_data['schema_tokens']:,} tokens"
        )


        # ----------------------------------------------------
        # SYSTEM PROMPT
        # ----------------------------------------------------

        st.write(
            f"**System Prompt:** "
            f"{token_data['system_prompt_tokens']:,} tokens"
        )


        # ----------------------------------------------------
        # JD
        # ----------------------------------------------------

        st.write(
            f"**Job Description:** "
            f"{token_data['job_description_tokens']:,} tokens"
        )


        # ----------------------------------------------------
        # RESUMES
        # ----------------------------------------------------

        for file_name, token_count in (
            token_data["resume_tokens"].items()
        ):

            st.write(
                f"**{file_name}:** "
                f"{token_count:,} tokens"
            )


        # ----------------------------------------------------
        # USER PROMPTS
        # ----------------------------------------------------

        st.write(
            "### User Prompt Tokens"
        )


        for file_name, token_count in (
            token_data["user_prompt_tokens"].items()
        ):

            st.write(
                f"**{file_name}:** "
                f"{token_count:,} tokens"
            )


    # ========================================================
    # CONTEXT WINDOW
    # ========================================================

    if "context_data" in result:

        st.subheader(
            "🧠 Context Window"
        )


        context_data = (
            result["context_data"]
        )


        for file_name, data in context_data.items():

            with st.expander(
                f"📄 {file_name}"
            ):

                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Context Window",
                        f"{data['context_window']:,}"
                    )


                with col2:

                    st.metric(
                        "Total Tokens",
                        f"{data['total_tokens']:,}"
                    )


                with col3:

                    st.metric(
                        "Available",
                        f"{data['available_tokens']:,}"
                    )


                if data["fits_context"]:

                    st.success(
                        "✅ Fits Context Window"
                    )

                else:

                    st.warning(
                        "⚠️ Exceeds Context Window"
                    )


    # ========================================================
    # CHUNKING
    # ========================================================

    if "chunk_data" in result:

        st.subheader(
            "✂️ Dynamic Chunking"
        )


        chunk_data = (
            result["chunk_data"]
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Chunk Size",
                f"{chunk_data['chunk_size']:,}"
            )


        with col2:

            st.metric(
                "Chunk Overlap",
                f"{chunk_data['chunk_overlap']:,}"
            )


        with col3:

            st.metric(
                "Top K",
                chunk_data["top_k"]
            )


        with col4:

            st.metric(
                "Retrieval Budget",
                f"{chunk_data['retrieval_budget']:,}"
            )


    # ========================================================
    # RESUME CHUNKS
    # ========================================================

    if "resume_chunks" in result:

        st.subheader(
            "📦 Resume Chunks"
        )


        resume_chunks = (
            result["resume_chunks"]
        )


        for file_name, chunks in resume_chunks.items():

            with st.expander(
                f"📄 {file_name} "
                f"({len(chunks)} chunks)"
            ):

                for chunk in chunks:

                    st.write(
                        f"### Chunk "
                        f"{chunk['chunk_id']}"
                    )


                    st.caption(
                        f"Token Count: "
                        f"{chunk['token_count']}"
                    )


                    st.text_area(
                        "Chunk Text",
                        chunk["text"],
                        height=150,
                        key=(
                            f"{file_name}_"
                            f"{chunk['chunk_id']}"
                        )
                    )


    # ========================================================
    # FULL RESUME TEXT
    # ========================================================

    if "resume_text" in result:

        st.subheader(
            "📝 Full Resume Text"
        )


        for file_name, resume_text in (
            result["resume_text"].items()
        ):

            with st.expander(
                f"📄 {file_name}"
            ):

                st.text_area(
                    "Resume Text",
                    resume_text,
                    height=300,
                    key=f"full_{file_name}"
                )


    # ========================================================
    # JOB DESCRIPTION
    # ========================================================

    if "job_description" in result:

        st.subheader(
            "📋 Extracted Job Description"
        )


        st.text_area(
            "Job Description",
            result["job_description"],
            height=300
        )