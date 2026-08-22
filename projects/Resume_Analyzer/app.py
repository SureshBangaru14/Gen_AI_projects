# ============================================================
# app.py
# RESUME ANALYZER
# ============================================================

import streamlit as st
import pandas as pd

from document_process import DocumentProcess


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        color: #666;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📄 Resume Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'AI-powered Resume Parsing & Job Matching System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "processing_result" not in st.session_state:
    st.session_state.processing_result = None


if "selected_result" not in st.session_state:
    st.session_state.selected_result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.divider()

    # ========================================================
    # EMBEDDING METHOD
    # ========================================================

    embedding_methods = [

        "OpenAI Embeddings",

        "Transformer-based Embeddings",

        "Sentence-BERT (SBERT)",

        "Other Modern Text Embedding Models",

        "Multilingual Embeddings",

        "Domain-Specific Embeddings",

        "Sparse Embeddings",

        "Hybrid Dense + Sparse Embeddings",

        "Late-Interaction Embeddings",

        "Multimodal Embeddings",

        "TF-IDF",

        "Word2Vec"

    ]

    embedding_method = st.selectbox(
        "Embedding Method",
        embedding_methods,
        index=0
    )


    # ========================================================
    # EMBEDDING MODEL
    # ========================================================

    embedding_model_options = {

        "OpenAI Embeddings": [

            "text-embedding-3-small",

            "text-embedding-3-large"

        ],

        "Transformer-based Embeddings": [

            "sentence-transformers/all-mpnet-base-v2",

            "sentence-transformers/all-MiniLM-L6-v2"

        ],

        "Sentence-BERT (SBERT)": [

            "sentence-transformers/all-MiniLM-L6-v2",

            "sentence-transformers/all-mpnet-base-v2"

        ],

        "Other Modern Text Embedding Models": [

            "sentence-transformers/all-mpnet-base-v2",

            "sentence-transformers/all-MiniLM-L6-v2"

        ],

        "Multilingual Embeddings": [

            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"

        ],

        "Domain-Specific Embeddings": [

            "sentence-transformers/all-mpnet-base-v2"

        ],

        "Sparse Embeddings": [

            "TF-IDF"

        ],

        "Hybrid Dense + Sparse Embeddings": [

            "text-embedding-3-small",

            "text-embedding-3-large"

        ],

        "Late-Interaction Embeddings": [

            "sentence-transformers/all-MiniLM-L6-v2"

        ],

        "Multimodal Embeddings": [

            "Not configured"

        ],

        "TF-IDF": [

            "TF-IDF"

        ],

        "Word2Vec": [

            "Word2Vec"

        ]

    }


    selected_models = embedding_model_options.get(

        embedding_method,

        []

    )


    embedding_model = st.selectbox(

        "Embedding Model",

        selected_models

    )


    # ========================================================
    # OPENAI API KEY
    # ========================================================

    st.markdown("### 🔑 OpenAI")

    openai_api_key = st.text_input(

        "OpenAI API Key",

        type="password",

        help=(
            "Required for OpenAI embeddings "
            "and LLM candidate analysis."
        )

    )


    # ========================================================
    # RETRIEVAL / SIMILARITY
    # ========================================================

    st.markdown(
        "### 🔎 Retrieval / Similarity"
    )


    similarity_methods = [

        "Cosine Similarity",

        "Dot Product / Inner Product",

        "BM25",

        "Hybrid Search",

        "Cross-Encoder Reranking",

        "ANN (Approximate Nearest Neighbor)",

        "Dense Vector Retrieval",

        "Sparse Vector Retrieval",

        "Dense + Sparse Hybrid Retrieval",

        "MMR (Maximal Marginal Relevance)",

        "Euclidean Distance (L2)",

        "ColBERT / Late-Interaction Retrieval"

    ]


    similarity_method = st.selectbox(

        "Similarity Search Method",

        similarity_methods,

        index=0

    )


    # ========================================================
    # TOP K
    # ========================================================

    top_k = st.slider(

        "Top Matching Results",

        min_value=1,

        max_value=20,

        value=5

    )


    # ========================================================
    # GENERATION MODEL
    # ========================================================

    generation_model = st.selectbox(

        "LLM Generation Model",

        [

            "gpt-4o-mini",

            "gpt-4o"

        ],

        index=0

    )


    # ========================================================
    # CONTEXT SAFETY BUFFER
    # ========================================================

    st.markdown(
        "### 🛡 Context Safety Buffer"
    )


    safety_buffer_method = st.selectbox(

        "Safety Buffer Method",

        [

            "Percentage",

            "Fixed Tokens",

            "Hybrid"

        ]

    )


    if safety_buffer_method == "Percentage":

        safety_buffer_percent = st.select_slider(

            "Safety Buffer",

            options=[

                10,

                20,

                30,

                40,

                50,

                60

            ],

            value=20,

            format_func=lambda x:
                f"{x}%"

        )

        fixed_safety_buffer = None

        maximum_safety_buffer = None


    elif safety_buffer_method == "Fixed Tokens":

        fixed_safety_buffer = st.number_input(

            "Safety Buffer Tokens",

            min_value=100,

            max_value=10000,

            value=1000,

            step=100

        )

        safety_buffer_percent = 20

        maximum_safety_buffer = None


    else:

        safety_buffer_percent = st.select_slider(

            "Percentage",

            options=[

                10,

                20,

                30,

                40,

                50,

                60

            ],

            value=20,

            format_func=lambda x:
                f"{x}%"

        )

        maximum_safety_buffer = st.number_input(

            "Maximum Buffer Tokens",

            min_value=100,

            max_value=10000,

            value=3000,

            step=100

        )

        fixed_safety_buffer = None


    st.divider()


    st.caption(

        "The safety buffer protects the "
        "model context window from overflow."

    )


# ============================================================
# MAIN LAYOUT
# ============================================================

left_column, right_column = st.columns(

    [1, 3],

    gap="large"

)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_column:

    st.markdown(

        '<div class="section-title">'
        '📤 Upload Documents'
        '</div>',

        unsafe_allow_html=True

    )


    # ========================================================
    # RESUME UPLOAD
    # ========================================================

    resume_files = st.file_uploader(

        "Upload Resume PDFs",

        type=["pdf"],

        accept_multiple_files=True,

        help=(
            "Upload one or multiple resume PDFs."
        )

    )


    # ========================================================
    # JD INPUT METHOD
    # ========================================================

    st.markdown(
        "### 📋 Job Description"
    )


    jd_input_method = st.radio(

        "Job Description Input",

        [

            "📄 Upload PDF",

            "📝 Upload DOCX",

            "📃 Upload TXT",

            "✍️ Paste Text"

        ],

        index=0

    )


    # ========================================================
    # PDF
    # ========================================================

    jd_file = None

    jd_pasted_text = ""


    if jd_input_method == "📄 Upload PDF":

        jd_file = st.file_uploader(

            "Upload Job Description PDF",

            type=["pdf"],

            accept_multiple_files=False,

            key="jd_pdf"

        )


    # ========================================================
    # DOCX
    # ========================================================

    elif jd_input_method == "📝 Upload DOCX":

        jd_file = st.file_uploader(

            "Upload Job Description DOCX",

            type=["docx"],

            accept_multiple_files=False,

            key="jd_docx"

        )


    # ========================================================
    # TXT
    # ========================================================

    elif jd_input_method == "📃 Upload TXT":

        jd_file = st.file_uploader(

            "Upload Job Description TXT",

            type=["txt"],

            accept_multiple_files=False,

            key="jd_txt"

        )


    # ========================================================
    # PASTE TEXT
    # ========================================================

    elif jd_input_method == "✍️ Paste Text":

        jd_pasted_text = st.text_area(

            "Paste Job Description",

            height=350,

            placeholder=(
                "Paste the complete Job Description here..."
            )

        )


    st.markdown("")


    # ========================================================
    # BUTTONS
    # ========================================================

    parse_button = st.button(

        "🚀 Parse & Match",

        use_container_width=True,

        type="primary"

    )


    clear_button = st.button(

        "🗑 Clear Results",

        use_container_width=True

    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with right_column:

    st.markdown(

        '<div class="section-title">'
        '📋 Job Description Preview'
        '</div>',

        unsafe_allow_html=True

    )


    # ========================================================
    # UPLOADED JD PREVIEW
    # ========================================================

    if jd_file:

        st.success(

            f"✅ JD uploaded: {jd_file.name}"

        )

        st.caption(

            "Input type: "
            +
            jd_input_method

        )


    # ========================================================
    # PASTED JD PREVIEW
    # ========================================================

    elif jd_pasted_text.strip():

        st.success(
            "✅ Job Description text entered."
        )

        st.caption(
            f"Characters: {len(jd_pasted_text):,}"
        )

        with st.expander(
            "Preview Job Description"
        ):

            st.write(
                jd_pasted_text
            )


    else:

        st.info(

            "Select a Job Description input method "
            "and provide the Job Description."

        )


    # ========================================================
    # RESUME STATUS
    # ========================================================

    if resume_files:

        st.success(

            f"📄 {len(resume_files)} "
            f"resume(s) uploaded."

        )

    else:

        st.info(
            "No resumes uploaded yet."
        )


# ============================================================
# CLEAR RESULTS
# ============================================================

if clear_button:

    st.session_state.processing_result = None

    st.session_state.selected_result = None

    st.rerun()


# ============================================================
# PROCESS
# ============================================================

if parse_button:

    # ========================================================
    # VALIDATE RESUMES
    # ========================================================

    if not resume_files:

        st.error(

            "Please upload at least "
            "one resume PDF."

        )

        st.stop()


    # ========================================================
    # VALIDATE JD
    # ========================================================

    if (
        jd_input_method != "✍️ Paste Text"
        and
        not jd_file
    ):

        st.error(

            "Please upload the "
            "Job Description file."

        )

        st.stop()


    if (
        jd_input_method == "✍️ Paste Text"
        and
        not jd_pasted_text.strip()
    ):

        st.error(

            "Please paste the "
            "Job Description text."

        )

        st.stop()


    # ========================================================
    # OPENAI KEY
    # ========================================================

    if not openai_api_key:

        st.error(

            "OpenAI API key is required "
            "for Candidate Analyzer."

        )

        st.stop()


    # ========================================================
    # DETERMINE INTERNAL JD METHOD
    # ========================================================

    if jd_input_method == "📄 Upload PDF":

        internal_jd_method = "Upload PDF"

    elif jd_input_method == "📝 Upload DOCX":

        internal_jd_method = "Upload DOCX"

    elif jd_input_method == "📃 Upload TXT":

        internal_jd_method = "Upload TXT"

    else:

        internal_jd_method = "Paste Text"


    # ========================================================
    # PROCESSING UI
    # ========================================================

    progress = st.progress(0)

    status = st.empty()


    try:

        # ====================================================
        # STEP 1
        # ====================================================

        status.write(
            "🔄 Initializing Resume Analyzer..."
        )

        progress.progress(5)


        # ====================================================
        # STEP 2
        # ====================================================

        status.write(
            "📄 Reading Resume PDFs..."
        )

        progress.progress(10)


        # ====================================================
        # DOCUMENT PROCESSOR
        # ====================================================

        processor = DocumentProcess(

            resume_files=resume_files,

            jd_input_method=
                internal_jd_method,

            jd_file_name=
                jd_file
                if jd_file
                else jd_pasted_text,

            openai_api_key=
                openai_api_key,

            generation_model=
                generation_model,

            embedding_method=
                embedding_method,

            embedding_model=
                embedding_model,

            similarity_method=
                similarity_method,

            top_k=
                top_k,

            safety_buffer_method=
                safety_buffer_method,

            safety_buffer_percent=
                safety_buffer_percent,

            fixed_safety_buffer=
                fixed_safety_buffer,

            maximum_safety_buffer=
                maximum_safety_buffer

        )


        # ====================================================
        # STEP 3
        # ====================================================

        status.write(
            "📋 Processing Job Description..."
        )

        progress.progress(20)


        # ====================================================
        # RUN
        # ====================================================

        result = processor.process()


        # ====================================================
        # COMPLETE
        # ====================================================

        status.write(
            "✅ Resume matching completed."
        )

        progress.progress(100)


        st.session_state.processing_result = (
            result
        )

        st.session_state.selected_result = None


        st.success(

            "🎉 Resume matching completed successfully."

        )


    except Exception as error:

        progress.empty()

        status.empty()


        st.error(

            "❌ Processing failed: "
            + str(error)

        )


        with st.expander(
            "🔍 Technical Error"
        ):

            st.exception(
                error
            )


# ============================================================
# RESULTS
# ============================================================

result = (
    st.session_state.processing_result
)


if result:

    st.divider()


    st.markdown(

        '<div class="section-title">'
        '🏆 Matching Results'
        '</div>',

        unsafe_allow_html=True

    )


    matching_results = result.get(

        "matching_results",

        []

    )


    # ========================================================
    # NO RESULTS
    # ========================================================

    if not matching_results:

        st.warning(

            "No matching results were generated."

        )


    else:

        # ====================================================
        # TABLE
        # ====================================================

        table_data = []


        for item in matching_results:

            match_value = item.get(

                "match_percentage",

                0

            )


            try:

                match_value = float(
                    match_value
                )

            except (
                TypeError,
                ValueError
            ):

                match_value = 0.0


            table_data.append({

                "Rank":
                    item.get(
                        "rank",
                        0
                    ),

                "Resume":
                    item.get(
                        "file_name",
                        ""
                    ),

                "Candidate":
                    item.get(
                        "candidate_name",
                        ""
                    ),

                "Match %":
                    f"{match_value:.2f}%",

                "Retrieval":
                    item.get(
                        "retrieval_method",
                        ""
                    )

            })


        dataframe = pd.DataFrame(
            table_data
        )


        st.dataframe(

            dataframe,

            use_container_width=True,

            hide_index=True

        )


        # ====================================================
        # CANDIDATE SELECTION
        # ====================================================

        candidate_names = []


        for item in matching_results:

            candidate_name = (

                item.get(
                    "candidate_name"
                )

                or

                item.get(
                    "file_name",
                    "Unknown Candidate"
                )

            )


            candidate_names.append(
                candidate_name
            )


        selected_candidate = st.selectbox(

            "Select Candidate",

            candidate_names

        )


        selected_index = (
            candidate_names.index(
                selected_candidate
            )
        )


        selected_result = (
            matching_results[
                selected_index
            ]
        )


        st.session_state.selected_result = (
            selected_result
        )


        # ====================================================
        # METRICS
        # ====================================================

        st.divider()


        score_col1, score_col2, score_col3 = (
            st.columns(3)
        )


        with score_col1:

            match_value = (

                selected_result.get(
                    "match_percentage",
                    0
                )

            )


            try:

                match_value = float(
                    match_value
                )

            except (
                TypeError,
                ValueError
            ):

                match_value = 0.0


            st.metric(

                "Match Percentage",

                f"{match_value:.2f}%"

            )


        with score_col2:

            retrieval_value = (

                selected_result.get(
                    "retrieval_score"
                )

            )


            if retrieval_value is None:

                retrieval_display = "N/A"

            else:

                try:

                    retrieval_display = (
                        f"{float(retrieval_value):.4f}"
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    retrieval_display = "N/A"


            st.metric(

                "Retrieval Score",

                retrieval_display

            )


        with score_col3:

            st.metric(

                "Retrieval Method",

                selected_result.get(

                    "retrieval_method",

                    "N/A"

                )

            )


        # ====================================================
        # CANDIDATE INFORMATION
        # ====================================================

        st.markdown(
            "### 👤 Candidate Information"
        )


        candidate_col1, candidate_col2 = (
            st.columns(2)
        )


        with candidate_col1:

            st.write(

                "**Candidate:** "
                +
                str(
                    selected_result.get(
                        "candidate_name",
                        "Unknown"
                    )
                )

            )


        with candidate_col2:

            st.write(

                "**Resume:** "
                +
                str(
                    selected_result.get(
                        "file_name",
                        ""
                    )
                )

            )


        # ====================================================
        # COMPONENT SCORES
        # ====================================================

        st.markdown(
            "### 📊 Component Scores"
        )


        component_scores = (

            selected_result.get(

                "component_scores",

                {}

            )

        )


        if component_scores:

            component_rows = []


            for key, value in (
                component_scores.items()
            ):

                try:

                    numeric_value = float(
                        value
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    numeric_value = 0.0


                component_rows.append({

                    "Component":
                        key.replace(
                            "_",
                            " "
                        ).title(),

                    "Score":
                        f"{numeric_value:.2f}%"

                })


            component_dataframe = (
                pd.DataFrame(
                    component_rows
                )
            )


            st.dataframe(

                component_dataframe,

                use_container_width=True,

                hide_index=True

            )


        else:

            st.info(
                "No component scores available."
            )


        # ====================================================
        # SKILLS
        # ====================================================

        skills_col1, skills_col2 = (
            st.columns(2)
        )


        # ====================================================
        # MATCHED SKILLS
        # ====================================================

        with skills_col1:

            st.markdown(
                "### ✅ Matched Skills"
            )


            matched_skills = (

                selected_result.get(

                    "matched_skills",

                    []

                )

            )


            if matched_skills:

                for skill in matched_skills:

                    st.success(
                        str(skill)
                    )

            else:

                st.info(
                    "No matched skills found."
                )


        # ====================================================
        # MISSING SKILLS
        # ====================================================

        with skills_col2:

            st.markdown(
                "### ⚠️ Missing Skills"
            )


            missing_skills = (

                selected_result.get(

                    "missing_skills",

                    []

                )

            )


            if missing_skills:

                for skill in missing_skills:

                    st.warning(
                        str(skill)
                    )

            else:

                st.success(

                    "No major skill gaps identified."

                )


        # ====================================================
        # EXPERIENCE
        # ====================================================

        st.markdown(
            "### 💼 Experience"
        )


        experience_col1, experience_col2 = (
            st.columns(2)
        )


        with experience_col1:

            st.write(

                "**Required Experience:** "
                +
                str(
                    selected_result.get(
                        "required_years",
                        0
                    )
                )
                +
                " years"

            )


        with experience_col2:

            st.write(

                "**Candidate Experience:** "
                +
                str(
                    selected_result.get(
                        "candidate_years",
                        0
                    )
                )
                +
                " years"

            )


        # ====================================================
        # RESPONSIBILITIES
        # ====================================================

        st.markdown(
            "### 📋 Responsibilities"
        )


        responsibility_match = (

            selected_result.get(

                "responsibility_match",

                ""

            )

        )


        st.write(

            "**Responsibility Match:** "
            +
            str(
                responsibility_match
            )

        )


        # ====================================================
        # EDUCATION
        # ====================================================

        st.markdown(
            "### 🎓 Education"
        )


        education_col1, education_col2 = (
            st.columns(2)
        )


        with education_col1:

            st.write(

                "**Required:** "
                +
                str(
                    selected_result.get(
                        "required_education",
                        ""
                    )
                )

            )


        with education_col2:

            st.write(

                "**Candidate:** "
                +
                str(
                    selected_result.get(
                        "candidate_education",
                        ""
                    )
                )

            )


        # ====================================================
        # PROJECTS
        # ====================================================

        st.markdown(
            "### 🚀 Relevant Projects"
        )


        relevant_projects = (

            selected_result.get(

                "relevant_projects",

                []

            )

        )


        if relevant_projects:

            for project in relevant_projects:

                st.write(
                    f"• {project}"
                )

        else:

            st.info(
                "No relevant projects identified."
            )


        # ====================================================
        # STRENGTHS
        # ====================================================

        st.markdown(
            "### 💪 Strengths"
        )


        strengths = (

            selected_result.get(

                "strengths",

                []

            )

        )


        if strengths:

            for strength in strengths:

                st.write(
                    f"• {strength}"
                )

        else:

            st.info(
                "No strengths identified."
            )


        # ====================================================
        # SKILL GAPS
        # ====================================================

        st.markdown(
            "### ⚠️ Skill Gaps"
        )


        skill_gaps = (

            selected_result.get(

                "skill_gaps",

                []

            )

        )


        if skill_gaps:

            for gap in skill_gaps:

                st.write(
                    f"• {gap}"
                )

        else:

            st.success(

                "No significant skill gaps identified."

            )


        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        st.markdown(
            "### 💡 Recommendations"
        )


        recommendations = (

            selected_result.get(

                "recommendations",

                []

            )

        )


        if recommendations:

            for recommendation in recommendations:

                st.write(
                    f"• {recommendation}"
                )

        else:

            st.info(
                "No recommendations available."
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.markdown(
            "### 📝 Summary"
        )


        summary = (

            selected_result.get(

                "summary",

                ""

            )

        )


        if summary:

            st.info(
                summary
            )

        else:

            st.info(
                "No summary available."
            )


        # ====================================================
        # DOWNLOAD RESUME
        # ====================================================

        st.markdown(
            "### 📥 Resume"
        )


        selected_file_name = (

            selected_result.get(

                "file_name",

                ""

            )

        )


        matching_upload = None


        if resume_files:

            for uploaded_file in resume_files:

                if (

                    uploaded_file.name
                    ==
                    selected_file_name

                ):

                    matching_upload = (
                        uploaded_file
                    )

                    break


        if matching_upload:

            st.download_button(

                label=
                    "📥 Download Resume",

                data=
                    matching_upload.getvalue(),

                file_name=
                    matching_upload.name,

                mime=
                    "application/pdf",

                use_container_width=True

            )

        else:

            st.info(

                "Original uploaded resume "
                "is not available for download."

            )


# ============================================================
# PIPELINE INFORMATION
# ============================================================

if result:

    st.divider()


    with st.expander(
        "🔧 Pipeline Information"
    ):

        chunk_data = (

            result.get(

                "chunk_data",

                {}

            )

        )


        context_data = (

            result.get(

                "context_data",

                {}

            )

        )


        vector_data = (

            result.get(

                "vector_data",

                {}

            )

        )


        st.write(

            "**Embedding Method:**",

            chunk_data.get(

                "embedding_method",

                "N/A"

            )

        )


        st.write(

            "**Embedding Model:**",

            chunk_data.get(

                "embedding_model",

                "N/A"

            )

        )


        st.write(

            "**Similarity Method:**",

            chunk_data.get(

                "similarity_method",

                "N/A"

            )

        )


        st.write(

            "**Safety Buffer Method:**",

            chunk_data.get(

                "safety_buffer_method",

                "N/A"

            )

        )


        st.write(

            "**Safety Buffer:**",

            str(

                chunk_data.get(

                    "safety_buffer_percent",

                    "N/A"

                )

            )
            +
            "%"

        )


        st.write(

            "**Safety Buffer Tokens:**",

            chunk_data.get(

                "safety_buffer_tokens",

                "N/A"

            )

        )


        st.write(

            "**Chunk Size:**",

            chunk_data.get(

                "chunk_size",

                "N/A"

            )

        )


        st.write(

            "**Chunk Overlap:**",

            chunk_data.get(

                "chunk_overlap",

                "N/A"

            )

        )


        st.write(

            "**Total Chunks:**",

            vector_data.get(

                "total_chunks",

                "N/A"

            )

        )


        st.write(

            "**Embedding Dimension:**",

            vector_data.get(

                "embedding_dimension",

                "N/A"

            )

        )


        if context_data:

            st.write(
                "**Context Calculation:**"
            )

            st.json(
                context_data
            )