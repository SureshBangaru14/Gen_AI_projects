import streamlit as st
from document_process import DocumentProcess



st.set_page_config(page_title="Financial Document Processing", page_icon="📄", layout="wide")

st.title("Financial Document Processing")

# ============================================================
# 1. SELECT FINANCIAL DOCUMENT TYPE
# ============================================================

financial_doc_type = st.selectbox("Select Financial Document Type",
    ["BOL", "INVOICE", "CLAIM", "POD"])

# ============================================================
# 2. embedding_model_options
# ============================================================

embedding_methods = ["OpenAI Embeddings", "Sentence-BERT (SBERT)"]

embedding_method = st.selectbox("Embedding Method", embedding_methods, index=0)


embedding_model_options = {

                            "OpenAI Embeddings": [

                                "text-embedding-3-small",

                                "text-embedding-3-large"

                            ],

                            "Sentence-BERT (SBERT)": [

                                "sentence-transformers/all-MiniLM-L6-v2",

                                "sentence-transformers/all-mpnet-base-v2"

                            ]

                        }


selected_models = embedding_model_options.get(embedding_method, [])


embedding_model = st.selectbox("Embedding Model", selected_models)


# ========================================================
# OPENAI API KEY
# ========================================================

# Show the API-key field only when OpenAI Embeddings
# is selected.
openai_api_key = ""

if embedding_method == "OpenAI Embeddings":

    st.markdown("### 🔑 OpenAI")

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Required when OpenAI Embeddings is selected."
    )

    if openai_api_key:
        st.success("✅ OpenAI API key provided.")
    else:
        st.warning(
            "⚠️ OpenAI API key is required for OpenAI Embeddings."
        )



# ============================================================
# 3. FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(f"Upload {financial_doc_type} document", type=["pdf", "png", "jpg", "jpeg", "xlsx", "csv", "docx"])

# ============================================================
# 3. SHOW UPLOAD STATUS
# ============================================================

if uploaded_file:

    st.success(
        f"✅ {financial_doc_type} document uploaded: "
        f"{uploaded_file.name}"
    )

    # ========================================================
    # 4. START ENGINE BUTTON
    # ========================================================

    start_engine = st.button("🚀 START ENGINE", type="primary", use_container_width=True)

    # ========================================================
    # 5. START PROCESSING
    # ========================================================

    if start_engine:

        st.info(f"🚀 Starting {financial_doc_type} processing...")

        # ----------------------------------------------------
        # Step 1: Read uploaded file
        # ----------------------------------------------------

        st.write("📥 Step 1: Reading document...")
        # ====================================================
        # DOCUMENT PROCESSOR
        # ====================================================

        processor_result = DocumentProcess(input_file=uploaded_file, financial_doc_type = financial_doc_type, openai_api_key = openai_api_key, 
                                    embedding_method = embedding_method, embedding_model = embedding_model).process()


        st.subheader("📊 Processing Result")

        result = {
            "document_type" : financial_doc_type,
            "file_name" : uploaded_file.name,
            "status" : "SUCCESS",
            "processor_result" : processor_result
        }

        st.json(result)

else:

    st.info(
        f"Please upload a {financial_doc_type} document to continue."
    )
