import os
import json
import chromadb
import ollama
import pytesseract

from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "/home/suresh/Gen_AI_Practice/Generative-AI/Vector_Database/HR_Policy.pdf"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral"

CHROMA_DIR = "chroma_basic_hr"
COLLECTION_NAME = "hr_policy_basic"

OCR_DPI = 250
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 5


# ============================================================
# PRODUCTION RESPONSE SCHEMA
# ============================================================

SCHEMA = {
    "working_hours": {
        "start_time": "string",
        "end_time": "string",
        "timezone": "string",
        "working_days": ["string"]
    },
    "source_pages": ["integer"],
    "confidence": "float"
}


# ============================================================
# PDF OCR
# ============================================================

def extract_pdf_with_ocr(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    print("Converting PDF pages to images...")

    pages = convert_from_path(
        pdf_path,
        dpi=OCR_DPI
    )

    page_data = []

    for page_no, image in enumerate(
        pages,
        start=1
    ):
        print(
            f"OCR processing page "
            f"{page_no}/{len(pages)}..."
        )

        text = pytesseract.image_to_string(
            image,
            config="--psm 6"
        )

        text = text.strip()

        if text:
            page_data.append({
                "page_no": page_no,
                "text": text
            })

    return page_data


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(page_data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for page in page_data:
        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk_no, chunk in enumerate(
            page_chunks,
            start=1
        ):
            chunks.append({
                "id": (
                    f"page_{page['page_no']}"
                    f"_chunk_{chunk_no}"
                ),
                "text": chunk,
                "metadata": {
                    "file_name": os.path.basename(
                        PDF_PATH
                    ),
                    "page_no": page["page_no"],
                    "chunk_no": chunk_no
                }
            })

    return chunks


# ============================================================
# EMBEDDING MODEL
# ============================================================

def load_embedding_model():
    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# CHROMADB
# ============================================================

def create_vector_database():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "HR Policy Basic RAG"
        }
    )

    return collection


# ============================================================
# DOCUMENT INGESTION
# ============================================================

def ingest_documents(collection, embedding_model):
    print("\nStarting PDF OCR...")

    page_data = extract_pdf_with_ocr(
        PDF_PATH
    )

    print(
        f"Total OCR pages: "
        f"{len(page_data)}"
    )

    chunks = create_chunks(
        page_data
    )

    print(
        f"Total chunks: "
        f"{len(chunks)}"
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "Generating embeddings..."
    )

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    collection.upsert(
        ids=[
            chunk["id"]
            for chunk in chunks
        ],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            chunk["metadata"]
            for chunk in chunks
        ]
    )

    print(
        f"ChromaDB count: "
        f"{collection.count()}"
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    collection,
    embedding_model,
    question
):
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for document, metadata in zip(
        documents,
        metadatas
    ):
        context_parts.append(
            f"[Page {metadata['page_no']}, "
            f"Chunk {metadata['chunk_no']}]\n"
            f"{document}"
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt():
    schema = json.dumps(
        SCHEMA,
        indent=2
    )

    return f"""
You are a production-grade HR Policy RAG assistant.

Your task is to extract the answer from the
provided HR Policy context.

You MUST follow these rules:

1. Use ONLY the provided retrieved context.
2. Do NOT use external knowledge.
3. Do NOT hallucinate missing information.
4. If working hours are not available in the
   retrieved context, return empty strings,
   empty working_days and confidence 0.
5. Extract only information explicitly supported
   by the context.
6. Return the source page numbers containing
   the supporting information.
7. Confidence must be a number between 0 and 1.
8. Return ONLY valid JSON.
9. Follow the response schema exactly.
10. Do not add additional fields.

RESPONSE SCHEMA:

{schema}
"""


# ============================================================
# USER PROMPT
# ============================================================

def build_user_prompt(
    question,
    context
):
    schema = json.dumps(
        SCHEMA,
        indent=2
    )

    return f"""
Extract the requested information from the
HR Policy document.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

REQUIRED RESPONSE SCHEMA:
{schema}

Return ONLY valid JSON.
"""


# ============================================================
# LLM GENERATION
# ============================================================

def generate_answer(
    question,
    context
):
    system_prompt = build_system_prompt()

    user_prompt = build_user_prompt(
        question,
        context
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        format="json"
    )

    response_text = (
        response["message"]["content"]
    )

    try:
        result = json.loads(
            response_text
        )

        return validate_response(
            result
        )

    except json.JSONDecodeError:
        return create_empty_response()


# ============================================================
# RESPONSE VALIDATION
# ============================================================

def create_empty_response():
    return {
        "working_hours": {
            "start_time": "",
            "end_time": "",
            "timezone": "",
            "working_days": []
        },
        "source_pages": [],
        "confidence": 0.0
    }


def validate_response(result):
    working_hours = result.get(
        "working_hours",
        {}
    )

    source_pages = result.get(
        "source_pages",
        []
    )

    confidence = result.get(
        "confidence",
        0.0
    )

    validated = {
        "working_hours": {
            "start_time": working_hours.get(
                "start_time",
                ""
            ),
            "end_time": working_hours.get(
                "end_time",
                ""
            ),
            "timezone": working_hours.get(
                "timezone",
                ""
            ),
            "working_days": working_hours.get(
                "working_days",
                []
            )
        },
        "source_pages": source_pages,
        "confidence": confidence
    }

    return validated


# ============================================================
# DISPLAY RETRIEVED DOCUMENTS
# ============================================================

def display_retrieved_documents(
    results
):
    print("\nRetrieved Chunks")
    print("-" * 70)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (
        document,
        metadata,
        distance
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances
        ),
        start=1
    ):
        print(
            f"\nRank: {i}"
        )

        print(
            f"Page: "
            f"{metadata['page_no']}"
        )

        print(
            f"Chunk: "
            f"{metadata['chunk_no']}"
        )

        print(
            f"Distance: "
            f"{distance:.4f}"
        )

        print(
            f"Text: "
            f"{document[:300]}..."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("BASIC RAG - PRODUCTION STYLE")
    print("=" * 70)

    print(
        f"PDF: {PDF_PATH}"
    )

    print(
        "OCR: Tesseract"
    )

    print(
        f"Embedding: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        "Vector DB: ChromaDB"
    )

    print(
        f"LLM: {LLM_MODEL}"
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    # --------------------------------------------------------
    # Create vector database
    # --------------------------------------------------------

    collection = (
        create_vector_database()
    )

    # --------------------------------------------------------
    # OCR + Chunk + Embedding + Store
    # --------------------------------------------------------

    ingest_documents(
        collection,
        embedding_model
    )

    # --------------------------------------------------------
    # Question loop
    # --------------------------------------------------------

    while True:
        print(
            "\n" + "=" * 70
        )

        question = input(
            "Enter your question "
            "(type 'exit' to stop): "
        ).strip()

        if question.lower() == "exit":
            print("Exiting...")
            break

        if not question:
            continue

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        print(
            "\nSearching ChromaDB..."
        )

        results = retrieve_documents(
            collection,
            embedding_model,
            question
        )

        # ----------------------------------------------------
        # Display retrieved chunks
        # ----------------------------------------------------

        display_retrieved_documents(
            results
        )

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context = build_context(
            results
        )

        # ----------------------------------------------------
        # Schema
        # System Prompt
        # User Prompt
        # LLM
        # ----------------------------------------------------

        print(
            "\nGenerating structured response..."
        )

        result = generate_answer(
            question,
            context
        )

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "PRODUCTION JSON RESPONSE"
        )

        print(
            "=" * 70
        )

        print(
            json.dumps(
                result,
                indent=4
            )
        )


if __name__ == "__main__":
    main()