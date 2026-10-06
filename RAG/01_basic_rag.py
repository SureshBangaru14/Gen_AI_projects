import os
import json
import chromadb
import ollama
import pytesseract

from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIG
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
# SCHEMA
# ============================================================

SCHEMA = {
    "answer": "string",
    "source": [
        {
            "page_number": "integer",
            "chunk_number": "integer",
            "chunk_id": "string",
            "chunk_text": "string",
            "relevance_score": "float"
        }
    ],
    "confidence": "float"
}


# ============================================================
# PDF + TESSERACT OCR
# ============================================================

def extract_pdf_with_ocr(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    print("Converting PDF pages to images...")

    # converting PDF pages into images
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

        # extracting text from image using Tesseract OCR
        text = pytesseract.image_to_string(
            image,
            config="--psm 6"
        )

        # storing page number and OCR text
        page_data.append({
            "page_no": page_no,
            "text": text.strip()
        })

    return page_data


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(page_data):
    # creating recursive text splitter
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
        if not page["text"]:
            continue

        # creating chunks from page text
        page_chunks = splitter.split_text(
            page["text"]
        )

        for chunk_no, chunk in enumerate(
            page_chunks,
            start=1
        ):
            # creating chunk metadata
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

    # loading sentence transformer embedding model
    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# CHROMADB
# ============================================================

def create_vector_database():
    # creating persistent ChromaDB client
    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    # creating or loading ChromaDB collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


# ============================================================
# INGEST DOCUMENTS
# ============================================================

def ingest_documents(
    collection,
    embedding_model
):
    print(
        "\nStarting PDF OCR..."
    )

    # extracting text from PDF using Tesseract OCR
    page_data = extract_pdf_with_ocr(
        PDF_PATH
    )

    print(
        f"Total pages: {len(page_data)}"
    )

    # creating chunks from extracted text
    chunks = create_chunks(
        page_data
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    # getting text from all chunks
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "\nGenerating embeddings..."
    )

    # generating embeddings for document chunks
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    # storing chunks and embeddings into vector data base
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
    # converting user question into embedding
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    # searching relevant chunks from vector data base
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
# LLM GENERATION
# ============================================================

def generate_answer(
    question,
    results
):
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        # converting ChromaDB distance into relevance score
        relevance_score = 1 / (1 + distance)

        # adding retrieved chunk to LLM context
        context_parts.append(
            f"""
Page Number: {metadata['page_no']}
Chunk Number: {metadata['chunk_no']}
Chunk ID: page_{metadata['page_no']}_chunk_{metadata['chunk_no']}
Relevance Score: {relevance_score:.4f}

Chunk Text:
{document}
"""
        )

    # preparing retrieved context
    context = "\n".join(
        context_parts
    )

    # creating system prompt
    system_prompt = f"""
You are a production-level HR Policy RAG assistant.

Your task is to answer the user's question using ONLY
the retrieved HR Policy context.

Rules:
1. Use only the provided retrieved context.
2. Do not use outside knowledge.
3. Do not invent or hallucinate information.
4. If the answer is not available, say:
   "I don't know based on HR_Policy.pdf."
5. Use the retrieved chunks as evidence.
6. Return the page number of supporting chunks.
7. Return the chunk number of supporting chunks.
8. Return the exact chunk text from the provided context.
9. Return the chunk ID.
10. Return the relevance score provided in the context.
11. Confidence must be between 0.0 and 1.0.
12. Return only valid JSON.
13. Follow the schema exactly.
14. Do not add additional fields.

Schema:
{json.dumps(SCHEMA, indent=2)}
"""

    # creating user prompt
    user_prompt = f"""
USER QUESTION:
{question}

RETRIEVED HR POLICY CONTEXT:
{context}

Use the retrieved context to answer the question.

Return only valid JSON using this schema:

{json.dumps(SCHEMA, indent=2)}
"""

    # sending system prompt and user prompt to LLM
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

    # getting LLM response
    response_text = response["message"]["content"]

    try:
        # converting JSON string into Python dictionary
        return json.loads(
            response_text
        )

    except json.JSONDecodeError:
        # returning empty response if JSON parsing fails
        return {
            "answer": "I don't know based on HR_Policy.pdf.",
            "source": [],
            "confidence": 0.0
        }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("BASIC RAG - HR POLICY")
    print("=" * 60)

    print(
        f"PDF: {PDF_PATH}"
    )

    print(
        f"Embedding: {EMBEDDING_MODEL}"
    )

    print(
        f"LLM: {LLM_MODEL}"
    )

    # loading embedding model
    embedding_model = load_embedding_model()

    # creating vector data base
    collection = create_vector_database()

    # data updating into vector data base
    ingest_documents(
        collection,
        embedding_model
    )

    while True:
        print(
            "\n" + "=" * 60
        )

        question = input(
            "Enter your question "
            "(type 'exit' to stop): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        print(
            "\nRetrieving relevant chunks..."
        )

        # searching relevant chunks from vector data base
        results = retrieve_documents(
            collection,
            embedding_model,
            question
        )

        print(
            "\nRetrieved chunks:"
        )

        for i, (
            metadata,
            distance
        ) in enumerate(
            zip(
                results["metadatas"][0],
                results["distances"][0]
            ),
            start=1
        ):
            # calculating relevance score
            relevance_score = (
                1 / (1 + distance)
            )

            print(
                f"{i}. "
                f"Page={metadata['page_no']} "
                f"Chunk={metadata['chunk_no']} "
                f"Distance={distance:.4f} "
                f"Relevance={relevance_score:.4f}"
            )

        print(
            "\nGenerating answer..."
        )

        # generating answer using LLM
        answer = generate_answer(
            question,
            results
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "FINAL JSON RESPONSE"
        )

        print(
            "=" * 60
        )

        # displaying structured JSON response
        print(
            json.dumps(
                answer,
                indent=4
            )
        )


if __name__ == "__main__":
    main()