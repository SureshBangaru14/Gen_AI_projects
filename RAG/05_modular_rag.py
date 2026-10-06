# ====================================================================
# MODULAR RAG - METHOD 3
#
# Query Module
# + Retrieval Module
# + Reranking Module
# + Context Module
# + Generation Module
# + Validation Module
#
# ====================================================================
#
# COMPLETE FLOW
#
# PDF
#   ↓
# FLOW STEP 1: OCR MODULE
#   ↓
# FLOW STEP 2: CHUNKING MODULE
#   ↓
# FLOW STEP 3: EMBEDDING MODULE
#   ↓
# FLOW STEP 4: VECTOR DATABASE MODULE
#   ↓
# FLOW STEP 5: RETRIEVAL MODULE
#   ↓
# FLOW STEP 6: USER QUESTION
#   ↓
# FLOW STEP 7: QUERY MODULE
#   ↓
# FLOW STEP 8: RERANKING MODULE
#   ↓
# FLOW STEP 9: CONTEXT MODULE
#   ↓
# FLOW STEP 10: GENERATION MODULE
#   ↓
# FLOW STEP 11: JSON PARSING MODULE
#   ↓
# FLOW STEP 12: VALIDATION MODULE
#   ↓
# FLOW STEP 13: FINAL ANSWER
#   ↓
# FLOW STEP 14: WHILE LOOP
#
# ====================================================================
#
# MODULAR ARCHITECTURE
#
#                  USER QUESTION
#                        ↓
#                ┌───────────────┐
#                │ QUERY MODULE  │
#                └───────┬───────┘
#                        ↓
#                ┌───────────────┐
#                │ RETRIEVAL     │
#                │ MODULE        │
#                └───────┬───────┘
#                        ↓
#                ┌───────────────┐
#                │ RERANKING     │
#                │ MODULE        │
#                └───────┬───────┘
#                        ↓
#                ┌───────────────┐
#                │ CONTEXT       │
#                │ MODULE        │
#                └───────┬───────┘
#                        ↓
#                ┌───────────────┐
#                │ GENERATION    │
#                │ MODULE        │
#                └───────┬───────┘
#                        ↓
#                ┌───────────────┐
#                │ VALIDATION    │
#                │ MODULE        │
#                └───────┬───────┘
#                        ↓
#                   FINAL JSON
#
# ====================================================================


# ====================================================================
# IMPORT LIBRARIES
# ====================================================================

import os
import json

import chromadb
import ollama
import pytesseract

from pdf2image import convert_from_path

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


# ====================================================================
# CONFIGURATION
# ====================================================================

PDF_PATH = (
    "/home/suresh/Gen_AI_Practice/"
    "Generative-AI/Vector_Database/"
    "HR_Policy.pdf"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

LLM_MODEL = "mistral"

CHROMA_DIR = "chroma_modular_hr"

COLLECTION_NAME = "hr_policy_modular"

OCR_DPI = 250

CHUNK_SIZE = 600

CHUNK_OVERLAP = 100

RETRIEVAL_TOP_K = 10

FINAL_TOP_K = 5


# ====================================================================
# JSON SCHEMA
# ====================================================================

FIELD_NAMES = [
    "Answer",
    "Source",
    "Confidence"
]


SCHEMA = {

    "type": "object",

    "properties": {

        "Answer": {
            "type": "string"
        },

        "Source": {

            "type": "array",

            "items": {

                "type": "object",

                "properties": {

                    "Page Number": {
                        "type": "integer"
                    },

                    "Chunk Number": {
                        "type": "integer"
                    },

                    "Chunk ID": {
                        "type": "string"
                    },

                    "Chunk Text": {
                        "type": "string"
                    },

                    "Relevance Score": {
                        "type": "number"
                    }
                },

                "required": [
                    "Page Number",
                    "Chunk Number",
                    "Chunk ID",
                    "Chunk Text",
                    "Relevance Score"
                ],

                "additionalProperties": False
            }
        },

        "Confidence": {
            "type": "number"
        }
    },

    "required": FIELD_NAMES,

    "additionalProperties": False
}


ARRAY_SCHEMA = {

    "type": "array",

    "items": SCHEMA
}


# ====================================================================
# FLOW STEP 1: OCR MODULE
# ====================================================================
#
# PDF
#   ↓
# PDF Pages
#   ↓
# OCR
#   ↓
# Page-wise Text
#
# ====================================================================

def extract_text_with_ocr(pdf_path):

    print("\n" + "=" * 70)
    print("FLOW STEP 1: OCR MODULE")
    print("=" * 70)

    print(
        f"Reading PDF: {pdf_path}"
    )

    images = convert_from_path(
        pdf_path,
        dpi=OCR_DPI
    )

    pages = []

    for page_no, image in enumerate(
        images,
        start=1
    ):

        print(
            f"OCR processing page {page_no}"
        )

        text = pytesseract.image_to_string(
            image,
            config="--psm 6"
        )

        text = text.strip()

        pages.append(
            {
                "page_no": page_no,
                "text": text
            }
        )

    print(
        f"Total pages: {len(pages)}"
    )

    return pages


# ====================================================================
# FLOW STEP 2: CHUNKING MODULE
# ====================================================================
#
# Page-wise Text
#   ↓
# RecursiveCharacterTextSplitter
#   ↓
# Chunks
#
# ====================================================================

def create_chunks(pages):

    print("\n" + "=" * 70)
    print("FLOW STEP 2: CHUNKING MODULE")
    print("=" * 70)

    splitter = RecursiveCharacterTextSplitter(

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    for page in pages:

        page_no = page["page_no"]

        page_text = page["text"]

        page_chunks = splitter.split_text(
            page_text
        )

        for chunk_no, chunk in enumerate(
            page_chunks,
            start=1
        ):

            chunk = chunk.strip()

            if not chunk:
                continue

            chunk_id = (
                f"page_{page_no}_chunk_{chunk_no}"
            )

            chunks.append(
                {
                    "id": chunk_id,

                    "text": chunk,

                    "metadata": {

                        "file_name":
                            os.path.basename(
                                PDF_PATH
                            ),

                        "page_no":
                            page_no,

                        "chunk_no":
                            chunk_no
                    }
                }
            )

    print(
        f"Total chunks: {len(chunks)}"
    )

    return chunks


# ====================================================================
# FLOW STEP 3: EMBEDDING MODULE
# ====================================================================
#
# Chunks
#   ↓
# SentenceTransformer
#   ↓
# Embeddings
#
# ====================================================================

def load_embedding_model():

    print("\n" + "=" * 70)
    print("FLOW STEP 3: EMBEDDING MODULE")
    print("=" * 70)

    print(
        f"Loading: {EMBEDDING_MODEL}"
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )

    return embedding_model


# ====================================================================
# FLOW STEP 4: VECTOR DATABASE MODULE
# ====================================================================
#
# Embeddings
#   ↓
# ChromaDB
#
# ====================================================================

def create_vector_database():

    print("\n" + "=" * 70)
    print("FLOW STEP 4: VECTOR DATABASE MODULE")
    print("=" * 70)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    return collection


def store_documents(
    collection,
    embedding_model,
    chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 4: STORE DOCUMENTS")
    print("=" * 70)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    print(
        "Creating embeddings..."
    )

    embeddings = embedding_model.encode(

        texts,

        normalize_embeddings=True,

        show_progress_bar=True
    ).tolist()

    collection.upsert(

        ids=ids,

        documents=texts,

        embeddings=embeddings,

        metadatas=metadatas
    )

    print(
        f"Stored chunks: {len(chunks)}"
    )

    print(
        f"Database count: "
        f"{collection.count()}"
    )


# ====================================================================
# FLOW STEP 5: RETRIEVAL MODULE
# ====================================================================
#
# Question
#   ↓
# Query Embedding
#   ↓
# ChromaDB
#   ↓
# Retrieved Candidates
#
# ====================================================================

def retrieve_documents(
    question,
    embedding_model,
    collection
):

    print("\n" + "=" * 70)
    print("FLOW STEP 5: RETRIEVAL MODULE")
    print("=" * 70)

    query_embedding = (
        embedding_model.encode(

            [question],

            normalize_embeddings=True
        )[0].tolist()
    )

    results = collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=RETRIEVAL_TOP_K,

        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = (
        results["documents"][0]
    )

    metadatas = (
        results["metadatas"][0]
    )

    distances = (
        results["distances"][0]
    )

    retrieved = []

    for document, metadata, distance in zip(

        documents,

        metadatas,

        distances
    ):

        chunk_id = (
            f"page_{metadata['page_no']}"
            f"_chunk_{metadata['chunk_no']}"
        )

        retrieved.append(
            {
                "id": chunk_id,

                "text": document,

                "metadata": metadata,

                "distance":
                    float(distance)
            }
        )

    print(
        f"Retrieved candidates: "
        f"{len(retrieved)}"
    )

    return retrieved


# ====================================================================
# FLOW STEP 6: USER QUESTION
# ====================================================================
#
# User
#   ↓
# Question
#
# ====================================================================

def get_user_question():

    print("\n" + "=" * 70)
    print("FLOW STEP 6: USER QUESTION")
    print("=" * 70)

    question = input(
        "\nEnter your question: "
    ).strip()

    return question


# ====================================================================
# FLOW STEP 7: QUERY MODULE
# ====================================================================
#
# User Question
#   ↓
# Query Analysis
#   ↓
# Search Query
#
# ====================================================================

def query_module(question):

    print("\n" + "=" * 70)
    print("FLOW STEP 7: QUERY MODULE")
    print("=" * 70)

    system_prompt = """
You are a query processing module for a RAG system.

Convert the user's question into a concise
search query.

Rules:

1. Preserve the original meaning.
2. Do not answer the question.
3. Do not add facts.
4. Return only the search query.
"""

    user_prompt = f"""
User Question:

{question}

Create the best search query for retrieving
relevant information from HR_Policy.pdf.
"""

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
        ]
    )

    search_query = (
        response["message"]["content"]
        .strip()
    )

    print(
        f"\nOriginal Question:\n{question}"
    )

    print(
        f"\nSearch Query:\n{search_query}"
    )

    return search_query


# ====================================================================
# FLOW STEP 8: RERANKING MODULE
# ====================================================================
#
# Question
#       +
# Retrieved Candidates
#       ↓
# CrossEncoder
#       ↓
# Reranked Candidates
#
# ====================================================================

def load_reranker():

    print("\n" + "=" * 70)
    print("FLOW STEP 8: LOAD RERANKER")
    print("=" * 70)

    print(
        f"Loading: {RERANKER_MODEL}"
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print(
        "CrossEncoder loaded."
    )

    return reranker


def rerank_documents(
    question,
    documents,
    reranker
):

    print("\n" + "=" * 70)
    print("FLOW STEP 8: RERANKING MODULE")
    print("=" * 70)

    if not documents:

        return []

    pairs = []

    for document in documents:

        pairs.append(
            (
                question,
                document["text"]
            )
        )

    scores = reranker.predict(
        pairs
    )

    reranked = []

    for document, score in zip(
        documents,
        scores
    ):

        item = dict(document)

        item["reranker_score"] = (
            float(score)
        )

        reranked.append(
            item
        )

    reranked.sort(

        key=lambda item:
            item["reranker_score"],

        reverse=True
    )

    for rank, document in enumerate(
        reranked,
        start=1
    ):

        print(
            f"{rank}. "
            f"{document['id']} | "
            f"Score: "
            f"{document['reranker_score']:.6f}"
        )

    return reranked


# ====================================================================
# FLOW STEP 9: CONTEXT MODULE
# ====================================================================
#
# Reranked Documents
#   ↓
# Top 5
#   ↓
# LLM Context
#
# ====================================================================

def context_module(
    reranked_documents
):

    print("\n" + "=" * 70)
    print("FLOW STEP 9: CONTEXT MODULE")
    print("=" * 70)

    top_documents = (
        reranked_documents[
            :FINAL_TOP_K
        ]
    )

    context_parts = []

    for rank, document in enumerate(
        top_documents,
        start=1
    ):

        metadata = document["metadata"]

        page_no = metadata[
            "page_no"
        ]

        chunk_no = metadata[
            "chunk_no"
        ]

        chunk_id = document["id"]

        chunk_text = document["text"]

        score = document[
            "reranker_score"
        ]

        context = f"""
============================================================
SOURCE {rank}

Page Number: {page_no}

Chunk Number: {chunk_no}

Chunk ID: {chunk_id}

Relevance Score: {score:.6f}

Chunk Text:

{chunk_text}
============================================================
"""

        context_parts.append(
            context
        )

    final_context = "\n".join(
        context_parts
    )

    print(final_context)

    return (
        top_documents,
        final_context
    )


# ====================================================================
# FLOW STEP 10: GENERATION MODULE
# ====================================================================
#
# System Prompt
#       +
# User Prompt
#       +
# Context
#       ↓
# Mistral
#       ↓
# JSON Response
#
# ====================================================================

def generation_module(
    question,
    context
):

    print("\n" + "=" * 70)
    print("FLOW STEP 10: GENERATION MODULE")
    print("=" * 70)

    schema_text = json.dumps(
        ARRAY_SCHEMA,
        indent=4
    )

    system_prompt = f"""
You are an HR policy question answering system.

Use ONLY the retrieved context.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not present in the
   retrieved context, say:

   "I don't know based on HR_Policy.pdf."

4. Return ONLY valid JSON.
5. Top-level response must be an array.
6. Return exactly one answer record.
7. Source must contain supporting chunks.
8. Preserve Page Number.
9. Preserve Chunk Number.
10. Preserve Chunk ID.
11. Include Chunk Text.
12. Relevance Score must be the
    CrossEncoder score.
13. Confidence must be between 0 and 1.
14. Do not create fake sources.

Required JSON Schema:

{schema_text}
"""

    user_prompt = f"""
Question:

{question}

Retrieved Context:

{context}

Answer the question using only this context.
"""

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

    raw_response = (
        response["message"]["content"]
        .strip()
    )

    print(
        "\nRaw LLM Response:"
    )

    print(raw_response)

    return raw_response


# ====================================================================
# FLOW STEP 11: JSON PARSING MODULE
# ====================================================================
#
# Raw LLM Response
#   ↓
# JSON
#   ↓
# Python Object
#
# ====================================================================

def parse_json_response(
    raw_response
):

    print("\n" + "=" * 70)
    print("FLOW STEP 11: JSON PARSING MODULE")
    print("=" * 70)

    try:

        data = json.loads(
            raw_response
        )

    except json.JSONDecodeError as error:

        print(
            f"JSON parsing error: {error}"
        )

        return []

    if isinstance(
        data,
        list
    ):

        return data

    if isinstance(
        data,
        dict
    ):

        if (
            data.get("type") == "array"
            and "items" in data
        ):

            return data["items"]

        if "items" in data:

            return data["items"]

        if "results" in data:

            return data["results"]

        if "data" in data:

            return data["data"]

        if "Answer" in data:

            return [
                data
            ]

    return []


# ====================================================================
# FLOW STEP 12: VALIDATION MODULE
# ====================================================================
#
# LLM Response
#   ↓
# Source Validation
#   ↓
# Confidence Validation
#   ↓
# Final Valid JSON
#
# ====================================================================

def validation_module(
    parsed_response,
    top_documents
):

    print("\n" + "=" * 70)
    print("FLOW STEP 12: VALIDATION MODULE")
    print("=" * 70)

    if not parsed_response:

        print(
            "No valid response received."
        )

        return []

    valid_chunk_ids = {

        document["id"]

        for document in top_documents
    }

    validated_records = []

    for record in parsed_response:

        if not isinstance(
            record,
            dict
        ):

            continue

        answer = str(
            record.get(
                "Answer",
                ""
            )
        ).strip()

        confidence = record.get(
            "Confidence",
            0
        )

        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError
        ):

            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )

        sources = record.get(
            "Source",
            []
        )

        validated_sources = []

        if isinstance(
            sources,
            list
        ):

            for source in sources:

                if not isinstance(
                    source,
                    dict
                ):

                    continue

                chunk_id = source.get(
                    "Chunk ID",
                    ""
                )

                if chunk_id not in valid_chunk_ids:

                    print(
                        f"Rejected source: "
                        f"{chunk_id}"
                    )

                    continue

                try:

                    page_number = int(
                        source.get(
                            "Page Number"
                        )
                    )

                    chunk_number = int(
                        source.get(
                            "Chunk Number"
                        )
                    )

                    relevance_score = float(
                        source.get(
                            "Relevance Score",
                            0
                        )
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    continue

                validated_sources.append(
                    {
                        "Page Number":
                            page_number,

                        "Chunk Number":
                            chunk_number,

                        "Chunk ID":
                            chunk_id,

                        "Chunk Text":
                            str(
                                source.get(
                                    "Chunk Text",
                                    ""
                                )
                            ),

                        "Relevance Score":
                            relevance_score
                    }
                )

        validated_records.append(
            {
                "Answer":
                    answer,

                "Source":
                    validated_sources,

                "Confidence":
                    confidence
            }
        )

    return validated_records


# ====================================================================
# FLOW STEP 13: FINAL ANSWER
# ====================================================================
#
# Validated Response
#   ↓
# Final JSON
#
# ====================================================================

def print_final_answer(
    final_response
):

    print("\n" + "=" * 70)
    print("FLOW STEP 13: FINAL ANSWER")
    print("=" * 70)

    print(
        json.dumps(
            final_response,
            indent=4,
            ensure_ascii=False
        )
    )


# ====================================================================
# MODULAR RAG PIPELINE
# ====================================================================

def run_modular_rag(
    question,
    embedding_model,
    collection,
    reranker
):

    # ------------------------------------------------------------
    # FLOW STEP 7: QUERY MODULE
    # ------------------------------------------------------------

    search_query = query_module(
        question
    )

    # ------------------------------------------------------------
    # FLOW STEP 5: RETRIEVAL MODULE
    #
    # The rewritten search query is used for retrieval.
    # ------------------------------------------------------------

    retrieved_documents = retrieve_documents(

        search_query,

        embedding_model,

        collection
    )

    # ------------------------------------------------------------
    # FLOW STEP 8: RERANKING MODULE
    # ------------------------------------------------------------

    reranked_documents = rerank_documents(

        question,

        retrieved_documents,

        reranker
    )

    # ------------------------------------------------------------
    # FLOW STEP 9: CONTEXT MODULE
    # ------------------------------------------------------------

    (
        top_documents,
        context
    ) = context_module(
        reranked_documents
    )

    # ------------------------------------------------------------
    # FLOW STEP 10: GENERATION MODULE
    # ------------------------------------------------------------

    raw_response = generation_module(

        question,

        context
    )

    # ------------------------------------------------------------
    # FLOW STEP 11: JSON PARSING MODULE
    # ------------------------------------------------------------

    parsed_response = parse_json_response(
        raw_response
    )

    # ------------------------------------------------------------
    # FLOW STEP 12: VALIDATION MODULE
    # ------------------------------------------------------------

    final_response = validation_module(

        parsed_response,

        top_documents
    )

    # ------------------------------------------------------------
    # FLOW STEP 13: FINAL ANSWER
    # ------------------------------------------------------------

    print_final_answer(
        final_response
    )

    return final_response


# ====================================================================
# FLOW STEP 14: WHILE LOOP
# ====================================================================
#
# User Question
#   ↓
# Modular RAG
#   ↓
# Final Answer
#   ↓
# Next Question
#
# ====================================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "MODULAR RAG - METHOD 3"
    )

    print(
        "QUERY + RETRIEVAL + RERANKING "
        "+ CONTEXT + GENERATION + VALIDATION"
    )

    print("=" * 70)

    # ------------------------------------------------------------
    # FLOW STEP 1: OCR
    # ------------------------------------------------------------

    pages = extract_text_with_ocr(
        PDF_PATH
    )

    # ------------------------------------------------------------
    # FLOW STEP 2: CHUNKING
    # ------------------------------------------------------------

    chunks = create_chunks(
        pages
    )

    if not chunks:

        print(
            "No chunks created."
        )

        return

    # ------------------------------------------------------------
    # FLOW STEP 3: EMBEDDING
    # ------------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    # ------------------------------------------------------------
    # FLOW STEP 4: VECTOR DATABASE
    # ------------------------------------------------------------

    collection = (
        create_vector_database()
    )

    store_documents(

        collection,

        embedding_model,

        chunks
    )

    # ------------------------------------------------------------
    # FLOW STEP 8: CROSSENCODER
    # ------------------------------------------------------------

    reranker = load_reranker()

    # ------------------------------------------------------------
    # FLOW STEP 14: WHILE LOOP
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FLOW STEP 14: ASK QUESTIONS")
    print("=" * 70)

    while True:

        question = input(
            "\nEnter your question "
            "(type 'exit' to stop): "
        ).strip()

        # --------------------------------------------------------
        # EXIT
        # --------------------------------------------------------

        if question.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print(
                "\nExiting Modular RAG."
            )

            break

        # --------------------------------------------------------
        # EMPTY QUESTION
        # --------------------------------------------------------

        if not question:

            print(
                "Please enter a question."
            )

            continue

        # --------------------------------------------------------
        # RUN MODULAR RAG
        # --------------------------------------------------------

        run_modular_rag(

            question,

            embedding_model,

            collection,

            reranker
        )


# ====================================================================
# APPLICATION ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()