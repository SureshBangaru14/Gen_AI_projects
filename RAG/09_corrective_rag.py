# ====================================================================
# FILE: 08_corrective_rag.py
# ====================================================================
#
# CORRECTIVE RAG (CRAG)
#
# Complete pipeline:
#
# PDF
#   ↓
# OCR
#   ↓
# Recursive Chunking
#   ↓
# SentenceTransformer
#   ↓
# ChromaDB
#   ↓
# User Question
#   ↓
# Initial Retrieval
#   ↓
# CrossEncoder Reranking
#   ↓
# Relevance Evaluation
#   ↓
# ┌─────────────────────────────┐
# │ Is retrieval relevant?      │
# └──────────────┬──────────────┘
#                │
#        ┌───────┴────────┐
#        ↓                ↓
#      YES                NO
#        ↓                ↓
#  USE CONTEXT       CORRECT RETRIEVAL
#        │                │
#        │          Query Rewriting
#        │                ↓
#        │          New Retrieval
#        │                ↓
#        │          CrossEncoder
#        │                ↓
#        │         Corrected Context
#        │                │
#        └────────┬───────┘
#                 ↓
#               MISTRAL
#                 ↓
#            JSON Answer
#                 ↓
#         Source + Confidence
#
# ====================================================================


# ====================================================================
# INSTALLATION
# ====================================================================
#
# pip install pytesseract
# pip install pdf2image
# pip install pillow
# pip install sentence-transformers
# pip install chromadb
# pip install langchain-text-splitters
# pip install cross-encoder
# pip install ollama
#
# Ubuntu:
#
# sudo apt update
# sudo apt install tesseract-ocr
# sudo apt install poppler-utils
#
# ====================================================================


import os
import json

import chromadb
import pytesseract

from pdf2image import convert_from_path

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

import ollama


# ====================================================================
# CONFIGURATION
# ====================================================================

PDF_PATH = (
    "/home/suresh/Gen_AI_Practice/Generative-AI/"
    "Vector_Database/HR_Policy.pdf"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

LLM_MODEL = "mistral"

CHROMA_PATH = "chroma_corrective_hr"

COLLECTION_NAME = "hr_policy_corrective"

OCR_DPI = 250

CHUNK_SIZE = 600

CHUNK_OVERLAP = 100

RETRIEVAL_TOP_K = 10

FINAL_TOP_K = 5

MAX_CORRECTION_ATTEMPTS = 2


# ====================================================================
# FLOW STEP 1: OCR
# ====================================================================
#
# PDF
#   ↓
# PDF Pages
#   ↓
# Images
#   ↓
# Tesseract OCR
#   ↓
# Page-wise Text
#
# ====================================================================


def extract_text_from_pdf(pdf_path):

    print("\n" + "=" * 70)
    print("FLOW STEP 1: OCR")
    print("=" * 70)

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

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
            f"Processing page "
            f"{page_no}/{len(pages)}..."
        )

        text = pytesseract.image_to_string(
            image,
            config="--psm 6"
        )

        text = text.strip()

        page_data.append(
            {
                "page_no": page_no,
                "page_data": text
            }
        )

    print(
        f"OCR completed: "
        f"{len(page_data)} pages"
    )

    return page_data


# ====================================================================
# FLOW STEP 2: RECURSIVE CHUNKING
# ====================================================================
#
# Page Text
#   ↓
# RecursiveCharacterTextSplitter
#   ↓
# Chunks
#
# ====================================================================


def create_chunks(page_data):

    print("\n" + "=" * 70)
    print("FLOW STEP 2: RECURSIVE CHUNKING")
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

    for page in page_data:

        page_no = page["page_no"]

        text = page["page_data"]

        if not text.strip():

            continue

        page_chunks = splitter.split_text(
            text
        )

        for chunk_no, chunk_text in enumerate(
            page_chunks,
            start=1
        ):

            chunk_id = (
                f"page_{page_no}_chunk_{chunk_no}"
            )

            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "page_no": page_no,
                    "chunk_no": chunk_no,
                    "file_name": os.path.basename(
                        PDF_PATH
                    )
                }
            )

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    return chunks


# ====================================================================
# FLOW STEP 3: LOAD EMBEDDING MODEL
# ====================================================================
#
# Chunk
#   ↓
# SentenceTransformer
#   ↓
# Embedding Vector
#
# ====================================================================


def load_embedding_model():

    print("\n" + "=" * 70)
    print("FLOW STEP 3: LOAD EMBEDDING MODEL")
    print("=" * 70)

    print(
        f"Loading model: "
        f"{EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return model


# ====================================================================
# FLOW STEP 4: CREATE / UPDATE CHROMADB
# ====================================================================
#
# Chunks
#   ↓
# Embeddings
#   ↓
# ChromaDB
#
# ====================================================================


def create_vector_database(
    chunks,
    embedding_model
):

    print("\n" + "=" * 70)
    print("FLOW STEP 4: CHROMADB")
    print("=" * 70)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "file_name": chunk["file_name"],
            "page_no": chunk["page_no"],
            "chunk_no": chunk["chunk_no"]
        }
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(
        f"ChromaDB records: "
        f"{collection.count()}"
    )

    return collection


# ====================================================================
# FLOW STEP 5: LOAD CROSSENCODER
# ====================================================================
#
# Question + Chunk
#       ↓
# CrossEncoder
#       ↓
# Relevance Score
#
# ====================================================================


def load_reranker():

    print("\n" + "=" * 70)
    print("FLOW STEP 5: LOAD CROSSENCODER")
    print("=" * 70)

    print(
        f"Loading model: "
        f"{RERANKER_MODEL}"
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    return reranker


# ====================================================================
# FLOW STEP 6: INITIAL RETRIEVAL
# ====================================================================
#
# USER QUESTION
#       ↓
# SentenceTransformer
#       ↓
# Query Embedding
#       ↓
# ChromaDB
#       ↓
# Top 10 Candidate Chunks
#
# ====================================================================


def retrieve_chunks(
    question,
    collection,
    embedding_model
):

    print("\n" + "=" * 70)
    print("FLOW STEP 6: INITIAL RETRIEVAL")
    print("=" * 70)

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=RETRIEVAL_TOP_K
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    chunks = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        page_no = metadata.get(
            "page_no"
        )

        chunk_no = metadata.get(
            "chunk_no"
        )

        chunk_id = (
            f"page_{page_no}_chunk_{chunk_no}"
        )

        vector_score = (
            1 / (1 + distance)
        )

        chunks.append(
            {
                "id": chunk_id,
                "text": document,
                "page_no": page_no,
                "chunk_no": chunk_no,
                "file_name": metadata.get(
                    "file_name",
                    os.path.basename(
                        PDF_PATH
                    )
                ),
                "vector_distance": float(
                    distance
                ),
                "vector_score": float(
                    vector_score
                )
            }
        )

    print(
        f"Retrieved candidates: "
        f"{len(chunks)}"
    )

    return chunks


# ====================================================================
# FLOW STEP 7: CROSSENCODER RERANKING
# ====================================================================
#
# Question
#    +
# Candidates
#    ↓
# CrossEncoder
#    ↓
# Relevance Scores
#    ↓
# Top 5
#
# ====================================================================


def rerank_chunks(
    question,
    chunks,
    reranker
):

    print("\n" + "=" * 70)
    print("FLOW STEP 7: CROSSENCODER RERANKING")
    print("=" * 70)

    if not chunks:

        return []

    pairs = [
        [
            question,
            chunk["text"]
        ]
        for chunk in chunks
    ]

    scores = reranker.predict(
        pairs
    )

    ranked_chunks = []

    for chunk, score in zip(
        chunks,
        scores
    ):

        item = dict(chunk)

        item[
            "rerank_score"
        ] = float(score)

        ranked_chunks.append(
            item
        )

    ranked_chunks.sort(
        key=lambda x:
            x["rerank_score"],
        reverse=True
    )

    ranked_chunks = ranked_chunks[
        :FINAL_TOP_K
    ]

    for rank, chunk in enumerate(
        ranked_chunks,
        start=1
    ):

        print(
            f"{rank}. "
            f"{chunk['id']} | "
            f"Score="
            f"{chunk['rerank_score']:.4f}"
        )

    return ranked_chunks


# ====================================================================
# FLOW STEP 8: CRAG RELEVANCE EVALUATION
# ====================================================================
#
# Retrieved Chunks
#       +
# Question
#       ↓
# Relevance Evaluator
#       ↓
# ┌─────────────────────┐
# │ Relevant?           │
# └──────────┬──────────┘
#            │
#       ┌────┴────┐
#       ↓         ↓
#      YES        NO
#       ↓         ↓
#    Accept     Correct
#
# ====================================================================


def evaluate_retrieval(
    question,
    ranked_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 8: CRAG RELEVANCE EVALUATION")
    print("=" * 70)

    if not ranked_chunks:

        return {
            "decision": "incorrect",
            "score": 0.0,
            "reason":
                "No documents were retrieved."
        }

    context = ""

    for index, chunk in enumerate(
        ranked_chunks,
        start=1
    ):

        context += f"""
SOURCE {index}
Page: {chunk['page_no']}
Chunk: {chunk['chunk_no']}
Chunk ID: {chunk['id']}

Text:
{chunk['text']}
"""

    prompt = f"""
You are a retrieval evaluator in a
Corrective RAG system.

Your job is to determine whether the
retrieved documents are relevant enough
to answer the user's question.

QUESTION:
{question}

RETRIEVED DOCUMENTS:
{context}

Classify the retrieval as:

"correct"
- The retrieved documents contain useful
  information needed to answer the question.

"incorrect"
- The retrieved documents do not contain
  sufficient useful information.

Return ONLY JSON:

{{
    "decision": "correct",
    "score": 0.95,
    "reason": "short explanation"
}}

The score must be between 0 and 1.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json"
    )

    raw_response = response[
        "message"
    ][
        "content"
    ]

    try:

        result = json.loads(
            raw_response
        )

        decision = result.get(
            "decision",
            "incorrect"
        )

        score = result.get(
            "score",
            0.0
        )

        reason = result.get(
            "reason",
            ""
        )

        if decision not in {
            "correct",
            "incorrect"
        }:

            decision = "incorrect"

        try:

            score = float(score)

        except (
            TypeError,
            ValueError
        ):

            score = 0.0

        score = max(
            0.0,
            min(
                1.0,
                score
            )
        )

        print(
            f"CRAG decision: "
            f"{decision}"
        )

        print(
            f"CRAG score: "
            f"{score:.4f}"
        )

        print(
            f"Reason: {reason}"
        )

        return {
            "decision": decision,
            "score": score,
            "reason": reason
        }

    except json.JSONDecodeError:

        print(
            "Could not parse CRAG evaluation."
        )

        return {
            "decision": "incorrect",
            "score": 0.0,
            "reason":
                "Invalid evaluator response."
        }


# ====================================================================
# FLOW STEP 9: QUERY CORRECTION
# ====================================================================
#
# Incorrect Retrieval
#       ↓
# Mistral
#       ↓
# Query Rewrite
#       ↓
# Corrected Query
#
# ====================================================================


def correct_query(
    question,
    evaluation_reason
):

    print("\n" + "=" * 70)
    print("FLOW STEP 9: QUERY CORRECTION")
    print("=" * 70)

    prompt = f"""
You are a query correction component
in a Corrective RAG system.

The original question was:

{question}

The retrieval evaluator reported:

{evaluation_reason}

Rewrite the question into a better
search query that is more likely to retrieve
the correct information from an HR Policy
document.

Do not answer the question.

Return ONLY JSON:

{{
    "query": "corrected search query"
}}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json"
    )

    raw_response = response[
        "message"
    ][
        "content"
    ]

    try:

        result = json.loads(
            raw_response
        )

        corrected_query = result.get(
            "query",
            question
        )

        print(
            f"Original query:\n"
            f"{question}"
        )

        print(
            f"\nCorrected query:\n"
            f"{corrected_query}"
        )

        return corrected_query

    except json.JSONDecodeError:

        print(
            "Query correction failed."
        )

        return question


# ====================================================================
# FLOW STEP 10: CORRECTIVE RETRIEVAL
# ====================================================================
#
# Corrected Query
#       ↓
# SentenceTransformer
#       ↓
# ChromaDB
#       ↓
# New Candidate Chunks
#
# ====================================================================


def corrective_retrieval(
    corrected_query,
    collection,
    embedding_model
):

    print("\n" + "=" * 70)
    print("FLOW STEP 10: CORRECTIVE RETRIEVAL")
    print("=" * 70)

    return retrieve_chunks(
        corrected_query,
        collection,
        embedding_model
    )


# ====================================================================
# FLOW STEP 11: BUILD FINAL CONTEXT
# ====================================================================
#
# Correct / Corrected Chunks
#       ↓
# Structured Context
#       ↓
# Mistral
#
# ====================================================================


def build_context(
    ranked_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 11: BUILD FINAL CONTEXT")
    print("=" * 70)

    context_parts = []

    for index, chunk in enumerate(
        ranked_chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {index}

Page Number:
{chunk['page_no']}

Chunk Number:
{chunk['chunk_no']}

Chunk ID:
{chunk['id']}

Vector Score:
{chunk.get('vector_score', 0.0):.4f}

CrossEncoder Relevance Score:
{chunk.get('rerank_score', 0.0):.4f}

Chunk Text:
{chunk['text']}
"""
        )

    return "\n".join(
        context_parts
    )


# ====================================================================
# FLOW STEP 12: SYSTEM PROMPT
# ====================================================================
#
# Final answer generation rules.
#
# ====================================================================


def build_system_prompt():

    return """
You are an HR Policy question-answering assistant.

Answer ONLY from the supplied HR Policy context.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Every factual statement must be
   supported by the supplied context.
4. If the context does not contain the answer,
   return:

   "I don't know based on HR_Policy.pdf."

5. Always provide source information.
6. Confidence must be between 0 and 1.
7. Return ONLY valid JSON.
8. Return a JSON array.
9. Use exactly these field names:

   Answer
   Source
   Confidence

Source must contain:

Page Number
Chunk Number
Chunk ID
Chunk Text
Relevance Score

Required format:

[
  {
    "Answer": "string",
    "Source": [
      {
        "Page Number": 1,
        "Chunk Number": 1,
        "Chunk ID": "page_1_chunk_1",
        "Chunk Text": "string",
        "Relevance Score": 0.0
      }
    ],
    "Confidence": 0.0
  }
]
"""


# ====================================================================
# FLOW STEP 13: FINAL GENERATION
# ====================================================================
#
# Question
#    +
# Corrected Context
#    ↓
# Mistral
#    ↓
# Answer
#
# ====================================================================


def generate_answer(
    question,
    ranked_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 13: MISTRAL GENERATION")
    print("=" * 70)

    context = build_context(
        ranked_chunks
    )

    prompt = f"""
Answer the following HR Policy question.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Return ONLY the required JSON array.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content":
                    build_system_prompt()
            },
            {
                "role": "user",
                "content":
                    prompt
            }
        ],
        format="json"
    )

    return response[
        "message"
    ][
        "content"
    ]


# ====================================================================
# FLOW STEP 14: JSON PARSING
# ====================================================================
#
# LLM Response
#       ↓
# JSON
#
# Handles:
#
# Direct array
#
# [
#   {...}
# ]
#
# And schema wrapper:
#
# {
#   "type": "array",
#   "items": [...]
# }
#
# ====================================================================


def parse_json_response(
    raw_response
):

    print("\n" + "=" * 70)
    print("FLOW STEP 14: JSON PARSING")
    print("=" * 70)

    try:

        data = json.loads(
            raw_response
        )

        # -----------------------------------------------------------
        # Handle array schema wrapper
        # -----------------------------------------------------------

        if (
            isinstance(data, dict)
            and data.get("type") == "array"
            and "items" in data
        ):

            data = data["items"]

        # -----------------------------------------------------------
        # Handle common wrappers
        # -----------------------------------------------------------

        elif isinstance(data, dict):

            if "items" in data:

                data = data["items"]

            elif "results" in data:

                data = data["results"]

            elif "data" in data:

                data = data["data"]

            else:

                data = [data]

        if not isinstance(
            data,
            list
        ):

            data = [data]

        return data

    except json.JSONDecodeError:

        print(
            "JSON parsing failed."
        )

        return [
            {
                "Answer":
                    "I don't know based on "
                    "HR_Policy.pdf.",
                "Source": [],
                "Confidence": 0.0
            }
        ]


# ====================================================================
# FLOW STEP 15: SOURCE VALIDATION
# ====================================================================
#
# LLM Source
#       ↓
# Compare with actual retrieved chunks
#       ↓
# Valid Source
#
# ====================================================================


def validate_sources(
    answer_data,
    ranked_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 15: SOURCE VALIDATION")
    print("=" * 70)

    valid_ids = {
        chunk["id"]
        for chunk in ranked_chunks
    }

    for item in answer_data:

        if not isinstance(
            item,
            dict
        ):

            continue

        sources = item.get(
            "Source",
            []
        )

        valid_sources = []

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
                    "Chunk ID"
                )

                if chunk_id in valid_ids:

                    valid_sources.append(
                        source
                    )

        item["Source"] = (
            valid_sources
        )

        if "Answer" not in item:

            item["Answer"] = (
                "I don't know based on "
                "HR_Policy.pdf."
            )

        try:

            confidence = float(
                item.get(
                    "Confidence",
                    0.0
                )
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

        item["Confidence"] = (
            confidence
        )

    return answer_data


# ====================================================================
# FLOW STEP 16: FINAL ANSWER
# ====================================================================
#
# Validated JSON
#       ↓
# Final Output
#
# ====================================================================


def print_final_answer(
    answer_data
):

    print("\n" + "=" * 70)
    print("FLOW STEP 16: FINAL ANSWER")
    print("=" * 70)

    print(
        json.dumps(
            answer_data,
            indent=2,
            ensure_ascii=False
        )
    )


# ====================================================================
# FLOW STEP 17: COMPLETE CRAG PIPELINE
# ====================================================================
#
# QUESTION
#    ↓
# INITIAL RETRIEVAL
#    ↓
# RERANK
#    ↓
# EVALUATE
#    ↓
# ┌─────────────────────┐
# │ Retrieval Correct?  │
# └──────────┬──────────┘
#            │
#       ┌────┴────┐
#       ↓         ↓
#      YES        NO
#       ↓         ↓
#    ACCEPT     CORRECT
#       │         │
#       │     QUERY REWRITE
#       │         │
#       │     RETRIEVE AGAIN
#       │         │
#       │       RERANK
#       │         │
#       └────┬────┘
#            ↓
#       FINAL CONTEXT
#            ↓
#          MISTRAL
#            ↓
#          JSON
#
# ====================================================================


def corrective_rag(
    question,
    collection,
    embedding_model,
    reranker
):

    print("\n")
    print("=" * 70)
    print("CORRECTIVE RAG STARTED")
    print("=" * 70)

    current_query = question

    best_chunks = []

    # ---------------------------------------------------------------
    # CRAG correction loop
    # ---------------------------------------------------------------

    for attempt in range(
        0,
        MAX_CORRECTION_ATTEMPTS + 1
    ):

        print("\n")
        print("=" * 70)

        print(
            f"CRAG ATTEMPT "
            f"{attempt + 1}/"
            f"{MAX_CORRECTION_ATTEMPTS + 1}"
        )

        print("=" * 70)

        # -----------------------------------------------------------
        # INITIAL / CORRECTIVE RETRIEVAL
        # -----------------------------------------------------------

        retrieved_chunks = (
            retrieve_chunks(
                current_query,
                collection,
                embedding_model
            )
        )

        # -----------------------------------------------------------
        # RERANK
        # -----------------------------------------------------------

        ranked_chunks = (
            rerank_chunks(
                current_query,
                retrieved_chunks,
                reranker
            )
        )

        # -----------------------------------------------------------
        # Save latest result
        # -----------------------------------------------------------

        if ranked_chunks:

            best_chunks = ranked_chunks

        # -----------------------------------------------------------
        # CRAG EVALUATION
        # -----------------------------------------------------------

        evaluation = (
            evaluate_retrieval(
                question,
                ranked_chunks
            )
        )

        # -----------------------------------------------------------
        # CORRECT RETRIEVAL
        # -----------------------------------------------------------

        if evaluation[
            "decision"
        ] == "correct":

            print(
                "\nCRAG:"
            )

            print(
                "Retrieval is CORRECT."
            )

            print(
                "Using retrieved context."
            )

            break

        # -----------------------------------------------------------
        # INCORRECT RETRIEVAL
        # -----------------------------------------------------------

        print(
            "\nCRAG:"
        )

        print(
            "Retrieval is INCORRECT."
        )

        # -----------------------------------------------------------
        # Maximum correction attempts
        # -----------------------------------------------------------

        if attempt >= MAX_CORRECTION_ATTEMPTS:

            print(
                "Maximum correction attempts reached."
            )

            break

        # -----------------------------------------------------------
        # Correct query
        # -----------------------------------------------------------

        current_query = (
            correct_query(
                question,
                evaluation["reason"]
            )
        )

        print(
            "\nCRAG will perform "
            "corrective retrieval."
        )

    # =================================================================
    # NO USABLE CONTEXT
    # =================================================================

    if not best_chunks:

        print(
            "\nNo usable context found."
        )

        return [
            {
                "Answer":
                    "I don't know based on "
                    "HR_Policy.pdf.",
                "Source": [],
                "Confidence": 0.0
            }
        ]

    # =================================================================
    # FINAL GENERATION
    # =================================================================

    raw_response = generate_answer(
        question,
        best_chunks
    )

    print(
        "\nRaw LLM Response:"
    )

    print(
        raw_response
    )

    # =================================================================
    # JSON
    # =================================================================

    answer_data = (
        parse_json_response(
            raw_response
        )
    )

    # =================================================================
    # SOURCE VALIDATION
    # =================================================================

    answer_data = (
        validate_sources(
            answer_data,
            best_chunks
        )
    )

    return answer_data


# ====================================================================
# FLOW STEP 18: MAIN
# ====================================================================
#
# Complete initialization:
#
# PDF
#   ↓
# OCR
#   ↓
# Chunking
#   ↓
# Embeddings
#   ↓
# ChromaDB
#   ↓
# CrossEncoder
#   ↓
# CRAG
#
# ====================================================================


def main():

    print("\n")
    print("=" * 70)
    print("CORRECTIVE RAG (CRAG) - HR POLICY")
    print("=" * 70)

    # ---------------------------------------------------------------
    # OCR
    # ---------------------------------------------------------------

    page_data = (
        extract_text_from_pdf(
            PDF_PATH
        )
    )

    # ---------------------------------------------------------------
    # Chunking
    # ---------------------------------------------------------------

    chunks = create_chunks(
        page_data
    )

    # ---------------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    # ---------------------------------------------------------------
    # ChromaDB
    # ---------------------------------------------------------------

    collection = (
        create_vector_database(
            chunks,
            embedding_model
        )
    )

    # ---------------------------------------------------------------
    # CrossEncoder
    # ---------------------------------------------------------------

    reranker = load_reranker()

    # ---------------------------------------------------------------
    # QUESTION LOOP
    # ---------------------------------------------------------------

    while True:

        print("\n")
        print("=" * 70)
        print("CRAG QUESTION")
        print(
            "Type 'exit', 'quit', or 'q' to stop."
        )
        print("=" * 70)

        question = input(
            "\nQuestion: "
        ).strip()

        # -----------------------------------------------------------
        # Exit
        # -----------------------------------------------------------

        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "\nExiting Corrective RAG..."
            )

            break

        # -----------------------------------------------------------
        # Empty question
        # -----------------------------------------------------------

        if not question:

            print(
                "\nPlease enter a question."
            )

            continue

        # -----------------------------------------------------------
        # Run CRAG
        # -----------------------------------------------------------

        answer = corrective_rag(
            question,
            collection,
            embedding_model,
            reranker
        )

        # -----------------------------------------------------------
        # Final JSON
        # -----------------------------------------------------------

        print_final_answer(
            answer
        )


# ====================================================================
# PROGRAM ENTRY POINT
# ====================================================================


if __name__ == "__main__":

    main()