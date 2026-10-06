# ====================================================================
# FILE: 07_self_rag.py
# ====================================================================
#
# SELF-RAG FLOW
#
# PDF
#   ↓
# OCR
#   ↓
# Recursive Chunking
#   ↓
# SentenceTransformer Embeddings
#   ↓
# ChromaDB
#   ↓
# USER QUESTION
#   ↓
# INITIAL RETRIEVAL
#   ↓
# SELF-RELEVANCE CHECK
#   ↓
# ┌──────────────────────────────┐
# │ Are retrieved chunks useful? │
# └──────────────┬───────────────┘
#                │
#         ┌──────┴──────┐
#         ↓             ↓
#        YES            NO
#         ↓             ↓
#     RERANK        RE-RETRIEVE
#         ↓             │
#      GENERATE         │
#         ↓             │
#  SELF-SUPPORT CHECK   │
#         ↓             │
#    ┌────┴────┐        │
#    ↓         ↓        │
#   YES        NO       │
#    ↓         ↓        │
# FINAL     RE-RETRIEVE─┘
# ANSWER
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

CHROMA_PATH = "chroma_self_rag_hr"

COLLECTION_NAME = "hr_policy_self_rag"

OCR_DPI = 250

CHUNK_SIZE = 600

CHUNK_OVERLAP = 100

RETRIEVAL_TOP_K = 10

FINAL_TOP_K = 5

MAX_SELF_RAG_ITERATIONS = 3


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
        f"Total chunks: {len(chunks)}"
    )

    return chunks


# ====================================================================
# FLOW STEP 3: LOAD EMBEDDING MODEL
# ====================================================================
#
# Chunk Text
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
        f"Loading: {EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return model


# ====================================================================
# FLOW STEP 4: STORE IN CHROMADB
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
# FLOW STEP 5: USER QUESTION
# ====================================================================
#
# User
#   ↓
# Question
#
# ====================================================================


def get_user_question():

    print("\n" + "=" * 70)
    print("FLOW STEP 5: USER QUESTION")
    print("=" * 70)

    question = input(
        "\nQuestion: "
    ).strip()

    return question


# ====================================================================
# FLOW STEP 6: INITIAL RETRIEVAL
# ====================================================================
#
# Question
#   ↓
# SentenceTransformer
#   ↓
# ChromaDB
#   ↓
# Top 10 Candidates
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
                    1 / (1 + distance)
                )
            }
        )

    print(
        f"Retrieved candidates: "
        f"{len(chunks)}"
    )

    return chunks


# ====================================================================
# FLOW STEP 7: LOAD CROSSENCODER
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
    print("FLOW STEP 7: LOAD CROSSENCODER")
    print("=" * 70)

    print(
        f"Loading: {RERANKER_MODEL}"
    )

    return CrossEncoder(
        RERANKER_MODEL
    )


# ====================================================================
# FLOW STEP 8: RERANKING
# ====================================================================
#
# Question
#    +
# Retrieved Chunks
#    ↓
# CrossEncoder
#    ↓
# Ranked Chunks
#
# ====================================================================


def rerank_chunks(
    question,
    chunks,
    reranker
):

    print("\n" + "=" * 70)
    print("FLOW STEP 8: CROSSENCODER RERANKING")
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

    print(
        f"Final candidates: "
        f"{len(ranked_chunks)}"
    )

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
# FLOW STEP 9: SELF-RELEVANCE EVALUATION
# ====================================================================
#
# This is the first major Self-RAG component.
#
# Retrieved Context
#       ↓
# LLM evaluates
#       ↓
# Is context relevant?
#
# Result:
#
# {
#   "relevant": true,
#   "reason": "..."
# }
#
# ====================================================================


def evaluate_retrieval_relevance(
    question,
    ranked_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 9: SELF-RELEVANCE EVALUATION")
    print("=" * 70)

    if not ranked_chunks:

        return {
            "relevant": False,
            "reason":
                "No chunks were retrieved."
        }

    context = ""

    for index, chunk in enumerate(
        ranked_chunks,
        start=1
    ):

        context += (
            f"\nSOURCE {index}\n"
            f"Page: {chunk['page_no']}\n"
            f"Chunk: {chunk['chunk_no']}\n"
            f"Text: {chunk['text']}\n"
        )

    prompt = f"""
You are a retrieval evaluator.

Determine whether the retrieved context
contains information relevant to answering
the question.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Return ONLY JSON:

{{
    "relevant": true,
    "reason": "short explanation"
}}

Use false if the context does not contain
useful information for answering the question.
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

        relevant = result.get(
            "relevant",
            False
        )

        reason = result.get(
            "reason",
            ""
        )

        print(
            f"Relevant: {relevant}"
        )

        print(
            f"Reason: {reason}"
        )

        return {
            "relevant":
                bool(relevant),
            "reason":
                reason
        }

    except json.JSONDecodeError:

        print(
            "Could not parse relevance evaluation."
        )

        return {
            "relevant": False,
            "reason":
                "Invalid evaluator response."
        }


# ====================================================================
# FLOW STEP 10: QUERY REWRITE
# ====================================================================
#
# If retrieval is poor:
#
# Original Question
#       ↓
# Mistral
#       ↓
# Better Search Query
#
# ====================================================================


def rewrite_query(
    question
):

    print("\n" + "=" * 70)
    print("FLOW STEP 10: QUERY REWRITE")
    print("=" * 70)

    prompt = f"""
Rewrite this HR Policy question into
a better search query.

Do not answer the question.

Make the query specific and preserve
the original intent.

QUESTION:
{question}

Return ONLY JSON:

{{
    "query": "rewritten search query"
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

        data = json.loads(
            raw_response
        )

        rewritten_query = data.get(
            "query",
            question
        )

        print(
            f"Original: "
            f"{question}"
        )

        print(
            f"Rewritten: "
            f"{rewritten_query}"
        )

        return rewritten_query

    except json.JSONDecodeError:

        print(
            "Query rewrite failed."
        )

        return question


# ====================================================================
# FLOW STEP 11: BUILD CONTEXT
# ====================================================================
#
# Top Ranked Chunks
#       ↓
# Structured Context
#
# ====================================================================


def build_context(
    ranked_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 11: BUILD CONTEXT")
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

Relevance Score:
{chunk['rerank_score']:.4f}

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
# LLM answer rules.
#
# ====================================================================


def build_system_prompt():

    return """
You are an HR Policy question-answering assistant.

Answer ONLY using the supplied HR Policy context.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Every factual statement must be supported
   by the retrieved context.
4. If the answer cannot be supported,
   return:

   "I don't know based on HR_Policy.pdf."

5. Always provide source information.
6. Confidence must be between 0 and 1.
7. Return ONLY valid JSON.
8. Return a JSON array.
9. Use the exact field names.

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
# FLOW STEP 13: ANSWER GENERATION
# ====================================================================
#
# Question
#    +
# Context
#    ↓
# Mistral
#    ↓
# Draft Answer
#
# ====================================================================


def generate_answer(
    question,
    ranked_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 13: ANSWER GENERATION")
    print("=" * 70)

    context = build_context(
        ranked_chunks
    )

    prompt = f"""
Answer this HR Policy question.

QUESTION:
{question}

CONTEXT:
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
# FLOW STEP 14: SELF-ANSWER EVALUATION
# ====================================================================
#
# This is the second major Self-RAG component.
#
# Draft Answer
#       +
# Retrieved Context
#       ↓
# LLM evaluates
#       ↓
# Is answer supported?
#
# ====================================================================


def evaluate_answer_support(
    question,
    raw_answer,
    ranked_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 14: SELF-ANSWER EVALUATION")
    print("=" * 70)

    context = build_context(
        ranked_chunks
    )

    prompt = f"""
You are an answer quality evaluator.

Determine whether the generated answer
is fully supported by the supplied HR Policy
context.

QUESTION:
{question}

GENERATED ANSWER:
{raw_answer}

CONTEXT:
{context}

Check:

1. Is the answer relevant?
2. Are the factual claims supported?
3. Did the answer invent information?
4. Does the context actually support the answer?

Return ONLY JSON:

{{
    "supported": true,
    "score": 0.95,
    "reason": "short explanation"
}}

Score must be between 0 and 1.
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

        supported = result.get(
            "supported",
            False
        )

        score = result.get(
            "score",
            0.0
        )

        reason = result.get(
            "reason",
            ""
        )

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
            f"Supported: {supported}"
        )

        print(
            f"Support Score: "
            f"{score:.4f}"
        )

        print(
            f"Reason: {reason}"
        )

        return {
            "supported":
                bool(supported),
            "score":
                score,
            "reason":
                reason
        }

    except json.JSONDecodeError:

        print(
            "Could not parse answer evaluation."
        )

        return {
            "supported": False,
            "score": 0.0,
            "reason":
                "Invalid evaluator response."
        }


# ====================================================================
# FLOW STEP 15: JSON PARSING
# ====================================================================
#
# LLM response
#       ↓
# JSON object
#
# Handles:
#
# [
#   {...}
# ]
#
# and:
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
    print("FLOW STEP 15: JSON PARSING")
    print("=" * 70)

    try:

        data = json.loads(
            raw_response
        )

        # -----------------------------------------------------------
        # Schema wrapper
        # -----------------------------------------------------------

        if (
            isinstance(data, dict)
            and data.get("type") == "array"
            and "items" in data
        ):

            data = data["items"]

        # -----------------------------------------------------------
        # Common wrappers
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
# FLOW STEP 16: SOURCE VALIDATION
# ====================================================================
#
# LLM Sources
#       ↓
# Compare against retrieved chunks
#       ↓
# Valid Sources
#
# ====================================================================


def validate_sources(
    answer_data,
    ranked_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 16: SOURCE VALIDATION")
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
# FLOW STEP 17: FINAL ANSWER
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
    print("FLOW STEP 17: FINAL ANSWER")
    print("=" * 70)

    print(
        json.dumps(
            answer_data,
            indent=2,
            ensure_ascii=False
        )
    )


# ====================================================================
# FLOW STEP 18: SELF-RAG PIPELINE
# ====================================================================
#
# QUESTION
#    ↓
# RETRIEVE
#    ↓
# SELF-RELEVANCE CHECK
#    ↓
# ┌───────────────┐
# │ Relevant?     │
# └───────┬───────┘
#         │
#     ┌───┴────┐
#     ↓        ↓
#    YES       NO
#     ↓        ↓
#  RERANK    REWRITE
#     │        │
#     │        └────→ RETRIEVE AGAIN
#     ↓
#  GENERATE
#     ↓
# SELF-ANSWER CHECK
#     ↓
# ┌───────────────┐
# │ Supported?    │
# └───────┬───────┘
#         │
#     ┌───┴────┐
#     ↓        ↓
#    YES       NO
#     ↓        ↓
# FINAL      RETRIEVE AGAIN
#
# ====================================================================


def self_rag(
    question,
    collection,
    embedding_model,
    reranker
):

    print("\n")
    print("=" * 70)
    print("SELF-RAG STARTED")
    print("=" * 70)

    current_question = question

    best_chunks = []

    # ---------------------------------------------------------------
    # Self-RAG iteration loop
    # ---------------------------------------------------------------

    for iteration in range(
        1,
        MAX_SELF_RAG_ITERATIONS + 1
    ):

        print("\n")
        print("=" * 70)

        print(
            f"SELF-RAG ITERATION "
            f"{iteration}/"
            f"{MAX_SELF_RAG_ITERATIONS}"
        )

        print("=" * 70)

        # -----------------------------------------------------------
        # STEP A: RETRIEVAL
        # -----------------------------------------------------------

        retrieved_chunks = (
            retrieve_chunks(
                current_question,
                collection,
                embedding_model
            )
        )

        # -----------------------------------------------------------
        # STEP B: RERANK
        # -----------------------------------------------------------

        ranked_chunks = (
            rerank_chunks(
                current_question,
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
        # STEP C: SELF-RELEVANCE CHECK
        # -----------------------------------------------------------

        relevance = (
            evaluate_retrieval_relevance(
                current_question,
                ranked_chunks
            )
        )

        # -----------------------------------------------------------
        # If context is NOT relevant:
        # rewrite query and retrieve again.
        # -----------------------------------------------------------

        if not relevance["relevant"]:

            print(
                "\nSelf-RAG decision:"
            )

            print(
                "Retrieved context is NOT relevant."
            )

            if iteration < MAX_SELF_RAG_ITERATIONS:

                current_question = (
                    rewrite_query(
                        current_question
                    )
                )

                print(
                    "\nSelf-RAG will retrieve again."
                )

                continue

            break

        # -----------------------------------------------------------
        # STEP D: GENERATE ANSWER
        # -----------------------------------------------------------

        raw_answer = generate_answer(
            question,
            ranked_chunks
        )

        print(
            "\nDraft Answer:"
        )

        print(
            raw_answer
        )

        # -----------------------------------------------------------
        # STEP E: SELF-ANSWER CHECK
        # -----------------------------------------------------------

        support = (
            evaluate_answer_support(
                question,
                raw_answer,
                ranked_chunks
            )
        )

        # -----------------------------------------------------------
        # Answer is supported
        # -----------------------------------------------------------

        if support["supported"]:

            print(
                "\nSelf-RAG decision:"
            )

            print(
                "Answer is supported by context."
            )

            answer_data = (
                parse_json_response(
                    raw_answer
                )
            )

            answer_data = (
                validate_sources(
                    answer_data,
                    ranked_chunks
                )
            )

            return answer_data

        # -----------------------------------------------------------
        # Answer is NOT supported
        # -----------------------------------------------------------

        print(
            "\nSelf-RAG decision:"
        )

        print(
            "Answer is NOT sufficiently supported."
        )

        if iteration < MAX_SELF_RAG_ITERATIONS:

            current_question = (
                rewrite_query(
                    question
                )
            )

            print(
                "\nSelf-RAG will retrieve "
                "additional context."
            )

            continue

        # -----------------------------------------------------------
        # Maximum iterations reached
        # -----------------------------------------------------------

        break

    # =================================================================
    # FALLBACK
    # =================================================================
    #
    # If Self-RAG could not establish sufficient
    # support, do NOT blindly return the draft.
    #
    # =================================================================

    print("\n" + "=" * 70)
    print("SELF-RAG FALLBACK")
    print("=" * 70)

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
# FLOW STEP 19: MAIN
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
# Embedding
#   ↓
# ChromaDB
#   ↓
# CrossEncoder
#   ↓
# Self-RAG
#
# ====================================================================


def main():

    print("\n")
    print("=" * 70)
    print("SELF-RAG - HR POLICY")
    print("=" * 70)

    # ---------------------------------------------------------------
    # STEP 1: OCR
    # ---------------------------------------------------------------

    page_data = (
        extract_text_from_pdf(
            PDF_PATH
        )
    )

    # ---------------------------------------------------------------
    # STEP 2: Chunking
    # ---------------------------------------------------------------

    chunks = create_chunks(
        page_data
    )

    # ---------------------------------------------------------------
    # STEP 3: Embedding
    # ---------------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    # ---------------------------------------------------------------
    # STEP 4: ChromaDB
    # ---------------------------------------------------------------

    collection = (
        create_vector_database(
            chunks,
            embedding_model
        )
    )

    # ---------------------------------------------------------------
    # STEP 7: CrossEncoder
    # ---------------------------------------------------------------

    reranker = load_reranker()

    # ---------------------------------------------------------------
    # QUESTION LOOP
    # ---------------------------------------------------------------

    while True:

        print("\n")
        print("=" * 70)
        print("SELF-RAG QUESTION")
        print(
            "Type 'exit', 'quit', or 'q' to stop."
        )
        print("=" * 70)

        question = get_user_question()

        # -----------------------------------------------------------
        # Exit
        # -----------------------------------------------------------

        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "\nExiting Self-RAG..."
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
        # Run Self-RAG
        # -----------------------------------------------------------

        answer = self_rag(
            question,
            collection,
            embedding_model,
            reranker
        )

        # -----------------------------------------------------------
        # Final answer
        # -----------------------------------------------------------

        print_final_answer(
            answer
        )


# ====================================================================
# PROGRAM ENTRY POINT
# ====================================================================


if __name__ == "__main__":

    main()