# ====================================================================
# ADVANCED RAG - METHOD 3
#
# HYBRID SEARCH
# Dense Retrieval + BM25 + CrossEncoder Reranking
#
# ====================================================================
#
# COMPLETE FLOW
#
# PDF
#   ↓
# FLOW STEP 1: OCR
#   ↓
# FLOW STEP 2: RECURSIVE CHARACTER CHUNKING
#   ↓
# FLOW STEP 3: LOAD EMBEDDING MODEL
#   ↓
# FLOW STEP 4: CHROMADB UPSERT
#   ↓
# FLOW STEP 5: BUILD BM25 INDEX
#   ↓
# FLOW STEP 6: USER QUESTION
#   ↓
# FLOW STEP 7: DENSE RETRIEVAL
#   ↓
# FLOW STEP 8: BM25 SPARSE RETRIEVAL
#   ↓
# FLOW STEP 9: HYBRID MERGE
#   ↓
# FLOW STEP 10: DEDUPLICATION
#   ↓
# FLOW STEP 11: LOAD CROSSENCODER
#   ↓
# FLOW STEP 12: CROSSENCODER RERANKING
#   ↓
# FLOW STEP 13: SELECT TOP 5
#   ↓
# FLOW STEP 14: BUILD CONTEXT
#   ↓
# FLOW STEP 15: SYSTEM PROMPT
#   ↓
# FLOW STEP 16: USER PROMPT
#   ↓
# FLOW STEP 17: MISTRAL LLM
#   ↓
# FLOW STEP 18: JSON PARSING
#   ↓
# FLOW STEP 19: SOURCE VALIDATION
#   ↓
# FLOW STEP 20: FINAL ANSWER
#   ↓
# FLOW STEP 21: WHILE LOOP
#
# ====================================================================


# ====================================================================
# IMPORT LIBRARIES
# ====================================================================

import os
import json
import re

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

from rank_bm25 import BM25Okapi


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

CHROMA_DIR = "chroma_hybrid_hr"

COLLECTION_NAME = "hr_policy_hybrid"

OCR_DPI = 250

CHUNK_SIZE = 600

CHUNK_OVERLAP = 100

DENSE_TOP_K = 5

BM25_TOP_K = 5

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
# FLOW STEP 1: OCR
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
    print("FLOW STEP 1: OCR")
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
        f"Total pages processed: {len(pages)}"
    )

    return pages


# ====================================================================
# FLOW STEP 2: RECURSIVE CHARACTER CHUNKING
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
    print("FLOW STEP 2: RECURSIVE CHARACTER CHUNKING")
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
        f"Total chunks created: {len(chunks)}"
    )

    for chunk in chunks:

        print(
            f"{chunk['id']} -> "
            f"{len(chunk['text'])} characters"
        )

    return chunks


# ====================================================================
# FLOW STEP 3: LOAD EMBEDDING MODEL
# ====================================================================
#
# SentenceTransformer
#   ↓
# Embedding Model
#
# ====================================================================

def load_embedding_model():

    print("\n" + "=" * 70)
    print("FLOW STEP 3: LOAD EMBEDDING MODEL")
    print("=" * 70)

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded successfully."
    )

    return model


# ====================================================================
# FLOW STEP 4: CHROMADB UPSERT
# ====================================================================
#
# Chunks
#   ↓
# SentenceTransformer
#   ↓
# Embeddings
#   ↓
# ChromaDB
#
# ====================================================================

def create_chroma_db():

    print("\n" + "=" * 70)
    print("FLOW STEP 4: CREATE CHROMADB")
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


def ingest_chunks(
    collection,
    embedding_model,
    chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 4: CHROMADB UPSERT")
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
        "Creating document embeddings..."
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
        f"Upserted chunks: {len(chunks)}"
    )

    print(
        f"ChromaDB total chunks: "
        f"{collection.count()}"
    )


# ====================================================================
# FLOW STEP 5: BUILD BM25 INDEX
# ====================================================================
#
# Chunks
#   ↓
# Tokenization
#   ↓
# BM25
#   ↓
# Sparse Keyword Index
#
# ====================================================================

def tokenize_text(text):

    text = text.lower()

    tokens = re.findall(
        r"\b\w+\b",
        text
    )

    return tokens


def build_bm25_index(chunks):

    print("\n" + "=" * 70)
    print("FLOW STEP 5: BUILD BM25 INDEX")
    print("=" * 70)

    tokenized_documents = []

    for chunk in chunks:

        tokens = tokenize_text(
            chunk["text"]
        )

        tokenized_documents.append(
            tokens
        )

    bm25 = BM25Okapi(
        tokenized_documents
    )

    print(
        f"BM25 indexed documents: "
        f"{len(chunks)}"
    )

    return bm25


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
# FLOW STEP 7: DENSE RETRIEVAL
# ====================================================================
#
# User Question
#   ↓
# SentenceTransformer
#   ↓
# Query Embedding
#   ↓
# ChromaDB
#   ↓
# Dense Top 5
#
# ====================================================================

def dense_retrieval(
    question,
    embedding_model,
    collection
):

    print("\n" + "=" * 70)
    print("FLOW STEP 7: DENSE RETRIEVAL")
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

        n_results=DENSE_TOP_K,

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

    dense_results = []

    for document, metadata, distance in zip(

        documents,

        metadatas,

        distances
    ):

        chunk_id = (
            f"page_{metadata['page_no']}"
            f"_chunk_{metadata['chunk_no']}"
        )

        dense_results.append(
            {
                "id": chunk_id,

                "text": document,

                "metadata": metadata,

                "dense_distance":
                    float(distance)
            }
        )

    print(
        f"Dense results: "
        f"{len(dense_results)}"
    )

    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['id']} "
            f"Distance: "
            f"{result['dense_distance']:.6f}"
        )

    return dense_results


# ====================================================================
# FLOW STEP 8: BM25 SPARSE RETRIEVAL
# ====================================================================
#
# User Question
#   ↓
# Tokenization
#   ↓
# BM25
#   ↓
# Keyword Matching
#   ↓
# Sparse Top 5
#
# ====================================================================

def bm25_retrieval(
    question,
    chunks,
    bm25
):

    print("\n" + "=" * 70)
    print("FLOW STEP 8: BM25 SPARSE RETRIEVAL")
    print("=" * 70)

    query_tokens = tokenize_text(
        question
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(

        range(len(scores)),

        key=lambda index:
            scores[index],

        reverse=True
    )

    top_indices = ranked_indices[
        :BM25_TOP_K
    ]

    bm25_results = []

    for index in top_indices:

        chunk = chunks[index]

        bm25_results.append(
            {
                "id": chunk["id"],

                "text": chunk["text"],

                "metadata":
                    chunk["metadata"],

                "bm25_score":
                    float(scores[index])
            }
        )

    print(
        f"BM25 results: "
        f"{len(bm25_results)}"
    )

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['id']} "
            f"BM25 Score: "
            f"{result['bm25_score']:.6f}"
        )

    return bm25_results


# ====================================================================
# FLOW STEP 9: HYBRID MERGE
# ====================================================================
#
# Dense Results
#       +
# BM25 Results
#       ↓
# Hybrid Results
#
# ====================================================================

def hybrid_merge(
    dense_results,
    bm25_results
):

    print("\n" + "=" * 70)
    print("FLOW STEP 9: HYBRID MERGE")
    print("=" * 70)

    hybrid_results = {}

    # ------------------------------------------------------------
    # Add Dense Results
    # ------------------------------------------------------------

    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        chunk_id = result["id"]

        if chunk_id not in hybrid_results:

            hybrid_results[chunk_id] = {

                "id":
                    chunk_id,

                "text":
                    result["text"],

                "metadata":
                    result["metadata"],

                "dense_rank":
                    rank,

                "bm25_rank":
                    None,

                "dense_distance":
                    result["dense_distance"],

                "bm25_score":
                    0.0
            }

        else:

            hybrid_results[
                chunk_id
            ]["dense_rank"] = rank

    # ------------------------------------------------------------
    # Add BM25 Results
    # ------------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        chunk_id = result["id"]

        if chunk_id not in hybrid_results:

            hybrid_results[chunk_id] = {

                "id":
                    chunk_id,

                "text":
                    result["text"],

                "metadata":
                    result["metadata"],

                "dense_rank":
                    None,

                "bm25_rank":
                    rank,

                "dense_distance":
                    None,

                "bm25_score":
                    result["bm25_score"]
            }

        else:

            hybrid_results[
                chunk_id
            ]["bm25_rank"] = rank

            hybrid_results[
                chunk_id
            ]["bm25_score"] = (
                result["bm25_score"]
            )

    results = list(
        hybrid_results.values()
    )

    print(
        f"Hybrid candidates: "
        f"{len(results)}"
    )

    for result in results:

        print(
            f"{result['id']} | "
            f"Dense Rank: "
            f"{result['dense_rank']} | "
            f"BM25 Rank: "
            f"{result['bm25_rank']}"
        )

    return results


# ====================================================================
# FLOW STEP 10: DEDUPLICATION
# ====================================================================
#
# Hybrid Results
#   ↓
# Unique Chunk IDs
#   ↓
# Deduplicated Results
#
# ====================================================================

def deduplicate_results(
    hybrid_results
):

    print("\n" + "=" * 70)
    print("FLOW STEP 10: DEDUPLICATION")
    print("=" * 70)

    unique_results = {}

    for result in hybrid_results:

        chunk_id = result["id"]

        if chunk_id not in unique_results:

            unique_results[chunk_id] = result

    results = list(
        unique_results.values()
    )

    print(
        f"Unique chunks: "
        f"{len(results)}"
    )

    return results


# ====================================================================
# FLOW STEP 11: LOAD CROSSENCODER
# ====================================================================
#
# CrossEncoder Model
#   ↓
# Load Reranker
#
# ====================================================================

def load_reranker():

    print("\n" + "=" * 70)
    print("FLOW STEP 11: LOAD CROSSENCODER")
    print("=" * 70)

    print(
        f"Loading reranker: "
        f"{RERANKER_MODEL}"
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print(
        "CrossEncoder loaded successfully."
    )

    return reranker


# ====================================================================
# FLOW STEP 12: CROSSENCODER RERANKING
# ====================================================================
#
# User Question
#       +
# Hybrid Candidates
#       ↓
# CrossEncoder
#       ↓
# Reranked Results
#
# ====================================================================

def rerank_results(
    question,
    results,
    reranker
):

    print("\n" + "=" * 70)
    print("FLOW STEP 12: CROSSENCODER RERANKING")
    print("=" * 70)

    if not results:

        return []

    pairs = []

    for result in results:

        pairs.append(
            (
                question,
                result["text"]
            )
        )

    scores = reranker.predict(
        pairs
    )

    reranked_results = []

    for result, score in zip(
        results,
        scores
    ):

        item = dict(result)

        item["reranker_score"] = (
            float(score)
        )

        reranked_results.append(
            item
        )

    reranked_results.sort(

        key=lambda item:
            item["reranker_score"],

        reverse=True
    )

    for rank, result in enumerate(
        reranked_results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['id']} | "
            f"CrossEncoder Score: "
            f"{result['reranker_score']:.6f}"
        )

    return reranked_results


# ====================================================================
# FLOW STEP 13: SELECT TOP 5
# ====================================================================
#
# Reranked Results
#   ↓
# Top 5
#
# ====================================================================

def select_top_chunks(
    reranked_results
):

    print("\n" + "=" * 70)
    print("FLOW STEP 13: SELECT TOP 5")
    print("=" * 70)

    top_chunks = reranked_results[
        :FINAL_TOP_K
    ]

    for rank, result in enumerate(
        top_chunks,
        start=1
    ):

        print(
            f"TOP {rank}: "
            f"{result['id']} | "
            f"Score: "
            f"{result['reranker_score']:.6f}"
        )

    return top_chunks


# ====================================================================
# FLOW STEP 14: BUILD CONTEXT
# ====================================================================
#
# Top 5 Chunks
#   ↓
# Context
#
# ====================================================================

def build_context(
    top_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 14: BUILD CONTEXT")
    print("=" * 70)

    context_parts = []

    for rank, result in enumerate(
        top_chunks,
        start=1
    ):

        metadata = result["metadata"]

        page_no = metadata[
            "page_no"
        ]

        chunk_no = metadata[
            "chunk_no"
        ]

        chunk_id = result["id"]

        chunk_text = result["text"]

        relevance_score = (
            result["reranker_score"]
        )

        context = f"""
============================================================
Retrieved Chunk Rank: {rank}

Page Number: {page_no}

Chunk Number: {chunk_no}

Chunk ID: {chunk_id}

Relevance Score: {relevance_score:.6f}

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

    return final_context


# ====================================================================
# FLOW STEP 15: SYSTEM PROMPT
# ====================================================================
#
# System Prompt
#   +
# JSON Schema
#   ↓
# LLM Instructions
#
# ====================================================================

def create_system_prompt():

    print("\n" + "=" * 70)
    print("FLOW STEP 15: SYSTEM PROMPT")
    print("=" * 70)

    schema_text = json.dumps(
        ARRAY_SCHEMA,
        indent=4
    )

    system_prompt = f"""
You are an HR policy question answering system.

Answer ONLY using the retrieved context.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the
   retrieved context, say:

   "I don't know based on HR_Policy.pdf."

4. Return ONLY valid JSON.
5. The top-level response MUST be an array.
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

JSON Schema:

{schema_text}
"""

    return system_prompt


# ====================================================================
# FLOW STEP 16: USER PROMPT
# ====================================================================
#
# User Question
#       +
# Retrieved Context
#       ↓
# User Prompt
#
# ====================================================================

def create_user_prompt(
    question,
    context
):

    print("\n" + "=" * 70)
    print("FLOW STEP 16: USER PROMPT")
    print("=" * 70)

    user_prompt = f"""
Answer the following question using ONLY
the retrieved context.

Question:

{question}

Retrieved Context:

{context}
"""

    return user_prompt


# ====================================================================
# FLOW STEP 17: MISTRAL LLM
# ====================================================================
#
# System Prompt
#       +
# User Prompt
#       ↓
# Mistral
#       ↓
# Raw JSON
#
# ====================================================================

def call_llm(
    system_prompt,
    user_prompt
):

    print("\n" + "=" * 70)
    print("FLOW STEP 17: MISTRAL LLM")
    print("=" * 70)

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
# FLOW STEP 18: JSON PARSING
# ====================================================================
#
# Raw JSON
#   ↓
# json.loads()
#   ↓
# Python Object
#
# ====================================================================

def parse_llm_response(
    raw_response
):

    print("\n" + "=" * 70)
    print("FLOW STEP 18: JSON PARSING")
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
# FLOW STEP 19: SOURCE VALIDATION
# ====================================================================
#
# LLM JSON
#   ↓
# Validate Source
#   ↓
# Valid JSON
#
# ====================================================================

def validate_response(
    parsed_response,
    top_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 19: SOURCE VALIDATION")
    print("=" * 70)

    if not parsed_response:

        return []

    valid_chunk_ids = {

        result["id"]

        for result in top_chunks
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
# FLOW STEP 20: FINAL ANSWER
# ====================================================================
#
# Validated JSON
#   ↓
# Final Answer
#
# ====================================================================

def print_final_answer(
    validated_response
):

    print("\n" + "=" * 70)
    print("FLOW STEP 20: FINAL ANSWER")
    print("=" * 70)

    print(
        json.dumps(
            validated_response,
            indent=4,
            ensure_ascii=False
        )
    )


# ====================================================================
# COMPLETE QUESTION PIPELINE
# ====================================================================

def process_question(
    question,
    embedding_model,
    collection,
    chunks,
    bm25,
    reranker
):

    # ------------------------------------------------------------
    # FLOW STEP 7: DENSE RETRIEVAL
    # ------------------------------------------------------------

    dense_results = dense_retrieval(

        question,

        embedding_model,

        collection
    )

    # ------------------------------------------------------------
    # FLOW STEP 8: BM25 RETRIEVAL
    # ------------------------------------------------------------

    bm25_results = bm25_retrieval(

        question,

        chunks,

        bm25
    )

    # ------------------------------------------------------------
    # FLOW STEP 9: HYBRID MERGE
    # ------------------------------------------------------------

    hybrid_results = hybrid_merge(

        dense_results,

        bm25_results
    )

    # ------------------------------------------------------------
    # FLOW STEP 10: DEDUPLICATION
    # ------------------------------------------------------------

    unique_results = (
        deduplicate_results(
            hybrid_results
        )
    )

    # ------------------------------------------------------------
    # FLOW STEP 12: CROSSENCODER
    # ------------------------------------------------------------

    reranked_results = rerank_results(

        question,

        unique_results,

        reranker
    )

    # ------------------------------------------------------------
    # FLOW STEP 13: TOP 5
    # ------------------------------------------------------------

    top_chunks = select_top_chunks(
        reranked_results
    )

    # ------------------------------------------------------------
    # FLOW STEP 14: CONTEXT
    # ------------------------------------------------------------

    context = build_context(
        top_chunks
    )

    # ------------------------------------------------------------
    # FLOW STEP 15: SYSTEM PROMPT
    # ------------------------------------------------------------

    system_prompt = (
        create_system_prompt()
    )

    # ------------------------------------------------------------
    # FLOW STEP 16: USER PROMPT
    # ------------------------------------------------------------

    user_prompt = create_user_prompt(

        question,

        context
    )

    # ------------------------------------------------------------
    # FLOW STEP 17: MISTRAL
    # ------------------------------------------------------------

    raw_response = call_llm(

        system_prompt,

        user_prompt
    )

    # ------------------------------------------------------------
    # FLOW STEP 18: JSON PARSING
    # ------------------------------------------------------------

    parsed_response = (
        parse_llm_response(
            raw_response
        )
    )

    # ------------------------------------------------------------
    # FLOW STEP 19: SOURCE VALIDATION
    # ------------------------------------------------------------

    validated_response = (
        validate_response(

            parsed_response,

            top_chunks
        )
    )

    # ------------------------------------------------------------
    # FLOW STEP 20: FINAL ANSWER
    # ------------------------------------------------------------

    print_final_answer(
        validated_response
    )

    return validated_response


# ====================================================================
# FLOW STEP 21: WHILE LOOP
# ====================================================================
#
# Ask Question
#   ↓
# Dense + BM25
#   ↓
# Hybrid Search
#   ↓
# CrossEncoder
#   ↓
# Mistral
#   ↓
# Final Answer
#   ↓
# Ask Next Question
#
# ====================================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "ADVANCED RAG - METHOD 3"
    )

    print(
        "HYBRID SEARCH"
    )

    print(
        "DENSE + BM25 + CROSSENCODER"
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
    # FLOW STEP 3: EMBEDDING MODEL
    # ------------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    # ------------------------------------------------------------
    # FLOW STEP 4: CHROMADB
    # ------------------------------------------------------------

    collection = create_chroma_db()

    ingest_chunks(

        collection,

        embedding_model,

        chunks
    )

    # ------------------------------------------------------------
    # FLOW STEP 5: BM25
    # ------------------------------------------------------------

    bm25 = build_bm25_index(
        chunks
    )

    # ------------------------------------------------------------
    # FLOW STEP 11: CROSSENCODER
    # ------------------------------------------------------------

    reranker = load_reranker()

    # ------------------------------------------------------------
    # FLOW STEP 21: WHILE LOOP
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FLOW STEP 21: ASK QUESTIONS")
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
                "\nExiting Advanced RAG."
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
        # PROCESS QUESTION
        # --------------------------------------------------------

        process_question(

            question,

            embedding_model,

            collection,

            chunks,

            bm25,

            reranker
        )


# ====================================================================
# APPLICATION ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()