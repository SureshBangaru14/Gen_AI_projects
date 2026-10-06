# ====================================================================
# FILE: 06_agentic_rag.py
# ====================================================================
#
# AGENTIC RAG FLOW
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
# Knowledge Graph
#   ↓
# ┌──────────────────────────────────────────────────────────────┐
# │                         USER QUESTION                        │
# └──────────────────────────────────────────────────────────────┘
#                              ↓
#                           AGENT
#                              ↓
#                    ┌─────────┼─────────┐
#                    ↓         ↓         ↓
#                VECTOR      GRAPH    REWRITE
#                SEARCH      SEARCH    QUERY
#                    ↓         ↓         ↓
#                    └─────────┼─────────┘
#                              ↓
#                         MERGE RESULTS
#                              ↓
#                         CROSSENCODER
#                              ↓
#                         TOP 5 CHUNKS
#                              ↓
#                         QUALITY CHECK
#                              ↓
#                    ┌─────────┴─────────┐
#                    ↓                   ↓
#                 GOOD                  BAD
#                    ↓                   ↓
#                 ANSWER           AGENT RETRIES
#                    │                   │
#                    └─────────┬─────────┘
#                              ↓
#                           MISTRAL
#                              ↓
#                            JSON
#                              ↓
#                     SOURCE + CONFIDENCE
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
import re
import json
from collections import defaultdict

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

CHROMA_PATH = "chroma_agentic_hr"

COLLECTION_NAME = "hr_policy_agentic"

OCR_DPI = 250

CHUNK_SIZE = 600

CHUNK_OVERLAP = 100

VECTOR_TOP_K = 10

GRAPH_TOP_K = 10

FINAL_TOP_K = 5

MAX_AGENT_STEPS = 3

MIN_GOOD_RESULTS = 1

MIN_GOOD_SCORE = 0.0


# ====================================================================
# GLOBAL KNOWLEDGE GRAPH
# ====================================================================

GRAPH = defaultdict(set)

ENTITY_TO_CHUNKS = defaultdict(set)

RELATIONSHIPS = []

CHUNK_STORE = {}


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
# Tesseract
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
# FLOW STEP 2: CHUNKING
# ====================================================================
#
# Page Text
#   ↓
# RecursiveCharacterTextSplitter
#   ↓
# Chunks
#
# Example:
#
# page_1_chunk_1
# page_1_chunk_2
# page_2_chunk_1
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

            record = {
                "id": chunk_id,
                "text": chunk_text,
                "page_no": page_no,
                "chunk_no": chunk_no,
                "file_name": os.path.basename(
                    PDF_PATH
                )
            }

            chunks.append(record)

            CHUNK_STORE[chunk_id] = record

    print(
        f"Total chunks: {len(chunks)}"
    )

    return chunks


# ====================================================================
# FLOW STEP 3: EMBEDDING MODEL
# ====================================================================
#
# Chunk
#   ↓
# SentenceTransformer
#   ↓
# Dense Embedding
#
# ====================================================================


def load_embedding_model():

    print("\n" + "=" * 70)
    print("FLOW STEP 3: LOAD EMBEDDING MODEL")
    print("=" * 70)

    print(
        f"Model: {EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return model


# ====================================================================
# FLOW STEP 4: CHROMADB
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
# FLOW STEP 5: ENTITY EXTRACTION
# ====================================================================
#
# Chunk
#   ↓
# Entity Extraction
#   ↓
# HR Entities
#
# ====================================================================


def extract_entities(text):

    entities = set()

    patterns = [

        r"\bemployees?\b",

        r"\bworking hours?\b",

        r"\bscheduled hours?\b",

        r"\bmeal breaks?\b",

        r"\bmeal break\b",

        r"\bleave\b",

        r"\bvacation\b",

        r"\bsick leave\b",

        r"\battendance\b",

        r"\bovertime\b",

        r"\bholidays?\b",

        r"\bmanager\b",

        r"\bsupervisor\b",

        r"\bpolicy\b",

        r"\bwork\b",

        r"\bhours?\b",

        r"\bbreak\b"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:

            entity = (
                match
                .strip()
                .lower()
            )

            entity = re.sub(
                r"\s+",
                " ",
                entity
            )

            entities.add(entity)

    number_patterns = [

        r"\b\d+\s*hours?\b",

        r"\b\d+\s*days?\b",

        r"\b\d+\s*minutes?\b"
    ]

    for pattern in number_patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:

            entities.add(
                match
                .strip()
                .lower()
            )

    return sorted(entities)


# ====================================================================
# FLOW STEP 6: RELATIONSHIP EXTRACTION
# ====================================================================
#
# Entity
#   ↓
# Relationship
#   ↓
# Knowledge Graph
#
# ====================================================================


def extract_relationships(chunk):

    text = chunk["text"]

    entities = extract_entities(
        text
    )

    relationships = []

    # ---------------------------------------------------------------
    # Employee → expected_to_work → X hours
    # ---------------------------------------------------------------

    if (
        "employee" in entities
        and any(
            "hour" in entity
            for entity in entities
        )
    ):

        for entity in entities:

            if "hour" in entity:

                relationships.append(
                    {
                        "subject": "employee",
                        "relation":
                            "expected_to_work",
                        "object": entity,
                        "chunk_id":
                            chunk["id"]
                    }
                )

    # ---------------------------------------------------------------
    # Employee → available_during →
    # scheduled hours
    # ---------------------------------------------------------------

    if (
        "employee" in entities
        and "scheduled hours" in entities
    ):

        relationships.append(
            {
                "subject": "employee",
                "relation":
                    "available_during",
                "object":
                    "scheduled hours",
                "chunk_id":
                    chunk["id"]
            }
        )

    # ---------------------------------------------------------------
    # Employee → has → meal break
    # ---------------------------------------------------------------

    if (
        "employee" in entities
        and "meal break" in entities
    ):

        relationships.append(
            {
                "subject": "employee",
                "relation": "has",
                "object":
                    "meal break",
                "chunk_id":
                    chunk["id"]
            }
        )

    # ---------------------------------------------------------------
    # Working hours → excludes →
    # meal break
    # ---------------------------------------------------------------

    if (
        "working hours" in entities
        and "meal break" in entities
    ):

        relationships.append(
            {
                "subject":
                    "working hours",
                "relation":
                    "excludes",
                "object":
                    "meal break",
                "chunk_id":
                    chunk["id"]
            }
        )

    return entities, relationships


# ====================================================================
# FLOW STEP 7: BUILD KNOWLEDGE GRAPH
# ====================================================================
#
# Chunks
#   ↓
# Entities
#   ↓
# Relationships
#   ↓
# Graph
#
# ====================================================================


def build_knowledge_graph(chunks):

    print("\n" + "=" * 70)
    print("FLOW STEP 7: BUILD KNOWLEDGE GRAPH")
    print("=" * 70)

    GRAPH.clear()

    ENTITY_TO_CHUNKS.clear()

    RELATIONSHIPS.clear()

    for chunk in chunks:

        entities, relationships = (
            extract_relationships(
                chunk
            )
        )

        for entity in entities:

            ENTITY_TO_CHUNKS[
                entity
            ].add(
                chunk["id"]
            )

        for relationship in relationships:

            subject = relationship[
                "subject"
            ]

            object_ = relationship[
                "object"
            ]

            GRAPH[
                subject
            ].add(
                object_
            )

            GRAPH[
                object_
            ].add(
                subject
            )

            RELATIONSHIPS.append(
                relationship
            )

    print(
        f"Entities: "
        f"{len(ENTITY_TO_CHUNKS)}"
    )

    print(
        f"Relationships: "
        f"{len(RELATIONSHIPS)}"
    )


# ====================================================================
# FLOW STEP 8: VECTOR SEARCH TOOL
# ====================================================================
#
# Agent Tool #1
#
# Question
#   ↓
# Embedding
#   ↓
# ChromaDB
#   ↓
# Candidate Chunks
#
# ====================================================================


def vector_search(
    question,
    collection,
    embedding_model
):

    print("\n" + "=" * 70)
    print("AGENT TOOL: VECTOR SEARCH")
    print("=" * 70)

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=VECTOR_TOP_K
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

    output = []

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

        output.append(
            {
                "id": chunk_id,
                "text": document,
                "page_no": page_no,
                "chunk_no": chunk_no,
                "file_name":
                    metadata.get(
                        "file_name",
                        os.path.basename(
                            PDF_PATH
                        )
                    ),
                "retrieval_method":
                    "vector",
                "retrieval_score":
                    1 / (1 + distance)
            }
        )

    print(
        f"Vector results: "
        f"{len(output)}"
    )

    return output


# ====================================================================
# FLOW STEP 9: GRAPH SEARCH TOOL
# ====================================================================
#
# Agent Tool #2
#
# Question
#   ↓
# Entities
#   ↓
# Graph
#   ↓
# Related Chunks
#
# ====================================================================


def graph_search(question):

    print("\n" + "=" * 70)
    print("AGENT TOOL: GRAPH SEARCH")
    print("=" * 70)

    query_entities = extract_entities(
        question
    )

    print(
        f"Query entities: "
        f"{query_entities}"
    )

    candidate_ids = set()

    # ---------------------------------------------------------------
    # Direct entity → chunk
    # ---------------------------------------------------------------

    for entity in query_entities:

        candidate_ids.update(
            ENTITY_TO_CHUNKS.get(
                entity,
                set()
            )
        )

    # ---------------------------------------------------------------
    # One-hop graph traversal
    # ---------------------------------------------------------------

    for entity in query_entities:

        neighbours = GRAPH.get(
            entity,
            set()
        )

        for neighbour in neighbours:

            candidate_ids.update(
                ENTITY_TO_CHUNKS.get(
                    neighbour,
                    set()
                )
            )

    results = []

    for chunk_id in candidate_ids:

        if chunk_id in CHUNK_STORE:

            chunk = dict(
                CHUNK_STORE[chunk_id]
            )

            chunk[
                "retrieval_method"
            ] = "graph"

            chunk[
                "retrieval_score"
            ] = 1.0

            results.append(
                chunk
            )

    print(
        f"Graph results: "
        f"{len(results)}"
    )

    return results[:GRAPH_TOP_K]


# ====================================================================
# FLOW STEP 10: QUERY REWRITE TOOL
# ====================================================================
#
# Agent Tool #3
#
# Original Question
#       ↓
# Mistral
#       ↓
# Improved Search Query
#
# ====================================================================


def rewrite_query(question):

    print("\n" + "=" * 70)
    print("AGENT TOOL: QUERY REWRITE")
    print("=" * 70)

    prompt = f"""
Rewrite the following HR Policy question
into a concise search query.

Do not answer the question.

Return ONLY JSON:

{{
  "query": "rewritten query"
}}

Question:
{question}
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

    raw = response[
        "message"
    ][
        "content"
    ]

    try:

        data = json.loads(raw)

        if isinstance(data, dict):

            rewritten = data.get(
                "query",
                question
            )

            print(
                f"Rewritten query: "
                f"{rewritten}"
            )

            return rewritten

    except json.JSONDecodeError:

        pass

    return question


# ====================================================================
# FLOW STEP 11: MERGE AGENT RESULTS
# ====================================================================
#
# Vector Results
#       +
# Graph Results
#       +
# Rewritten Query Results
#       ↓
# Deduplicate
#
# ====================================================================


def merge_results(
    *result_sets
):

    print("\n" + "=" * 70)
    print("FLOW STEP 11: MERGE AGENT RESULTS")
    print("=" * 70)

    merged = {}

    for result_set in result_sets:

        for result in result_set:

            chunk_id = result["id"]

            if chunk_id not in merged:

                merged[chunk_id] = result

            else:

                # ---------------------------------------------------
                # If a chunk was found by multiple
                # tools, increase its priority.
                # ---------------------------------------------------

                existing = merged[
                    chunk_id
                ]

                existing[
                    "retrieval_method"
                ] = (
                    existing.get(
                        "retrieval_method",
                        ""
                    )
                    + "+"
                    + result.get(
                        "retrieval_method",
                        ""
                    )
                )

                existing[
                    "retrieval_score"
                ] = max(
                    existing.get(
                        "retrieval_score",
                        0.0
                    ),
                    result.get(
                        "retrieval_score",
                        0.0
                    )
                )

    results = list(
        merged.values()
    )

    print(
        f"Merged unique chunks: "
        f"{len(results)}"
    )

    return results


# ====================================================================
# FLOW STEP 12: LOAD CROSSENCODER
# ====================================================================
#
# Candidate Chunks
#       ↓
# CrossEncoder
#
# ====================================================================


def load_reranker():

    print("\n" + "=" * 70)
    print("FLOW STEP 12: LOAD CROSSENCODER")
    print("=" * 70)

    print(
        f"Model: {RERANKER_MODEL}"
    )

    return CrossEncoder(
        RERANKER_MODEL
    )


# ====================================================================
# FLOW STEP 13: RERANK
# ====================================================================
#
# Question + Candidate
#       ↓
# CrossEncoder
#       ↓
# Relevance Score
#       ↓
# Top 5
#
# ====================================================================


def rerank_chunks(
    question,
    chunks,
    reranker
):

    print("\n" + "=" * 70)
    print("FLOW STEP 13: CROSSENCODER RERANKING")
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

    ranked = []

    for chunk, score in zip(
        chunks,
        scores
    ):

        item = dict(chunk)

        item[
            "rerank_score"
        ] = float(score)

        ranked.append(
            item
        )

    ranked.sort(
        key=lambda x:
            x["rerank_score"],
        reverse=True
    )

    ranked = ranked[
        :FINAL_TOP_K
    ]

    for rank, item in enumerate(
        ranked,
        start=1
    ):

        print(
            f"{rank}. "
            f"{item['id']} | "
            f"Score="
            f"{item['rerank_score']:.4f} | "
            f"Method="
            f"{item.get('retrieval_method')}"
        )

    return ranked


# ====================================================================
# FLOW STEP 14: AGENT QUALITY CHECK
# ====================================================================
#
# Retrieved Results
#       ↓
# Check whether useful
#       ↓
# GOOD / BAD
#
# ====================================================================


def evaluate_results(
    ranked_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 14: AGENT QUALITY CHECK")
    print("=" * 70)

    if not ranked_chunks:

        print(
            "Result quality: BAD"
        )

        return False

    if len(ranked_chunks) < MIN_GOOD_RESULTS:

        print(
            "Result quality: BAD"
        )

        return False

    best_score = ranked_chunks[0][
        "rerank_score"
    ]

    print(
        f"Best rerank score: "
        f"{best_score:.4f}"
    )

    # ---------------------------------------------------------------
    # CrossEncoder scores are model-specific.
    # We use a permissive threshold here.
    # The agent mainly checks whether retrieval
    # returned usable candidates.
    # ---------------------------------------------------------------

    if best_score < MIN_GOOD_SCORE:

        print(
            "Result quality: BAD"
        )

        return False

    print(
        "Result quality: GOOD"
    )

    return True


# ====================================================================
# FLOW STEP 15: BUILD CONTEXT
# ====================================================================
#
# Top Chunks
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
    print("FLOW STEP 15: BUILD CONTEXT")
    print("=" * 70)

    parts = []

    for index, chunk in enumerate(
        ranked_chunks,
        start=1
    ):

        parts.append(
            f"""
SOURCE {index}

Page Number:
{chunk['page_no']}

Chunk Number:
{chunk['chunk_no']}

Chunk ID:
{chunk['id']}

Retrieval Method:
{chunk.get('retrieval_method', 'unknown')}

Relevance Score:
{chunk['rerank_score']:.4f}

Chunk Text:
{chunk['text']}
"""
        )

    return "\n".join(parts)


# ====================================================================
# FLOW STEP 16: SYSTEM PROMPT
# ====================================================================
#
# Defines answer rules.
#
# ====================================================================


def build_system_prompt():

    return """
You are an HR Policy question-answering assistant.

Answer ONLY using the supplied HR Policy context.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not supported by the
   retrieved context, say:

   "I don't know based on HR_Policy.pdf."

4. Always provide source information.
5. Confidence must be between 0 and 1.
6. Return ONLY valid JSON.
7. Return a JSON ARRAY.
8. Use the exact field names.

Required structure:

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
# FLOW STEP 17: USER PROMPT
# ====================================================================
#
# Question
#       +
# Retrieved Context
#       ↓
# Final LLM Prompt
#
# ====================================================================


def build_user_prompt(
    question,
    context
):

    return f"""
Answer this HR Policy question:

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Return ONLY the JSON array.
"""


# ====================================================================
# FLOW STEP 18: LLM GENERATION
# ====================================================================
#
# Context
#   +
# Question
#   ↓
# Mistral
#   ↓
# JSON
#
# ====================================================================


def generate_answer(
    question,
    context
):

    print("\n" + "=" * 70)
    print("FLOW STEP 18: MISTRAL GENERATION")
    print("=" * 70)

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
                    build_user_prompt(
                        question,
                        context
                    )
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
# FLOW STEP 19: JSON PARSING
# ====================================================================
#
# LLM JSON
#   ↓
# Python Object
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
    print("FLOW STEP 19: JSON PARSING")
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
        # Other wrappers
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

        return [
            {
                "Answer":
                    raw_response,
                "Source": [],
                "Confidence": 0.0
            }
        ]


# ====================================================================
# FLOW STEP 20: SOURCE VALIDATION
# ====================================================================
#
# LLM Source
#       ↓
# Compare with retrieved chunks
#       ↓
# Valid Source
#
# ====================================================================


def validate_sources(
    answer_data,
    ranked_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 20: SOURCE VALIDATION")
    print("=" * 70)

    valid_ids = {
        chunk["id"]
        for chunk in ranked_chunks
    }

    for item in answer_data:

        sources = item.get(
            "Source",
            []
        )

        valid_sources = []

        for source in sources:

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

        item["Confidence"] = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )

    return answer_data


# ====================================================================
# FLOW STEP 21: FINAL ANSWER
# ====================================================================
#
# Validated Answer
#       ↓
# Pretty JSON
#
# ====================================================================


def print_final_answer(
    answer_data
):

    print("\n" + "=" * 70)
    print("FLOW STEP 21: FINAL ANSWER")
    print("=" * 70)

    print(
        json.dumps(
            answer_data,
            indent=2,
            ensure_ascii=False
        )
    )


# ====================================================================
# FLOW STEP 22: AGENT DECISION
# ====================================================================
#
# This is the most important Agentic RAG function.
#
# The agent decides:
#
# VECTOR
# GRAPH
# REWRITE
#
# ====================================================================


def agent_decide(
    question,
    attempt
):

    print("\n" + "=" * 70)
    print(
        f"FLOW STEP 22: AGENT DECISION "
        f"(ATTEMPT {attempt})"
    )
    print("=" * 70)

    # ---------------------------------------------------------------
    # On first attempt, ask Mistral to decide.
    # ---------------------------------------------------------------

    prompt = f"""
You are a retrieval agent for an HR Policy system.

Choose the best retrieval strategy for this question.

Available tools:

1. vector
   - Semantic vector search.
   - Good for normal policy questions.

2. graph
   - Knowledge graph search.
   - Good when the question contains entities
     or relationships.

3. rewrite
   - Rewrite the question into a better
     search query.
   - Good for vague or poorly phrased questions.

4. vector_graph
   - Use both vector and graph search.

Question:
{question}

Attempt:
{attempt}

Return ONLY JSON:

{{
  "strategy": "vector"
}}

Allowed values:

vector
graph
rewrite
vector_graph
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

    raw = response[
        "message"
    ][
        "content"
    ]

    try:

        data = json.loads(
            raw
        )

        strategy = data.get(
            "strategy",
            "vector"
        )

        allowed = {
            "vector",
            "graph",
            "rewrite",
            "vector_graph"
        }

        if strategy not in allowed:

            strategy = "vector"

        print(
            f"Agent selected: "
            f"{strategy}"
        )

        return strategy

    except json.JSONDecodeError:

        print(
            "Agent decision parsing failed."
        )

        return "vector"


# ====================================================================
# FLOW STEP 23: EXECUTE AGENT TOOL
# ====================================================================
#
# Agent Decision
#       ↓
# Execute selected tool
#
# ====================================================================


def execute_agent_strategy(
    strategy,
    question,
    collection,
    embedding_model
):

    print("\n" + "=" * 70)
    print("FLOW STEP 23: EXECUTE AGENT STRATEGY")
    print("=" * 70)

    # ---------------------------------------------------------------
    # VECTOR
    # ---------------------------------------------------------------

    if strategy == "vector":

        return vector_search(
            question,
            collection,
            embedding_model
        )

    # ---------------------------------------------------------------
    # GRAPH
    # ---------------------------------------------------------------

    if strategy == "graph":

        return graph_search(
            question
        )

    # ---------------------------------------------------------------
    # REWRITE
    # ---------------------------------------------------------------

    if strategy == "rewrite":

        rewritten_question = (
            rewrite_query(
                question
            )
        )

        return vector_search(
            rewritten_question,
            collection,
            embedding_model
        )

    # ---------------------------------------------------------------
    # VECTOR + GRAPH
    # ---------------------------------------------------------------

    if strategy == "vector_graph":

        vector_results = (
            vector_search(
                question,
                collection,
                embedding_model
            )
        )

        graph_results = (
            graph_search(
                question
            )
        )

        return merge_results(
            vector_results,
            graph_results
        )

    return vector_search(
        question,
        collection,
        embedding_model
    )


# ====================================================================
# FLOW STEP 24: AGENTIC RAG PIPELINE
# ====================================================================
#
# Question
#   ↓
# Agent Decision
#   ↓
# Tool
#   ↓
# Reranker
#   ↓
# Quality Check
#   ↓
# Good?
#   ├── YES → Generate
#   │
#   └── NO  → Agent retries
#
# ====================================================================


def agentic_rag(
    question,
    collection,
    embedding_model,
    reranker
):

    print("\n")
    print("=" * 70)
    print("AGENTIC RAG STARTED")
    print("=" * 70)

    current_question = question

    ranked_chunks = []

    # ---------------------------------------------------------------
    # Agent loop
    # ---------------------------------------------------------------

    for attempt in range(
        1,
        MAX_AGENT_STEPS + 1
    ):

        print("\n")
        print(
            "=" * 70
        )

        print(
            f"AGENT ITERATION: "
            f"{attempt}/{MAX_AGENT_STEPS}"
        )

        print(
            "=" * 70
        )

        # -----------------------------------------------------------
        # Agent decides strategy
        # -----------------------------------------------------------

        strategy = agent_decide(
            current_question,
            attempt
        )

        # -----------------------------------------------------------
        # Agent executes tool
        # -----------------------------------------------------------

        candidates = (
            execute_agent_strategy(
                strategy,
                current_question,
                collection,
                embedding_model
            )
        )

        # -----------------------------------------------------------
        # If rewrite was selected,
        # current question is improved.
        # -----------------------------------------------------------

        if strategy == "rewrite":

            current_question = (
                rewrite_query(
                    current_question
                )
            )

        # -----------------------------------------------------------
        # Reranking
        # -----------------------------------------------------------

        ranked_chunks = (
            rerank_chunks(
                question,
                candidates,
                reranker
            )
        )

        # -----------------------------------------------------------
        # Quality check
        # -----------------------------------------------------------

        is_good = evaluate_results(
            ranked_chunks
        )

        if is_good:

            print(
                "\nAgent found useful context."
            )

            break

        # -----------------------------------------------------------
        # Retrieval failed.
        # Agent modifies strategy on
        # next iteration.
        # -----------------------------------------------------------

        print(
            "\nAgent decided that retrieval "
            "quality was insufficient."
        )

        # -----------------------------------------------------------
        # Force a different strategy
        # by changing the query.
        # -----------------------------------------------------------

        if attempt < MAX_AGENT_STEPS:

            current_question = (
                rewrite_query(
                    current_question
                )
            )

            print(
                f"\nAgent retry query:"
                f"\n{current_question}"
            )

    # ---------------------------------------------------------------
    # No useful results
    # ---------------------------------------------------------------

    if not ranked_chunks:

        return [
            {
                "Answer":
                    "I don't know based on "
                    "HR_Policy.pdf.",
                "Source": [],
                "Confidence": 0.0
            }
        ]

    # ---------------------------------------------------------------
    # Context
    # ---------------------------------------------------------------

    context = build_context(
        ranked_chunks
    )

    # ---------------------------------------------------------------
    # LLM
    # ---------------------------------------------------------------

    raw_response = generate_answer(
        question,
        context
    )

    print(
        "\nRaw LLM Response:"
    )

    print(
        raw_response
    )

    # ---------------------------------------------------------------
    # JSON
    # ---------------------------------------------------------------

    answer_data = (
        parse_json_response(
            raw_response
        )
    )

    # ---------------------------------------------------------------
    # Validate sources
    # ---------------------------------------------------------------

    answer_data = (
        validate_sources(
            answer_data,
            ranked_chunks
        )
    )

    return answer_data


# ====================================================================
# FLOW STEP 25: MAIN INITIALIZATION
# ====================================================================
#
# PDF
#   ↓
# OCR
#   ↓
# Chunk
#   ↓
# Embedding
#   ↓
# ChromaDB
#   ↓
# Knowledge Graph
#   ↓
# CrossEncoder
#   ↓
# Agent
#
# ====================================================================


def main():

    print("\n")
    print("=" * 70)
    print("AGENTIC RAG - HR POLICY")
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
    # Embedding model
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
    # Knowledge Graph
    # ---------------------------------------------------------------

    build_knowledge_graph(
        chunks
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
        print("AGENTIC RAG QUESTION")
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
                "\nExiting Agentic RAG..."
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
        # Run Agentic RAG
        # -----------------------------------------------------------

        answer = agentic_rag(
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