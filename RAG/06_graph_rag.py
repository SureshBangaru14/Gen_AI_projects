# ====================================================================
# FILE: 05_graph_rag.py
# ====================================================================
#
# GRAPH RAG FLOW
#
# PDF
#   ↓
# OCR
#   ↓
# Page-wise Text
#   ↓
# Recursive Chunking
#   ↓
# SentenceTransformer Embeddings
#   ↓
# ChromaDB
#   ↓
# Entity Extraction
#   ↓
# Relationship Extraction
#   ↓
# Knowledge Graph
#   ↓
# User Question
#   ↓
# Query Entity Extraction
#   ↓
# Graph Traversal
#   ↓
# Candidate Chunks
#   ↓
# CrossEncoder Reranking
#   ↓
# Top 5 Context
#   ↓
# Mistral
#   ↓
# JSON Answer
#   ↓
# Source + Confidence
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
import hashlib
from collections import defaultdict

import chromadb
import pytesseract

from pdf2image import convert_from_path

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from langchain_text_splitters import RecursiveCharacterTextSplitter

import ollama


# ====================================================================
# CONFIGURATION
# ====================================================================

PDF_PATH = (
    "/home/suresh/Gen_AI_Practice/Generative-AI/"
    "Vector_Database/HR_Policy.pdf"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_MODEL = "mistral"

CHROMA_PATH = "chroma_graph_hr"

COLLECTION_NAME = "hr_policy_graph"

OCR_DPI = 250

CHUNK_SIZE = 600

CHUNK_OVERLAP = 100

RETRIEVAL_TOP_K = 10

FINAL_TOP_K = 5


# ====================================================================
# GRAPH STORAGE
# ====================================================================

# Entity → related entities
GRAPH = defaultdict(set)

# Entity → source chunks
ENTITY_TO_CHUNKS = defaultdict(set)

# Relationship records
RELATIONSHIPS = []

# Chunk ID → complete chunk record
CHUNK_STORE = {}


# ====================================================================
# FLOW STEP 1: OCR
# ====================================================================
#
# PDF
#   ↓
# Convert PDF pages into images
#   ↓
# Tesseract OCR
#   ↓
# Page-wise text
#
# ====================================================================


def extract_text_from_pdf(pdf_path):

    print("\n" + "=" * 70)
    print("FLOW STEP 1: OCR")
    print("=" * 70)

    print(f"PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    pages = convert_from_path(
        pdf_path,
        dpi=OCR_DPI
    )

    page_data = []

    for page_no, image in enumerate(pages, start=1):

        print(
            f"Processing page {page_no}/{len(pages)}..."
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
        f"OCR completed. Pages extracted: {len(page_data)}"
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
# Small meaningful chunks
#
# Chunk ID:
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

        page_chunks = splitter.split_text(text)

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
        f"Total chunks created: {len(chunks)}"
    )

    return chunks


# ====================================================================
# FLOW STEP 3: EMBEDDING
# ====================================================================
#
# Chunk Text
#   ↓
# SentenceTransformer
#   ↓
# Dense Vector
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
# FLOW STEP 4: CHROMADB
# ====================================================================
#
# Chunk
#   ↓
# Embedding
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
        item["text"]
        for item in chunks
    ]

    ids = [
        item["id"]
        for item in chunks
    ]

    metadatas = [
        {
            "file_name": item["file_name"],
            "page_no": item["page_no"],
            "chunk_no": item["chunk_no"]
        }
        for item in chunks
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
        f"ChromaDB records: {collection.count()}"
    )

    return collection


# ====================================================================
# FLOW STEP 5: ENTITY EXTRACTION
# ====================================================================
#
# Chunk
#   ↓
# Detect important entities
#   ↓
# Employee
# Working Hours
# Meal Break
# Scheduled Hours
# Leave
# etc.
#
# This implementation uses a lightweight rule-based extractor.
# The purpose is to make the Graph RAG pipeline easy to understand.
#
# ====================================================================


def extract_entities(text):

    entities = set()

    # ---------------------------------------------------------------
    # Common HR concepts
    # ---------------------------------------------------------------

    patterns = [
        r"\bemployees?\b",
        r"\bemployee\b",
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

            normalized = match.strip().lower()

            normalized = re.sub(
                r"\s+",
                " ",
                normalized
            )

            entities.add(normalized)

    # ---------------------------------------------------------------
    # Numbers with units
    # ---------------------------------------------------------------

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
                match.strip().lower()
            )

    return sorted(entities)


# ====================================================================
# FLOW STEP 6: RELATIONSHIP EXTRACTION
# ====================================================================
#
# Entities
#   ↓
# Relationships
#   ↓
# Knowledge Graph
#
# Example:
#
# Employee
#    │
#    └── expected_to_work
#                 │
#                 ▼
#              8 hours
#
# ====================================================================


def extract_relationships(
    chunk
):

    text = chunk["text"]

    entities = extract_entities(text)

    relationships = []

    # ---------------------------------------------------------------
    # Rule 1:
    #
    # employee → expected_to_work → 8 hours
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
                        "relation": "expected_to_work",
                        "object": entity,
                        "chunk_id": chunk["id"]
                    }
                )

    # ---------------------------------------------------------------
    # Rule 2:
    #
    # employee → available_during → scheduled hours
    # ---------------------------------------------------------------

    if (
        "employee" in entities
        and "scheduled hours" in entities
    ):

        relationships.append(
            {
                "subject": "employee",
                "relation": "available_during",
                "object": "scheduled hours",
                "chunk_id": chunk["id"]
            }
        )

    # ---------------------------------------------------------------
    # Rule 3:
    #
    # employee → has → meal break
    # ---------------------------------------------------------------

    if (
        "employee" in entities
        and "meal break" in entities
    ):

        relationships.append(
            {
                "subject": "employee",
                "relation": "has",
                "object": "meal break",
                "chunk_id": chunk["id"]
            }
        )

    # ---------------------------------------------------------------
    # Rule 4:
    #
    # working hours → excludes → meal breaks
    # ---------------------------------------------------------------

    if (
        "working hours" in entities
        and "meal break" in entities
    ):

        relationships.append(
            {
                "subject": "working hours",
                "relation": "excludes",
                "object": "meal break",
                "chunk_id": chunk["id"]
            }
        )

    return entities, relationships


# ====================================================================
# FLOW STEP 7: BUILD KNOWLEDGE GRAPH
# ====================================================================
#
# Chunks
#   ↓
# Entity Extraction
#   ↓
# Relationship Extraction
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
            extract_relationships(chunk)
        )

        # -----------------------------------------------------------
        # Store entity → chunk relationship
        # -----------------------------------------------------------

        for entity in entities:

            ENTITY_TO_CHUNKS[
                entity
            ].add(
                chunk["id"]
            )

        # -----------------------------------------------------------
        # Store graph relationships
        # -----------------------------------------------------------

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
        f"Unique entities: "
        f"{len(ENTITY_TO_CHUNKS)}"
    )

    print(
        f"Relationships: "
        f"{len(RELATIONSHIPS)}"
    )

    print("\nKnowledge Graph:")

    for entity, neighbours in GRAPH.items():

        print(
            f"  {entity} -> "
            f"{sorted(neighbours)}"
        )


# ====================================================================
# FLOW STEP 8: USER QUESTION
# ====================================================================
#
# User
#   ↓
# Question
#
# ====================================================================


def get_user_question():

    print("\n" + "=" * 70)
    print("FLOW STEP 8: USER QUESTION")
    print("=" * 70)

    question = input(
        "\nAsk your HR Policy question: "
    ).strip()

    return question


# ====================================================================
# FLOW STEP 9: QUERY ENTITY EXTRACTION
# ====================================================================
#
# Question
#   ↓
# Extract entities
#   ↓
# Graph entities
#
# ====================================================================


def extract_query_entities(question):

    print("\n" + "=" * 70)
    print("FLOW STEP 9: QUERY ENTITY EXTRACTION")
    print("=" * 70)

    entities = extract_entities(
        question
    )

    print(
        f"Query entities: {entities}"
    )

    return entities


# ====================================================================
# FLOW STEP 10: GRAPH RETRIEVAL
# ====================================================================
#
# Question Entities
#       ↓
# Knowledge Graph
#       ↓
# Related Entities
#       ↓
# Entity → Chunk
#       ↓
# Candidate Chunks
#
# ====================================================================


def graph_retrieval(
    query_entities
):

    print("\n" + "=" * 70)
    print("FLOW STEP 10: GRAPH RETRIEVAL")
    print("=" * 70)

    candidate_chunk_ids = set()

    visited_entities = set()

    # ---------------------------------------------------------------
    # Direct entity → chunk retrieval
    # ---------------------------------------------------------------

    for entity in query_entities:

        if entity in ENTITY_TO_CHUNKS:

            candidate_chunk_ids.update(
                ENTITY_TO_CHUNKS[entity]
            )

            visited_entities.add(
                entity
            )

    # ---------------------------------------------------------------
    # One-hop graph traversal
    # ---------------------------------------------------------------

    for entity in list(
        visited_entities
    ):

        neighbours = GRAPH.get(
            entity,
            set()
        )

        for neighbour in neighbours:

            candidate_chunk_ids.update(
                ENTITY_TO_CHUNKS.get(
                    neighbour,
                    set()
                )
            )

    print(
        f"Graph candidate chunks: "
        f"{len(candidate_chunk_ids)}"
    )

    results = []

    for chunk_id in candidate_chunk_ids:

        if chunk_id in CHUNK_STORE:

            results.append(
                CHUNK_STORE[chunk_id]
            )

    return results


# ====================================================================
# FLOW STEP 11: VECTOR FALLBACK RETRIEVAL
# ====================================================================
#
# If graph retrieval finds nothing:
#
# Question
#   ↓
# SentenceTransformer
#   ↓
# ChromaDB
#   ↓
# Candidate chunks
#
# ====================================================================


def vector_fallback_retrieval(
    question,
    collection,
    embedding_model
):

    print("\n" + "=" * 70)
    print("FLOW STEP 11: VECTOR FALLBACK RETRIEVAL")
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

        relevance_score = (
            1 / (1 + distance)
        )

        output.append(
            {
                "id": chunk_id,
                "text": document,
                "page_no": page_no,
                "chunk_no": chunk_no,
                "file_name": metadata.get(
                    "file_name",
                    os.path.basename(PDF_PATH)
                ),
                "retrieval_score":
                    relevance_score
            }
        )

    return output


# ====================================================================
# FLOW STEP 12: LOAD CROSSENCODER
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
    print("FLOW STEP 12: LOAD CROSSENCODER")
    print("=" * 70)

    print(
        f"Loading: {RERANKER_MODEL}"
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    return reranker


# ====================================================================
# FLOW STEP 13: CROSSENCODER RERANKING
# ====================================================================
#
# Question
#    +
# Candidate Chunks
#    ↓
# CrossEncoder
#    ↓
# Relevance Scores
#    ↓
# Sorted Chunks
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

    pairs = []

    for chunk in chunks:

        pairs.append(
            [
                question,
                chunk["text"]
            ]
        )

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
        key=lambda x: x[
            "rerank_score"
        ],
        reverse=True
    )

    return ranked_chunks[
        :FINAL_TOP_K
    ]


# ====================================================================
# FLOW STEP 14: BUILD GRAPH CONTEXT
# ====================================================================
#
# Top Graph Chunks
#       ↓
# Context
#       ↓
# LLM
#
# ====================================================================


def build_context(
    ranked_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 14: BUILD CONTEXT")
    print("=" * 70)

    context_parts = []

    for index, chunk in enumerate(
        ranked_chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {index}
Page Number: {chunk['page_no']}
Chunk Number: {chunk['chunk_no']}
Chunk ID: {chunk['id']}
Relevance Score: {chunk['rerank_score']:.4f}

Chunk Text:
{chunk['text']}
"""
        )

    return "\n".join(
        context_parts
    )


# ====================================================================
# FLOW STEP 15: SYSTEM PROMPT
# ====================================================================
#
# Defines how Mistral should answer.
#
# ====================================================================


def build_system_prompt():

    return """
You are an HR Policy question-answering assistant.

You must answer ONLY from the supplied context.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer cannot be found in the context,
   return:
   "I don't know based on HR_Policy.pdf."
4. Always provide source information.
5. Confidence must be between 0 and 1.
6. Return ONLY valid JSON.
7. Follow the requested JSON structure exactly.

JSON structure:

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
# FLOW STEP 16: USER PROMPT
# ====================================================================
#
# Question
#    +
# Context
#    ↓
# LLM prompt
#
# ====================================================================


def build_user_prompt(
    question,
    context
):

    return f"""
Answer the following HR Policy question.

QUESTION:
{question}

RETRIEVED GRAPH CONTEXT:
{context}

Return only the required JSON array.
"""


# ====================================================================
# FLOW STEP 17: MISTRAL GENERATION
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
    print("FLOW STEP 17: MISTRAL GENERATION")
    print("=" * 70)

    system_prompt = (
        build_system_prompt()
    )

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

    return response["message"]["content"]


# ====================================================================
# FLOW STEP 18: JSON PARSING
# ====================================================================
#
# Mistral response
#       ↓
# JSON parsing
#       ↓
# Python object
#
# Handles:
#
# 1. Direct array
#
# [
#   {...}
# ]
#
# 2. Schema wrapper
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
    print("FLOW STEP 18: JSON PARSING")
    print("=" * 70)

    try:

        data = json.loads(
            raw_response
        )

        # -----------------------------------------------------------
        # Handle schema wrapper
        # -----------------------------------------------------------

        if (
            isinstance(data, dict)
            and data.get("type") == "array"
            and "items" in data
        ):

            data = data["items"]

        # -----------------------------------------------------------
        # Handle object containing answer array
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

    except json.JSONDecodeError as error:

        print(
            "JSON parsing failed:"
        )

        print(error)

        return [
            {
                "Answer": raw_response,
                "Source": [],
                "Confidence": 0.0
            }
        ]


# ====================================================================
# FLOW STEP 19: VALIDATE SOURCES
# ====================================================================
#
# LLM Answer
#       ↓
# Validate source against
# retrieved chunks
#
# ====================================================================


def validate_sources(
    answer_data,
    ranked_chunks
):

    print("\n" + "=" * 70)
    print("FLOW STEP 19: SOURCE VALIDATION")
    print("=" * 70)

    valid_source_ids = {
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

            if chunk_id in valid_source_ids:

                valid_sources.append(
                    source
                )

        item["Source"] = (
            valid_sources
        )

        confidence = item.get(
            "Confidence",
            0.0
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

        item["Confidence"] = (
            confidence
        )

        if "Answer" not in item:

            item["Answer"] = (
                "I don't know based on "
                "HR_Policy.pdf."
            )

    return answer_data


# ====================================================================
# FLOW STEP 20: FINAL ANSWER
# ====================================================================
#
# Validated JSON
#       ↓
# Pretty JSON
#
# ====================================================================


def print_final_answer(
    answer_data
):

    print("\n" + "=" * 70)
    print("FLOW STEP 20: FINAL ANSWER")
    print("=" * 70)

    print(
        json.dumps(
            answer_data,
            indent=2,
            ensure_ascii=False
        )
    )


# ====================================================================
# FLOW STEP 21: SINGLE QUESTION PIPELINE
# ====================================================================
#
# Question
#   ↓
# Query Entities
#   ↓
# Graph Retrieval
#   ↓
# Vector Fallback
#   ↓
# Reranking
#   ↓
# Context
#   ↓
# Mistral
#   ↓
# JSON
#   ↓
# Validation
#   ↓
# Final Answer
#
# ====================================================================


def answer_question(
    question,
    collection,
    embedding_model,
    reranker
):

    # ---------------------------------------------------------------
    # Extract entities from question
    # ---------------------------------------------------------------

    query_entities = (
        extract_query_entities(
            question
        )
    )

    # ---------------------------------------------------------------
    # Graph retrieval
    # ---------------------------------------------------------------

    graph_chunks = graph_retrieval(
        query_entities
    )

    # ---------------------------------------------------------------
    # If graph retrieval finds nothing,
    # use vector retrieval as fallback.
    # ---------------------------------------------------------------

    if not graph_chunks:

        print(
            "\nGraph retrieval returned "
            "no chunks."
        )

        graph_chunks = (
            vector_fallback_retrieval(
                question,
                collection,
                embedding_model
            )
        )

    # ---------------------------------------------------------------
    # If graph retrieval returns too few
    # candidates, add vector results.
    # ---------------------------------------------------------------

    if len(graph_chunks) < RETRIEVAL_TOP_K:

        vector_chunks = (
            vector_fallback_retrieval(
                question,
                collection,
                embedding_model
            )
        )

        existing_ids = {
            chunk["id"]
            for chunk in graph_chunks
        }

        for chunk in vector_chunks:

            if (
                chunk["id"]
                not in existing_ids
            ):

                graph_chunks.append(
                    chunk
                )

                existing_ids.add(
                    chunk["id"]
                )

    # ---------------------------------------------------------------
    # Limit candidates
    # ---------------------------------------------------------------

    graph_chunks = graph_chunks[
        :RETRIEVAL_TOP_K
    ]

    print(
        f"\nCandidates for reranking: "
        f"{len(graph_chunks)}"
    )

    # ---------------------------------------------------------------
    # CrossEncoder reranking
    # ---------------------------------------------------------------

    ranked_chunks = rerank_chunks(
        question,
        graph_chunks,
        reranker
    )

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
    # Build context
    # ---------------------------------------------------------------

    context = build_context(
        ranked_chunks
    )

    # ---------------------------------------------------------------
    # Generate answer
    # ---------------------------------------------------------------

    raw_response = generate_answer(
        question,
        context
    )

    print(
        "\nRaw LLM Response:"
    )

    print(raw_response)

    # ---------------------------------------------------------------
    # Parse JSON
    # ---------------------------------------------------------------

    answer_data = parse_json_response(
        raw_response
    )

    # ---------------------------------------------------------------
    # Validate source
    # ---------------------------------------------------------------

    answer_data = validate_sources(
        answer_data,
        ranked_chunks
    )

    return answer_data


# ====================================================================
# FLOW STEP 22: MAIN
# ====================================================================
#
# Complete initialization:
#
# PDF
#  ↓
# OCR
#  ↓
# Chunking
#  ↓
# Embedding
#  ↓
# ChromaDB
#  ↓
# Graph
#  ↓
# Reranker
#  ↓
# Question Loop
#
# ====================================================================


def main():

    print("\n")
    print("=" * 70)
    print("GRAPH RAG - HR POLICY")
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
    # STEP 3: Embedding model
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
    # STEP 5-7: Build Knowledge Graph
    # ---------------------------------------------------------------

    build_knowledge_graph(
        chunks
    )

    # ---------------------------------------------------------------
    # STEP 12: Load CrossEncoder
    # ---------------------------------------------------------------

    reranker = load_reranker()

    # ---------------------------------------------------------------
    # STEP 22: Question Loop
    # ---------------------------------------------------------------

    while True:

        print("\n")
        print("=" * 70)
        print("ASK HR POLICY QUESTION")
        print("Type 'exit', 'quit', or 'q' to stop.")
        print("=" * 70)

        question = input(
            "\nQuestion: "
        ).strip()

        # -----------------------------------------------------------
        # Exit condition
        # -----------------------------------------------------------

        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "\nExiting Graph RAG..."
            )

            break

        if not question:

            print(
                "\nPlease enter a question."
            )

            continue

        # -----------------------------------------------------------
        # Run Graph RAG
        # -----------------------------------------------------------

        answer = answer_question(
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