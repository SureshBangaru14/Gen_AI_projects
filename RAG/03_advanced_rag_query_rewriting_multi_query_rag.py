# ====================================================================
# ADVANCED RAG - METHOD 2
#
# Query Rewriting + Multi-Query Retrieval + CrossEncoder Reranking
#
# FLOW
#
# PDF
#   ↓
# OCR
#   ↓
# Page-wise Text
#   ↓
# Recursive Character Chunking
#   ↓
# SentenceTransformer Embeddings
#   ↓
# ChromaDB
#   ↓
# User Question
#   ↓
# Query Rewriting
#   ↓
# Generate Multiple Queries
#   ↓
# Query Embeddings
#   ↓
# Dense Retrieval for Each Query
#   ↓
# Merge + Deduplicate
#   ↓
# CrossEncoder Reranking
#   ↓
# Top 5 Chunks
#   ↓
# Context
#   ↓
# System Prompt + Schema
#   ↓
# User Prompt
#   ↓
# Mistral
#   ↓
# JSON Parsing
#   ↓
# Source Validation
#   ↓
# Final Answer
#
# ====================================================================


# ====================================================================
# 1. IMPORT LIBRARIES
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
# 2. CONFIGURATION
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

CHROMA_DIR = "chroma_multi_query_hr"

COLLECTION_NAME = "hr_policy_multi_query"

OCR_DPI = 250

CHUNK_SIZE = 600

CHUNK_OVERLAP = 100

TOP_K_PER_QUERY = 5

FINAL_TOP_K = 5

NUMBER_OF_QUERIES = 4


# ====================================================================
# 3. JSON SCHEMA
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
# 4. OCR
# ====================================================================

def extract_text_with_ocr(pdf_path):
    """
    STEP 1

    PDF
      ↓
    PDF Pages
      ↓
    OCR
      ↓
    Page-wise Text
    """

    print("\n" + "=" * 70)
    print("STEP 1 - OCR")
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
# 5. RECURSIVE CHARACTER CHUNKING
# ====================================================================

def create_chunks(pages):
    """
    STEP 2

    Page-wise Text
        ↓
    RecursiveCharacterTextSplitter
        ↓
    Chunks
    """

    print("\n" + "=" * 70)
    print("STEP 2 - RECURSIVE CHARACTER CHUNKING")
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

                        "file_name": os.path.basename(
                            PDF_PATH
                        ),

                        "page_no": page_no,

                        "chunk_no": chunk_no
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
# 6. LOAD SENTENCE TRANSFORMER
# ====================================================================

def load_embedding_model():
    """
    STEP 3

    SentenceTransformer
        ↓
    Embedding Model
    """

    print("\n" + "=" * 70)
    print("STEP 3 - LOAD EMBEDDING MODEL")
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
# 7. CREATE CHROMADB
# ====================================================================

def create_chroma_db():
    """
    STEP 4

    ChromaDB
        ↓
    Persistent Vector Database
    """

    print("\n" + "=" * 70)
    print("STEP 4 - CHROMADB")
    print("=" * 70)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(
        f"ChromaDB collection: "
        f"{COLLECTION_NAME}"
    )

    return collection


# ====================================================================
# 8. EMBED AND UPSERT DOCUMENTS
# ====================================================================

def ingest_chunks(
    collection,
    embedding_model,
    chunks
):
    """
    STEP 4

    Chunks
      ↓
    SentenceTransformer
      ↓
    Embeddings
      ↓
    ChromaDB
    """

    print("\n" + "=" * 70)
    print("STEP 4 - CHROMADB UPSERT")
    print("=" * 70)

    if not chunks:

        print(
            "No chunks available."
        )

        return

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
# 9. USER QUESTION
# ====================================================================

def get_user_question():
    """
    STEP 5

    User
      ↓
    Question
    """

    print("\n" + "=" * 70)
    print("STEP 5 - USER QUESTION")
    print("=" * 70)

    question = input(
        "\nEnter your question: "
    ).strip()

    return question


# ====================================================================
# 10. QUERY REWRITING
# ====================================================================

def rewrite_query(question):
    """
    STEP 6

    Original Question
        ↓
    LLM Query Rewriting
        ↓
    Improved Search Query
    """

    print("\n" + "=" * 70)
    print("STEP 6 - QUERY REWRITING")
    print("=" * 70)

    system_prompt = """
You are a search query rewriting system.

Your task is to rewrite the user's question
into one clear and concise search query.

Rules:

1. Preserve the original meaning.
2. Do not answer the question.
3. Do not add unsupported information.
4. Return only the rewritten query.
5. Do not use JSON.
"""

    user_prompt = f"""
Original question:

{question}

Rewrite this question for semantic document retrieval.
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

    rewritten_query = (
        response["message"]["content"]
        .strip()
    )

    print(
        "\nOriginal Query:"
    )

    print(question)

    print(
        "\nRewritten Query:"
    )

    print(rewritten_query)

    return rewritten_query


# ====================================================================
# 11. GENERATE MULTIPLE QUERIES
# ====================================================================

def generate_multiple_queries(
    question,
    rewritten_query
):
    """
    STEP 7

    Original Question
        ↓
    Rewritten Query
        ↓
    Multiple Query Generation
        ↓
    Query 1
    Query 2
    Query 3
    Query 4
    """

    print("\n" + "=" * 70)
    print("STEP 7 - MULTI-QUERY GENERATION")
    print("=" * 70)

    system_prompt = f"""
You are a search query generation system.

Generate exactly {NUMBER_OF_QUERIES}
different search queries for the same question.

The queries should represent different
ways of searching the HR policy document.

Rules:

1. Preserve the original intent.
2. Do not answer the question.
3. Do not add unsupported facts.
4. Return ONLY a JSON array.
5. Each item must be a string.

Example:

[
    "query one",
    "query two",
    "query three",
    "query four"
]
"""

    user_prompt = f"""
Original Question:

{question}

Rewritten Query:

{rewritten_query}

Generate {NUMBER_OF_QUERIES}
different retrieval queries.
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
        "\nRaw Multi-Query Response:"
    )

    print(raw_response)

    try:

        data = json.loads(
            raw_response
        )

    except json.JSONDecodeError:

        print(
            "Could not parse multi-query JSON."
        )

        return [
            rewritten_query
        ]

    queries = []

    if isinstance(
        data,
        list
    ):

        queries = data

    elif isinstance(
        data,
        dict
    ):

        if "queries" in data:

            queries = data["queries"]

        elif "items" in data:

            queries = data["items"]

    queries = [
        str(query).strip()
        for query in queries
        if str(query).strip()
    ]

    if not queries:

        queries = [
            rewritten_query
        ]

    print(
        "\nGenerated Queries:"
    )

    for index, query in enumerate(
        queries,
        start=1
    ):

        print(
            f"{index}. {query}"
        )

    return queries


# ====================================================================
# 12. EMBED SEARCH QUERIES
# ====================================================================

def embed_queries(
    queries,
    embedding_model
):
    """
    STEP 8

    Multiple Queries
        ↓
    SentenceTransformer
        ↓
    Query Embeddings
    """

    print("\n" + "=" * 70)
    print("STEP 8 - QUERY EMBEDDINGS")
    print("=" * 70)

    embeddings = embedding_model.encode(

        queries,

        normalize_embeddings=True,

        show_progress_bar=False
    ).tolist()

    print(
        f"Created {len(embeddings)} "
        f"query embeddings."
    )

    return embeddings


# ====================================================================
# 13. MULTI-QUERY DENSE RETRIEVAL
# ====================================================================

def multi_query_retrieval(
    collection,
    queries,
    query_embeddings
):
    """
    STEP 9

    Query 1 ──→ ChromaDB ──→ Top 5
    Query 2 ──→ ChromaDB ──→ Top 5
    Query 3 ──→ ChromaDB ──→ Top 5
    Query 4 ──→ ChromaDB ──→ Top 5
    """

    print("\n" + "=" * 70)
    print("STEP 9 - MULTI-QUERY DENSE RETRIEVAL")
    print("=" * 70)

    all_results = []

    for query_index, (
        query,
        query_embedding
    ) in enumerate(
        zip(
            queries,
            query_embeddings
        ),
        start=1
    ):

        print(
            f"\nQuery {query_index}:"
        )

        print(query)

        results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=TOP_K_PER_QUERY,

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

        for document, metadata, distance in zip(

            documents,

            metadatas,

            distances
        ):

            chunk_id = (
                f"page_{metadata['page_no']}"
                f"_chunk_{metadata['chunk_no']}"
            )

            all_results.append(

                {
                    "id": chunk_id,

                    "text": document,

                    "metadata": metadata,

                    "distance": distance,

                    "query": query
                }
            )

    print(
        f"\nTotal retrieved results: "
        f"{len(all_results)}"
    )

    return all_results


# ====================================================================
# 14. MERGE + DEDUPLICATE
# ====================================================================

def merge_and_deduplicate(
    results
):
    """
    STEP 10

    Query Results
        ↓
    Merge
        ↓
    Deduplicate by Chunk ID
    """

    print("\n" + "=" * 70)
    print("STEP 10 - MERGE + DEDUPLICATE")
    print("=" * 70)

    unique_chunks = {}

    for result in results:

        chunk_id = result["id"]

        if chunk_id not in unique_chunks:

            unique_chunks[chunk_id] = result

        else:

            existing = (
                unique_chunks[chunk_id]
            )

            if (
                result["distance"]
                <
                existing["distance"]
            ):

                unique_chunks[chunk_id] = (
                    result
                )

    merged_results = list(
        unique_chunks.values()
    )

    print(
        f"Before deduplication: "
        f"{len(results)}"
    )

    print(
        f"After deduplication: "
        f"{len(merged_results)}"
    )

    return merged_results


# ====================================================================
# 15. LOAD CROSSENCODER
# ====================================================================

def load_reranker():
    """
    STEP 11

    CrossEncoder Model
        ↓
    Load Reranker
    """

    print("\n" + "=" * 70)
    print("STEP 11 - LOAD CROSSENCODER")
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
# 16. CROSSENCODER RERANKING
# ====================================================================

def rerank_chunks(
    question,
    chunks,
    reranker
):
    """
    STEP 12

    User Question
        +
    Retrieved Chunks
        ↓
    CrossEncoder
        ↓
    Reranked Chunks
    """

    print("\n" + "=" * 70)
    print("STEP 12 - CROSSENCODER RERANKING")
    print("=" * 70)

    if not chunks:

        return []

    pairs = []

    for chunk in chunks:

        pairs.append(
            (
                question,
                chunk["text"]
            )
        )

    print(
        f"Reranking {len(pairs)} chunks..."
    )

    scores = reranker.predict(
        pairs
    )

    reranked = []

    for chunk, score in zip(
        chunks,
        scores
    ):

        item = dict(chunk)

        item["reranker_score"] = (
            float(score)
        )

        reranked.append(item)

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    print(
        "\nReranked Results:"
    )

    for rank, chunk in enumerate(
        reranked,
        start=1
    ):

        print(
            f"{rank}. "
            f"{chunk['id']} "
            f"Score: "
            f"{chunk['reranker_score']:.6f}"
        )

    return reranked


# ====================================================================
# 17. SELECT TOP 5
# ====================================================================

def select_top_chunks(
    reranked_chunks
):
    """
    STEP 13

    Reranked Chunks
        ↓
    Top 5
    """

    print("\n" + "=" * 70)
    print("STEP 13 - SELECT TOP 5")
    print("=" * 70)

    top_chunks = reranked_chunks[
        :FINAL_TOP_K
    ]

    print(
        f"Selected top "
        f"{len(top_chunks)} chunks."
    )

    return top_chunks


# ====================================================================
# 18. BUILD CONTEXT
# ====================================================================

def build_context(
    top_chunks
):
    """
    STEP 14

    Top 5 Chunks
        ↓
    Context
    """

    print("\n" + "=" * 70)
    print("STEP 14 - BUILD CONTEXT")
    print("=" * 70)

    context_parts = []

    for rank, chunk in enumerate(
        top_chunks,
        start=1
    ):

        metadata = chunk["metadata"]

        page_no = metadata[
            "page_no"
        ]

        chunk_no = metadata[
            "chunk_no"
        ]

        chunk_id = chunk["id"]

        chunk_text = chunk["text"]

        reranker_score = (
            chunk["reranker_score"]
        )

        context = f"""
============================================================
Retrieved Chunk Rank: {rank}
Page Number: {page_no}
Chunk Number: {chunk_no}
Chunk ID: {chunk_id}
Relevance Score: {reranker_score:.6f}

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
# 19. SYSTEM PROMPT
# ====================================================================

def create_system_prompt():
    """
    STEP 15

    System Prompt
        +
    Schema Rules
    """

    print("\n" + "=" * 70)
    print("STEP 15 - SYSTEM PROMPT")
    print("=" * 70)

    schema_text = json.dumps(
        ARRAY_SCHEMA,
        indent=4
    )

    system_prompt = f"""
You are an HR policy question answering system.

You must answer ONLY using the retrieved context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the
   retrieved context, say:

   "I don't know based on HR_Policy.pdf."

4. Return ONLY valid JSON.
5. The top-level response MUST be an array.
6. Return exactly one answer record.
7. Source MUST contain the chunks that support
   the answer.
8. Preserve the source page number.
9. Preserve the source chunk number.
10. Preserve the exact chunk ID.
11. Include the chunk text.
12. Relevance Score must be the CrossEncoder score.
13. Confidence must be between 0 and 1.
14. Do not create fake sources.

JSON Schema:

{schema_text}
"""

    return system_prompt


# ====================================================================
# 20. USER PROMPT
# ====================================================================

def create_user_prompt(
    question,
    context
):
    """
    STEP 16

    User Question
        +
    Retrieved Context
        ↓
    LLM Prompt
    """

    print("\n" + "=" * 70)
    print("STEP 16 - USER PROMPT")
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
# 21. CALL MISTRAL
# ====================================================================

def call_llm(
    system_prompt,
    user_prompt
):
    """
    STEP 17

    System Prompt
        +
    User Prompt
        ↓
    Mistral
        ↓
    JSON Response
    """

    print("\n" + "=" * 70)
    print("STEP 17 - MISTRAL LLM")
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
# 22. JSON PARSING
# ====================================================================

def parse_llm_response(
    raw_response
):
    """
    STEP 18

    Raw LLM Response
        ↓
    JSON Parsing
        ↓
    Python Object
    """

    print("\n" + "=" * 70)
    print("STEP 18 - JSON PARSING")
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

    # ------------------------------------------------------------
    # Handle direct array
    # ------------------------------------------------------------

    if isinstance(
        data,
        list
    ):

        return data

    # ------------------------------------------------------------
    # Handle object containing items
    # ------------------------------------------------------------

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

        # --------------------------------------------------------
        # Handle a single answer object
        # --------------------------------------------------------

        if "Answer" in data:

            return [
                data
            ]

    return []


# ====================================================================
# 23. SOURCE VALIDATION
# ====================================================================

def validate_response(
    parsed_response,
    top_chunks
):
    """
    STEP 19

    LLM JSON
        ↓
    Validate
        ↓
    Valid Final JSON
    """

    print("\n" + "=" * 70)
    print("STEP 19 - SOURCE VALIDATION")
    print("=" * 70)

    if not parsed_response:

        return []

    valid_chunk_ids = {
        chunk["id"]
        for chunk in top_chunks
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
                "Answer": answer,

                "Source":
                    validated_sources,

                "Confidence":
                    confidence
            }
        )

    return validated_records


# ====================================================================
# 24. FINAL ANSWER
# ====================================================================

def process_question(
    question,
    embedding_model,
    collection,
    reranker
):
    """
    STEP 20

    Complete Advanced RAG Pipeline
    """

    # ------------------------------------------------------------
    # STEP 6 - Query Rewriting
    # ------------------------------------------------------------

    rewritten_query = rewrite_query(
        question
    )

    # ------------------------------------------------------------
    # STEP 7 - Multi Query Generation
    # ------------------------------------------------------------

    queries = generate_multiple_queries(
        question,
        rewritten_query
    )

    # ------------------------------------------------------------
    # STEP 8 - Query Embeddings
    # ------------------------------------------------------------

    query_embeddings = embed_queries(
        queries,
        embedding_model
    )

    # ------------------------------------------------------------
    # STEP 9 - Multi Query Retrieval
    # ------------------------------------------------------------

    retrieved_chunks = multi_query_retrieval(

        collection,

        queries,

        query_embeddings
    )

    # ------------------------------------------------------------
    # STEP 10 - Merge + Deduplicate
    # ------------------------------------------------------------

    unique_chunks = merge_and_deduplicate(
        retrieved_chunks
    )

    # ------------------------------------------------------------
    # STEP 12 - CrossEncoder Reranking
    # ------------------------------------------------------------

    reranked_chunks = rerank_chunks(

        question,

        unique_chunks,

        reranker
    )

    # ------------------------------------------------------------
    # STEP 13 - Top 5
    # ------------------------------------------------------------

    top_chunks = select_top_chunks(
        reranked_chunks
    )

    # ------------------------------------------------------------
    # STEP 14 - Context
    # ------------------------------------------------------------

    context = build_context(
        top_chunks
    )

    # ------------------------------------------------------------
    # STEP 15 - System Prompt
    # ------------------------------------------------------------

    system_prompt = (
        create_system_prompt()
    )

    # ------------------------------------------------------------
    # STEP 16 - User Prompt
    # ------------------------------------------------------------

    user_prompt = create_user_prompt(

        question,

        context
    )

    # ------------------------------------------------------------
    # STEP 17 - Mistral
    # ------------------------------------------------------------

    raw_response = call_llm(

        system_prompt,

        user_prompt
    )

    # ------------------------------------------------------------
    # STEP 18 - JSON Parsing
    # ------------------------------------------------------------

    parsed_response = parse_llm_response(
        raw_response
    )

    # ------------------------------------------------------------
    # STEP 19 - Source Validation
    # ------------------------------------------------------------

    validated_response = validate_response(

        parsed_response,

        top_chunks
    )

    # ------------------------------------------------------------
    # STEP 20 - Final Answer
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 20 - FINAL ANSWER")
    print("=" * 70)

    print(
        json.dumps(
            validated_response,
            indent=4,
            ensure_ascii=False
        )
    )

    return validated_response


# ====================================================================
# 25. MAIN
# ====================================================================

def main():
    """
    STEP 21

    Complete Application Flow

    PDF
      ↓
    OCR
      ↓
    Chunking
      ↓
    Embeddings
      ↓
    ChromaDB
      ↓
    CrossEncoder
      ↓
    WHILE LOOP
      ↓
    Question
      ↓
    Query Rewriting
      ↓
    Multi Query
      ↓
    Retrieval
      ↓
    Reranking
      ↓
    Mistral
      ↓
    Final JSON
    """

    print("\n")
    print("=" * 70)
    print("ADVANCED RAG - METHOD 2")
    print("QUERY REWRITING + MULTI-QUERY RAG")
    print("=" * 70)

    # ================================================================
    # STEP 1 - OCR
    # ================================================================

    pages = extract_text_with_ocr(
        PDF_PATH
    )

    # ================================================================
    # STEP 2 - CHUNKING
    # ================================================================

    chunks = create_chunks(
        pages
    )

    if not chunks:

        print(
            "No chunks created."
        )

        return

    # ================================================================
    # STEP 3 - EMBEDDING MODEL
    # ================================================================

    embedding_model = (
        load_embedding_model()
    )

    # ================================================================
    # STEP 4 - CHROMADB
    # ================================================================

    collection = create_chroma_db()

    ingest_chunks(

        collection,

        embedding_model,

        chunks
    )

    # ================================================================
    # STEP 11 - CROSSENCODER
    # ================================================================

    reranker = load_reranker()

    # ================================================================
    # STEP 21 - WHILE LOOP
    # ================================================================

    print("\n" + "=" * 70)
    print("STEP 21 - ASK QUESTIONS")
    print("=" * 70)

    while True:

        question = input(
            "\nEnter your question "
            "(type 'exit' to stop): "
        ).strip()

        # ------------------------------------------------------------
        # Exit
        # ------------------------------------------------------------

        if question.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print(
                "\nExiting Advanced RAG."
            )

            break

        # ------------------------------------------------------------
        # Empty question
        # ------------------------------------------------------------

        if not question:

            print(
                "Please enter a question."
            )

            continue

        # ------------------------------------------------------------
        # Process question
        # ------------------------------------------------------------

        process_question(

            question,

            embedding_model,

            collection,

            reranker
        )


# ====================================================================
# 26. APPLICATION ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()