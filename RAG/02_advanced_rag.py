"""
======================================================================
ADVANCED RAG - HR POLICY
======================================================================

COMPLETE ADVANCED RAG FLOW
======================================================================

STEP 1
PDF
  |
  v
Tesseract OCR
  |
  v

STEP 2
Page-wise Text
  |
  v
Recursive Character Chunking
  |
  v

STEP 3
SentenceTransformer
all-MiniLM-L6-v2
  |
  v
Normalized Embeddings
  |
  v

STEP 4
ChromaDB
  |
  v

STEP 5
USER QUESTION
  |
  v

STEP 6
QUERY EMBEDDING
  |
  v

STEP 7
DENSE RETRIEVAL
TOP 10
  |
  v

STEP 8
CROSS ENCODER RERANKING
  |
  v

STEP 9
TOP 5 RELEVANT CHUNKS
  |
  v

STEP 10
CONTEXT BUILDING
  |
  v

STEP 11
SYSTEM PROMPT
  |
  v

STEP 12
USER PROMPT
  |
  v

STEP 13
MISTRAL LLM
OLLAMA
  |
  v

STEP 14
JSON PARSING
  |
  v

STEP 15
SOURCE VALIDATION
  |
  v

STEP 16
FINAL ANSWER
  |
  v

STEP 17
ASK NEXT QUESTION
  |
  +----------------------+
                         |
                         v
                    WHILE LOOP
======================================================================
"""


# ====================================================================
# 1. IMPORTS
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


# ====================================================================
# 2. CONFIGURATION
# ====================================================================

# PDF file path
PDF_PATH = (
    "/home/suresh/Gen_AI_Practice/Generative-AI/"
    "Vector_Database/HR_Policy.pdf"
)

# SentenceTransformer embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# CrossEncoder reranker model
RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Ollama LLM
LLM_MODEL = "mistral"

# ChromaDB persistent directory
CHROMA_DIR = "chroma_advanced_hr"

# ChromaDB collection name
COLLECTION_NAME = "hr_policy_advanced"

# OCR DPI
OCR_DPI = 250

# Chunk size
CHUNK_SIZE = 600

# Chunk overlap
CHUNK_OVERLAP = 100

# Number of chunks retrieved from ChromaDB
RETRIEVAL_TOP_K = 10

# Number of chunks selected after reranking
RERANK_TOP_K = 5


# ====================================================================
# 3. FIELD NAMES
# ====================================================================

FIELD_NAMES = [
    "Answer",
    "Source",
    "Confidence"
]


# ====================================================================
# 4. SOURCE SCHEMA
# ====================================================================

SOURCE_SCHEMA = {
    "type": "array",
    "description": (
        "List of document chunks supporting "
        "the generated answer."
    ),
    "items": {
        "type": "object",

        "properties": {

            "Page Number": {
                "type": "integer",
                "description": (
                    "PDF page number."
                )
            },

            "Chunk Number": {
                "type": "integer",
                "description": (
                    "Chunk number within the page."
                )
            },

            "Chunk ID": {
                "type": "string",
                "description": (
                    "Unique chunk identifier."
                )
            },

            "Chunk Text": {
                "type": "string",
                "description": (
                    "Actual retrieved chunk text."
                )
            },

            "Relevance Score": {
                "type": "number",
                "description": (
                    "CrossEncoder relevance score."
                )
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
}


# ====================================================================
# 5. FINAL OBJECT SCHEMA
# ====================================================================

SCHEMA = {
    "type": "object",

    "properties": {

        "Answer": {
            "type": "string",
            "description": (
                "Answer generated only from "
                "retrieved HR policy context."
            )
        },

        "Source": SOURCE_SCHEMA,

        "Confidence": {
            "type": "number",
            "description": (
                "Confidence score between 0 and 1."
            )
        }
    },

    "required": FIELD_NAMES,

    "additionalProperties": False
}


# ====================================================================
# 6. ARRAY SCHEMA
# ====================================================================

ARRAY_SCHEMA = {
    "type": "array",

    "description": (
        "Array containing one or more answer records."
    ),

    "items": SCHEMA
}


# ====================================================================
# 7. STEP 1 - OCR
# ====================================================================

def extract_text_with_ocr(pdf_path):
    """
    STEP 1

    PDF
      ↓
    PDF Pages
      ↓
    Images
      ↓
    Tesseract OCR
      ↓
    Page-wise Text
    """

    print("\n" + "=" * 70)
    print("STEP 1 - OCR")
    print("=" * 70)

    # Convert PDF pages into images
    images = convert_from_path(
        pdf_path,
        dpi=OCR_DPI
    )

    pages = []

    # Process every page
    for page_no, image in enumerate(
        images,
        start=1
    ):

        print(
            f"OCR processing page {page_no}..."
        )

        # Extract text using Tesseract
        text = pytesseract.image_to_string(
            image,
            config="--psm 6"
        )

        text = text.strip()

        # Store page number and text
        pages.append({
            "page_no": page_no,
            "text": text
        })

        print(
            f"Page {page_no}: "
            f"{len(text)} characters"
        )

    print(
        f"\nTotal pages: {len(pages)}"
    )

    return pages


# ====================================================================
# 8. STEP 2 - RECURSIVE CHARACTER CHUNKING
# ====================================================================

def create_chunks(pages):
    """
    STEP 2

    Page-wise Text
        ↓
    RecursiveCharacterTextSplitter
        ↓
    Document Chunks
    """

    print("\n" + "=" * 70)
    print("STEP 2 - RECURSIVE CHARACTER CHUNKING")
    print("=" * 70)

    # Create recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(
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

    # Process each page
    for page in pages:

        page_no = page["page_no"]
        page_text = page["text"]

        # Skip empty pages
        if not page_text:
            continue

        # Split page text
        page_chunks = text_splitter.split_text(
            page_text
        )

        # Create metadata
        for chunk_no, chunk_text in enumerate(
            page_chunks,
            start=1
        ):

            # Create unique chunk ID
            chunk_id = (
                f"page_{page_no}_chunk_{chunk_no}"
            )

            chunks.append({
                "id": chunk_id,

                "text": chunk_text,

                "metadata": {
                    "file_name": os.path.basename(
                        PDF_PATH
                    ),
                    "page_no": page_no,
                    "chunk_no": chunk_no
                }
            })

    print(
        f"Total chunks created: {len(chunks)}"
    )

    # Display chunks
    for chunk in chunks:

        print("\n" + "-" * 70)

        print(
            f"Chunk ID: {chunk['id']}"
        )

        print(
            f"Page: "
            f"{chunk['metadata']['page_no']}"
        )

        print(
            f"Chunk Number: "
            f"{chunk['metadata']['chunk_no']}"
        )

        print(
            f"Text: "
            f"{chunk['text'][:300]}"
        )

    return chunks


# ====================================================================
# 9. STEP 3 - LOAD SENTENCETRANSFORMER
# ====================================================================

def load_embedding_model():
    """
    STEP 3

    Document Chunks
        ↓
    SentenceTransformer
        ↓
    Embeddings
    """

    print("\n" + "=" * 70)
    print("STEP 3 - SENTENCETRANSFORMER")
    print("=" * 70)

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    # Load embedding model
    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )

    return model


# ====================================================================
# 10. STEP 4 - CHROMADB
# ====================================================================

def create_chroma_collection():
    """
    STEP 4

    Embeddings
        ↓
    Persistent ChromaDB
        ↓
    Collection
    """

    print("\n" + "=" * 70)
    print("STEP 4 - CHROMADB")
    print("=" * 70)

    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Existing chunks: "
        f"{collection.count()}"
    )

    return collection


# ====================================================================
# 11. STEP 4 - CREATE DOCUMENT EMBEDDINGS
# ====================================================================

def create_embeddings(
    embedding_model,
    chunks
):
    """
    STEP 4

    Document Chunks
        ↓
    SentenceTransformer
        ↓
    Normalized Embeddings
    """

    print("\n" + "=" * 70)
    print("STEP 4 - CREATE DOCUMENT EMBEDDINGS")
    print("=" * 70)

    # Extract chunk text
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Creating embeddings for "
        f"{len(texts)} chunks..."
    )

    # Create normalized embeddings
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    print(
        f"Embedding dimension: "
        f"{len(embeddings[0])}"
    )

    return embeddings


# ====================================================================
# 12. STEP 4 - CHROMADB UPSERT
# ====================================================================

def ingest_chunks(
    collection,
    chunks,
    embeddings
):
    """
    STEP 4

    Chunks
      +
    Embeddings
      +
    Metadata
      ↓
    ChromaDB
    """

    print("\n" + "=" * 70)
    print("STEP 4 - CHROMADB UPSERT")
    print("=" * 70)

    # Prepare IDs
    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    # Prepare documents
    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    # Prepare metadata
    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    # Insert or update chunks
    collection.upsert(
        ids=ids,
        documents=documents,
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
# 13. STEP 5 - USER QUESTION
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

    # Ask question
    question = input(
        "\nEnter your question: "
    ).strip()

    return question


# ====================================================================
# 14. STEP 6 - QUERY EMBEDDING
# ====================================================================

def create_query_embedding(
    embedding_model,
    question
):
    """
    STEP 6

    User Question
        ↓
    SentenceTransformer
        ↓
    Query Embedding
    """

    print("\n" + "=" * 70)
    print("STEP 6 - QUERY EMBEDDING")
    print("=" * 70)

    # Create normalized query embedding
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    return query_embedding


# ====================================================================
# 15. STEP 7 - DENSE RETRIEVAL TOP 10
# ====================================================================

def dense_retrieval(
    collection,
    embedding_model,
    question
):
    """
    STEP 7

    Query Embedding
        ↓
    ChromaDB
        ↓
    Dense Retrieval
        ↓
    TOP 10
    """

    print("\n" + "=" * 70)
    print("STEP 7 - DENSE RETRIEVAL TOP 10")
    print("=" * 70)

    print(
        f"Question: {question}"
    )

    # Create query embedding
    query_embedding = create_query_embedding(
        embedding_model,
        question
    )

    # Search ChromaDB
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

    # Extract documents
    documents = results.get(
        "documents",
        [[]]
    )[0]

    # Extract metadata
    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    # Extract distances
    distances = results.get(
        "distances",
        [[]]
    )[0]

    retrieved_chunks = []

    # Convert results
    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        # Convert distance to relevance score
        retrieval_score = (
            1 / (1 + distance)
        )

        # Read page number
        page_no = int(
            metadata["page_no"]
        )

        # Read chunk number
        chunk_no = int(
            metadata["chunk_no"]
        )

        # Create chunk ID
        chunk_id = (
            f"page_{page_no}_chunk_{chunk_no}"
        )

        retrieved_chunks.append({
            "id": chunk_id,

            "text": document,

            "page_no": page_no,

            "chunk_no": chunk_no,

            "distance": float(
                distance
            ),

            "retrieval_score": float(
                retrieval_score
            )
        })

    print(
        f"Retrieved chunks: "
        f"{len(retrieved_chunks)}"
    )

    # Display dense retrieval results
    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"Dense Rank: {rank}"
        )

        print(
            f"Chunk ID: {chunk['id']}"
        )

        print(
            f"Page: {chunk['page_no']}"
        )

        print(
            f"Chunk Number: "
            f"{chunk['chunk_no']}"
        )

        print(
            f"Distance: "
            f"{chunk['distance']:.6f}"
        )

        print(
            f"Retrieval Score: "
            f"{chunk['retrieval_score']:.6f}"
        )

        print(
            f"Text: "
            f"{chunk['text'][:300]}"
        )

    return retrieved_chunks


# ====================================================================
# 16. STEP 8 - LOAD CROSSENCODER
# ====================================================================

def load_reranker():
    """
    STEP 8

    Question + Document
        ↓
    CrossEncoder
    """

    print("\n" + "=" * 70)
    print("STEP 8 - LOAD CROSSENCODER")
    print("=" * 70)

    print(
        f"Loading reranker: "
        f"{RERANKER_MODEL}"
    )

    # Load CrossEncoder
    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print(
        "CrossEncoder loaded."
    )

    return reranker


# ====================================================================
# 17. STEP 8 - CROSSENCODER RERANKING
# ====================================================================

def rerank_chunks(
    reranker,
    question,
    retrieved_chunks
):
    """
    STEP 8

    TOP 10 Retrieved Chunks
        ↓
    CrossEncoder
        ↓
    Relevance Scores
        ↓
    Sort
    """

    print("\n" + "=" * 70)
    print("STEP 8 - CROSSENCODER RERANKING")
    print("=" * 70)

    # No chunks
    if not retrieved_chunks:

        return []

    # Create question-document pairs
    pairs = []

    for chunk in retrieved_chunks:

        pairs.append([
            question,
            chunk["text"]
        ])

    # Calculate CrossEncoder scores
    scores = reranker.predict(
        pairs
    )

    reranked_chunks = []

    # Attach scores
    for chunk, score in zip(
        retrieved_chunks,
        scores
    ):

        reranked_chunk = dict(
            chunk
        )

        reranked_chunk[
            "rerank_score"
        ] = float(
            score
        )

        reranked_chunks.append(
            reranked_chunk
        )

    # Sort highest score first
    reranked_chunks.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # Keep TOP 5
    reranked_chunks = reranked_chunks[
        :RERANK_TOP_K
    ]

    print(
        f"Final reranked chunks: "
        f"{len(reranked_chunks)}"
    )

    # Display reranked results
    for rank, chunk in enumerate(
        reranked_chunks,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"Rerank: {rank}"
        )

        print(
            f"Chunk ID: {chunk['id']}"
        )

        print(
            f"Page: {chunk['page_no']}"
        )

        print(
            f"Chunk Number: "
            f"{chunk['chunk_no']}"
        )

        print(
            f"CrossEncoder Score: "
            f"{chunk['rerank_score']:.6f}"
        )

        print(
            f"Text: "
            f"{chunk['text'][:400]}"
        )

    return reranked_chunks


# ====================================================================
# 18. STEP 9 - TOP 5 RELEVANT CHUNKS
# ====================================================================

def select_top_chunks(
    reranked_chunks
):
    """
    STEP 9

    Reranked Chunks
        ↓
    TOP 5
    """

    print("\n" + "=" * 70)
    print("STEP 9 - TOP 5 RELEVANT CHUNKS")
    print("=" * 70)

    # Select top 5
    top_chunks = reranked_chunks[
        :RERANK_TOP_K
    ]

    print(
        f"Selected chunks: "
        f"{len(top_chunks)}"
    )

    return top_chunks


# ====================================================================
# 19. STEP 10 - BUILD CONTEXT
# ====================================================================

def build_context(
    reranked_chunks
):
    """
    STEP 10

    TOP 5 Chunks
        ↓
    Context
        ↓
    LLM
    """

    print("\n" + "=" * 70)
    print("STEP 10 - BUILD CONTEXT")
    print("=" * 70)

    context_parts = []

    # Build context from every chunk
    for index, chunk in enumerate(reranked_chunks, start=1):

        context = f"""
==================================================
RETRIEVED SOURCE {index}
==================================================

Page Number: {chunk['page_no']}

Chunk Number: {chunk['chunk_no']}

Chunk ID: {chunk['id']}

CrossEncoder Relevance Score:
{chunk['rerank_score']:.6f}

Chunk Text:
{chunk['text']}
"""

        context_parts.append(
            context
        )

    # Join chunks
    final_context = "\n".join(
        context_parts
    )

    print(
        f"Context characters: "
        f"{len(final_context)}"
    )

    return final_context


# ====================================================================
# 20. STEP 11 - SYSTEM PROMPT
# ====================================================================

def build_system_prompt():
    """
    STEP 11

    System Prompt
        +
    JSON Schema
        ↓
    LLM Instructions
    """

    print("\n" + "=" * 70)
    print("STEP 11 - SYSTEM PROMPT")
    print("=" * 70)

    # Convert schema to JSON text
    schema_text = json.dumps(
        ARRAY_SCHEMA,
        indent=2
    )

    system_prompt = f"""
You are an HR Policy question-answering assistant.

Answer the user's question using ONLY the retrieved
context from HR_Policy.pdf.

STRICT RULES:

1. Do not use outside knowledge.

2. Do not invent information.

3. If the answer is not available in the retrieved
   context, answer:

   "I don't know based on HR_Policy.pdf."

4. Return ONLY valid JSON.

5. Return a JSON ARRAY.

6. Every record must contain:

   Answer
   Source
   Confidence

7. Answer must contain the actual answer to the
   user's question.

8. Source must contain supporting chunks.

9. Do not invent Page Number.

10. Do not invent Chunk Number.

11. Do not invent Chunk ID.

12. Do not invent Chunk Text.

13. Confidence must be between 0 and 1.

14. Relevance Score must use the provided
    CrossEncoder score.

15. If one answer is sufficient, return one record.

16. If multiple distinct answers are required,
    return multiple records.

EXPECTED JSON SCHEMA:

{schema_text}
"""

    return system_prompt


# ====================================================================
# 21. STEP 12 - USER PROMPT
# ====================================================================

def build_user_prompt(
    question,
    context
):
    """
    STEP 12

    User Question
        +
    Retrieved Context
        ↓
    User Prompt
    """

    print("\n" + "=" * 70)
    print("STEP 12 - USER PROMPT")
    print("=" * 70)

    # Build user prompt
    user_prompt = f"""
USER QUESTION:

{question}


RETRIEVED HR POLICY CONTEXT:

{context}


TASK:

Answer the user's question using ONLY the retrieved
context.

Return ONLY a JSON ARRAY.
"""

    return user_prompt


# ====================================================================
# 22. STEP 13 - MISTRAL LLM
# ====================================================================

def generate_answer(question, context):
    """
    STEP 13

    System Prompt
          +
    User Prompt
          ↓
    Ollama
          ↓
    Mistral
          ↓
    JSON Response
    """

    print("\n" + "=" * 70)
    print("STEP 13 - MISTRAL LLM")
    print("=" * 70)

    # Create system prompt
    system_prompt = build_system_prompt()

    # Create user prompt
    user_prompt = build_user_prompt(
        question,
        context
    )

    print(
        f"Ollama model: {LLM_MODEL}"
    )

    # Call Ollama
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

    # Extract response text
    content = response[
        "message"
    ][
        "content"
    ]

    return content


# ====================================================================
# 23. STEP 14 - JSON PARSING
# ====================================================================

def parse_json_response(
    response_text
):
    """
    STEP 14

    Ollama JSON
        ↓
    Parse JSON
        ↓
    JSON Array
    """

    print("\n" + "=" * 70)
    print("STEP 14 - JSON PARSING")
    print("=" * 70)

    # Remove whitespace
    cleaned = response_text.strip()

    # Remove ```json
    if cleaned.startswith(
        "```json"
    ):

        cleaned = cleaned[
            len("```json"):
        ].strip()

    # Remove ```
    elif cleaned.startswith(
        "```"
    ):

        cleaned = cleaned[
            len("```"):
        ].strip()

    # Remove ending ```
    if cleaned.endswith(
        "```"
    ):

        cleaned = cleaned[
            :-3
        ].strip()

    # Parse JSON
    data = json.loads(
        cleaned
    )

    # --------------------------------------------------------------
    # FORMAT:
    #
    # {
    #     "type": "array",
    #     "items": [...]
    # }
    # --------------------------------------------------------------

    if isinstance(
        data,
        dict
    ):

        if (
            data.get("type") == "array"
            and "items" in data
        ):

            data = data[
                "items"
            ]

    # --------------------------------------------------------------
    # FORMAT:
    #
    # {
    #     "Answer": "...",
    #     ...
    # }
    # --------------------------------------------------------------

    if isinstance(
        data,
        dict
    ):

        if "Answer" in data:

            data = [
                data
            ]

    # --------------------------------------------------------------
    # FORMAT:
    #
    # [
    #     {...}
    # ]
    # --------------------------------------------------------------

    if isinstance(
        data,
        list
    ):

        print(
            f"Parsed records: {len(data)}"
        )

        return data

    # Invalid JSON structure
    raise ValueError(
        "LLM response is not a valid answer array."
    )


# ====================================================================
# 24. STEP 15 - SOURCE VALIDATION
# ====================================================================

def validate_result(
    data,
    reranked_chunks
):
    """
    STEP 15

    LLM Answer
        +
    Retrieved Chunks
        ↓
    Source Validation
        ↓
    Trusted JSON
    """

    print("\n" + "=" * 70)
    print("STEP 15 - SOURCE VALIDATION")
    print("=" * 70)

    # Create trusted lookup
    source_lookup = {}

    for chunk in reranked_chunks:

        source_lookup[
            chunk["id"]
        ] = chunk

    validated_records = []

    # Process every record
    for record in data:

        # Skip invalid records
        if not isinstance(
            record,
            dict
        ):

            continue

        # ----------------------------------------------------------
        # GET ANSWER
        # ----------------------------------------------------------

        answer = record.get(
            "Answer",
            ""
        )

        if answer is None:

            answer = ""

        answer = str(
            answer
        ).strip()

        # ----------------------------------------------------------
        # GET CONFIDENCE
        # ----------------------------------------------------------

        confidence = record.get(
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

        # Keep confidence between 0 and 1
        confidence = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )

        # ----------------------------------------------------------
        # GET LLM SOURCES
        # ----------------------------------------------------------

        llm_sources = record.get(
            "Source",
            []
        )

        trusted_sources = []

        if isinstance(
            llm_sources,
            list
        ):

            for source in llm_sources:

                if not isinstance(
                    source,
                    dict
                ):

                    continue

                # Get chunk ID
                chunk_id = source.get(
                    "Chunk ID"
                )

                # Check source against retrieved data
                if chunk_id in source_lookup:

                    chunk = source_lookup[
                        chunk_id
                    ]

                    trusted_sources.append({
                        "Page Number": (
                            chunk["page_no"]
                        ),

                        "Chunk Number": (
                            chunk["chunk_no"]
                        ),

                        "Chunk ID": (
                            chunk["id"]
                        ),

                        "Chunk Text": (
                            chunk["text"]
                        ),

                        "Relevance Score": (
                            chunk["rerank_score"]
                        )
                    })

        # ----------------------------------------------------------
        # FALLBACK SOURCE
        # ----------------------------------------------------------

        if not trusted_sources:

            if reranked_chunks:

                chunk = reranked_chunks[0]

                trusted_sources.append({
                    "Page Number": (
                        chunk["page_no"]
                    ),

                    "Chunk Number": (
                        chunk["chunk_no"]
                    ),

                    "Chunk ID": (
                        chunk["id"]
                    ),

                    "Chunk Text": (
                        chunk["text"]
                    ),

                    "Relevance Score": (
                        chunk["rerank_score"]
                    )
                })

        # ----------------------------------------------------------
        # CREATE FINAL RECORD
        # ----------------------------------------------------------

        validated_records.append({
            "Answer": answer,

            "Source": trusted_sources,

            "Confidence": confidence
        })

    # --------------------------------------------------------------
    # FALLBACK IF NO RECORD
    # --------------------------------------------------------------

    if not validated_records:

        fallback_sources = []

        if reranked_chunks:

            chunk = reranked_chunks[0]

            fallback_sources.append({
                "Page Number": (
                    chunk["page_no"]
                ),

                "Chunk Number": (
                    chunk["chunk_no"]
                ),

                "Chunk ID": (
                    chunk["id"]
                ),

                "Chunk Text": (
                    chunk["text"]
                ),

                "Relevance Score": (
                    chunk["rerank_score"]
                )
            })

        validated_records.append({
            "Answer": (
                "I don't know based on HR_Policy.pdf."
            ),

            "Source": fallback_sources,

            "Confidence": 0.0
        })

    return validated_records


# ====================================================================
# 25. STEP 16 - FINAL ANSWER
# ====================================================================

def print_final_result(
    result
):
    """
    STEP 16

    Validated JSON
        ↓
    Final Answer
    """

    print("\n" + "=" * 70)
    print("STEP 16 - FINAL ANSWER")
    print("=" * 70)

    # Print final JSON
    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )


# ====================================================================
# 26. STEP 17 - PROCESS ONE QUESTION
# ====================================================================

def process_question(
    question,
    collection,
    embedding_model,
    reranker
):
    """
    STEP 17

    Complete question-processing pipeline:

    Question
       ↓
    Query Embedding
       ↓
    Dense Retrieval TOP 10
       ↓
    CrossEncoder
       ↓
    TOP 5
       ↓
    Context
       ↓
    Mistral
       ↓
    JSON
       ↓
    Validation
    """

    # ==============================================================
    # STEP 6 - QUERY EMBEDDING
    # ==============================================================

    # Query embedding happens inside dense retrieval.

    # ==============================================================
    # STEP 7 - DENSE RETRIEVAL
    # ==============================================================

    retrieved_chunks = dense_retrieval(
        collection,
        embedding_model,
        question
    )

    # ==============================================================
    # STEP 8 - RERANKING
    # ==============================================================

    reranked_chunks = rerank_chunks(
        reranker,
        question,
        retrieved_chunks
    )

    # ==============================================================
    # STEP 9 - TOP 5
    # ==============================================================

    top_chunks = select_top_chunks(
        reranked_chunks
    )

    # No result
    if not top_chunks:

        return [{
            "Answer": (
                "I don't know based on HR_Policy.pdf."
            ),

            "Source": [],

            "Confidence": 0.0
        }]

    # ==============================================================
    # STEP 10 - CONTEXT
    # ==============================================================

    context = build_context(
        top_chunks
    )

    # ==============================================================
    # STEP 13 - LLM
    # ==============================================================

    raw_response = generate_answer(
        question,
        context
    )

    # Print raw response
    print("\n" + "=" * 70)
    print("RAW LLM RESPONSE")
    print("=" * 70)

    print(
        raw_response
    )

    # ==============================================================
    # STEP 14 - JSON PARSING
    # ==============================================================

    try:

        parsed_response = parse_json_response(
            raw_response
        )

    except Exception as error:

        print("\n" + "=" * 70)
        print("JSON PARSING ERROR")
        print("=" * 70)

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        parsed_response = [{
            "Answer": (
                "I don't know based on HR_Policy.pdf."
            ),

            "Source": [],

            "Confidence": 0.0
        }]

    # ==============================================================
    # STEP 15 - SOURCE VALIDATION
    # ==============================================================

    validated_result = validate_result(
        parsed_response,
        top_chunks
    )

    return validated_result


# ====================================================================
# 27. STEP 17 - MAIN PIPELINE
# ====================================================================

def main():

    print("\n")
    print("=" * 70)
    print("             ADVANCED RAG")
    print("              HR POLICY")
    print("=" * 70)

    print(
        f"\nPDF: {PDF_PATH}"
    )

    # ==============================================================
    # CHECK PDF
    # ==============================================================

    if not os.path.exists(
        PDF_PATH
    ):

        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    # ==============================================================
    # STEP 1 - OCR
    # ==============================================================

    pages = extract_text_with_ocr(
        PDF_PATH
    )

    # ==============================================================
    # STEP 2 - CHUNKING
    # ==============================================================

    chunks = create_chunks(
        pages
    )

    # ==============================================================
    # STEP 3 - EMBEDDING MODEL
    # ==============================================================

    embedding_model = load_embedding_model()

    # ==============================================================
    # STEP 4 - CHROMADB
    # ==============================================================

    collection = create_chroma_collection()

    # ==============================================================
    # STEP 4 - DOCUMENT EMBEDDINGS
    # ==============================================================

    embeddings = create_embeddings(
        embedding_model,
        chunks
    )

    # ==============================================================
    # STEP 4 - CHROMADB UPSERT
    # ==============================================================

    ingest_chunks(
        collection,
        chunks,
        embeddings
    )

    # ==============================================================
    # STEP 8 - LOAD CROSSENCODER
    # ==============================================================

    reranker = load_reranker()

    # ==============================================================
    # STEP 17 - WHILE LOOP
    # ==============================================================

    while True:

        print("\n")
        print("=" * 70)
        print("STEP 17 - ASK HR POLICY QUESTION")
        print("=" * 70)

        # ==========================================================
        # STEP 5 - USER QUESTION
        # ==========================================================

        question = get_user_question()

        # ==========================================================
        # EXIT
        # ==========================================================

        if question.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print(
                "\nExiting Advanced RAG..."
            )

            break

        # ==========================================================
        # EMPTY QUESTION
        # ==========================================================

        if not question:

            print(
                "\nPlease enter a question."
            )

            continue

        # ==========================================================
        # PROCESS QUESTION
        # ==========================================================

        try:

            result = process_question(
                question,
                collection,
                embedding_model,
                reranker
            )

            # ======================================================
            # STEP 16 - FINAL ANSWER
            # ======================================================

            print_final_result(
                result
            )

        except Exception as error:

            print("\n" + "=" * 70)
            print("ERROR")
            print("=" * 70)

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                "\nPlease try another question."
            )

    # ==============================================================
    # END
    # ==============================================================

    print("\n")
    print("=" * 70)
    print("ADVANCED RAG STOPPED")
    print("=" * 70)


# ====================================================================
# 28. ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    main()