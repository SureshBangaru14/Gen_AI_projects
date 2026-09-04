import chromadb

from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 01. WHAT IS A VECTOR DATABASE
# ============================================================

# Original document
document = """
Employees are eligible for annual leave after completing
12 months of continuous service. Employees must submit
their leave request through the HR portal before taking
planned leave. Managers are responsible for approving
employee leave requests.
"""

print("*"*100)
print(document)


# ============================================================
# 02. CHUNKING
# ============================================================

# A large document is divided into smaller pieces.
#
# Here we use:
#
# Recursive Character Text Splitting
#
# chunk_size   = maximum approximate character size
# chunk_overlap = characters repeated between neighboring chunks
#
# Example:
#
# Document
#    ↓
# Chunk 1
# Chunk 2
# Chunk 3

text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20, separators=["\n\n", "\n", ". ", " ", ""])


chunks = text_splitter.split_text(document)

print("=" * 70)


for i, chunk in enumerate(chunks, start=1):

    print(f"\nChunk {i}: ")
    print(chunk)


# ============================================================
# 03. EMBEDDING MODEL
# ============================================================

# The embedding model converts text into numbers.
#
# Text
#   ↓
# Embedding Model
#   ↓
# Vector
#
#
# Open-source embedding model:
#
# BAAI/bge-small-en-v1.5
#
# This model generates:
#
# 384-dimensional vectors
#
# Example:
#
# "Employees are eligible for annual leave."
#
#              ↓
#
# [0.12, -0.34, 0.56, ...]
#
#              ↑
#
#        384 numbers


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("=" * 70)

print("Embedding Model:",EMBEDDING_MODEL_NAME)


# ============================================================
# 04. VECTOR GENERATION
# ============================================================

# Convert every chunk into an embedding vector.
#
# Chunks
#   ↓
# Embedding Model
#   ↓
# Vectors
#
# Example:
#
# Chunk 1 → Vector 1
# Chunk 2 → Vector 2
# Chunk 3 → Vector 3


chunk_embeddings = model.encode(chunks, normalize_embeddings=True)

print("=" * 70)

for i, vector in enumerate(chunk_embeddings, start=1):

    print(f"Chunk {i} → " f"{len(vector)} dimensions")

    print("First 10 values:",vector[:10])


# ============================================================
# 05. VECTOR DIMENSIONS
# ============================================================

# A vector dimension means:
#
# HOW MANY NUMBERS EXIST IN ONE VECTOR?
#
# Example:
#
# [0.12, 0.45, -0.21, 0.67]
#
# This has:
#
# 4 dimensions
#
#
# Our embedding model:
#
# BAAI/bge-small-en-v1.5
#
# generates:
#
# 384 dimensions
#
#
# IMPORTANT:
#
# Every vector stored in the same collection must have
# the same dimensionality.


VECTOR_DIMENSIONS = len(chunk_embeddings[0])

print("=" * 70)

print("Vector Dimensions:", VECTOR_DIMENSIONS)


# ============================================================
# 06. QUERY
# ============================================================

# User asks a question.
#
# Query:
#
# "Who is eligible for annual leave?"
#
# The query must also be converted into a vector.
#
# Query
#   ↓
# Same Embedding Model
#   ↓
# Query Vector
#
# IMPORTANT:
#
# The SAME embedding model should normally be used for:
#
# Documents → Embeddings
# Query     → Embedding
#
# so that they exist in the same vector space.


query = "Who is eligible for annual leave?"


query_embedding = model.encode(query, normalize_embeddings=True)

print("=" * 70)

print("\nQuery Vector Dimensions:", len(query_embedding))


# ============================================================
# 07. CREATE PERSISTENT CHROMADB CLIENT
# ============================================================

# ChromaDB is our Vector Database.
#
# PersistentClient means:
#
# Data is stored on disk.
#
# Database location:
#
# ./chroma_db
#
#
# Flow:
#
# Embeddings
#     ↓
# ChromaDB
#     ↓
# Persistent Storage


client = chromadb.PersistentClient(path="./chroma_db")

print("=" * 70)

print("Persistent ChromaDB path: ./chroma_db")


# ============================================================
# 08. COLLECTION
# ============================================================

# A collection is a logical container for related
# vector records.
#
# Example:
#
# ChromaDB
#    │
#    ├── hr_documents
#    ├── bol_documents
#    ├── resume_documents
#    └── invoice_documents
#
#
# Our collection:
#
# hr_documents


COLLECTION_NAME = "hr_documents"


# collection = client.get_or_create_collection(name=COLLECTION_NAME)

# ============================================================
# CREATE / GET CHROMADB COLLECTION
# ============================================================

collection = client.get_or_create_collection(

    # --------------------------------------------------------
    # name
    # --------------------------------------------------------
    #
    # Name of the collection.
    #
    # A collection is a logical container in ChromaDB used
    # to store related:
    #
    #   - Documents / Chunks
    #   - Embeddings / Vectors
    #   - IDs
    #   - Metadata
    #
    # Example collections:
    #
    #   hr_documents
    #   bol_documents
    #   resume_documents
    #   invoice_documents
    #
    # In this example:
    #
    #   hr_documents
    #
    # means this collection contains HR documents.
    #
    name=COLLECTION_NAME,


    # --------------------------------------------------------
    # configuration
    # --------------------------------------------------------
    #
    # Configuration defines how ChromaDB should handle the
    # vector index/search for this collection.
    #
    configuration={
        

        # ----------------------------------------------------
        # hnsw
        # ----------------------------------------------------
        #
        # HNSW stands for:
        #
        # Hierarchical Navigable Small World
        #
        # HNSW is a vector indexing algorithm.
        #
        # Its purpose is to make vector similarity search
        # faster and more efficient.
        #
        # Concept:
        #
        # Query Vector
        #      ↓
        #   HNSW Index
        #      ↓
        # Find nearby vectors
        #      ↓
        # Top-K Results
        #
        "hnsw": {


            # ------------------------------------------------
            # space
            # ------------------------------------------------
            #
            # "space" defines how the distance between two
            # vectors is calculated.
            #
            # "l2" means:
            #
            # L2 / Euclidean Distance
            #
            # Common options:
            #
            #   "l2"      → Euclidean distance
            #   "cosine"  → Cosine distance
            #   "ip"      → Inner Product
            #
            # In this example:
            #
            # HNSW  = Indexing algorithm
            # L2    = Distance metric
            #
            "space": "l2"
        }
    }
)

print("=" * 70)

print("Collection:", COLLECTION_NAME)


# ============================================================
# 09. INDEX CONFIGURATION
# ============================================================

# This section explains the important concepts involved
# when vectors are stored and searched.
# 
#
# ------------------------------------------------------------
# EMBEDDING MODEL
# ------------------------------------------------------------
#
# Model:
#
# BAAI/bge-small-en-v1.5
#
# Purpose:
#
# Converts text → numerical vector
#
#
# ------------------------------------------------------------
# VECTOR DIMENSIONS
# ------------------------------------------------------------
#
# Dimension:
#
# 384
#
# Meaning:
#
# Every vector contains 384 numerical values.
#
#
# Example:
#
# Vector =
#
# [0.12, -0.34, 0.56, ...]
#
#        ↑
#    384 values
#
#
# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------
#
# We used:
#
# normalize_embeddings=True
#
# This normalizes the generated embeddings.
#
# This is useful when using cosine-style similarity.
#
#
# ------------------------------------------------------------
# DISTANCE / SIMILARITY
# ------------------------------------------------------------
#
# Vector search needs a way to determine how close two
# vectors are.
#
# Common concepts:
#
# 1. Cosine similarity
# 2. Euclidean distance
# 3. Dot product
#
#
# Cosine similarity:
#
# Measures the angle/directional similarity between vectors.
#
#
# Example:
#
# Query Vector
#      ↓
#      ●
#     / \
#    /   \
#   ●     ●
#   ↑     ↑
# Similar  Less Similar
#
#
# IMPORTANT CHROMADB POINT:
#
# ChromaDB manages the underlying vector indexing and
# similarity-search infrastructure for the collection.
#
# In this basic example, we explicitly provide the
# embeddings and let ChromaDB handle the vector storage
# and search.
#
#
# ------------------------------------------------------------
# COLLECTION vs INDEX
# ------------------------------------------------------------
#
# COLLECTION:
#
# Where the vector records logically belong.
#
# INDEX:
#
# The search structure used internally to make vector
# retrieval efficient.
#
#
# Simple concept:
#
# Collection
#     ↓
# Stores vector records
#
# Index
#     ↓
# Helps search vectors efficiently

print("09. INDEX CONFIGURATION")
print("=" * 70)

print("Embedding Model :", EMBEDDING_MODEL_NAME)

print("Vector Dimensions:", VECTOR_DIMENSIONS)

print("Embeddings normalized: True")

print("Search concept: Vector similarity")

print("Index management: Handled by ChromaDB")


# ============================================================
# 10. DOCUMENT IDs
# ============================================================

# Every chunk needs a unique ID.
#
# Example:
#
# hr_policy_chunk_001
# hr_policy_chunk_002
# hr_policy_chunk_003


ids = []

for i in range(len(chunks)):

    ids.append(f"hr_policy_chunk_{i + 1:03d}")

print("10. DOCUMENT IDs")
print("=" * 70)

for document_id in ids:

    print(document_id)


# ============================================================
# 11. METADATA
# ============================================================

# Metadata gives additional information about a chunk.
#
# Example:
#
# file_name
# page_number
# document_id
# chunk_number
# document_type
#
#
# Metadata is NOT the embedding.
#
# It is additional information associated with the vector.


metadatas = []


for i in range(len(chunks)):

    metadata = { "document_id": "hr_policy_001",
                
                 "file_name": "hr_policy.pdf",
                 
                 "page_number": 1,
                 
                 "chunk_number": i + 1,
                 
                 "document_type": "HR Policy" }

    metadatas.append(metadata)


# ============================================================
# 12. STORE VECTOR + DOCUMENT + METADATA + ID
# ============================================================

# Each record contains:
#
# ID
#    ↓
# Vector
#    ↓
# Document / Chunk
#    ↓
# Metadata
#
#
# Conceptual record:
#
# {
#     "id": "hr_policy_chunk_001",
#
#     "vector": [
#         0.12,
#         -0.34,
#         ...
#     ],
#
#     "document":
#         "Employees are eligible...",
#
#     "metadata": {
#         "file_name": "hr_policy.pdf",
#         "page_number": 1
#     }
# }


collection.add( ids=ids,
               
                embeddings=[vector.tolist() for vector in chunk_embeddings],
                
                documents=chunks,
                
                metadatas=metadatas )

print("12. DATA STORED")
print("=" * 70)

print("Vectors + Documents + Metadata + IDs " "stored successfully.")


# ============================================================
# 13. COLLECTION COUNT
# ============================================================

count = collection.count()

print("13. COLLECTION COUNT")
print("=" * 70)

print("Total records:", count)


# ============================================================
# 14. SEMANTIC SEARCH
# ============================================================

# User query:
#
# "Who is eligible for annual leave?"
#
# Query
#   ↓
# Query Embedding
#   ↓
# Query Vector
#   ↓
# ChromaDB
#   ↓
# Vector Similarity Search
#   ↓
# Top-K Results = 3


results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=3, where={"document_type": "HR Policy"})


# ============================================================
# 15. SEARCH RESULTS
# ============================================================
print("15. SEMANTIC SEARCH RESULTS")
print("=" * 70)
print("====results====: ",results)

for i in range(len(results["documents"][0])):

    print(f"\nResult {i + 1}")

    print("ID:", results["ids"][0][i])

    print("Document:",results["documents"][0][i])

    print("Metadata:", results["metadatas"][0][i])

    print("Distance:",results["distances"][0][i])
