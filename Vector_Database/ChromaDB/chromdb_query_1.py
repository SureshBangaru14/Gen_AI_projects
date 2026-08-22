"""
============================================================
                 CHROMADB COMPLETE FLOW
============================================================


YOUR DOCUMENTS
      |
      ▼
┌─────────────────────┐
│   Embedding Model   │
└─────────────────────┘
      |
      ▼
   Vectors
      |
      ▼
  ┌───────────┐
  │ ChromaDB  │
  └───────────┘
      |
      |
      |                         USER QUERY
      |                              |
      |                              ▼
      |                    ┌─────────────────────┐
      |                    │   Embedding Model   │
      |                    └─────────────────────┘
      |                              |
      |                              ▼
      |                         Query Vector
      |                              |
      └──────────────┬───────────────┘
                     |
                     ▼
             Similarity Search
                     |
                     ▼
              Top-K Documents


============================================================
                 COLLECTION.ADD() FLOW
============================================================

                    collection.add()
                           |
             ┌─────────────┴─────────────┐
             ▼                           ▼
        Documents                       IDs
             |                           |
             ▼                           ▼
      Embedding Model             Unique identifiers
             |
             ▼
          Vectors
             |
             └──────────────┐
                            ▼
                         ChromaDB
                            |
                 ┌──────────┴──────────┐
                 ▼                     ▼
             Documents             Embeddings


============================================================
                 CHROMADB DATA
============================================================

ID       Document                     Vector
------------------------------------------------------------
car1     Car runs on land             [...]
plane1   Plane flies in the sky       [...]
boat1    Boat travels on water        [...]
bus1     Bus is public transport...   [...]


============================================================
              QUERY SEARCH FLOW
============================================================

User Query
    |
    ▼
"Which vehicle travels on water?"
    |
    ▼
Embedding Model
    |
    ▼
Query Vector
    |
    ▼
Compare with stored vectors
    |
    ▼
Similarity / Distance
    |
    ▼
Rank results
    |
    ▼
Top-K Documents
    |
    ▼
"Boat travels on water"


============================================================
             MOST IMPORTANT CONCEPTS
============================================================

Client
    → Interface used to communicate with ChromaDB

Collection
    → Container that stores documents, IDs, embeddings,
      and metadata

documents
    → Original text data

ids
    → Unique identifier for every document

embedding
    → Numerical representation of text

vector
    → Array of numerical values representing text

embedding_function
    → Converts text into vectors

query_texts
    → User's search/query text

n_results
    → Number of matching documents to return

add()
    → Adds documents to a collection

get()
    → Retrieves stored data

query()
    → Performs similarity search

count()
    → Returns number of records in collection

distance
    → Indicates how close/far the query is from stored vectors

dimension
    → Number of values in an embedding vector
"""


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import chromadb

from chromadb.utils import embedding_functions


# ============================================================
# 1. CREATE CHROMADB CLIENT
# ============================================================

# chromadb.Client()
#
# Creates a ChromaDB client.
#
# Client is the main interface through which our Python
# application communicates with ChromaDB.
#
#
# Current example:
#
#     client = chromadb.Client()
#
# This is an in-memory client.
#
# For learning:
#     Good
#
# For production:
#     Usually use PersistentClient or Chroma server.
#
#
# Persistent example:
#
# client = chromadb.PersistentClient(
#     path="./chroma_db"
# )


client = chromadb.Client()


# ============================================================
# 2. CREATE EMBEDDING FUNCTION
# ============================================================

# An embedding function converts text into numerical vectors.
#
#
# Example:
#
# "Boat travels on water"
#              |
#              ▼
#       Embedding Model
#              |
#              ▼
# [0.12, -0.45, 0.78, ...]
#
#
# DefaultEmbeddingFunction()
#
# tells ChromaDB to use its default embedding function.
#
# We explicitly create it here so that we can see and reuse
# the SAME embedding function for:
#
#     1. Document embeddings
#     2. Query embeddings
#
#
# IMPORTANT:
#
# Document embedding and query embedding should be generated
# using the same embedding model/function.

embedding_function = (
    embedding_functions.DefaultEmbeddingFunction()
)


# ============================================================
# 3. CREATE COLLECTION
# ============================================================

# A Collection is a container for related data.
#
# Similar concept in SQL:
#
#     Database
#         |
#         └── Table
#
#
# ChromaDB:
#
#     ChromaDB
#         |
#         └── Collection
#
#
# Our collection:
#
#     vehicles
#
# will contain:
#
#     IDs
#     Documents
#     Embeddings
#     Metadata (if provided)


collection = client.create_collection(
    name="vehicles",
    embedding_function=embedding_function
)


# Display the embedding function being used.

print("=" * 70)
print("EMBEDDING FUNCTION")
print("=" * 70)

print(embedding_function)


# Display collection name.

print("\nCollection Created:")
print(collection.name)


# ============================================================
# 4. ADD DOCUMENTS
# ============================================================

# We are adding four documents.
#
# These are the ORIGINAL TEXT values.
#
#
# ChromaDB will use the embedding function to convert
# these documents into numerical vectors.
#
#
# Conceptually:
#
#
# "Car runs on land"
#         |
#         ▼
# Embedding Model
#         |
#         ▼
# [0.12, 0.45, -0.23, ...]
#
#
# "Boat travels on water"
#         |
#         ▼
# Embedding Model
#         |
#         ▼
# [0.91, -0.12, 0.44, ...]


collection.add(

    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------

    # Original text that we want to store.

    documents=[
        "Car runs on land",
        "Plane flies in the sky",
        "Boat travels on water",
        "Bus is public transport on road"
    ],


    # --------------------------------------------------------
    # IDS
    # --------------------------------------------------------

    # Every document must have a unique ID.
    #
    # ID allows us to identify a particular document.
    #
    #
    # Example:
    #
    # boat1
    #   |
    #   └── Boat travels on water

    ids=[
        "car1",
        "plane1",
        "boat1",
        "bus1"
    ]
)


# ============================================================
# 5. COUNT DOCUMENTS
# ============================================================

# count() tells us how many records are currently stored
# inside the collection.

print("\nDocuments Added:")
print(collection.count())


# Expected:

# Documents Added:
# 4


# ============================================================
# 6. GET STORED DOCUMENTS + EMBEDDINGS
# ============================================================

# collection.get()
#
# retrieves stored information from ChromaDB.
#
#
# include=["documents", "embeddings"]
#
# means:
#
#     Return original documents
#     Return their embedding vectors
#
#
# IDs are returned automatically as identifiers.

stored_data = collection.get(
    include=[
        "documents",
        "embeddings"
    ]
)


# ============================================================
# 7. DISPLAY DOCUMENT EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("STORED DOCUMENT EMBEDDINGS")
print("=" * 70)


for i in range(len(stored_data["documents"])):

    # Get document text.

    document = stored_data["documents"][i]


    # Get embedding vector for this document.

    embedding = stored_data["embeddings"][i]


    # Get document ID.

    document_id = stored_data["ids"][i]


    print("\n" + "-" * 70)

    print("ID:")
    print(document_id)

    print("\nDocument:")
    print(document)

    print("\nFirst 10 Embedding Values:")

    # Display only first 10 values.
    #
    # We don't print the complete vector because embeddings
    # can contain hundreds/thousands of numbers.

    print(embedding[:10])


    print("\nEmbedding Dimension:")

    # len(vector)
    #
    # tells us how many numerical values exist in the vector.

    print(len(embedding))


# ============================================================
# 8. CREATE USER QUERY
# ============================================================

# This is the question entered by the user.

query = "Which vehicle travels on water?"


# ============================================================
# 9. CREATE QUERY EMBEDDING
# ============================================================

# We use the SAME embedding function used for documents.
#
#
# Query:
#
# "Which vehicle travels on water?"
#               |
#               ▼
#       Embedding Function
#               |
#               ▼
#       Query Vector
#
#
# This vector will then be compared with the stored
# document vectors.

query_embedding = embedding_function([query])


# ============================================================
# 10. DISPLAY QUERY EMBEDDING
# ============================================================

print("\n" + "=" * 70)
print("QUERY EMBEDDING")
print("=" * 70)


print("\nQuery:")
print(query)


print("\nFirst 10 Query Embedding Values:")

# query_embedding is a list containing one embedding.
#
# query_embedding[0]
#       |
#       └── actual vector
#
# [:10]
#       |
#       └── first 10 values

print(query_embedding[0][:10])


print("\nQuery Embedding Dimension:")

# Number of dimensions in query vector.

print(len(query_embedding[0]))


# ============================================================
# 11. PERFORM SIMILARITY SEARCH
# ============================================================

# collection.query()
#
# performs semantic similarity search.
#
#
# The query text is:
#
#     "Which vehicle travels on water?"
#
#
# ChromaDB internally:
#
#
# Query Text
#     |
#     ▼
# Query Embedding
#     |
#     ▼
# Compare with stored document embeddings
#     |
#     ▼
# Calculate distances
#     |
#     ▼
# Rank documents
#     |
#     ▼
# Return Top-K results


results = collection.query(

    # --------------------------------------------------------
    # QUERY TEXT
    # --------------------------------------------------------

    # IMPORTANT:
    #
    # Use the variable "query" instead of writing the same
    # text again.

    query_texts=[
        query
    ],


    # --------------------------------------------------------
    # TOP-K RESULTS
    # --------------------------------------------------------

    # Return the top 2 most relevant documents.

    n_results=2
)


# ============================================================
# 12. DISPLAY QUERY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SIMILARITY SEARCH RESULTS")
print("=" * 70)


print("\nUser Query:")
print(query)


# ------------------------------------------------------------
# MATCHING DOCUMENTS
# ------------------------------------------------------------

print("\nMatching Documents:")

for i, document in enumerate(
    results["documents"][0]
):

    print(
        f"{i + 1}. {document}"
    )


# ------------------------------------------------------------
# MATCHING IDS
# ------------------------------------------------------------

print("\nMatching IDs:")

for i, document_id in enumerate(
    results["ids"][0]
):

    print(
        f"{i + 1}. {document_id}"
    )


# ------------------------------------------------------------
# DISTANCES
# ------------------------------------------------------------

print("\nDistances:")

for i, distance in enumerate(
    results["distances"][0]
):

    print(
        f"{i + 1}. {distance}"
    )