"""
                    ┌─────────────────────┐
                    │   Documents         │
                    │ PDF / TXT / DOCX    │
                    └──────────┬──────────┘
                               ↓
                         Text Extraction
                               ↓
                     Recursive Chunking
                               ↓
                    SentenceTransformer
                               ↓
                         Embeddings
                               ↓
                    ┌──────────────────┐
                    │    ChromaDB      │
                    │ Persistent Store │
                    └────────┬─────────┘
                             │
User Query ──────────────────┘
     ↓
Query Embedding
     ↓
ChromaDB similarity search
     ↓
Initial Top-K = 20/50
     ↓
Distance threshold (optional)
     ↓
Cross-Encoder Reranker
     ↓
Reranker relevance score
     ↓
Sort descending
     ↓
Final Top-N = 5
     ↓
Context Construction
     ↓
LLM
     ↓
Final Answer + Sources


The important distinction is:

            ChromaDB
            = First-stage retrieval

            CrossEncoder
            = Actual second-stage reranking

            LLM
            = Generation

"""


# ============================================================
# REAL RAG RERANKING - LEARNING EXAMPLE
# ChromaDB + SentenceTransformer + CrossEncoder
# ============================================================

import chromadb

from sentence_transformers import (SentenceTransformer,CrossEncoder)


# ============================================================
# 1. SAMPLE DOCUMENTS
# ============================================================

documents = [
    "Employees are eligible for 26 weeks of maternity leave.",
    "Employees receive 15 days of annual leave every year.",
    "The company provides health insurance to all employees.",
    "Employees can work from home two days per week.",
    "Maternity leave must be requested through the HR portal.",
    "Employees receive a performance bonus based on yearly results.",
    "The company provides parental leave benefits to employees.",
    "Employees must submit leave requests to their manager."
]


# ============================================================
# 2. USER QUERY
# ============================================================

query = "How many weeks of maternity leave are provided?"


# ============================================================
# 3. CONFIGURATION
# ============================================================

TOP_K = 5

TOP_N = 3

CHROMA_PATH = "./chroma_learning_db"

COLLECTION_NAME = "employee_policy"


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 5. LOAD RERANKER
# ============================================================

print("Loading reranker...")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ============================================================
# 6. CREATE CHROMADB CLIENT
# ============================================================

client = chromadb.PersistentClient(path=CHROMA_PATH)


# ============================================================
# 7. CREATE COLLECTION
# ============================================================

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,

    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)


# ============================================================
# 8. CREATE DOCUMENT EMBEDDINGS
# ============================================================

document_embeddings = embedding_model.encode(documents, normalize_embeddings=True).tolist()


# ============================================================
# 9. STORE DOCUMENTS IN CHROMADB
# ============================================================

ids = [f"doc_{i}" for i in range(len(documents))]


metadatas = [
                {
                    "document_id": i + 1
                }

                for i in range(len(documents))
            ]


collection.upsert(

    ids=ids,

    documents=documents,

    embeddings=document_embeddings,

    metadatas=metadatas
)


print("\nDocuments stored in ChromaDB.")


# ============================================================
# 10. CREATE QUERY EMBEDDING
# ============================================================

query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()


# ============================================================
# 11. FIRST-STAGE RETRIEVAL
# ============================================================

results = collection.query(

    query_embeddings=[
        query_embedding
    ],

    n_results=TOP_K,

    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)


retrieved_documents = results["documents"][0]

retrieved_metadatas = results["metadatas"][0]

retrieved_distances = results["distances"][0]


# ============================================================
# 12. DISPLAY TOP-K RETRIEVAL
# ============================================================

print("\n")
print("=" * 70)
print("1. TOP-K RETRIEVAL")
print("=" * 70)

print(f"\nQuery: {query}")

print(f"TOP_K = {TOP_K}")


for rank, (document, metadata, distance) in enumerate(zip(retrieved_documents, retrieved_metadatas, retrieved_distances),start=1):

    # Chroma cosine distance
    # Lower distance = more similar

    similarity = 1 - distance


    print(
        f"\nRank: {rank}"
    )

    print(
        f"Document ID: "
        f"{metadata['document_id']}"
    )

    print(
        f"Chroma Distance: "
        f"{distance:.4f}"
    )

    print(
        f"Similarity Score: "
        f"{similarity:.4f}"
    )

    print(
        f"Document: "
        f"{document}"
    )


# ============================================================
# 13. PREPARE CANDIDATES FOR RERANKING
# ============================================================

candidates = []


for document, metadata, distance in zip(

    retrieved_documents,
    retrieved_metadatas,
    retrieved_distances

):

    candidates.append({

        "document": document,

        "metadata": metadata,

        "distance": float(
            distance
        ),

        "similarity": float(
            1 - distance
        )

    })


# ============================================================
# 14. CREATE QUERY-DOCUMENT PAIRS
# ============================================================

pairs = [

    (
        query,
        candidate["document"]
    )

    for candidate in candidates

]


# ============================================================
# 15. CROSS-ENCODER RERANKING
# ============================================================

rerank_scores = reranker.predict(
    pairs
)


# ============================================================
# 16. ADD RERANKING SCORES
# ============================================================

for candidate, score in zip(

    candidates,
    rerank_scores

):

    candidate["rerank_score"] = float(
        score
    )


# ============================================================
# 17. SORT BY RERANKING SCORE
# ============================================================

reranked_results = sorted(

    candidates,

    key=lambda x:
        x["rerank_score"],

    reverse=True

)


# ============================================================
# 18. DISPLAY RERANKED RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("2. CROSS-ENCODER RERANKING")
print("=" * 70)


for rank, result in enumerate(

    reranked_results,

    start=1

):

    print(
        f"\nRank: {rank}"
    )

    print(
        f"Document ID: "
        f"{result['metadata']['document_id']}"
    )

    print(
        f"Chroma Similarity: "
        f"{result['similarity']:.4f}"
    )

    print(
        f"Rerank Score: "
        f"{result['rerank_score']:.4f}"
    )

    print(
        f"Document: "
        f"{result['document']}"
    )


# ============================================================
# 19. FINAL TOP-N
# ============================================================

final_results = reranked_results[
    :TOP_N
]


# ============================================================
# 20. FINAL CONTEXT
# ============================================================

print("\n")
print("=" * 70)
print("3. FINAL TOP-N CONTEXT")
print("=" * 70)


for rank, result in enumerate(

    final_results,

    start=1

):

    print(
        f"\nFinal Rank: {rank}"
    )

    print(
        f"Rerank Score: "
        f"{result['rerank_score']:.4f}"
    )

    print(
        f"Document: "
        f"{result['document']}"
    )


# ============================================================
# 21. BUILD LLM CONTEXT
# ============================================================

context = "\n\n".join(

    [
        result["document"]

        for result in final_results
    ]

)


print("\n")
print("=" * 70)
print("4. CONTEXT SENT TO LLM")
print("=" * 70)

print(context)


# ============================================================
# 22. COMPLETE FLOW
# ============================================================

print("\n")
print("=" * 70)
print("COMPLETE RAG FLOW")
print("=" * 70)

print(
    """
User Query
    ↓
Query Embedding
    ↓
ChromaDB
    ↓
Top-K Retrieval
    ↓
Candidate Documents
    ↓
Cross-Encoder
    ↓
Rerank Scores
    ↓
Sort Descending
    ↓
Top-N
    ↓
Final Context
    ↓
LLM
"""
)