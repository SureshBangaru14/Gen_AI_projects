import faiss
import numpy as np
import time

from sentence_transformers import SentenceTransformer


# ============================================================
# 1. DOCUMENTS
# ============================================================

documents = [
    "Car runs on petrol",
    "Bus carries passengers",
    "Bicycle runs without fuel",
    "Boat travels on water",
    "Plane flies in the sky"
]


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

# SentenceTransformer converts text into numerical vectors.

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ============================================================
# 3. CREATE DOCUMENT EMBEDDINGS
# ============================================================

# Convert documents into vectors.

embeddings = model.encode(documents, convert_to_numpy=True).astype("float32")


# ============================================================
# 4. GET VECTOR DIMENSION
# ============================================================

# all-MiniLM-L6-v2 produces 384-dimensional vectors.

dimension = embeddings.shape[1]

print("Embedding Shape:", embeddings.shape)

print("Vector Dimension:", dimension)


# ============================================================
# 5. NORMALIZE DOCUMENT EMBEDDINGS
# ============================================================

# Normalize vectors so that Inner Product can be used
# as Cosine Similarity.

faiss.normalize_L2(embeddings)


# ============================================================
# 6. WHAT IS HNSW?
# ============================================================

# HNSW =
# Hierarchical Navigable Small World
#
# HNSW is a graph-based Approximate Nearest Neighbor
# search algorithm.


# ============================================================
# 7. CREATE HNSW INDEX
# ============================================================

# M controls graph connectivity.

M = 32

base_index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)


# ============================================================
# 8. GRAPH-BASED SEARCH
# ============================================================

# HNSW creates a graph of connected vectors.
#
# Similar vectors are connected through graph relationships.
#
# Query:
#
#        Query
#          |
#          ↓
#      Entry Point
#          |
#          ↓
#       Graph
#       /   \
#     Node  Node
#       \   /
#        Node
#
# HNSW navigates the graph to find nearest vectors.


# ============================================================
# 9. HNSW LAYERS
# ============================================================

# HNSW uses hierarchical graph layers.
#
# Upper layers:
#     Fast/coarse navigation
#
# Lower layers:
#     Detailed navigation
#
# Layer 0:
#     Detailed graph connections.

print("Maximum HNSW Level:", base_index.hnsw.max_level)


# ============================================================
# 10. NODES
# ============================================================

# Every document embedding becomes a node.

for node_id, document in enumerate(documents):

    print(f"Node {node_id} -> {document}")


# ============================================================
# 11. CONNECTIONS
# ============================================================

# M controls the graph connectivity.

print("M:", M)


# ============================================================
# 12. SET efConstruction
# ============================================================

# efConstruction controls the effort used
# while constructing the HNSW graph.

base_index.hnsw.efConstruction = 100

print("efConstruction:", base_index.hnsw.efConstruction)


# ============================================================
# 13. SET efSearch
# ============================================================

# efSearch controls the search effort.
#
# Higher efSearch:
#     Usually improves recall
#     Usually increases latency

base_index.hnsw.efSearch = 32

print("efSearch:", base_index.hnsw.efSearch)


# ============================================================
# 14. DYNAMIC ID WRAPPER
# ============================================================

# IndexIDMap2 allows us to use custom IDs
# instead of positional IDs.

index = faiss.IndexIDMap2(base_index)


# ============================================================
# 15. GENERATE DYNAMIC IDs
# ============================================================

# Start IDs from 1001.

next_id = 1001

ids = np.arange(next_id, next_id + len(documents), dtype="int64")


# ============================================================
# 16. ADD EMBEDDINGS WITH DYNAMIC IDs
# ============================================================

index.add_with_ids(embeddings, ids)


# ============================================================
# 17. DISPLAY DOCUMENT → FAISS ID
# ============================================================

print("\nDocument -> FAISS ID")

for faiss_id, document in zip(ids, documents):

    print(f"{faiss_id} -> {document}")


# ============================================================
# 18. CREATE TEXT QUERY
# ============================================================

query_text = (
    "vehicle that carries passengers"
)


# ============================================================
# 19. QUERY → EMBEDDING
# ============================================================

# Convert the query text into the same
# embedding space as the documents.

query_embedding = model.encode([query_text], convert_to_numpy=True).astype("float32")


# ============================================================
# 20. NORMALIZE QUERY EMBEDDING
# ============================================================

# Query must be normalized in the same way
# as document embeddings.

faiss.normalize_L2(query_embedding)


# ============================================================
# 21. DEFINE TOP-K
# ============================================================

top_k = 3


# ============================================================
# 22. START LATENCY TIMER
# ============================================================

start_time = time.perf_counter()


# ============================================================
# 23. HNSW SEARCH
# ============================================================

scores, result_ids = index.search(query_embedding, top_k)


# ============================================================
# 24. STOP LATENCY TIMER
# ============================================================

end_time = time.perf_counter()


# ============================================================
# 25. CALCULATE LATENCY
# ============================================================

latency_ms = (end_time - start_time) * 1000


# ============================================================
# 26. DISPLAY SEARCH RESULTS
# ============================================================

print("\nHNSW Search Results")

for rank, (faiss_id, score) in enumerate(zip(result_ids[0], scores[0]), start=1):

    # Find original document position.

    position = np.where(ids == faiss_id)[0][0]

    print("\n" + "-" * 60)

    print("Rank:", rank)

    print("FAISS ID:", faiss_id)

    print("Document:", documents[position])

    print("Cosine Similarity:", round(float(score), 4))


# ============================================================
# 27. DISPLAY LATENCY
# ============================================================

print("\nSearch Latency:")

print(round(latency_ms, 4), "ms")


# ============================================================
# 28. CREATE EXACT INDEX
# ============================================================

# IndexFlatIP performs exact Inner Product search.
#
# Because vectors are normalized,
# Inner Product = Cosine Similarity.

exact_index = faiss.IndexFlatIP(dimension)


# ============================================================
# 29. ADD DOCUMENT EMBEDDINGS
# ============================================================

exact_index.add(embeddings)


# ============================================================
# 30. EXACT SEARCH
# ============================================================

exact_scores, exact_ids = (exact_index.search(query_embedding, top_k))


# ============================================================
# 31. GET HNSW RESULTS
# ============================================================

hnsw_ids = result_ids[0]


# ============================================================
# 32. CONVERT EXACT POSITIONS TO IDs
# ============================================================

exact_faiss_ids = ids[exact_ids[0]]


# ============================================================
# 33. CALCULATE RECALL@K
# ============================================================

ground_truth_set = set(exact_faiss_ids)

hnsw_set = set(hnsw_ids)

recall = (len(ground_truth_set & hnsw_set)/ top_k)


# ============================================================
# 34. DISPLAY RECALL
# ============================================================

print(f"\nRecall@{top_k}:", round(recall, 4))


# ============================================================
# 35. DISPLAY FINAL HNSW CONFIGURATION
# ============================================================

print("\n" + "=" * 60)

print("HNSW Configuration")

print(
    "Embedding Model:", "all-MiniLM-L6-v2")

print("Vector Dimension:", dimension)

print("Total Vectors:", index.ntotal)

print("M:", M)

print("efConstruction:", base_index.hnsw.efConstruction)

print("efSearch:", base_index.hnsw.efSearch)

print("Top-K:", top_k)

print("Latency:", round(latency_ms, 4), "ms")

print(f"Recall@{top_k}:", round(recall, 4))