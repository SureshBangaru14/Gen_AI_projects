import faiss
import numpy as np

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

# Convert text into numerical vectors.

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ============================================================
# 3. CREATE DOCUMENT EMBEDDINGS
# ============================================================

embeddings = model.encode(documents, convert_to_numpy=True).astype("float32")


# ============================================================
# 4. GET VECTOR DIMENSION
# ============================================================

dimension = embeddings.shape[1]

print("Embedding Shape:", embeddings.shape)

print("Vector Dimension:", dimension)


# ============================================================
# 5. NORMALIZE EMBEDDINGS
# ============================================================

# Normalization allows Inner Product to represent
# Cosine Similarity.

faiss.normalize_L2(embeddings)


# ============================================================
# 6. DEFINE nlist
# ============================================================

# nlist = number of clusters.
#
# IMPORTANT:
# For only 5 documents, we use nlist=2
# only for learning purposes.
#
# Real production datasets need much more
# training data.

nlist = 2


# ============================================================
# 7. CREATE QUANTIZER
# ============================================================

# The quantizer finds the nearest cluster.

quantizer = faiss.IndexFlatIP(dimension)


# ============================================================
# 8. CREATE IVF INDEX
# ============================================================

# IndexIVFFlat:
#
#     quantizer
#     dimension
#     nlist
#     similarity metric

base_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)


# ============================================================
# 9. CHECK TRAINING STATUS
# ============================================================

print("Before training:", base_index.is_trained)


# ============================================================
# 10. TRAIN IVF INDEX
# ============================================================

# IVF must learn the cluster centroids
# before vectors can be added.

base_index.train(embeddings)


# ============================================================
# 11. CHECK TRAINING STATUS
# ============================================================

print("After training:", base_index.is_trained)


# ============================================================
# 12. SET nprobe
# ============================================================

# nprobe = number of clusters searched.
#
# nprobe=1:
#     Search only one cluster.
#
# nprobe=nlist:
#     Search all clusters.

base_index.nprobe = 2


# ============================================================
# 13. DYNAMIC ID WRAPPER
# ============================================================

# IndexIDMap2 allows custom FAISS IDs.

index = faiss.IndexIDMap2(base_index)


# ============================================================
# 14. GENERATE DYNAMIC IDs
# ============================================================

next_id = 1001

ids = np.arange(next_id, next_id + len(documents), dtype="int64")


# ============================================================
# 15. ADD EMBEDDINGS WITH DYNAMIC IDs
# ============================================================

index.add_with_ids(embeddings, ids)


# ============================================================
# 16. DISPLAY DOCUMENT → FAISS ID
# ============================================================

print("\nDocument -> FAISS ID")

for faiss_id, document in zip(ids, documents):

    print(f"{faiss_id} -> {document}")


# ============================================================
# 17. CREATE TEXT QUERY
# ============================================================

query_text = ("vehicle that carries passengers")


# ============================================================
# 18. QUERY → EMBEDDING
# ============================================================

query_embedding = model.encode([query_text], convert_to_numpy=True).astype("float32")


# ============================================================
# 19. NORMALIZE QUERY
# ============================================================

faiss.normalize_L2(query_embedding)


# ============================================================
# 20. DEFINE TOP-K
# ============================================================

top_k = 3


# ============================================================
# 21. IVF SEARCH
# ============================================================

scores, result_ids = index.search(query_embedding, top_k)


# ============================================================
# 22. DISPLAY SEARCH RESULTS
# ============================================================

print("\nIVF Search Results")

for rank, (faiss_id, score) in enumerate(zip(result_ids[0], scores[0]), start=1):

    # Find original document position.

    position = np.where(ids == faiss_id)[0][0]

    print("\n" + "-" * 60)

    print("Rank:", rank)

    print("FAISS ID:", faiss_id)

    print("Document:", documents[position])

    print("Cosine Similarity:", round(float(score), 4))


# ============================================================
# 23. DISPLAY IVF CONFIGURATION
# ============================================================

print("\n" + "=" * 60)

print("IVF Configuration")

print("Embedding Model:", "all-MiniLM-L6-v2")

print("Vector Dimension:", dimension)

print("Total Vectors:", index.ntotal)

print("nlist:", nlist)

print("nprobe:", base_index.nprobe)

print("Is Trained:", base_index.is_trained)