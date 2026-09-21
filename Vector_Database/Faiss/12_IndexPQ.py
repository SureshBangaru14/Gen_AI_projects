import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# 51. PRODUCT QUANTIZATION (PQ)
# ============================================================

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
# 2. LOAD REAL EMBEDDING MODEL
# ============================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================
# 3. CREATE DOCUMENT EMBEDDINGS
# ============================================================

embeddings = model.encode(
    documents,
    convert_to_numpy=True
).astype("float32")

print("Embedding shape:", embeddings.shape)

# ============================================================
# 4. NORMALIZE EMBEDDINGS
# ============================================================

# Normalization allows us to work with cosine-like similarity.
faiss.normalize_L2(embeddings)

# ============================================================
# 5. GET VECTOR DIMENSION
# ============================================================

dimension = embeddings.shape[1]

print("Vector dimension:", dimension)

# ============================================================
# 6. PRODUCT QUANTIZATION SETTINGS
# ============================================================

# Number of sub-vectors.
m = 8

# Number of bits used for each sub-vector.
nbits = 8

print("PQ sub-vectors:", m)
print("Bits per sub-vector:", nbits)

# ============================================================
# 7. CHECK DIMENSION COMPATIBILITY
# ============================================================

if dimension % m != 0:
    raise ValueError(
        f"Dimension {dimension} must be divisible by m={m}"
    )

print(
    "Dimensions per sub-vector:",
    dimension // m
)

# ============================================================
# 8. CREATE TRAINING VECTORS
# ============================================================

# PQ requires training data to learn the codebooks.
#
# We only have 5 real document vectors.
# Therefore, create additional training-only vectors
# around the original document embeddings.
#
# These vectors are NOT added to the FAISS database.

rng = np.random.default_rng(42)

training_vectors = np.vstack([
    embeddings + rng.normal(
        0,
        0.01,
        embeddings.shape
    ).astype("float32")
    for _ in range(1000)
])

training_vectors = training_vectors.astype("float32")

# Normalize training vectors.
faiss.normalize_L2(training_vectors)

print(
    "Training vectors:",
    training_vectors.shape
)

# ============================================================
# 9. CREATE PRODUCT QUANTIZATION INDEX
# ============================================================

base_index = faiss.IndexPQ(
    dimension,
    m,
    nbits,
    faiss.METRIC_INNER_PRODUCT
)

print("Index trained before training:", base_index.is_trained)

# ============================================================
# 10. TRAIN PQ CODEBOOK
# ============================================================

base_index.train(training_vectors)

print("Index trained after training:", base_index.is_trained)

# ============================================================
# 11. DYNAMIC ID WRAPPER
# ============================================================

index = faiss.IndexIDMap2(base_index)

# ============================================================
# 12. GENERATE DYNAMIC DOCUMENT IDS
# ============================================================

next_id = 1001

ids = np.arange(
    next_id,
    next_id + len(documents),
    dtype="int64"
)

print("Document IDs:", ids)

# ============================================================
# 13. ADD DOCUMENT EMBEDDINGS
# ============================================================

index.add_with_ids(
    embeddings,
    ids
)

print("Total vectors:", index.ntotal)

# ============================================================
# 14. METADATA MAPPING
# ============================================================

metadata = {
    1001: {
        "document": documents[0],
        "type": "vehicle"
    },
    1002: {
        "document": documents[1],
        "type": "vehicle"
    },
    1003: {
        "document": documents[2],
        "type": "vehicle"
    },
    1004: {
        "document": documents[3],
        "type": "vehicle"
    },
    1005: {
        "document": documents[4],
        "type": "vehicle"
    }
}

# ============================================================
# 15. CREATE QUERY
# ============================================================

query_text = "vehicle that carries passengers"

# ============================================================
# 16. CONVERT QUERY TO EMBEDDING
# ============================================================

query_embedding = model.encode(
    [query_text],
    convert_to_numpy=True
).astype("float32")

# ============================================================
# 17. NORMALIZE QUERY
# ============================================================

faiss.normalize_L2(query_embedding)

# ============================================================
# 18. SEARCH TOP-K
# ============================================================

top_k = 3

scores, result_ids = index.search(
    query_embedding,
    top_k
)

# ============================================================
# 19. DISPLAY RESULTS
# ============================================================

print("\nQuery:", query_text)

for rank, (score, result_id) in enumerate(
    zip(scores[0], result_ids[0]),
    start=1
):

    if result_id == -1:
        continue

    result_id = int(result_id)

    print("\nRank:", rank)
    print("ID:", result_id)
    print("Score:", float(score))
    print("Document:", metadata[result_id]["document"])

# ============================================================
# 20. SHOW PQ INFORMATION
# ============================================================

print("\n========== PQ INFORMATION ==========")

print("Original dimension:", dimension)
print("Number of sub-vectors:", m)
print(
    "Dimensions per sub-vector:",
    dimension // m
)
print("Bits per sub-vector:", nbits)

# ============================================================
# 21. COMPARE VECTOR STORAGE
# ============================================================

original_bytes = dimension * 4

pq_bytes = m * nbits // 8

print("\n========== MEMORY CONCEPT ==========")

print(
    "Original vector bytes:",
    original_bytes
)

print(
    "Approximate PQ code bytes:",
    pq_bytes
)

print(
    "Compression ratio:",
    round(
        original_bytes / pq_bytes,
        2
    )
)