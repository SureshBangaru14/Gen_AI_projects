import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# 52. IVFPQ
# Inverted File + Product Quantization
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

# Normalized vectors + Inner Product
# gives cosine-similarity-like behavior.

faiss.normalize_L2(embeddings)


# ============================================================
# 5. GET VECTOR DIMENSION
# ============================================================

dimension = embeddings.shape[1]

print("Vector dimension:", dimension)


# ============================================================
# 6. IVFPQ PARAMETERS
# ============================================================

# Number of IVF clusters.
nlist = 2

# Number of PQ sub-vectors.
m = 8

# Bits used for every PQ sub-vector.
nbits = 8

print("nlist:", nlist)
print("m:", m)
print("nbits:", nbits)


# ============================================================
# 7. CHECK PQ DIMENSION
# ============================================================

if dimension % m != 0:
    raise ValueError(
        f"Dimension {dimension} must be divisible by m={m}"
    )

print(
    "Dimensions per PQ sub-vector:",
    dimension // m
)


# ============================================================
# 8. CREATE TRAINING VECTORS
# ============================================================

# IVFPQ requires enough training vectors.
#
# We only have 5 real documents.
# Therefore we create training-only vectors
# around the real document embeddings.
#
# These vectors are ONLY used for training.
# They are NOT added as documents.

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

faiss.normalize_L2(training_vectors)

print(
    "Training vector shape:",
    training_vectors.shape
)


# ============================================================
# 9. CREATE IVF QUANTIZER
# ============================================================

# This index determines the nearest IVF cluster.

quantizer = faiss.IndexFlatIP(
    dimension
)


# ============================================================
# 10. CREATE IVFPQ BASE INDEX
# ============================================================

base_index = faiss.IndexIVFPQ(
    quantizer,
    dimension,
    nlist,
    m,
    nbits,
    faiss.METRIC_INNER_PRODUCT
)


# ============================================================
# 11. CHECK TRAINING STATUS
# ============================================================

print(
    "Before training:",
    base_index.is_trained
)


# ============================================================
# 12. TRAIN IVFPQ
# ============================================================

# IMPORTANT:
#
# train() does NOT train the embedding model.
#
# It trains:
#
# 1. IVF cluster centroids
# 2. PQ codebooks

base_index.train(
    training_vectors
)


# ============================================================
# 13. CHECK TRAINING STATUS
# ============================================================

print(
    "After training:",
    base_index.is_trained
)


# ============================================================
# 14. DYNAMIC ID WRAPPER
# ============================================================

index = faiss.IndexIDMap2(
    base_index
)


# ============================================================
# 15. GENERATE DYNAMIC DOCUMENT IDS
# ============================================================

next_id = 1001

ids = np.arange(
    next_id,
    next_id + len(documents),
    dtype="int64"
)

print("Document IDs:", ids)


# ============================================================
# 16. ADD DOCUMENT VECTORS
# ============================================================

index.add_with_ids(
    embeddings,
    ids
)

print(
    "Total vectors:",
    index.ntotal
)


# ============================================================
# 17. METADATA MAPPING
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
# 18. CREATE QUERY
# ============================================================

query_text = "vehicle that carries passengers"


# ============================================================
# 19. CREATE QUERY EMBEDDING
# ============================================================

query_embedding = model.encode(
    [query_text],
    convert_to_numpy=True
).astype("float32")


# ============================================================
# 20. NORMALIZE QUERY
# ============================================================

faiss.normalize_L2(
    query_embedding
)


# ============================================================
# 21. SET nprobe
# ============================================================

# nprobe = number of IVF clusters searched.
#
# nlist  = total clusters
# nprobe = clusters actually searched

ivfpq_index = index.index

ivfpq_index.nprobe = 1

print(
    "nlist:",
    ivfpq_index.nlist
)

print(
    "nprobe:",
    ivfpq_index.nprobe
)


# ============================================================
# 22. SEARCH TOP-K
# ============================================================

top_k = 3

scores, result_ids = index.search(
    query_embedding,
    top_k
)


# ============================================================
# 23. DISPLAY RESULTS
# ============================================================

print("\n========== IVFPQ RESULTS ==========")

print("Query:", query_text)

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
    print(
        "Document:",
        metadata[result_id]["document"]
    )


# ============================================================
# 24. CHANGE nprobe
# ============================================================

# Search more IVF clusters.
#
# Since nlist = 2,
# nprobe = 2 means searching all clusters.

ivfpq_index.nprobe = 2

print(
    "\nChanged nprobe to:",
    ivfpq_index.nprobe
)


# ============================================================
# 25. SEARCH AGAIN
# ============================================================

scores_2, result_ids_2 = index.search(
    query_embedding,
    top_k
)


# ============================================================
# 26. DISPLAY nprobe=2 RESULTS
# ============================================================

print("\n========== nprobe=2 RESULTS ==========")

for rank, (score, result_id) in enumerate(
    zip(scores_2[0], result_ids_2[0]),
    start=1
):

    if result_id == -1:
        continue

    result_id = int(result_id)

    print("\nRank:", rank)
    print("ID:", result_id)
    print("Score:", float(score))
    print(
        "Document:",
        metadata[result_id]["document"]
    )


# ============================================================
# 27. SHOW IVFPQ STRUCTURE
# ============================================================

print("\n========== IVFPQ CONFIGURATION ==========")

print(
    "Vector dimension:",
    dimension
)

print(
    "Number of IVF clusters:",
    ivfpq_index.nlist
)

print(
    "Number of searched clusters:",
    ivfpq_index.nprobe
)

print(
    "PQ sub-vectors:",
    m
)

print(
    "PQ bits:",
    nbits
)

print(
    "Vectors stored:",
    index.ntotal
)