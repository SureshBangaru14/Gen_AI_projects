import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# =====================================================
# 1. Documents
# =====================================================

documents = [
    "Car runs on petrol",
    "Bus carries passengers",
    "Bicycle runs without fuel",
    "Boat travels on water",
    "Plane flies in the sky"
]


# =====================================================
# 2. Embedding Model
# =====================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =====================================================
# 3. Documents → Embeddings
# =====================================================

embeddings = model.encode(
    documents,
    convert_to_numpy=True
).astype("float32")


# =====================================================
# 4. Normalize
# =====================================================

faiss.normalize_L2(
    embeddings
)


# =====================================================
# 5. Dynamic Dimension
# =====================================================

dimension = embeddings.shape[1]


# =====================================================
# 6. Base Index
# =====================================================

base_index = faiss.IndexFlatIP(
    dimension
)


# =====================================================
# 7. ID Map
# =====================================================

index = faiss.IndexIDMap(
    base_index
)


# =====================================================
# 8. Dynamic IDs
# =====================================================

start_id = 1001

ids = np.arange(
    start_id,
    start_id + len(documents),
    dtype="int64"
)


# =====================================================
# 9. Add Vectors + Dynamic IDs
# =====================================================

index.add_with_ids(
    embeddings,
    ids
)


print(
    "Total vectors:",
    index.ntotal
)

print(
    "Dynamic IDs:",
    ids
)


# =====================================================
# 10. Metadata Mapping
# =====================================================

metadata = {}

for document, vector_id in zip(
    documents,
    ids
):

    metadata[str(vector_id)] = {
        "text": document
    }


# =====================================================
# 11. Query
# =====================================================

query = "vehicle that carries people"


query_embedding = model.encode(
    [query],
    convert_to_numpy=True
).astype("float32")


faiss.normalize_L2(
    query_embedding
)


# =====================================================
# 12. Search
# =====================================================

k = min(
    3,
    index.ntotal
)


scores, result_ids = index.search(
    query_embedding,
    k
)


# =====================================================
# 13. Retrieve Using Dynamic IDs
# =====================================================

for rank, (
    score,
    vector_id
) in enumerate(
    zip(
        scores[0],
        result_ids[0]
    ),
    start=1
):

    vector_id = int(
        vector_id
    )

    info = metadata[
        str(vector_id)
    ]

    print(
        f"\nRank: {rank}"
    )

    print(
        f"ID: {vector_id}"
    )

    print(
        f"Score: {score:.4f}"
    )

    print(
        f"Text: {info['text']}"
    )