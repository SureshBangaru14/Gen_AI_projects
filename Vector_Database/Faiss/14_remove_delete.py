import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# 56. DELETE / REPLACE VECTORS
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
# 3. CREATE EMBEDDINGS
# ============================================================

embeddings = model.encode(
    documents,
    convert_to_numpy=True
).astype("float32")


# ============================================================
# 4. NORMALIZE EMBEDDINGS
# ============================================================

faiss.normalize_L2(
    embeddings
)


# ============================================================
# 5. GET DIMENSION
# ============================================================

dimension = embeddings.shape[1]

print(
    "Vector dimension:",
    dimension
)


# ============================================================
# 6. CREATE BASE INDEX
# ============================================================

base_index = faiss.IndexFlatIP(
    dimension
)


# ============================================================
# 7. DYNAMIC ID WRAPPER
# ============================================================

index = faiss.IndexIDMap2(
    base_index
)


# ============================================================
# 8. GENERATE DYNAMIC IDS
# ============================================================

next_id = 1001

ids = np.arange(
    next_id,
    next_id + len(documents),
    dtype="int64"
)

next_id += len(documents)


# ============================================================
# 9. ADD DOCUMENTS
# ============================================================

index.add_with_ids(
    embeddings,
    ids
)


# ============================================================
# 10. CREATE METADATA STORE
# ============================================================

metadata = {}

for document_id, document in zip(
    ids,
    documents
):

    metadata[int(document_id)] = {
        "text": document,
        "status": "active"
    }


# ============================================================
# 11. DISPLAY INITIAL INDEX
# ============================================================

print(
    "\n========== INITIAL INDEX =========="
)

print(
    "Total vectors:",
    index.ntotal
)

for document_id in sorted(metadata):

    print(
        document_id,
        "→",
        metadata[document_id]["text"]
    )


# ============================================================
# 12. DELETE A VECTOR
# ============================================================

delete_id = 1003

print(
    "\n========== DELETE VECTOR =========="
)

print(
    "Deleting ID:",
    delete_id
)


# ============================================================
# 13. CREATE ID ARRAY FOR REMOVAL
# ============================================================

delete_ids = np.array(
    [delete_id],
    dtype="int64"
)


# ============================================================
# 14. REMOVE VECTOR FROM FAISS
# ============================================================

removed_count = index.remove_ids(
    delete_ids
)

print(
    "Vectors removed:",
    removed_count
)


# ============================================================
# 15. REMOVE METADATA
# ============================================================

if delete_id in metadata:

    del metadata[delete_id]


# ============================================================
# 16. DISPLAY AFTER DELETE
# ============================================================

print(
    "\n========== AFTER DELETE =========="
)

print(
    "Total vectors:",
    index.ntotal
)

for document_id in sorted(metadata):

    print(
        document_id,
        "→",
        metadata[document_id]["text"]
    )


# ============================================================
# 17. REPLACE AN EXISTING DOCUMENT
# ============================================================

replace_id = 1002

new_text = (
    "Bus transports passengers between cities"
)


print(
    "\n========== REPLACE DOCUMENT =========="
)

print(
    "Old ID:",
    replace_id
)

print(
    "New text:",
    new_text
)


# ============================================================
# 18. REMOVE OLD VECTOR
# ============================================================

replace_ids = np.array(
    [replace_id],
    dtype="int64"
)

index.remove_ids(
    replace_ids
)


# ============================================================
# 19. CREATE NEW EMBEDDING
# ============================================================

new_embedding = model.encode(
    [new_text],
    convert_to_numpy=True
).astype("float32")


# ============================================================
# 20. NORMALIZE NEW EMBEDDING
# ============================================================

faiss.normalize_L2(
    new_embedding
)


# ============================================================
# 21. GENERATE NEW UNIQUE ID
# ============================================================

new_id = next_id

next_id += 1

new_id_array = np.array(
    [new_id],
    dtype="int64"
)


# ============================================================
# 22. ADD REPLACED DOCUMENT
# ============================================================

index.add_with_ids(
    new_embedding,
    new_id_array
)


# ============================================================
# 23. REMOVE OLD METADATA
# ============================================================

if replace_id in metadata:

    del metadata[replace_id]


# ============================================================
# 24. ADD NEW METADATA
# ============================================================

metadata[new_id] = {
    "text": new_text,
    "status": "active",
    "replaced_id": replace_id
}


# ============================================================
# 25. DISPLAY FINAL INDEX
# ============================================================

print(
    "\n========== FINAL INDEX =========="
)

print(
    "Total vectors:",
    index.ntotal
)

print(
    "Next available ID:",
    next_id
)

for document_id in sorted(metadata):

    print(
        document_id,
        "→",
        metadata[document_id]["text"]
    )


# ============================================================
# 26. SEARCH FINAL INDEX
# ============================================================

query_text = "bus carrying people"

query_embedding = model.encode(
    [query_text],
    convert_to_numpy=True
).astype("float32")


# ============================================================
# 27. NORMALIZE QUERY
# ============================================================

faiss.normalize_L2(
    query_embedding
)


# ============================================================
# 28. SEARCH
# ============================================================

top_k = 3

scores, result_ids = index.search(
    query_embedding,
    top_k
)


# ============================================================
# 29. DISPLAY SEARCH RESULTS
# ============================================================

print(
    "\n========== SEARCH RESULTS =========="
)

print(
    "Query:",
    query_text
)

for rank, (score, result_id) in enumerate(
    zip(scores[0], result_ids[0]),
    start=1
):

    if result_id == -1:
        continue

    result_id = int(result_id)

    print(
        f"\nRank: {rank}"
    )

    print(
        f"ID: {result_id}"
    )

    print(
        f"Score: {float(score):.4f}"
    )

    print(
        f"Document: {metadata[result_id]['text']}"
    )