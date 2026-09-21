import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# 1. INPUT DOCUMENTS
# =========================================================

documents = [
    "Car runs on petrol",
    "Bus carries passengers",
    "Bicycle runs without fuel",
    "Boat travels on water",
    "Plane flies in the sky"
]


file_name = "vehicles.txt"
page_no = 1


# =========================================================
# 2. LOAD EMBEDDING MODEL
# =========================================================

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# =========================================================
# 3. CREATE EMBEDDINGS
# =========================================================

embeddings = model.encode(documents, convert_to_numpy=True).astype("float32")


# =========================================================
# 4. NORMALIZE EMBEDDINGS
# =========================================================

faiss.normalize_L2(embeddings)


# =========================================================
# 5. CREATE DYNAMIC FAISS IDS
# =========================================================

start_id = 1001

ids = np.arange(start_id,start_id + len(documents), dtype="int64")


# =========================================================
# 6. CREATE DYNAMIC METADATA
# =========================================================

metadata = {}

for index, (document, vector_id) in enumerate(zip(documents, ids), start=1):

    chunk_id = f"{file_name}_p{page_no}_c{index}"

    metadata[str(vector_id)] = {
        "file_name": file_name,
        "page_no": page_no,
        "chunk_id": chunk_id,
        "text": document
    }


# =========================================================
# 7. CREATE FAISS INDEX
# =========================================================

dimension = embeddings.shape[1]

base_index = faiss.IndexFlatIP(dimension)

index = faiss.IndexIDMap(base_index)


# =========================================================
# 8. ADD VECTORS WITH IDS
# =========================================================

index.add_with_ids(embeddings,ids)


print("Total vectors:", index.ntotal)
print("Embedding dimension:", dimension)


# =========================================================
# 9. SAVE METADATA
# =========================================================

with open("metadata.json", "w", encoding="utf-8") as file:

    json.dump(metadata, file, indent=4)


# =========================================================
# 10. USER QUERY
# =========================================================

query = "Which vehicle runs without fuel?"


# =========================================================
# 11. QUERY → EMBEDDING
# =========================================================

query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")


# =========================================================
# 12. NORMALIZE QUERY
# =========================================================

faiss.normalize_L2(query_embedding)


# =========================================================
# 13. TOP-K SEARCH
# =========================================================

k = min(3, index.ntotal)

scores, result_ids = index.search(query_embedding, k)


# =========================================================
# 14. RETRIEVE METADATA
# =========================================================

print("\nQuery:", query)

print("\nSearch Results:")

for rank, vector_id in enumerate(result_ids[0], start=1):

    vector_id = int(vector_id)

    info = metadata[str(vector_id)]

    print("\n-----------------------------")

    print("Rank:", rank)

    print("FAISS ID:", vector_id)

    print("Similarity Score:", scores[0][rank - 1])

    print("File:", info["file_name"])

    print("Page:", info["page_no"])

    print("Chunk:", info["chunk_id"])

    print("Text:", info["text"])