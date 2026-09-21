import faiss
import json
from sentence_transformers import SentenceTransformer


# =========================================================
# LOAD FAISS INDEX
# =========================================================

index = faiss.read_index("vehicles.index")


# =========================================================
# LOAD METADATA
# =========================================================

with open("metadata_1.json", "r", encoding="utf-8") as file:

    metadata = json.load(file)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# =========================================================
# USER QUERY
# =========================================================

query = "Which vehicle runs without fuel?"


# =========================================================
# QUERY → EMBEDDING
# =========================================================

query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")


# Normalize query

faiss.normalize_L2(query_embedding)


# =========================================================
# SEARCH
# =========================================================

k = min(3,index.ntotal)

scores, result_ids = index.search(query_embedding, k)


# =========================================================
# RETRIEVE METADATA
# =========================================================

for rank, vector_id in enumerate(result_ids[0], start=1):

    vector_id = int(vector_id)

    info = metadata[str(vector_id)]

    print("\n----------------------")

    print("Rank:", rank)

    print("FAISS ID:", vector_id)

    print("Score:", scores[0][rank - 1])

    print("File:",info["file_name"])

    print("Page:",info["page_no"])

    print("Chunk:",info["chunk_id"])

    print("Text:", info["text"])
    
    
    
"""
                Application Start
                       ↓
             Check FAISS files
                       ↓
              ┌────────┴────────┐
              ↓                 ↓
        index.faiss        metadata.json
              ↓                 ↓
        Load FAISS          Load Metadata
              ↓                 ↓
              └────────┬────────┘
                       ↓
                   Ready
                       ↓
                     Query
                       ↓
                Query Embedding
                       ↓
                    FAISS
                       ↓
                 Top-K IDs
                       ↓
                  Metadata
                       ↓
               File/Page/Chunk
                       ↓
                  Chunk Text
                       ↓
                      LLM

"""