from sentence_transformers import SentenceTransformer
import faiss
import json


index = faiss.read_index("faiss_store/index.faiss")


with open("faiss_store/metadata.json", "r",encoding="utf-8") as f:

    metadata = json.load(f)


print("Loaded vectors:", index.ntotal)

print("Loaded metadata:", len(metadata))


model = SentenceTransformer("sentence-transformers/""all-MiniLM-L6-v2")


query = "What are the standard working hours?"


query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")


faiss.normalize_L2(query_embedding)


k = min(3, index.ntotal)


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

                    QUERY
                      │
                      ▼
       "What are the standard working hours?"
                      │
                      ▼
                Query Embedding
                      │
                      ▼
                 FAISS Search
                      │
                      ▼
                    Top-K
                      │
          ┌───────────┴──────────┐
          ▼                      ▼
      Vector IDs             Scores
          │
          ▼
    metadata.json
          │
          ▼
    File + Page + Chunk
          │
          ▼
       Context
          │
          ▼
         Qwen
          │
          ▼
       Final Answer

"""