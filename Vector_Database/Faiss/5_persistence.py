import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


documents = [
    "Car runs on petrol",
    "Bus carries passengers",
    "Bicycle runs without fuel",
    "Boat travels on water",
    "Plane flies in the sky"
]


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Create embeddings

embeddings = model.encode(documents, convert_to_numpy=True).astype("float32")


# Normalize

faiss.normalize_L2(embeddings)


# Dynamic IDs

start_id = 1001

ids = np.arange(start_id, start_id + len(documents), dtype="int64")


# Dynamic metadata

metadata = {}

for chunk_number, (text, vector_id) in enumerate(zip(documents, ids),start=1):

    metadata[str(vector_id)] = {

        "file_name": "vehicles.txt",

        "page_no": 1,

        "chunk_id": f"vehicles_p1_c{chunk_number}",

        "text": text
    }


# Create FAISS index

dimension = embeddings.shape[1]

base_index = faiss.IndexFlatIP(dimension)

index = faiss.IndexIDMap(base_index)


# Add vectors with IDs

index.add_with_ids(embeddings, ids)


# =========================================================
# SAVE FAISS INDEX
# =========================================================

faiss.write_index(index,"vehicles.index")


# =========================================================
# SAVE METADATA
# =========================================================

with open("metadata_1.json", "w", encoding="utf-8") as file:

    json.dump(metadata, file, indent=4)


print("FAISS index saved.")
print("Metadata saved.")