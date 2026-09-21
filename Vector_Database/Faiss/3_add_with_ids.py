import faiss
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

embeddings = model.encode(documents, convert_to_numpy=True).astype("float32")

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

base_index = faiss.IndexFlatIP(dimension)

index = faiss.IndexIDMap(base_index)

ids = np.array([101, 102, 103, 104, 105], dtype="int64")

index.add_with_ids(embeddings, ids)

query = "Which vehicle runs without fuel?"

query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

faiss.normalize_L2(query_embedding)

scores, result_ids = index.search(query_embedding, 3)

for rank, vector_id in enumerate(result_ids[0], start=1):

    print("Rank:", rank)
    print("Vector ID:", vector_id)
    print("Score:", scores[0][rank - 1])