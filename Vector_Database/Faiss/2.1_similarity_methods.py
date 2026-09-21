import faiss
from sentence_transformers import SentenceTransformer

documents = [
    "Car runs on petrol",
    "Bus carries passengers",
    "Bicycle runs without fuel",
    "Boat travels on water",
    "Plane flies in the sky"
]

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = model.encode(
    documents,
    convert_to_numpy=True
).astype("float32")

query = "Which vehicle runs without fuel?"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
).astype("float32")

faiss.normalize_L2(embeddings)

faiss.normalize_L2(query_embedding)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

scores, indices = index.search(
    query_embedding,
    3
)

for rank, idx in enumerate(indices[0], start=1):

    print("Rank:", rank)
    print("Cosine Similarity:", scores[0][rank - 1])
    print("Document:", documents[idx])
    print()