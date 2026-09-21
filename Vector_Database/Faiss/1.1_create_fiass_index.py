import faiss
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

print("Embedding shape:", embeddings.shape)
print("Embedding shape:", embeddings[1])

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

print("Is trained:", index.is_trained)

index.add(embeddings)

print("Number of vectors:", index.ntotal)

query = "Which vehicle runs without fuel?"

query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

print("Query embedding shape:", query_embedding.shape)

k = 3

distances, indices = index.search(query_embedding, k)

for rank, idx in enumerate(indices[0], start=1):

    print("Rank:", rank)
    print("Distance:", distances[0][rank - 1])
    print("Document:", documents[idx])
    print()
    
    
    """
                     DOCUMENTS

"Car runs on petrol"
"Bus carries passengers"
"Bicycle runs without fuel"
"Boat travels on water"
"Plane flies in the sky"
          │
          ↓
 ┌──────────────────────┐
 │  Embedding Model     │
 │  all-MiniLM-L6-v2    │
 └──────────────────────┘
          │
          ↓
    5 Embeddings
          │
          ↓
   384 Dimensions
          │
          ↓
 ┌──────────────────────┐
 │       FAISS          │
 │    IndexFlatL2       │
 └──────────────────────┘
          │
          │
          │
       QUERY
          │
          ↓
"Which vehicle runs
 without fuel?"
          │
          ↓
    Embedding Model
          │
          ↓
   Query Embedding
          │
          ↓
        FAISS
          │
          ↓
     Similarity Search
          │
          ↓
        Top-K
          │
          ↓
"Bicycle runs without fuel"
    
    """