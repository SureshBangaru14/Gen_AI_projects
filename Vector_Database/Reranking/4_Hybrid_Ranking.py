import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

documents = [
    "Employees are eligible for 26 weeks of maternity leave.",
    "Employees receive 15 days of annual leave every year.",
    "The company provides health insurance to all employees.",
    "Employees can work from home two days per week.",
    "Maternity leave must be requested through the HR portal.",
    "Employees receive a performance bonus based on yearly results.",
    "The company provides parental leave benefits to employees.",
    "Employees must submit leave requests to their manager."
]

query = "How many weeks of maternity leave are provided?"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection("hybrid_ranking")

collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=model.encode(documents).tolist()
)

semantic = collection.query(
    query_embeddings=[model.encode(query).tolist()],
    n_results=len(documents)
)

semantic_scores = {
    doc_id: 1 - distance
    for doc_id, distance in zip(
        semantic["ids"][0],
        semantic["distances"][0]
    )
}

bm25 = BM25Okapi([
    document.lower().split()
    for document in documents
])

keyword_scores = bm25.get_scores(query.lower().split())

def normalize(scores):
    minimum = min(scores)
    maximum = max(scores)
    return [
        (x - minimum) / (maximum - minimum)
        if maximum != minimum else 1
        for x in scores
    ]

keyword_scores = normalize(keyword_scores)

semantic_values = normalize([
    semantic_scores[f"doc_{i}"]
    for i in range(len(documents))
])

results = []

for i, document in enumerate(documents):
    hybrid_score = (
        0.4 * keyword_scores[i] +
        0.6 * semantic_values[i]
    )

    results.append((document, hybrid_score))

results.sort(key=lambda x: x[1], reverse=True)

for i, (document, score) in enumerate(results[:5], 1):
    print(i, round(score, 4), document)