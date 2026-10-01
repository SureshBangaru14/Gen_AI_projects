import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

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

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = chromadb.Client()
collection = client.get_or_create_collection("cross_encoder")

collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=embedding_model.encode(documents).tolist()
)

results = collection.query(
    query_embeddings=[embedding_model.encode(query).tolist()],
    n_results=5
)

candidates = results["documents"][0]

pairs = [[query, document] for document in candidates]
scores = reranker.predict(pairs)

reranked = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)

for i, (document, score) in enumerate(reranked[:2], 1):
    print(i, round(float(score), 4), document)