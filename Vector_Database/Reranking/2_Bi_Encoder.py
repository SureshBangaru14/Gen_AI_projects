import chromadb
from sentence_transformers import SentenceTransformer

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
collection = client.get_or_create_collection("bi_encoder")

embeddings = model.encode(documents).tolist()

collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=embeddings
)

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

for i, (doc, distance) in enumerate(
    zip(results["documents"][0], results["distances"][0]), 1
):
    print(i, round(1 - distance, 4), doc)