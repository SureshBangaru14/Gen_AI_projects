import os
import chromadb
import cohere
import ollama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


document = """
Employee Leave Policy

Employees are eligible for 26 weeks of maternity leave.
Maternity leave must be requested through the HR portal.

Employees receive 15 days of annual leave every year.
Annual leave requests must be submitted to the employee's manager.

The company provides health insurance to all employees.
Employees can work from home two days per week.

The company provides parental leave benefits to employees.
Employees receive a performance bonus based on yearly results.
"""

query = "How many weeks of maternity leave are provided?"


# Chunking

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(document)


# Embeddings

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(
    chunks
).tolist()

query_embedding = embedding_model.encode(
    query
).tolist()


# ChromaDB

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="cohere_rag"
)

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)


# First Stage

TOP_K = 5

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(TOP_K, len(chunks))
)

candidates = results["documents"][0]


# Cohere

co = cohere.ClientV2(
    api_key=os.environ["COHERE_API_KEY"]
)


# Second Stage

response = co.rerank(
    model="rerank-v3.5",
    query=query,
    documents=candidates,
    top_n=2
)


# Get ranked documents

final_results = []

for result in response.results:

    final_results.append(
        candidates[result.index]
    )


# Context

context = "\n\n".join(
    final_results
)


# Qwen

prompt = f"""
Answer the question using only the context.

Context:
{context}

Question:
{query}

Answer:
"""

answer = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


print("\nFINAL ANSWER:")
print(answer["message"]["content"])