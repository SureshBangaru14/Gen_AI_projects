import chromadb
import ollama
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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


# Embedding

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

collection_embeddings = embedding_model.encode(
    chunks
).tolist()

query_embedding = embedding_model.encode(
    query
).tolist()


# ChromaDB

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="bge_large_rag"
)

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=collection_embeddings
)


# First Stage

TOP_K = 5

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(TOP_K, len(chunks))
)

candidates = results["documents"][0]


# BGE Large

model_name = "BAAI/bge-reranker-large"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

reranker = AutoModelForSequenceClassification.from_pretrained(
    model_name
)

device = "cuda" if torch.cuda.is_available() else "cpu"

reranker.to(device)

reranker.eval()


# Second Stage

scores = []

for chunk in candidates:

    inputs = tokenizer(
        query,
        chunk,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = reranker(**inputs)

        score = output.logits.view(-1).float().cpu().item()

    scores.append(score)


reranked = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)


# Top-N

TOP_N = 2

final_results = reranked[:TOP_N]


# Context

context = "\n\n".join(
    chunk
    for chunk, score in final_results
)


# Qwen

prompt = f"""
Answer only from the context.

Context:
{context}

Question:
{query}

Answer:
"""

response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nFINAL ANSWER:")
print(response["message"]["content"])