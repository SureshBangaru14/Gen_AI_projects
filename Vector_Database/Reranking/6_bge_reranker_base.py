import chromadb
import ollama
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# DOCUMENT + QUERY
# ============================================================

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


# ============================================================
# CHUNKING
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(document)


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chunk_embeddings = embedding_model.encode(
    chunks
).tolist()

query_embedding = embedding_model.encode(
    query
).tolist()


# ============================================================
# CHROMADB
# ============================================================

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="bge_base_rag"
)

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=chunk_embeddings
)


# ============================================================
# FIRST STAGE
# ============================================================

TOP_K = 5

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(TOP_K, len(chunks))
)

candidate_chunks = results["documents"][0]
print("=========candidate_chunks=========")
print(candidate_chunks)

# ============================================================
# BGE RERANKER BASE
# ============================================================

model_name = "BAAI/bge-reranker-base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

reranker = AutoModelForSequenceClassification.from_pretrained(
    model_name
)

device = "cuda" if torch.cuda.is_available() else "cpu"

reranker.to(device)

reranker.eval()


# ============================================================
# SECOND STAGE
# ============================================================

scores = []

for chunk in candidate_chunks:

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


# ============================================================
# RERANK
# ============================================================

reranked = sorted(
    zip(candidate_chunks, scores),
    key=lambda x: x[1],
    reverse=True
)


# ============================================================
# TOP-N
# ============================================================

TOP_N = 2

final_results = reranked[:TOP_N]


# ============================================================
# CONTEXT
# ============================================================

context = "\n\n".join(
    chunk
    for chunk, score in final_results
)


# ============================================================
# QWEN / OLLAMA
# ============================================================

prompt = f"""
Answer the question using only the provided context.

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