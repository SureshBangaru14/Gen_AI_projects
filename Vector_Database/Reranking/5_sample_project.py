import chromadb
import ollama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# 1. DOCUMENT
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
# 2. LANGCHAIN RECURSIVE CHUNKING
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = splitter.split_text(document)


print("\n" + "=" * 70)
print("LANGCHAIN RECURSIVE CHUNKING")
print("=" * 70)

for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk)


# ============================================================
# 3. SENTENCE TRANSFORMER EMBEDDING
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
# 4. CHROMADB
# ============================================================

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="langchain_two_stage_rag"
)

collection.add(
    ids=[
        f"chunk_{i}"
        for i in range(len(chunks))
    ],
    documents=chunks,
    embeddings=chunk_embeddings
)


# ============================================================
# 5. FIRST STAGE
#    CHROMADB SEMANTIC RETRIEVAL
# ============================================================

TOP_K = 5

results = collection.query(
    query_embeddings=[
        query_embedding
    ],
    n_results=min(
        TOP_K,
        len(chunks)
    )
)


candidate_ids = results["ids"][0]

candidate_chunks = results["documents"][0]

candidate_distances = results["distances"][0]


print("\n" + "=" * 70)
print("FIRST STAGE - CHROMADB")
print("=" * 70)

for rank, (
    chunk_id,
    chunk,
    distance
) in enumerate(
    zip(
        candidate_ids,
        candidate_chunks,
        candidate_distances
    ),
    1
):

    similarity = 1 - distance

    print(f"\nRank       : {rank}")
    print(f"Chunk ID   : {chunk_id}")
    print(f"Similarity : {similarity:.4f}")
    print(f"Chunk      : {chunk}")


# ============================================================
# 6. SECOND STAGE
#    CROSS-ENCODER RERANKING
# ============================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


pairs = [
    [query, chunk]
    for chunk in candidate_chunks
]


rerank_scores = reranker.predict(
    pairs
)


reranked_results = sorted(
    zip(
        candidate_ids,
        candidate_chunks,
        rerank_scores
    ),
    key=lambda x: x[2],
    reverse=True
)


# ============================================================
# 7. TOP-N
# ============================================================

TOP_N = 2

final_results = reranked_results[
    :TOP_N
]


print("\n" + "=" * 70)
print("SECOND STAGE - CROSS-ENCODER")
print(final_results)
print("=" * 70)

for rank, (
    chunk_id,
    chunk,
    score
) in enumerate(
    final_results,
    1
):

    print(f"\nRank       : {rank}")
    print(f"Chunk ID   : {chunk_id}")
    print(f"Score      : {float(score):.4f}")
    print(f"Chunk      : {chunk}")


# ============================================================
# 8. FINAL CONTEXT
# ============================================================

context = "\n\n".join(chunk for _, chunk, _ in final_results)


print("\n" + "=" * 70)
print("FINAL CONTEXT")
print("=" * 70)

print(context)


# ============================================================
# 9. QWEN / OLLAMA
# ============================================================

prompt = f"""
You are a helpful question-answering assistant.

Answer the question using ONLY the context below.

If the answer is not available in the context,
say: "I don't know based on the provided context."

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


# ============================================================
# 10. FINAL ANSWER
# ============================================================

answer = response["message"]["content"]

print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)

print(answer)