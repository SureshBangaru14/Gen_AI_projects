import json
import chromadb
import ollama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder


class RAGReranker:

    def __init__(
        self,
        embedding_model="all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model="qwen3:8b",
        top_k=10,
        top_n=3
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(reranker_model)
        self.llm_model = llm_model
        self.top_k = top_k
        self.top_n = top_n

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75
        )

        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="rag_reranking",
            configuration={"hnsw": {"space": "cosine"}}
        )

    def create_chunks(self, document):
        return self.splitter.split_text(document)

    def store_documents(self, chunks):
        embeddings = self.embedding_model.encode(
            chunks,
            normalize_embeddings=True
        ).tolist()

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )

    def retrieve(self, query):
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "distances"]
        )

        documents = results["documents"][0]
        distances = results["distances"][0]

        candidates = []

        for document, distance in zip(documents, distances):
            candidates.append({
                "document": document,
                "retrieval_score": 1 - distance
            })

        return candidates

    def rerank(self, query, candidates):
        pairs = [
            [query, candidate["document"]]
            for candidate in candidates
        ]

        scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return candidates[:self.top_n]

    def create_context(self, results):
        return "\n\n".join(
            f"[Source Chunk {i}]\n{result['document']}"
            for i, result in enumerate(results, 1)
        )

    def generate_answer(self, query, results):

        context = self.create_context(results)

        SYSTEM_PROMPT = """
You are an enterprise RAG question-answering assistant.

Rules:
1. Answer only from the provided context.
2. Do not use outside knowledge.
3. If the answer is not available, return "Not found in context".
4. Keep the answer concise and factual.
5. Return valid JSON only.
6. Include supporting source chunks.
7. Confidence must be high, medium, or low.
"""

        SCHEMA = {
            "answer": "string",
            "confidence": "high | medium | low",
            "source_chunks": ["string"]
        }

        USER_PROMPT = f"""
RETRIEVED CONTEXT:
{context}

QUESTION:
{query}

OUTPUT SCHEMA:
{json.dumps(SCHEMA, indent=2)}

Return JSON only.
"""

        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": USER_PROMPT
                }
            ],
            format="json"
        )

        return json.loads(response["message"]["content"])

    def query(self, query):

        candidates = self.retrieve(query)

        reranked_results = self.rerank(
            query,
            candidates
        )

        answer = self.generate_answer(
            query,
            reranked_results
        )

        return {
            "first_stage": candidates,
            "second_stage": reranked_results,
            "answer": answer
        }


if __name__ == "__main__":

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

    rag = RAGReranker()

    chunks = rag.create_chunks(document)

    rag.store_documents(chunks)

    result = rag.query(query)

    print("\nFIRST STAGE - TOP K")

    for item in result["first_stage"]:
        print(
            f"Retrieval Score: {item['retrieval_score']:.4f}"
        )
        print(item["document"])
        print()

    print("\nSECOND STAGE - TOP N")

    for item in result["second_stage"]:
        print(
            f"Rerank Score: {item['rerank_score']:.4f}"
        )
        print(item["document"])
        print()

    print("\nFINAL ANSWER")
    print(json.dumps(result["answer"], indent=2))