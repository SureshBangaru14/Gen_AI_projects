import json
import time
import torch
import chromadb
import ollama

from functools import lru_cache
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)


class OptimizedRAG:

    def __init__(self):

        # ============================================================
        # CONFIGURATION
        # ============================================================

        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.reranker_model_name = "BAAI/bge-reranker-base"
        self.llm_model = "qwen3:8b"

        # ============================================================
        # 1. LATENCY REDUCTION
        # ============================================================
        #
        # Retrieve only TOP_K candidates from ChromaDB.
        #
        # Example:
        #
        # TOP_K = 100
        #     ↓
        # Reranker processes 100 documents
        #
        # TOP_K = 20
        #     ↓
        # Reranker processes only 20 documents
        #
        # Fewer candidates generally means lower reranking latency.
        #
        # However, TOP_K should not be too small because it can
        # reduce retrieval recall.
        # ============================================================

        self.TOP_K = 20

        # ============================================================
        # 2. BATCH RERANKING
        # ============================================================
        #
        # Process multiple query-document pairs together.
        #
        # Example:
        #
        # TOP_K = 20
        # BATCH_SIZE = 16
        #
        # Batch 1 → 16 documents
        # Batch 2 → 4 documents
        #
        # This can improve throughput, especially on GPU.
        # ============================================================

        self.RERANK_BATCH_SIZE = 16

        # ============================================================
        # 5. COST OPTIMIZATION
        # ============================================================
        #
        # After reranking, only TOP_N documents are sent to the LLM.
        #
        # TOP_K = 20
        #      ↓
        # Reranker
        #      ↓
        # TOP_N = 3
        #      ↓
        # LLM receives only 3 chunks
        #
        # Smaller context can reduce input tokens and LLM processing.
        # ============================================================

        self.TOP_N = 3

        # ============================================================
        # DEVICE DETECTION
        # ============================================================

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"\nReranker device: {self.device}")

        if self.device == "cuda":
            print(
                f"GPU: {torch.cuda.get_device_name(0)}"
            )
        else:
            print("GPU not available. Using CPU.")

        # ============================================================
        # EMBEDDING MODEL
        # ============================================================

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name
        )

        # ============================================================
        # RERANKER TOKENIZER
        # ============================================================

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.reranker_model_name,
            clean_up_tokenization_spaces=True
        )

        # ============================================================
        # 3. MODEL QUANTIZATION
        # ============================================================
        #
        # IMPORTANT:
        #
        # BitsAndBytes INT8 quantization requires a compatible GPU
        # in this Transformers environment.
        #
        # Therefore:
        #
        # GPU available
        #       ↓
        # INT8 quantization
        #
        # CPU only
        #       ↓
        # Normal FP32 model
        #
        # This prevents:
        #
        # RuntimeError:
        # No GPU found. A GPU is needed for quantization.
        # ============================================================

        if self.device == "cuda":

            print("Loading INT8 quantized reranker...")

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

            self.reranker = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    self.reranker_model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            )

        else:

            print(
                "Loading normal CPU reranker "
                "(INT8 quantization disabled)..."
            )

            self.reranker = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    self.reranker_model_name
                )
            )

            self.reranker.to("cpu")

        # ============================================================
        # CHUNKING
        # ============================================================

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )

        # ============================================================
        # CHROMADB
        # ============================================================

        self.client = chromadb.Client()

        self.collection = (
            self.client.get_or_create_collection(
                name="optimized_rag",
                configuration={
                    "hnsw": {
                        "space": "cosine"
                    }
                }
            )
        )

    # ================================================================
    # CREATE CHUNKS
    # ================================================================

    def create_chunks(self, document):

        chunks = self.splitter.split_text(
            document
        )

        return chunks

    # ================================================================
    # INDEX DOCUMENT
    # ================================================================

    def index_documents(self, document):

        chunks = self.create_chunks(
            document
        )

        print(
            f"\nNumber of chunks: {len(chunks)}"
        )

        embeddings = (
            self.embedding_model
            .encode(
                chunks,
                normalize_embeddings=True
            )
            .tolist()
        )

        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )

        print("Documents indexed successfully.")

    # ================================================================
    # FIRST-STAGE RETRIEVAL
    # ================================================================

    def retrieve(self, query):

        start = time.perf_counter()

        # ------------------------------------------------------------
        # LATENCY REDUCTION
        # ------------------------------------------------------------
        #
        # Only TOP_K candidates are retrieved.
        # This limits the number of documents sent to the reranker.
        # ------------------------------------------------------------

        query_embedding = (
            self.embedding_model
            .encode(
                query,
                normalize_embeddings=True
            )
            .tolist()
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.TOP_K,
            include=[
                "documents",
                "distances"
            ]
        )

        documents = results["documents"][0]
        distances = results["distances"][0]

        candidates = []

        for document, distance in zip(
            documents,
            distances
        ):

            candidates.append(
                {
                    "document": document,
                    "retrieval_score": (
                        1 - distance
                    )
                }
            )

        latency = time.perf_counter() - start

        print(
            f"\nFirst-stage retrieval:"
            f" {len(candidates)} documents"
        )

        print(
            f"Retrieval latency: "
            f"{latency:.4f} seconds"
        )

        return candidates

    # ================================================================
    # SECOND-STAGE RERANKING
    # ================================================================

    def rerank(
        self,
        query,
        candidates
    ):

        start = time.perf_counter()

        documents = [
            item["document"]
            for item in candidates
        ]

        pairs = [
            [query, document]
            for document in documents
        ]

        all_scores = []

        # ------------------------------------------------------------
        # BATCH RERANKING
        # ------------------------------------------------------------
        #
        # Instead of processing every document individually,
        # query-document pairs are processed in batches.
        #
        # Example:
        #
        # TOP_K = 20
        # BATCH_SIZE = 16
        #
        # Batch 1 → 16 pairs
        # Batch 2 → 4 pairs
        # ------------------------------------------------------------

        for i in range(
            0,
            len(pairs),
            self.RERANK_BATCH_SIZE
        ):

            batch = pairs[
                i:i + self.RERANK_BATCH_SIZE
            ]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            # --------------------------------------------------------
            # DEVICE
            # --------------------------------------------------------

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():

                output = self.reranker(
                    **inputs
                )

            scores = (
                output.logits
                .squeeze(-1)
                .float()
                .cpu()
                .tolist()
            )

            if isinstance(scores, float):
                scores = [scores]

            all_scores.extend(scores)

        # ------------------------------------------------------------
        # ADD RERANKING SCORE
        # ------------------------------------------------------------

        for candidate, score in zip(
            candidates,
            all_scores
        ):

            candidate["rerank_score"] = float(
                score
            )

        # ------------------------------------------------------------
        # SORT BY RERANKER SCORE
        # ------------------------------------------------------------

        candidates.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # ------------------------------------------------------------
        # COST OPTIMIZATION
        # ------------------------------------------------------------
        #
        # Do not send all TOP_K documents to the LLM.
        #
        # Example:
        #
        # ChromaDB → 20
        # Reranker → 20
        # Final context → 3
        #
        # This reduces the amount of context sent to the LLM.
        # ------------------------------------------------------------

        final_documents = candidates[
            :self.TOP_N
        ]

        latency = time.perf_counter() - start

        print(
            f"Reranking latency: "
            f"{latency:.4f} seconds"
        )

        print(
            f"Final documents: "
            f"{len(final_documents)}"
        )

        return final_documents

    # ================================================================
    # CREATE CONTEXT
    # ================================================================

    def create_context(
        self,
        ranked_documents
    ):

        context_parts = []

        for index, item in enumerate(
            ranked_documents,
            start=1
        ):

            context_parts.append(
                f"""
[Source Chunk {index}]
{item["document"]}
"""
            )

        return "\n".join(
            context_parts
        )

    # ================================================================
    # CACHE
    # ================================================================
    #
    # CACHE OPTIMIZATION
    #
    # If the exact same query is executed again, the previous
    # result is returned from the local cache.
    #
    # First request:
    #
    # Query → Embedding → ChromaDB → Reranker → LLM
    #
    # Second request:
    #
    # Query → CACHE HIT → Previous Result
    #
    # For multi-server production systems, a shared cache such as
    # Redis is generally more appropriate.
    # ================================================================

    @lru_cache(maxsize=1000)
    def ask(self, query):

        total_start = time.perf_counter()

        print("\nCACHE MISS → Running RAG pipeline")

        # ============================================================
        # FIRST STAGE
        # ============================================================

        candidates = self.retrieve(
            query
        )

        # ============================================================
        # SECOND STAGE
        # ============================================================

        ranked_documents = self.rerank(
            query,
            candidates
        )

        # ============================================================
        # FINAL CONTEXT
        # ============================================================

        context = self.create_context(
            ranked_documents
        )

        # ============================================================
        # SYSTEM PROMPT
        # ============================================================

        SYSTEM_PROMPT = """
You are an enterprise RAG question-answering assistant.

Rules:

1. Answer only using the provided context.
2. Do not use outside knowledge.
3. If the answer is not available in the context,
   return "Not found in context".
4. Keep the answer concise and factual.
5. Return valid JSON only.
6. Include the source chunks supporting the answer.
7. Confidence must be one of:
   "high", "medium", "low".
"""

        # ============================================================
        # SCHEMA
        # ============================================================

        SCHEMA = {
            "answer": "string",
            "confidence": "high | medium | low",
            "source_chunks": [
                "string"
            ]
        }

        # ============================================================
        # USER PROMPT
        # ============================================================

        USER_PROMPT = f"""
Use the retrieved context to answer the question.

RETRIEVED CONTEXT:

{context}

QUESTION:

{query}

OUTPUT SCHEMA:

{json.dumps(
    SCHEMA,
    indent=2
)}

Return JSON only.
"""

        # ============================================================
        # LLM
        # ============================================================

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

        total_latency = (
            time.perf_counter()
            - total_start
        )

        # ============================================================
        # FINAL RESULT
        # ============================================================

        result = {
            "answer": json.loads(
                response["message"]["content"]
            ),
            "metrics": {
                "retrieved_documents": len(
                    candidates
                ),
                "final_documents": len(
                    ranked_documents
                ),
                "total_latency_seconds": round(
                    total_latency,
                    4
                )
            }
        }

        return json.dumps(
            result,
            indent=2
        )


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":

    # ================================================================
    # SAMPLE DOCUMENT
    # ================================================================

    document = """
    Employee Leave Policy.

    Employees are eligible for 26 weeks of maternity leave.

    Maternity leave must be requested through the HR portal.

    Employees receive 15 days of annual leave every year.

    Annual leave requests must be submitted to the employee's manager.

    The company provides health insurance to all employees.

    Employees can work from home two days per week.

    The company provides parental leave benefits to employees.

    Employees receive a performance bonus based on yearly results.
    """

    # ================================================================
    # CREATE RAG
    # ================================================================

    rag = OptimizedRAG()

    # ================================================================
    # INDEX DOCUMENT
    # ================================================================

    rag.index_documents(
        document
    )

    # ================================================================
    # FIRST QUERY
    # ================================================================

    query = (
        "How many weeks of maternity leave "
        "are provided?"
    )

    print("\n" + "=" * 70)
    print("FIRST QUERY")
    print("=" * 70)

    result = rag.ask(
        query
    )

    print("\nFINAL RESULT:")
    print(result)

    # ================================================================
    # SECOND QUERY
    # ================================================================
    #
    # Same query.
    #
    # This should use the cached result.
    # ================================================================

    print("\n" + "=" * 70)
    print("SECOND QUERY - CACHE TEST")
    print("=" * 70)

    result = rag.ask(
        query
    )

    print("\nFINAL RESULT:")
    print(result)