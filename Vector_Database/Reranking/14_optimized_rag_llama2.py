# ============================================================
# 14_optimized_rag_llama2.py
#
# COMPLETE OPTIMIZED RAG PIPELINE
#
# Document
#    ↓
# RecursiveCharacterTextSplitter
#    ↓
# SentenceTransformer
#    ↓
# ChromaDB
#    ↓
# TOP_K
#    ↓
# BGE Reranker
#    ↓
# Batch Reranking
#    ↓
# TOP_N
#    ↓
# Context
#    ↓
# SYSTEM_PROMPT + SCHEMA + USER_PROMPT
#    ↓
# LangChain ChatOllama
#    ↓
# Llama 2
#    ↓
# JSON Answer
#
# Optimizations:
#
# 1. Latency Reduction
# 2. Batch Reranking
# 3. Model Quantization
# 4. Cache Results
# 5. Cost Optimization
# ============================================================


import json
import time
from functools import lru_cache

import torch
import chromadb

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama


# ============================================================
# OPTIONAL GPU QUANTIZATION IMPORT
# ============================================================
#
# BitsAndBytes is required only when CUDA is available.
#
# On CPU:
#
#     torch.cuda.is_available() == False
#
# We don't try INT8 GPU quantization.
#
# This prevents:
#
# RuntimeError:
# No GPU found. A GPU is needed for quantization.
# ============================================================

if torch.cuda.is_available():

    try:
        from transformers import BitsAndBytesConfig

        BITSANDBYTES_AVAILABLE = True

    except ImportError:

        BITSANDBYTES_AVAILABLE = False

else:

    BITSANDBYTES_AVAILABLE = False


# ============================================================
# SAMPLE DOCUMENT
# ============================================================

DOCUMENT = """
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


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an enterprise document question-answering assistant.

Answer the user's question ONLY using the provided context.

Rules:

1. Do not invent information.
2. Use only the provided context.
3. If the answer is not present in the context,
   say that the information is not available.
4. Keep the answer concise.
5. Return valid JSON only.
"""


# ============================================================
# SCHEMA
# ============================================================

SCHEMA = {
    "answer": "string",
    "confidence": "high | medium | low",
    "source_chunks": ["string"]
}


# ============================================================
# USER PROMPT TEMPLATE
# ============================================================
#
# IMPORTANT:
#
# JSON braces are escaped as:
#
# {{
# }}
#
# because Python .format() interprets { } as placeholders.
#
# Without escaping them:
#
# KeyError: '\n "answer"'
#
# will occur.
# ============================================================

USER_PROMPT_TEMPLATE = """
Context:

{context}


Question:

{query}


Return JSON using exactly this structure:

{{
    "answer": "string",
    "confidence": "high | medium | low",
    "source_chunks": ["chunk_id"]
}}
"""


# ============================================================
# RAG CLASS
# ============================================================

class OptimizedRAG:

    def __init__(self):

        print("\n")
        print("=" * 70)
        print("INITIALIZING OPTIMIZED RAG")
        print("=" * 70)

        # ========================================================
        # MODEL CONFIGURATION
        # ========================================================

        self.embedding_model_name = (
            "all-MiniLM-L6-v2"
        )

        self.reranker_model_name = (
            "BAAI/bge-reranker-base"
        )

        # ========================================================
        # 1. LATENCY REDUCTION
        # ========================================================
        #
        # ChromaDB first retrieves only TOP_K candidates.
        #
        # Example:
        #
        # Database
        #    ↓
        # 10000 chunks
        #    ↓
        # ChromaDB
        #    ↓
        # TOP_K = 20
        #    ↓
        # BGE Reranker
        #
        # We do NOT rerank all 10000 chunks.
        #
        # This reduces latency.
        # ========================================================

        self.TOP_K = 20

        # ========================================================
        # 2. BATCH RERANKING
        # ========================================================
        #
        # Process multiple query-document pairs together.
        #
        # Example:
        #
        # TOP_K = 20
        # BATCH_SIZE = 16
        #
        # Batch 1 = 16
        # Batch 2 = 4
        #
        # Larger batch:
        #     Better throughput
        #
        # Smaller batch:
        #     Lower memory usage
        # ========================================================

        self.RERANK_BATCH_SIZE = 16

        # ========================================================
        # 5. COST OPTIMIZATION
        # ========================================================
        #
        # Only send TOP_N chunks to the LLM.
        #
        # Example:
        #
        # ChromaDB = 20
        #       ↓
        # BGE = 20
        #       ↓
        # TOP_N = 3
        #       ↓
        # Llama 2 = 3 chunks
        #
        # This reduces context size and downstream computation.
        # ========================================================

        self.TOP_N = 3

        # ========================================================
        # CHUNKING CONFIGURATION
        # ========================================================

        self.CHUNK_SIZE = 400

        self.CHUNK_OVERLAP = 50

        # ========================================================
        # DEVICE DETECTION
        # ========================================================

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"\nDevice selected: {self.device}"
        )

        if self.device == "cuda":

            print(
                f"CUDA device: "
                f"{torch.cuda.get_device_name(0)}"
            )

        # ========================================================
        # EMBEDDING MODEL
        # ========================================================

        print("\nLoading SentenceTransformer...")

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.device
        )

        print(
            "Embedding model loaded:"
            f" {self.embedding_model_name}"
        )

        # ========================================================
        # CHROMADB
        # ========================================================
        #
        # In-memory ChromaDB is used here.
        #
        # This avoids persistent database permission problems
        # while learning/testing the RAG pipeline.
        #
        # Production:
        #
        # chromadb.PersistentClient(
        #     path="./chroma_db"
        # )
        # ========================================================

        print("\nInitializing ChromaDB...")

        self.chroma_client = chromadb.Client()

        self.collection = (
            self.chroma_client.get_or_create_collection(
                name="optimized_rag_llama2",
                configuration={
                    "hnsw": {
                        "space": "cosine"
                    }
                }
            )
        )

        print("ChromaDB initialized.")

        # ========================================================
        # TEXT SPLITTER
        # ========================================================

        self.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=self.CHUNK_SIZE,
                chunk_overlap=self.CHUNK_OVERLAP,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    ""
                ]
            )
        )

        # ========================================================
        # BGE RERANKER TOKENIZER
        # ========================================================

        print("\nLoading BGE tokenizer...")

        self.reranker_tokenizer = (
            AutoTokenizer.from_pretrained(
                self.reranker_model_name,
                clean_up_tokenization_spaces=True
            )
        )

        # ========================================================
        # 3. MODEL QUANTIZATION
        # ========================================================
        #
        # INT8 is attempted ONLY when CUDA is available.
        #
        # CPU:
        #
        #     Normal BGE model
        #
        # GPU:
        #
        #     Try INT8
        #
        # If INT8 fails:
        #
        #     Normal GPU model
        # ========================================================

        print("\nLoading BGE reranker...")

        if (
            self.device == "cuda"
            and BITSANDBYTES_AVAILABLE
        ):

            try:

                print(
                    "Trying INT8 quantization..."
                )

                quantization_config = (
                    BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                )

                self.reranker = (
                    AutoModelForSequenceClassification
                    .from_pretrained(
                        self.reranker_model_name,
                        quantization_config=(
                            quantization_config
                        ),
                        device_map="auto"
                    )
                )

                print(
                    "INT8 quantization enabled."
                )

            except Exception as e:

                print(
                    "\nINT8 quantization failed."
                )

                print(
                    f"Reason: {e}"
                )

                print(
                    "Loading normal GPU model..."
                )

                self.reranker = (
                    AutoModelForSequenceClassification
                    .from_pretrained(
                        self.reranker_model_name
                    )
                )

                self.reranker.to(
                    self.device
                )

        else:

            if self.device == "cpu":

                print(
                    "CPU detected."
                )

                print(
                    "Skipping GPU INT8 quantization."
                )

            elif not BITSANDBYTES_AVAILABLE:

                print(
                    "CUDA detected but bitsandbytes "
                    "is unavailable."
                )

                print(
                    "Loading normal GPU model."
                )

            self.reranker = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    self.reranker_model_name
                )
            )

            self.reranker.to(
                self.device
            )

        # ========================================================
        # EVALUATION MODE
        # ========================================================

        self.reranker.eval()

        print(
            "BGE reranker loaded:"
            f" {self.reranker_model_name}"
        )

        # ========================================================
        # LANGCHAIN + OLLAMA + LLAMA 2
        # ========================================================
        #
        # NO QWEN.
        #
        # Architecture:
        #
        # LangChain
        #     ↓
        # ChatOllama
        #     ↓
        # Ollama
        #     ↓
        # Llama 2
        # ========================================================

        print("\nInitializing Llama 2...")

        self.llm = ChatOllama(
            model="llama2",
            temperature=0
        )

        print(
            "Llama 2 initialized through "
            "LangChain ChatOllama."
        )


    # ============================================================
    # CREATE CHUNKS
    # ============================================================

    def create_chunks(
        self,
        document
    ):

        print("\n")
        print("=" * 70)
        print("CREATING DOCUMENT CHUNKS")
        print("=" * 70)

        chunks = (
            self.text_splitter.split_text(
                document
            )
        )

        print(
            f"\nTotal chunks created: "
            f"{len(chunks)}"
        )

        for i, chunk in enumerate(
            chunks
        ):

            print("\n")
            print(
                f"Chunk {i + 1}"
            )

            print("-" * 60)

            print(chunk)

        return chunks


    # ============================================================
    # INDEX DOCUMENTS
    # ============================================================

    def index_documents(
        self,
        chunks
    ):

        print("\n")
        print("=" * 70)
        print("INDEXING DOCUMENTS")
        print("=" * 70)

        # ========================================================
        # CREATE EMBEDDINGS
        # ========================================================

        print(
            "\nCreating embeddings..."
        )

        embeddings = (
            self.embedding_model.encode(
                chunks,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        )

        # ========================================================
        # CREATE IDS
        # ========================================================

        ids = [
            f"chunk_{i}"
            for i in range(
                len(chunks)
            )
        ]

        # ========================================================
        # METADATA
        # ========================================================

        metadatas = []

        for i, chunk in enumerate(
            chunks
        ):

            metadatas.append(
                {
                    "chunk_id": f"chunk_{i}",
                    "text": chunk
                }
            )

        # ========================================================
        # STORE IN CHROMADB
        # ========================================================

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas
        )

        print(
            f"\nSuccessfully indexed "
            f"{len(chunks)} chunks."
        )


    # ============================================================
    # FIRST STAGE RETRIEVAL
    # ============================================================

    def retrieve(
        self,
        query
    ):

        # ========================================================
        # CREATE QUERY EMBEDDING
        # ========================================================

        query_embedding = (
            self.embedding_model.encode(
                [query],
                normalize_embeddings=True
            )[0]
        )

        # ========================================================
        # CHROMADB SEARCH
        # ========================================================
        #
        # 1. LATENCY REDUCTION
        #
        # TOP_K limits how many chunks are passed to
        # the expensive BGE reranker.
        # ========================================================

        results = (
            self.collection.query(
                query_embeddings=[
                    query_embedding.tolist()
                ],
                n_results=self.TOP_K
            )
        )

        documents = (
            results["documents"][0]
        )

        metadatas = (
            results["metadatas"][0]
        )

        distances = (
            results["distances"][0]
        )

        candidates = []

        for i, document in enumerate(
            documents
        ):

            candidates.append(
                {
                    "text": document,
                    "metadata": metadatas[i],
                    "distance": distances[i]
                }
            )

        return candidates


    # ============================================================
    # BGE RERANKING
    # ============================================================

    def rerank(
        self,
        query,
        candidates
    ):

        if not candidates:

            return []

        # ========================================================
        # CREATE QUERY-DOCUMENT PAIRS
        # ========================================================

        pairs = [
            [
                query,
                item["text"]
            ]
            for item in candidates
        ]

        reranked_results = []

        # ========================================================
        # 2. BATCH RERANKING
        # ========================================================
        #
        # Instead of:
        #
        # model(query, doc1)
        # model(query, doc2)
        # model(query, doc3)
        #
        # We process:
        #
        # model([
        #     (query, doc1),
        #     (query, doc2),
        #     ...
        # ])
        #
        # This improves throughput.
        # ========================================================

        for start in range(
            0,
            len(pairs),
            self.RERANK_BATCH_SIZE
        ):

            batch_pairs = pairs[
                start:
                start + self.RERANK_BATCH_SIZE
            ]

            # ====================================================
            # TOKENIZATION
            # ====================================================

            encoded = (
                self.reranker_tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
            )

            # ====================================================
            # MOVE TO CPU/GPU
            # ====================================================

            encoded = {
                key: value.to(
                    self.device
                )
                for key, value
                in encoded.items()
            }

            # ====================================================
            # INFERENCE
            # ====================================================

            with torch.no_grad():

                outputs = (
                    self.reranker(
                        **encoded
                    )
                )

                batch_scores = (
                    outputs.logits.view(-1)
                )

            # ====================================================
            # SAVE SCORES
            # ====================================================

            for j, score in enumerate(
                batch_scores
            ):

                original_index = (
                    start + j
                )

                reranked_results.append(
                    {
                        "text": (
                            candidates[
                                original_index
                            ]["text"]
                        ),
                        "metadata": (
                            candidates[
                                original_index
                            ]["metadata"]
                        ),
                        "chroma_distance": (
                            candidates[
                                original_index
                            ]["distance"]
                        ),
                        "reranker_score": (
                            float(
                                score.cpu().item()
                            )
                        )
                    }
                )

        # ========================================================
        # SORT BY BGE SCORE
        # ========================================================

        reranked_results.sort(
            key=lambda x:
                x["reranker_score"],
            reverse=True
        )

        # ========================================================
        # 5. COST OPTIMIZATION
        # ========================================================
        #
        # Only TOP_N documents are passed to the LLM.
        # ========================================================

        return reranked_results[
            :self.TOP_N
        ]


    # ============================================================
    # BUILD CONTEXT
    # ============================================================

    def build_context(
        self,
        results
    ):

        context_parts = []

        for result in results:

            chunk_id = (
                result[
                    "metadata"
                ]["chunk_id"]
            )

            text = result[
                "text"
            ]

            context_parts.append(
                f"[{chunk_id}]\n{text}"
            )

        return "\n\n".join(
            context_parts
        )


    # ============================================================
    # GENERATE ANSWER USING LLAMA 2
    # ============================================================

    def generate_answer(
        self,
        query,
        results
    ):

        if not results:

            return {
                "answer": (
                    "No relevant information found."
                ),
                "confidence": "low",
                "source_chunks": []
            }

        # ========================================================
        # BUILD CONTEXT
        # ========================================================

        context = (
            self.build_context(
                results
            )
        )

        # ========================================================
        # BUILD USER PROMPT
        # ========================================================
        #
        # IMPORTANT:
        #
        # USER_PROMPT_TEMPLATE contains escaped JSON braces:
        #
        # {{
        # }}
        #
        # Therefore .format() works correctly.
        # ========================================================

        user_prompt = (
            USER_PROMPT_TEMPLATE.format(
                context=context,
                query=query
            )
        )

        # ========================================================
        # LANGCHAIN → OLLAMA → LLAMA 2
        # ========================================================

        response = (
            self.llm.invoke(
                [
                    (
                        "system",
                        SYSTEM_PROMPT
                    ),
                    (
                        "user",
                        user_prompt
                    )
                ]
            )
        )

        content = response.content

        # ========================================================
        # CLEAN POSSIBLE MARKDOWN JSON
        # ========================================================

        content = content.strip()

        if content.startswith(
            "```json"
        ):

            content = (
                content
                .replace(
                    "```json",
                    "",
                    1
                )
                .strip()
            )

            if content.endswith(
                "```"
            ):

                content = (
                    content[:-3]
                    .strip()
                )

        elif content.startswith(
            "```"
        ):

            content = (
                content
                .replace(
                    "```",
                    "",
                    1
                )
                .strip()
            )

            if content.endswith(
                "```"
            ):

                content = (
                    content[:-3]
                    .strip()
                )

        # ========================================================
        # PARSE JSON
        # ========================================================

        try:

            answer = json.loads(
                content
            )

        except json.JSONDecodeError:

            # ====================================================
            # LLAMA 2 FALLBACK
            # ====================================================
            #
            # Sometimes local LLMs don't return perfect JSON.
            #
            # Instead of crashing the complete RAG pipeline,
            # return the raw answer safely.
            # ====================================================

            answer = {
                "answer": content,
                "confidence": "medium",
                "source_chunks": [
                    result[
                        "metadata"
                    ]["chunk_id"]
                    for result in results
                ]
            }

        return answer


    # ============================================================
    # 4. CACHE RESULTS
    # ============================================================
    #
    # Identical query:
    #
    # First time:
    #
    # Query
    #   ↓
    # Embedding
    #   ↓
    # ChromaDB
    #   ↓
    # BGE
    #   ↓
    # Llama 2
    #
    # Second time:
    #
    # Query
    #   ↓
    # Cache
    #   ↓
    # Result
    #
    # This avoids repeating expensive computation.
    #
    # maxsize=1000:
    #
    # Up to 1000 query results are cached in memory.
    #
    # Production systems can use Redis for shared caching.
    # ============================================================

    @lru_cache(
        maxsize=1000
    )
    def cached_search(
        self,
        query
    ):

        print(
            "\n[CACHE MISS] "
            "Running RAG pipeline..."
        )

        # ========================================================
        # FIRST STAGE RETRIEVAL
        # ========================================================

        candidates = (
            self.retrieve(
                query
            )
        )

        print(
            f"Retrieved "
            f"{len(candidates)} "
            f"candidates from ChromaDB."
        )

        # ========================================================
        # SECOND STAGE RERANKING
        # ========================================================

        reranked = (
            self.rerank(
                query,
                candidates
            )
        )

        print(
            f"Reranked to "
            f"TOP_N = {len(reranked)}"
        )

        # ========================================================
        # GENERATE ANSWER
        # ========================================================

        answer = (
            self.generate_answer(
                query,
                reranked
            )
        )

        return answer


    # ============================================================
    # PUBLIC SEARCH
    # ============================================================

    def search(
        self,
        query
    ):

        start_time = (
            time.perf_counter()
        )

        # ========================================================
        # CACHE
        # ========================================================

        cache_before = (
            self.cached_search.cache_info()
        )

        result = (
            self.cached_search(
                query
            )
        )

        cache_after = (
            self.cached_search.cache_info()
        )

        end_time = (
            time.perf_counter()
        )

        # ========================================================
        # DETECT CACHE HIT
        # ========================================================

        if (
            cache_after.hits
            > cache_before.hits
        ):

            print(
                "\n[CACHE HIT] "
                "Returning cached result."
            )

        print(
            f"\nTotal pipeline time: "
            f"{end_time - start_time:.3f} seconds"
        )

        print(
            f"Cache statistics: "
            f"{cache_after}"
        )

        return result


    # ============================================================
    # DISPLAY FINAL RESULT
    # ============================================================

    def display_results(
        self,
        query,
        answer
    ):

        print("\n")

        print(
            "=" * 70
        )

        print(
            "QUERY"
        )

        print(
            "=" * 70
        )

        print(
            query
        )

        print("\n")

        print(
            "=" * 70
        )

        print(
            "FINAL LLAMA 2 ANSWER"
        )

        print(
            "=" * 70
        )

        print(
            json.dumps(
                answer,
                indent=4,
                ensure_ascii=False
            )
        )


# ================================================================
# MAIN
# ================================================================

def main():

    print("\n")

    print(
        "=" * 70
    )

    print(
        "OPTIMIZED RAG"
    )

    print(
        "BGE RERANKER + CHROMADB + LLAMA 2"
    )

    print(
        "=" * 70
    )

    # ============================================================
    # CREATE RAG SYSTEM
    # ============================================================

    rag = OptimizedRAG()

    # ============================================================
    # CREATE CHUNKS
    # ============================================================

    chunks = (
        rag.create_chunks(
            DOCUMENT
        )
    )

    # ============================================================
    # INDEX DOCUMENTS
    # ============================================================

    rag.index_documents(
        chunks
    )

    # ============================================================
    # QUERY
    # ============================================================

    query = (
        "How many weeks of maternity "
        "leave are provided?"
    )

    # ============================================================
    # FIRST REQUEST
    # ============================================================

    print("\n")

    print(
        "=" * 70
    )

    print(
        "FIRST QUERY"
    )

    print(
        "=" * 70
    )

    answer = (
        rag.search(
            query
        )
    )

    rag.display_results(
        query,
        answer
    )

    # ============================================================
    # SECOND REQUEST
    # ============================================================
    #
    # Same query.
    #
    # This demonstrates CACHE optimization.
    #
    # Expected:
    #
    # First request:
    #
    # [CACHE MISS]
    #
    # Second request:
    #
    # [CACHE HIT]
    # ============================================================

    print("\n")

    print(
        "=" * 70
    )

    print(
        "SECOND IDENTICAL QUERY"
    )

    print(
        "=" * 70
    )

    start = (
        time.perf_counter()
    )

    answer2 = (
        rag.search(
            query
        )
    )

    end = (
        time.perf_counter()
    )

    print(
        f"\nSecond request time: "
        f"{end - start:.6f} seconds"
    )

    rag.display_results(
        query,
        answer2
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()