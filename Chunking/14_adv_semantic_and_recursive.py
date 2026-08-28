from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

text = """
Python is a programming language.

Python is widely used for artificial intelligence
and machine learning.

Python provides libraries for machine learning.

FastAPI is a Python framework for building APIs.

FastAPI can be used to create REST APIs.

ChromaDB is a vector database.

ChromaDB stores and searches embeddings.
"""

file_name = "normal_text.txt"
file_type = "txt"

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

chunk_size = 300
chunk_overlap = 50

semantic_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

semantic_documents = semantic_splitter.create_documents([text])

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    keep_separator=False
)

final_documents = recursive_splitter.split_documents(semantic_documents)

chunk_documents = []

for i, document in enumerate(final_documents, start=1):

    chunk = document.page_content

    token_count = len(
        embeddings._client.tokenizer.encode(
            chunk,
            add_special_tokens=False
        )
    )

    dict_chunk = {}

    dict_chunk["id"] = f"id_{i:03d}"

    dict_chunk["text"] = chunk

    dict_chunk["metadata"] = {
        "document_id": "doc_id_001",
        "file_name": file_name,
        "file_type": file_type,
        "page_number": 1,
        "chunk_index": i,
        "chunk_method": "recursive_semantic_hybrid_chunking",
        "semantic_chunking": True,
        "recursive_chunking": True,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "character_count": len(chunk),
        "token_count": token_count,
        "embedding_model": embedding_model_name,
        "embedding_dimension": 384,
        "ocr_used": False
    }

    chunk_documents.append(dict_chunk)

print("=" * 60)
print("TOTAL CHUNKS:", len(chunk_documents))
print("=" * 60)

for item in chunk_documents:

    print("\n" + "-" * 60)
    print("ID:", item["id"])

    print("TEXT:")
    print(item["text"])

    print("METADATA:")
    print(item["metadata"])