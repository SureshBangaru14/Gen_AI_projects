from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

text = """Python is a programming language.

Python is widely used for AI and machine learning.

Python can be used to build APIs.

FastAPI is a popular Python framework.

FastAPI is commonly used for REST APIs.

RAG applications use Python for document processing.

Embeddings convert text into vectors.

Vector databases store and search embeddings.
"""

file_name = "normal_text.txt"
file_type = "txt"

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name
)

text_splitter = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type="percentile"
)

documents = text_splitter.create_documents([text])

chunk_documents = []

for i, document in enumerate(documents, start=1):

    chunk = document.page_content

    dict_chunk = {}

    dict_chunk["id"] = f"id_{i:03d}"

    dict_chunk["text"] = chunk

    dict_chunk["metadata"] = {
        "document_id": "doc_id_001",
        "file_name": file_name,
        "file_type": file_type,
        "page_number": 1,
        "chunk_index": i,
        "chunk_method": "semantic_chunking",
        "character_count": len(chunk),
        "embedding_model": embedding_model_name,
        "ocr_used": False
    }

    chunk_documents.append(dict_chunk)

print("=" * 60)
print("TOTAL CHUNKS:", len(chunk_documents))
print("=" * 60)

print(chunk_documents)




############################################################## Code: Semantic + Sentence Overlap



from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

text = """Python is a programming language.
Python is widely used for AI and machine learning.
Python can be used to build APIs.
FastAPI is a popular Python framework.
FastAPI is commonly used for REST APIs.
RAG applications use Python for document processing.
Embeddings convert text into vectors.
Vector databases store and search embeddings.
"""

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name
)

semantic_splitter = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type="percentile"
)

documents = semantic_splitter.create_documents([text])

semantic_chunks = [doc.page_content for doc in documents]

overlap_sentences = 1

final_chunks = []

for i, chunk in enumerate(semantic_chunks):

    if i == 0:
        final_chunk = chunk

    else:
        previous_sentences = semantic_chunks[i - 1].split(". ")

        overlap_text = previous_sentences[-overlap_sentences:]

        final_chunk = ". ".join(overlap_text) + ". " + chunk

    final_chunks.append(final_chunk)

for i, chunk in enumerate(final_chunks, start=1):

    print("\n" + "=" * 60)
    print("CHUNK:", i)
    print("=" * 60)
    print(chunk)