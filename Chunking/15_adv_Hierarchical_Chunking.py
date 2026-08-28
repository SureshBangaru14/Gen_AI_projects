from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Python is a programming language.

Python is widely used for artificial intelligence
and machine learning.

Python provides libraries such as TensorFlow,
PyTorch and scikit-learn.

FastAPI is a Python framework for building APIs.

FastAPI can be used to create REST APIs.

FastAPI provides automatic API documentation.

ChromaDB is a vector database.

ChromaDB can store and search embeddings.
"""

file_name = "normal_text.txt"
file_type = "txt"

parent_chunk_size = 500
parent_chunk_overlap = 50

child_chunk_size = 200
child_chunk_overlap = 30

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=parent_chunk_size,
    chunk_overlap=parent_chunk_overlap
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=child_chunk_size,
    chunk_overlap=child_chunk_overlap
)

parent_chunks = parent_splitter.split_text(text)

chunk_documents = []

for parent_index, parent_chunk in enumerate(parent_chunks, start=1):

    parent_id = f"parent_{parent_index:03d}"

    child_chunks = child_splitter.split_text(parent_chunk)

    for child_index, child_chunk in enumerate(child_chunks, start=1):

        child_id = f"{parent_id}_child_{child_index:03d}"

        dict_chunk = {}

        dict_chunk["id"] = child_id

        dict_chunk["text"] = child_chunk

        dict_chunk["metadata"] = {
            "document_id": "doc_id_001",
            "file_name": file_name,
            "file_type": file_type,
            "parent_id": parent_id,
            "parent_index": parent_index,
            "child_id": child_id,
            "child_index": child_index,
            "chunk_method": "parent_child_chunking",
            "parent_chunk_size": parent_chunk_size,
            "parent_chunk_overlap": parent_chunk_overlap,
            "child_chunk_size": child_chunk_size,
            "child_chunk_overlap": child_chunk_overlap,
            "character_count": len(child_chunk),
            "parent_character_count": len(parent_chunk),
            "ocr_used": False
        }

        chunk_documents.append(dict_chunk)

print("=" * 60)
print("TOTAL CHILD CHUNKS:", len(chunk_documents))
print("=" * 60)

for item in chunk_documents:

    print("\n" + "-" * 60)
    print("ID:", item["id"])
    print("TEXT:", item["text"])
    print("PARENT ID:", item["metadata"]["parent_id"])
    print("CHILD INDEX:", item["metadata"]["child_index"])