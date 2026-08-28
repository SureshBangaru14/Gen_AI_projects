from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. INPUT TEXT
# ============================================================

text = """

            Python is a programming language.
            It is widely used for AI and machine learning.
            
            Python can be used to build APIs.
            
            FastAPI is a popular Python framework.
            RAG applications use Python for document processing.
            Embeddings convert text into vectors.
            Vector databases store and search embeddings.

            """

# ============================================================
# 2. FILE INFORMATION
# ============================================================

file_name = "normal_text.txt"
file_type = "txt"


# ============================================================
# 3. CHUNKING CONFIGURATION
# ============================================================

chunk_size = 200
chunk_overlap = 50


# ============================================================
# 4. EMBEDDING MODEL
# ============================================================

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(embedding_model_name)


# ============================================================
# 5. RECURSIVE CHUNKING
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    keep_separator=False
)


# ============================================================
# 6. CREATE CHUNKS
# ============================================================

chunks_text = text_splitter.split_text(text)


print("=" * 60)
print("TOTAL CHUNKS:", len(chunks_text))
print("=" * 60)


# ============================================================
# 7. DISPLAY CHUNKS
# ============================================================

for i, chunk in enumerate(chunks_text, start=1):

    print("\n" + "-" * 60)

    print("CHUNK:", i)

    print("-" * 60)

    print(chunk)

    print("CHARACTERS:", len(chunk))


# ============================================================
# 8. CREATE DICTIONARY + EMBEDDING
# ============================================================

chunk_documents = []


for i, chunk in enumerate(chunks_text, start=1):

    # Create embedding
    # embedding = embedding_model.encode(chunk)

    # Token count
    token_count = len(
        embedding_model.tokenizer.encode(
            chunk,
            add_special_tokens=False
        )
    )

    # Create dictionary
    dict_chunk = {}

    dict_chunk["id"] = f"id_{i:03d}"

    dict_chunk["text"] = chunk

    # dict_chunk["embedding"] = embedding.tolist()

    dict_chunk["metadata"] = {
        "document_id": "doc_id_001",
        "file_name": file_name,
        "file_type": file_type,
        "page_number": 1,
        "chunk_index": i,
        "chunk_method": "recursive_chunking",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "character_count": len(chunk),
        "token_count": token_count,
        "embedding_model": embedding_model_name,
        # "embedding_dimension": len(embedding),
        "ocr_used": False
    }

    chunk_documents.append(dict_chunk)


print("&"*50)
print("chunk_documents : ",chunk_documents)

