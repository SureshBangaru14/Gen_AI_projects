text = """Python is a programming language.
        It is widely used for AI and machine learning.

        Python can be used to build APIs.
        FastAPI is a popular Python framework.

        RAG applications use Python for document processing.
        Embeddings convert text into vectors.

        Vector databases store and search embeddings."""


############################################ Basic Python Code


def paragraph_chunking(text):

    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if paragraph:
            chunks.append(paragraph)

    return chunks


chunks_text = paragraph_chunking(text)

print("chunks_text : ",chunks_text)

############################################ Using langchain


from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=6, chunk_overlap=3)


chunks_text = text_splitter.split_text(text)
print("#"*30)
print("chunks_text : ",chunks_text)






############################################ Using langchain Production Version

# You can combine paragraph boundaries with a token limit:

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=6, chunk_overlap=3, length_function=len)

chunks_text = text_splitter.split_text(text)
print("*"*30)
print("chunks_text : ",chunks_text)





######################################################### Using langchain Production Version chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


file_name = "normal_text.txt"

file_type = "txt"


chunk_size = 6

chunk_overlap = 2


embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(embedding_model_name)


text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=chunk_size, chunk_overlap=chunk_overlap, 
                                               length_function=len)

chunks_text = text_splitter.split_text(text)


print("=" * 50)

print("TOTAL CHUNKS:", len(chunks_text))

print("=" * 50)


# ============================================================
# CREATE CHUNK DOCUMENTS
# ============================================================

chunk_documents = []


for i, chunk in enumerate(chunks_text, start=1):

    dict_chunk = {}

    # --------------------------------------------------------
    # CREATE EMBEDDING FOR CURRENT CHUNK
    # --------------------------------------------------------

    embedding = embedding_model.encode(chunk)
    
    # --------------------------------------------------------
    # Chunk ID
    # --------------------------------------------------------

    dict_chunk["id"] = f"id_{i:03d}"


    # --------------------------------------------------------
    # Actual chunk text
    # --------------------------------------------------------

    dict_chunk["text"] = chunk


    # # Embedding
    # dict_chunk["embedding"] = embedding.tolist()


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    dict_chunk["metadata"] = {

        "document_id": f"doc_id_{i:03d}",

        "file_name": file_name,

        "file_type": file_type,

        "page_number": 1,

        "chunk_index": i,

        "chunk_method": "paragraph_based_chunking",

        "chunk_size": chunk_size,

        "chunk_overlap": chunk_overlap,

        "character_count": len(chunk),

        "token_count": len(embedding_model.tokenizer.encode(chunk, add_special_tokens=False)),
        
        "embedding_model": embedding_model_name,

        "ocr_used": False
    }


    # --------------------------------------------------------
    # Add dictionary to list
    # --------------------------------------------------------

    chunk_documents.append(
        dict_chunk
    )

print("&"*50)
print("chunk_documents : ",chunk_documents)