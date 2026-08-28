from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

text = """Employees are entitled to 20 days of annual leave.
Employees must submit leave requests through the HR portal.
Managers must approve leave requests before the leave begins.

Medical leave requires appropriate documentation.
Employees should inform their manager as soon as possible.

Remote work is available to eligible employees.
Employees must follow the company's remote work policy."""

file_name = "HR_Policy.pdf"
file_type = "pdf"

document_id = "doc_id_001"
page_number = 10
section = "HR Policy"

chunk_size = 200
chunk_overlap = 50

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(
    embedding_model_name
)

document = Document(
    page_content=text,
    metadata={
        "document_id": document_id,
        "file_name": file_name,
        "file_type": file_type,
        "page_number": page_number,
        "section": section
    }
)

print("*"*100)
print(document)
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    keep_separator=False
)

chunks = text_splitter.split_documents([document])

print("%"*100)
print("chunks : ",chunks)

chunk_documents = []

for i, chunk in enumerate(chunks, start=1):

    chunk_text = chunk.page_content

    embedding = embedding_model.encode(chunk_text)

    token_count = len(
        embedding_model.tokenizer.encode(
            chunk_text,
            add_special_tokens=False
        )
    )

    dict_chunk = {}

    dict_chunk["id"] = f"id_{i:03d}"

    dict_chunk["text"] = chunk_text

    # dict_chunk["embedding"] = embedding.tolist()

    dict_chunk["metadata"] = {
        "document_id": chunk.metadata["document_id"],
        "file_name": chunk.metadata["file_name"],
        "file_type": chunk.metadata["file_type"],
        "page_number": chunk.metadata["page_number"],
        "section": chunk.metadata["section"],
        "chunk_index": i,
        "chunk_method": "metadata_aware_chunking",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "character_count": len(chunk_text),
        "token_count": token_count,
        "embedding_model": embedding_model_name,
        "embedding_dimension": len(embedding),
        "ocr_used": False
    }

    chunk_documents.append(dict_chunk)


print("#"*100)
print(chunk_documents)