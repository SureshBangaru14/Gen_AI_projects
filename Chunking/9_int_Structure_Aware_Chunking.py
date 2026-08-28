from langchain_text_splitters import MarkdownHeaderTextSplitter

text = """# HR Policy

## Leave Policy

Employees are entitled to annual leave.

### Annual Leave

Employees receive 20 days of annual leave.

### Sick Leave

Employees can take sick leave.

## Attendance Policy

Employees must follow working hours.
"""

headers_to_split_on = [
    ("#", "title"),
    ("##", "section"),
    ("###", "subsection")
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

documents = splitter.split_text(text)

print("TOTAL CHUNKS:", documents)


chunk_documents = []

for i, document in enumerate(documents, start=1):

    chunk = document.page_content

    dict_chunk = {}

    dict_chunk["id"] = f"id_{i:03d}"

    dict_chunk["text"] = chunk

    dict_chunk["metadata"] = {
        "document_id": "doc_id_001",
        "file_name": "hr_policy.md",
        "file_type": "md",
        "page_number": 1,
        "chunk_index": i,
        "chunk_method": "structure_aware_chunking",
        "character_count": len(chunk),
        "title": document.metadata.get("title"),
        "section": document.metadata.get("section"),
        "subsection": document.metadata.get("subsection"),
        "ocr_used": False
    }

    chunk_documents.append(dict_chunk)

print(chunk_documents)





################################################# LangChain + Excel Structure-Aware Chunking


import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

file_name = "employees.xlsx"
file_type = "xlsx"

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name)

excel_file = pd.ExcelFile(file_name)

chunk_size = 200
chunk_overlap = 50

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    keep_separator=False
)

chunk_documents = []

global_chunk_index = 0

for sheet_name in excel_file.sheet_names:

    df = pd.read_excel(file_name, sheet_name=sheet_name)

    print("=" * 60)
    print("SHEET:", sheet_name)
    print("ROWS:", len(df))
    print("COLUMNS:", list(df.columns))
    print("=" * 60)

    rows_text = []

    for row_index, row in df.iterrows():

        row_text = f"Row: {row_index + 2}\n"

        for column in df.columns:

            row_text += f"{column}: {row[column]}\n"

        rows_text.append(row_text)

    structured_text = "\n".join(rows_text)

    chunks_text = text_splitter.split_text(structured_text)

    for chunk in chunks_text:

        global_chunk_index += 1

        embedding = embedding_model.encode(chunk)

        token_count = len(
            embedding_model.tokenizer.encode(
                chunk,
                add_special_tokens=False
            )
        )

        dict_chunk = {}

        dict_chunk["id"] = f"id_{global_chunk_index:03d}"

        dict_chunk["text"] = chunk

        # dict_chunk["embedding"] = embedding.tolist()

        dict_chunk["metadata"] = {
            "document_id": "doc_id_001",
            "file_name": file_name,
            "file_type": file_type,
            "sheet_name": sheet_name,
            "chunk_index": global_chunk_index,
            "chunk_method": "structure_aware_excel_chunking",
            "splitter": "RecursiveCharacterTextSplitter",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "character_count": len(chunk),
            "token_count": token_count,
            "columns": list(df.columns),
            "embedding_model": embedding_model_name,
            "embedding_dimension": len(embedding),
            "ocr_used": False
        }

        chunk_documents.append(dict_chunk)

print("\n" + "=" * 60)
print("TOTAL CHUNKS:",chunk_documents)
