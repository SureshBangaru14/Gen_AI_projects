from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from sentence_transformers import SentenceTransformer

code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class EmployeeModel:

    def __init__(self, data):
        self.data = data

    def preprocess(self):
        self.data = self.data.dropna()
        return self.data

    def train(self):
        X = self.data[["age", "experience"]]
        y = self.data["salary"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        return model

def calculate_average(values):

    return sum(values) / len(values)
"""

file_name = "employee_model.py"
file_type = "py"

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(embedding_model_name)

chunk_size = 500
chunk_overlap = 100

splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

chunks_text = splitter.split_text(code)

chunk_documents = []

for i, chunk in enumerate(chunks_text, start=1):

    token_count = len(embedding_model.tokenizer.encode(chunk, add_special_tokens=False))

    dict_chunk = {}

    dict_chunk["id"] = f"id_{i:03d}"

    dict_chunk["text"] = chunk

    dict_chunk["metadata"] = {
        "document_id": "doc_id_001",
        "file_name": file_name,
        "file_type": file_type,
        "page_number": 1,
        "chunk_index": i,
        "chunk_method": "code_chunking",
        "programming_language": "python",
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