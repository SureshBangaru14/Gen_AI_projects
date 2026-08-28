from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os


# ============================================================
# 1. LOAD .ENV
# ============================================================

load_dotenv("/home/suresh/Gen_AI_Practice/.env")


# ============================================================
# 2. CHECK API KEY
# ============================================================

api_key = os.getenv("open_api_key")

if not api_key:
    raise ValueError(
        "open_api_key not found in .env"
    )


# ============================================================
# 3. INPUT DOCUMENT
# ============================================================

text = """
SUMMARY

Experienced MLOps Engineer and Data Scientist
with experience in Python, machine learning,
Docker, Kubernetes and AWS.

SKILLS

Python, SQL, Machine Learning, NLP,
Docker, Kubernetes, AWS, GCP,
TensorFlow, scikit-learn, Git.

WORK EXPERIENCE

MLOps Engineer - ABC Technologies
2021 - Present

Developed machine learning pipelines using Python.
Built CI/CD pipelines for machine learning applications.
Deployed machine learning models using Docker and Kubernetes.
Managed AWS cloud infrastructure.
Worked on MLOps automation and model deployment.

Data Scientist - XYZ Company
2019 - 2021

Developed NLP applications using Python.
Created machine learning models.
Built data processing pipelines.
Performed data analysis and visualization.

EDUCATION

MSc in Data Science
Coventry University
2017 - 2018

PROJECTS

Resume Analyzer

Developed an NLP-based resume matching system
using Python, Sentence Transformers and ChromaDB.
"""


# ============================================================
# 4. FILE INFORMATION
# ============================================================

file_name = "resume.pdf"

file_type = "pdf"

document_id = "doc_id_001"


# ============================================================
# 5. BASE CHUNKING
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

base_chunks = text_splitter.split_text(text)


print("=" * 70)

print("BASE CHUNKS")

print("=" * 70)

print("TOTAL BASE CHUNKS:", len(base_chunks))


for i, chunk in enumerate(
    base_chunks,
    start=1
):

    print("\n" + "-" * 70)

    print("BASE CHUNK:", i)

    print("-" * 70)

    print(chunk)


# ============================================================
# 6. STRUCTURED OUTPUT
# ============================================================

class ChunkDecision(BaseModel):

    chunks: list[str] = Field(
        description="Logical self-contained document chunks"
    )


# ============================================================
# 7. OPENAI LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key 
)


# ============================================================
# 8. STRUCTURED LLM
# ============================================================

structured_llm = llm.with_structured_output(
    ChunkDecision
)


# ============================================================
# 9. AGENTIC CHUNKING
# ============================================================

agentic_chunks = []


for i, base_chunk in enumerate(
    base_chunks,
    start=1
):

    prompt = f"""
You are an expert document chunking agent.

Your task is to divide the provided document content
into meaningful, self-contained chunks.

Rules:

1. Preserve all original information.
2. Do not summarize.
3. Do not remove information.
4. Do not invent information.
5. Keep related information together.
6. Keep a person's job title with the job description.
7. Keep company name, job title and responsibilities together.
8. Keep education information together.
9. Keep project information together.
10. Do not mix unrelated sections.
11. Each chunk should be useful for semantic search.
12. Return only the final chunks.

DOCUMENT CONTENT:

{base_chunk}
"""

    result = structured_llm.invoke(prompt)

    agentic_chunks.extend(
        result.chunks
    )


# ============================================================
# 10. DISPLAY AGENTIC CHUNKS
# ============================================================

print("\n" + "=" * 70)

print("AGENTIC CHUNKS")

print("=" * 70)

print(
    "TOTAL AGENTIC CHUNKS:",
    len(agentic_chunks)
)


for i, chunk in enumerate(
    agentic_chunks,
    start=1
):

    print("\n" + "-" * 70)

    print("AGENTIC CHUNK:", i)

    print("-" * 70)

    print(chunk)

    print(
        "CHARACTERS:",
        len(chunk)
    )


# ============================================================
# 11. CREATE FINAL DICTIONARIES
# ============================================================

chunk_documents = []


for i, chunk in enumerate(
    agentic_chunks,
    start=1
):

    dict_chunk = {}

    dict_chunk["id"] = f"id_{i:03d}"

    dict_chunk["text"] = chunk

    dict_chunk["metadata"] = {

        "document_id": document_id,

        "file_name": file_name,

        "file_type": file_type,

        "page_number": None,

        "chunk_index": i,

        "chunk_method": "agentic_chunking",

        "base_chunk_method": "recursive_chunking",

        "base_chunk_size": 700,

        "base_chunk_overlap": 100,

        "character_count": len(chunk),

        "ocr_used": True
    }

    chunk_documents.append(
        dict_chunk
    )


# ============================================================
# 12. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("FINAL CHUNK DOCUMENTS")

print("=" * 70)


for document in chunk_documents:

    print("\n")

    print(document)