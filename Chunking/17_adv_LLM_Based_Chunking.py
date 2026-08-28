from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from rapidocr_onnxruntime import RapidOCR
from pdf2image import convert_from_path
import os


# ============================================================
# 1. LOAD .ENV
# ============================================================

load_dotenv(
    "/home/suresh/Gen_AI_Practice/.env"
)

api_key = os.getenv("open_api_key")

# ============================================================
# 2. CHECK API KEY
# ============================================================

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not found in .env"
    )


# ============================================================
# 3. FILE INFORMATION
# ============================================================

file_path = (
    "/home/suresh/GenAI Class Tutorial/Gen AI_3Y/73202660-2026-06-10.pdf"
)

file_name = os.path.basename(
    file_path
)

file_type = "pdf"

document_id = "doc_id_001"


# ============================================================
# 4. INITIALIZE RAPIDOCR
# ============================================================

ocr_engine = RapidOCR()


# ============================================================
# 5. CONVERT PDF TO IMAGES
# ============================================================

pages = convert_from_path(file_path, dpi=300)


# ============================================================
# 6. RAPIDOCR
# ============================================================

page_documents = []


for page_number, page in enumerate(pages, start=1):

    result, _ = ocr_engine(page)

    page_text = ""


    if result:

        text_lines = []

        for line in result:
            print("line[1] : ",line)
            text_lines.append(line[1])

        page_text = "\n".join(text_lines)


    page_documents.append(
        {
            "page_number": page_number,
            "text": page_text
        }
    )


    # print("OCR Characters:", page_text)


# ============================================================
# 7. STRUCTURED LLM OUTPUT
# ============================================================

class Chunk(BaseModel):

    text: str = Field(
        description=(
            "Original text belonging "
            "to the logical chunk"
        )
    )

    section: str = Field(
        description=(
            "Logical BOL section name"
        )
    )


class ChunkOutput(BaseModel):

    chunks: list[Chunk]


# ============================================================
# 8. INITIALIZE LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key = api_key
)


structured_llm = (
    llm.with_structured_output(
        ChunkOutput
    )
)


# ============================================================
# 9. LLM-BASED CHUNKING
# ============================================================

chunk_documents = []


for page_document in page_documents:

    page_number = page_document[
        "page_number"
    ]

    page_text = page_document[
        "text"
    ]


    if not page_text.strip():

        continue


    prompt = f"""
You are an expert Bill of Lading
document chunking system.

Divide the OCR text into meaningful,
self-contained logical chunks.

This is a BOL document.

Rules:

1. Preserve the original OCR text.
2. Do not summarize.
3. Do not remove information.
4. Do not invent information.
5. Keep related fields together.
6. Keep Origin information together.
7. Keep Carrier information together.
8. Keep Destination information together.
9. Keep Bill Third Party information together.
10. Keep Shipment Information together.
11. Keep Freight Charges together.
12. Keep Signature information together.
13. Do not mix unrelated sections.
14. Each chunk must be useful for RAG.
15. Preserve values exactly as extracted by OCR.
16. Return only logical chunks.

OCR TEXT:

{page_text}
"""


    result = structured_llm.invoke(
        prompt
    )


# ============================================================
# 10. CREATE CHUNK DICTIONARIES
# ============================================================

    for chunk_index, chunk in enumerate(
        result.chunks,
        start=1
    ):

        chunk_text = chunk.text


        dict_chunk = {}


        dict_chunk["id"] = (
            f"id_{page_number:03d}_"
            f"{chunk_index:03d}"
        )


        dict_chunk["text"] = (
            chunk_text
        )


        dict_chunk["metadata"] = {

            "document_id": document_id,

            "file_name": file_name,

            "file_type": file_type,

            "page_number": page_number,

            "chunk_index": chunk_index,

            "chunk_method": (
                "llm_based_chunking"
            ),

            "section": chunk.section,

            "character_count": (
                len(chunk_text)
            ),

            "ocr_method": (
                "rapidocr"
            ),

            "ocr_used": True
        }


        chunk_documents.append(
            dict_chunk
        )


# ============================================================
# 11. DISPLAY FINAL CHUNKS
# ============================================================

print("\n")

print("=" * 70)

print(
    "FINAL LLM-BASED CHUNKS"
)

print("=" * 70)

print(
    "TOTAL CHUNKS:",
    len(chunk_documents)
)


for document in chunk_documents:

    print("\n")

    print("-" * 70)

    print(
        "ID:",
        document["id"]
    )

    print(
        "SECTION:",
        document["metadata"]["section"]
    )

    print(
        "PAGE:",
        document["metadata"]["page_number"]
    )

    print("-" * 70)

    print(
        document["text"]
    )