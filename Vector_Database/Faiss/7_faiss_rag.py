import os
import json
import faiss
import numpy as np
import pytesseract

from pdf2image import convert_from_path

from sentence_transformers import SentenceTransformer

from langchain_text_splitters import (RecursiveCharacterTextSplitter)


# =========================================================
# 1. CONFIGURATION
# =========================================================

pdf_path = "/home/suresh/Gen_AI_Practice/Generative-AI/Vector_Database/HR_Policy.pdf"

store_dir = "faiss_store"

os.makedirs(store_dir, exist_ok=True)


index_path = os.path.join(store_dir,"index.faiss")

metadata_path = os.path.join(store_dir, "metadata.json")

config_path = os.path.join(store_dir, "config.json")


# =========================================================
# 2. PDF → IMAGES
# =========================================================

pages = convert_from_path(pdf_path, dpi=250)


# =========================================================
# 3. OCR PAGE BY PAGE
# =========================================================

page_data = []


for page_no, image in enumerate(pages, start=1):

    text = pytesseract.image_to_string(image)

    text = text.strip()


    if text:

        page_data.append({

            "file_name": os.path.basename(pdf_path),

            "page_no": page_no,

            "text": text
        })


print("Total Pages:", len(page_data))


# =========================================================
# 4. RECURSIVE CHUNKER
# =========================================================

chunk_size = 1000

chunk_overlap = 200


splitter = RecursiveCharacterTextSplitter(

    chunk_size=chunk_size,

    chunk_overlap=chunk_overlap,

    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


# =========================================================
# 5. PAGE-WISE CHUNKING
# =========================================================

chunks = []


for page in page_data:

    page_no = page["page_no"]

    text = page["text"]


    page_chunks = splitter.split_text(text)


    for chunk_no, chunk_text in enumerate(page_chunks, start=1):

        chunks.append({

            "file_name": page["file_name"],

            "page_no": page_no,

            "chunk_no": chunk_no,

            "chunk_id": f"p{page_no}_c{chunk_no}",

            "text": chunk_text
        })


print("Total Chunks:", len(chunks))
print("Total Chunks:", chunks)


# =========================================================
# 6. EMBEDDING MODEL
# =========================================================

model_name = ("sentence-transformers/""all-MiniLM-L6-v2")


model = SentenceTransformer(model_name)


# =========================================================
# 7. CHUNKS → EMBEDDINGS
# =========================================================

texts = [chunk["text"] for chunk in chunks]


embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")


# =========================================================
# 8. NORMALIZE
# =========================================================

faiss.normalize_L2(embeddings)


# =========================================================
# 9. CREATE FAISS INDEX
# =========================================================

dimension = embeddings.shape[1]


base_index = faiss.IndexFlatIP(dimension)


index = faiss.IndexIDMap(base_index)


# =========================================================
# 10. CREATE DYNAMIC IDS
# =========================================================

start_id = 1001


ids = np.arange(start_id, start_id + len(chunks), dtype="int64")


# =========================================================
# 11. ADD VECTORS
# =========================================================

index.add_with_ids(embeddings, ids)


print("FAISS vectors:", index.ntotal)


# =========================================================
# 12. METADATA
# =========================================================

metadata = {}


for chunk, vector_id in zip(chunks, ids):

    metadata[str(vector_id)] = {

        "file_name": chunk["file_name"],

        "page_no": chunk["page_no"],

        "chunk_no": chunk["chunk_no"],

        "chunk_id": chunk["chunk_id"],

        "text": chunk["text"]
    }


# =========================================================
# 13. SAVE FAISS INDEX
# =========================================================

faiss.write_index(index, index_path)


print("FAISS index saved:", index_path)


# =========================================================
# 14. SAVE METADATA
# =========================================================

with open(metadata_path, "w", encoding="utf-8") as f:

    json.dump(metadata, f, indent=4, ensure_ascii=False)


print("Metadata saved:", metadata_path)


# =========================================================
# 15. SAVE CONFIGURATION
# =========================================================

config = {

    "embedding_model": model_name,

    "dimension": dimension,

    "chunk_size": chunk_size,

    "chunk_overlap": chunk_overlap,

    "distance": "Inner Product",

    "similarity": "Cosine Similarity",

    "total_vectors": index.ntotal
}


with open(config_path, "w", encoding="utf-8") as f:

    json.dump(config, f, indent=4)


print("Config saved:", config_path)


"""

                 INGESTION
                    │
                    ▼
             HR_Policy.pdf
                    │
                    ▼
              PDF → Image
                    │
                    ▼
                Tesseract
                    │
                    ▼
             Page-wise Text
                    │
                    ▼
        Recursive Character Splitter
        chunk_size = 500
        overlap    = 100
                    │
                    ▼
              Page Chunks
                    │
                    ▼
        SentenceTransformer
        all-MiniLM-L6-v2
                    │
                    ▼
              Embeddings
                    │
                    ▼
             Normalize L2
                    │
                    ▼
              FAISS Index
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    index.faiss         metadata.json
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
              PERSISTENCE

"""