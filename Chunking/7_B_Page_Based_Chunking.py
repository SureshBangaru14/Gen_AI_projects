############################################  Page level chunking

from pdf2image import convert_from_path
import pytesseract
from sentence_transformers import SentenceTransformer


pdf_path = "/home/suresh/Gen_AI_Practice/Generative-AI/Chunking/resume-88.pdf"

file_name = "resume-88.pdf"
file_type = "pdf"

overlap_tokens = 150

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(embedding_model_name)


pages = convert_from_path(pdf_path, dpi=300)

print("=" * 60)
print("TOTAL PAGES:", len(pages))
print("=" * 60)


page_texts = []


for page_number, page_image in enumerate(pages, start=1):

    page_text = pytesseract.image_to_string(page_image).strip()

    page_texts.append(page_text)

    print("Page", page_number, "OCR characters:", len(page_text))


chunk_documents = []


for i, page_text in enumerate(page_texts, start=1):

    if not page_text:
        continue


    if i == 1:

        chunk_text = page_text

        overlap_used = 0

        overlap_from_page = None


    else:

        previous_page_text = page_texts[i - 2]

        previous_tokens = embedding_model.tokenizer.encode(previous_page_text, add_special_tokens=False)

        overlap_token_ids = previous_tokens[-overlap_tokens:]

        overlap_text = embedding_model.tokenizer.decode(overlap_token_ids)

        chunk_text = overlap_text + "\n\n" + page_text

        overlap_used = len(overlap_token_ids)

        overlap_from_page = i - 1


    embedding = embedding_model.encode(chunk_text)


    token_count = len(embedding_model.tokenizer.encode(chunk_text, add_special_tokens=False))


    dict_chunk = {}

    dict_chunk["id"] = f"id_{i:03d}"

    dict_chunk["text"] = chunk_text

    # dict_chunk["embedding"] = embedding.tolist()

    dict_chunk["metadata"] = {
        "document_id": "doc_id_001",
        "file_name": file_name,
        "file_type": file_type,
        "page_number": i,
        "chunk_index": i,
        "chunk_method": "page_based_chunking",
        "overlap_method": "token_based_overlap",
        "overlap_tokens": overlap_used,
        "overlap_from_page": overlap_from_page,
        "token_count": token_count,
        "character_count": len(chunk_text),
        "embedding_model": embedding_model_name,
        "embedding_dimension": len(embedding),
        "ocr_used": True
    }

    chunk_documents.append(dict_chunk)


print("&" * 50)
print("chunk_documents : ",chunk_documents)
