from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import pytesseract
import chromadb

PDF_PATH = "/home/suresh/Gen_AI_Practice/Generative-AI/Vector_Database/HR_Policy.pdf"

pages = convert_from_path(PDF_PATH, dpi=300)

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="hr_policy")

for page_no, page_image in enumerate(pages, 1):

    print("\n" + "=" * 80)
    print("PAGE:", page_no)
    print("=" * 80)

    page_text = pytesseract.image_to_string(page_image, config="--psm 6")

    page_text = page_text.strip()

    print(page_text)

    if not page_text:
        print("No text found")
        continue

    chunk_id = f"hr_policy_p{page_no}"

    embedding = model.encode(page_text).tolist()

    metadata = {
        "file_name": "HR_Policy.pdf",
        "page_no": page_no,
        "chunk_type": "page",
        "document_type": "HR Policy"
    }

    collection.upsert(
        ids=[chunk_id],
        documents=[page_text],
        embeddings=[embedding],
        metadatas=[metadata]
    )

    print("\nStored:", chunk_id)
    print("Characters:", len(page_text))
    print("Embedding dimension:", len(embedding))

print("\nTotal records:", collection.count())