from pdf2image import convert_from_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pytesseract
import chromadb

PDF_PATH = "/home/suresh/Gen_AI_Practice/Generative-AI/Vector_Database/HR_Policy.pdf"

pages = convert_from_path(PDF_PATH, dpi=300)

model = SentenceTransformer("all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="hr_policy2")

for page_no, page_image in enumerate(pages, 1):

    print("\n" + "=" * 80)
    print("PAGE:", page_no)
    print("=" * 80)

    page_text = pytesseract.image_to_string(page_image, config="--psm 3").strip()

    if not page_text:
        print("No text found")
        continue

    print("\nComplete Page Text:")
    print(page_text)

    chunks = splitter.split_text(page_text)

    print("\nNumber of chunks:", len(chunks))

    for chunk_no, chunk in enumerate(chunks, 1):

        chunk_id = f"hr_policy_p{page_no}_c{chunk_no}"

        embedding = model.encode(chunk).tolist()

        metadata = {
            "file_name": "HR_Policy.pdf",
            "page_no": page_no,
            "chunk_no": chunk_no,
            "chunk_type": "recursive",
            "document_type": "HR Policy"
        }

        collection.upsert(
            ids=[chunk_id],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[metadata]
        )

        print("\nChunk:", chunk_id)
        print(chunk)

print("\n" + "=" * 80)
print("Collection:", collection.name)
print("Total records:", collection.count())