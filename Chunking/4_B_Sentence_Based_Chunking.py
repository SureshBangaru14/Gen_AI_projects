text = """

            Python is a programming language.
            It is widely used for AI and machine learning.
            Python can be used to build APIs.
            FastAPI is a popular Python framework.
            RAG applications use Python for document processing.
            Embeddings convert text into vectors.
            Vector databases store and search embeddings.

            """


############################################ Basic Python Code


import re


def sentence_based_chunking(text, sentences_per_chunk=3):

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        
        print("i : ",i)
        print(i + sentences_per_chunk)
        chunk = " ".join(sentences[i:i + sentences_per_chunk])

        chunks.append(chunk)

    return chunks

chunks = sentence_based_chunking(text, sentences_per_chunk=2)

print("chunks ",chunks)




############################################ Using NLTK


import nltk

nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize


def sentence_based_chunking_1(text, sentences_per_chunk=3):

    sentences = sent_tokenize(text)

    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):

        chunk = " ".join(sentences[i:i + sentences_per_chunk])

        chunks.append(chunk)

    return chunks


chunks = sentence_based_chunking_1(text, sentences_per_chunk=2)
print("chunks_1 ",chunks)



############################################ Sentence-Based Chunking With Token Limit

import nltk
from transformers import AutoTokenizer


# Download NLTK sentence tokenizer data
nltk.download("punkt")
nltk.download("punkt_tab")


# --------------------------------------------------
# 2. Split the text into sentences
# --------------------------------------------------

sentences = nltk.sent_tokenize(text)

print("Sentences:", sentences)

# --------------------------------------------------
# 3. Create a tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")



def sentence_token_chunking(sentences, max_tokens, tokenizer):
    
    chunks = []
    chunk = []
    token_count = 0

    for sentence in sentences:
        tokens = len(tokenizer.encode(sentence))

        if token_count + tokens <= max_tokens:
            chunk.append(sentence)
            token_count += tokens
        else:
            chunks.append(" ".join(chunk))
            chunk = [sentence]
            token_count = tokens

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

max_tokens = 20
chunk = sentence_token_chunking(sentences, max_tokens,tokenizer)

print("chunk_2", chunk)



############################################ LangChain packages OLD METHOD


from langchain_text_splitters import SentenceTransformersTokenTextSplitter

# --------------------------------------------------
# 2. Chunking configuration
# --------------------------------------------------

tokens_per_chunk = 50
chunk_overlap = 10


# --------------------------------------------------
# 3. Sentence Transformers Token Splitter
# --------------------------------------------------

splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=tokens_per_chunk, chunk_overlap=chunk_overlap, 
                                                 model_name="sentence-transformers/all-MiniLM-L6-v2")


# --------------------------------------------------
# 4. Create chunks
# --------------------------------------------------

chunks_text = splitter.split_text(text)
print("chunks_text ",chunks_text)

# --------------------------------------------------
# 5. Store chunks
# --------------------------------------------------

chunk_documents = []


for i, chunk in enumerate(chunks_text, start=1):

    dict_chunk = {}

    # Chunk ID
    dict_chunk["id"] = (f"id_{i:03d}")

    # Actual chunk text
    dict_chunk["text"] = chunk

    # Metadata
    dict_chunk["metadata"] = {

        "document_id": (f"doc_id_{i:03d}"),

        "file_name": "normal text",

        "file_type": "txt",

        "chunk_index": i,

        "chunk_method": ("sentence_transformers_token_chunking"),

        "tokens_per_chunk": tokens_per_chunk,

        "chunk_overlap": chunk_overlap,

        "embedding_model": ("sentence-transformers/""all-MiniLM-L6-v2"),

    }

    # Add dictionary
    chunk_documents.append(dict_chunk)
    
print(chunk_documents)



############################################ LangChain packages NLTK LATEST

from langchain_text_splitters import NLTKTextSplitter

import nltk

nltk.download("punkt_tab")

chunk_size = 4
chunk_overlap = 1

splitter = NLTKTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

chunks_text = splitter.split_text(text)

chunk_documents = []

for i, chunk in enumerate(chunks_text, start=1):

    dict_chunk = {}

    # Chunk ID
    dict_chunk["id"] = (f"id_{i:03d}")

    # Actual chunk text
    dict_chunk["text"] = chunk

    # Metadata
    dict_chunk["metadata"] = {

        "document_id": (f"doc_id_{i:03d}"),

        "file_name": "normal text",

        "file_type": "txt",

        "chunk_index": i,

        "chunk_method": ("sentence_transformers_token_chunking"),

        "tokens_per_chunk": chunk_size,

        "chunk_overlap": chunk_overlap,

        "embedding_model": ("sentence-transformers/""all-MiniLM-L6-v2"),

    }

    # Add dictionary
    chunk_documents.append(dict_chunk)

print("*"*50)
print(chunk_documents)






############################################ SentenceTransformer

from sentence_transformers import SentenceTransformer


file_name = "normal_text.txt"
file_type = "txt"



tokens_per_chunk = 10
chunk_overlap = 3


embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(embedding_model_name)



tokenizer = model.tokenizer



token_ids = tokenizer.encode(text, add_special_tokens=False)


print("\nALL TOKENS:")

for index, token_id in enumerate(token_ids, start=1):

    token_text = tokenizer.decode([token_id])

    print(
        f"Token {index:03d} | "
        f"ID: {token_id:<8} | "
        f"Text: {repr(token_text)}"
    )




step = tokens_per_chunk - chunk_overlap

print("\n" + "=" * 70)
print("CHUNKING CONFIGURATION")
print("=" * 70)

print("Tokens per Chunk :", tokens_per_chunk)
print("Chunk Overlap    :", chunk_overlap)
print("Step Size        :", step)



chunks_text = []

for start in range(0, len(token_ids), step):

    end = start + tokens_per_chunk

    chunk_token_ids = token_ids[start:end]

    if not chunk_token_ids:
        break

    chunk_text = tokenizer.decode(
        chunk_token_ids,
        skip_special_tokens=True
    )

    chunks_text.append(chunk_text)



chunk_documents = []

for i, chunk in enumerate(chunks_text, start=1):

    dict_chunk = {}

    # Chunk ID
    dict_chunk["id"] = f"id_{i:03d}"

    # Actual chunk text
    dict_chunk["text"] = chunk

    # Metadata
    dict_chunk["metadata"] = {
        "document_id": f"doc_id_{i:03d}",
        "file_name": file_name,
        "file_type": file_type,
        "page_number": 1,
        "chunk_index": i,
        "chunk_method": "token_based_chunking",
        "tokens_per_chunk": tokens_per_chunk,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model_name,
        "ocr_used": False
    }

    chunk_documents.append(dict_chunk)


print("#"*50)
print(chunk_documents)

