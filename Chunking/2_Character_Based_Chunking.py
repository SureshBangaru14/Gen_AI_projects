text = """

            Python is a programming language.
            It is widely used for AI and machine learning.
            Python can be used to build APIs.
            FastAPI is a popular Python framework.
            RAG applications use Python for document processing.
            Embeddings convert text into vectors.
            Vector databases store and search embeddings.

            """


############################# Python — Basic Character-Based Version

def fixed_size_chunking(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):
        
        print("i",i)

        chunk = text[i:i+chunk_size]
        print(chunk)

        chunks.append(chunk)

    return chunks


chunks = fixed_size_chunking(text, chunk_size=100)

print("chunks ",chunks)



##################################### Fixed-Size Chunking With Overlap


def fixed_size_chunking_1(text, chunk_size=500, chunk_overlap = 50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


chunks = fixed_size_chunking_1(text, chunk_size=100, chunk_overlap = 50)

print("chunks ",chunks)




############################################ Token-Based Fixed-Size Chunking


import tiktoken

def token_fixed_chunking(text, tokenizer, chunk_size=500, overlap=50):

    tokens = tokenizer.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = start + chunk_size

        chunk_tokens = tokens[start:end]

        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


# Create tokenizer
"""cl100k_base
│
├── cl    cl → part of OpenAI's naming for this encoding.
├── 100k  100k → refers roughly to a vocabulary size of around 100,000 tokens.
└── base  base → it's the base tokenizer encoding, rather than a model itself.


"""
tokenizer = tiktoken.get_encoding("cl100k_base")


# Create chunks
chunks = token_fixed_chunking(text, tokenizer, chunk_size=50, overlap=10)

print("token based ",chunks)



##################################### LangChain Example



chunk_size = 100
chunk_overlap = 50

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

chunks_text = text_splitter.split_text(text)

chunk_documents = []

for i, chunk in enumerate(chunks_text, start=1):

    dict_chunk = {}

    # Chunk ID
    dict_chunk["id"] = f"chunk_{i}"

    # Actual chunk text
    dict_chunk["text"] = chunk

    # Metadata
    dict_chunk["metadata"] = {}

    dict_chunk["metadata"]["document_id"] = f"doc_id_{i}"
    dict_chunk["metadata"]["file_type"] = "text"
    dict_chunk["metadata"]["chunk_index"] = i
    dict_chunk["metadata"]["chunk_method"] = "character_chunking"
    dict_chunk["metadata"]["chunk_size"] = chunk_size
    dict_chunk["metadata"]["chunk_overlap"] = chunk_overlap

    # Add dictionary to new list
    chunk_documents.append(dict_chunk)

print("chunk_documents ",chunk_documents)