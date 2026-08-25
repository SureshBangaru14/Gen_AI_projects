text = """

            Python is a programming language.
            It is widely used for AI and machine learning.
            Python can be used to build APIs.
            FastAPI is a popular Python framework.
            RAG applications use Python for document processing.
            Embeddings convert text into vectors.
            Vector databases store and search embeddings.

            """


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

from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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
    dict_chunk["metadata"]["chunk_method"] = "token_based_chunking"
    dict_chunk["metadata"]["chunk_size"] = chunk_size
    dict_chunk["metadata"]["chunk_overlap"] = chunk_overlap

    # Add dictionary to new list
    chunk_documents.append(dict_chunk)

print("chunk_documents ",chunk_documents)