from langchain_text_splitters import HTMLHeaderTextSplitter

html = """
<h1>FastAPI Documentation</h1>

<h2>Introduction</h2>

<p>FastAPI is a modern Python web framework for building APIs.</p>

<h2>Installation</h2>

<p>Install FastAPI using pip.</p>

<pre>pip install fastapi</pre>

<h2>Creating an API</h2>

<p>Create a Python application using FastAPI.</p>

<h3>GET Request</h3>

<p>You can create a GET endpoint using the following code.</p>

<h2>Troubleshooting</h2>

<p>Check the Python environment if the application does not start.</p>
"""

file_name = "fastapi.html"
file_type = "html"

headers_to_split_on = [
    ("h1", "title"),
    ("h2", "section"),
    ("h3", "subsection")
]

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

documents = splitter.split_text(html)

chunk_documents = []

for i, document in enumerate(documents, start=1):

    chunk = document.page_content

    dict_chunk = {}

    dict_chunk["id"] = f"id_{i:03d}"
    dict_chunk["text"] = chunk

    dict_chunk["metadata"] = {
        "document_id": f"docid_{i:03d}",
        "file_name": file_name,
        "file_type": file_type,
        "page_number": 1,
        "chunk_index": i,
        "chunk_method": "html_chunking",
        "character_count": len(chunk),
        "title": document.metadata.get("title"),
        "section": document.metadata.get("section"),
        "subsection": document.metadata.get("subsection"),
        "ocr_used": False
    }

    chunk_documents.append(dict_chunk)

print("=" * 60)
print("TOTAL CHUNKS:",(chunk_documents))
print("=" * 60)





################################# html + recursive 


from langchain_text_splitters import (HTMLHeaderTextSplitter,RecursiveCharacterTextSplitter)

headers_to_split_on = [
    ("h1", "title"),
    ("h2", "section"),
    ("h3", "subsection")
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

documents = html_splitter.split_text(html)

recursive_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""],
                                                    chunk_size=500, chunk_overlap=100, length_function=len, 
                                                    keep_separator=False)

chunks = recursive_splitter.split_documents(documents)

print("#"*150)

print(chunks)