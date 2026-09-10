import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="transportation")

results = collection.query(query_texts=["Which vehicle carries passengers?"],
                           n_results=3)

print(results)