import chromadb

client = chromadb.PersistentClient(path="./chroma_db")



print("========================Cosine similarity=====================================")
# Cosine similarity
collection = client.get_or_create_collection(name="transportation", configuration={
    "hnsw": {"space": "cosine"}})

results = collection.query(query_texts=["Which vehicle carries passengers?"],
                           n_results=3)

print(results)
print("#"*50)

results = collection.query(query_texts=["Which vehicle carries passengers?"],
                           n_results=3,
                           where={
                               "vehicle": "bus"})

print(results)



print("========================Euclidean distance=====================================")
# Euclidean distance
collection = client.get_or_create_collection(name="transportation", configuration={
    "hnsw": {"space": "l2"}})

results = collection.query(query_texts=["Which vehicle carries passengers?"],
                           n_results=3)

print(results)
print("#"*50)

results = collection.query(query_texts=["Which vehicle carries passengers?"],
                           n_results=3,
                           where={
                               "vehicle": "bus"})

print(results)



print("========================Inner Product dot product=====================================")
# Euclidean distance
collection = client.get_or_create_collection(name="transportation", configuration={
    "hnsw": {"space": "ip"}})

results = collection.query(query_texts=["Which vehicle carries passengers?"],
                           n_results=3)

print(results)
print("#"*50)

results = collection.query(query_texts=["Which vehicle carries passengers?"],
                           n_results=3,
                           where={
                               "vehicle": "bus"})

print(results)