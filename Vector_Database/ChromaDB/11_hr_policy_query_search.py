import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="hr_policy", configuration={
    "hnsw": {"space": "l2"}})

results = collection.query(query_texts=["How many hours should employees work per day?"],
                           n_results=3, include=[
                                                    "documents",
                                                    "metadatas",
                                                    "distances",
                                                    "embeddings"
                                                ]
                                            )

print(results)

for i in range(len(results["ids"][0])):

    print("\n" + "-" * 80)

    print("Rank      :", i + 1)

    print("ID        :", results["ids"][0][i])

    print("Page      :",results["metadatas"][0][i]["page_no"])

    # Embedding
    embedding = results["embeddings"][0][i]

    print("Embedding dimension :",len(embedding))

    print("Embedding first 10  :", embedding[:10])

    # Distance
    print("Distance  :",results["distances"][0][i])

    # Document
    print("\nDocument:")

    print(results["documents"][0][i])