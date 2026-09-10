import chromadb

# --------------------------------------------------
# 1. Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(path="./chroma_db")

# --------------------------------------------------
# 2. Get existing collection
# --------------------------------------------------

collection = client.get_collection(name="hr_policy")

print("Collection name:", collection.name)
print("Total records:", collection.count())

# --------------------------------------------------
# 3. Get all stored records
# --------------------------------------------------

results = collection.get(
                        include=[
                            "documents",
                            "metadatas",
                            "embeddings"
                        ]
                    )

# --------------------------------------------------
# 4. Display records
# --------------------------------------------------

for i in range(len(results["ids"])):

    print("\n" + "=" * 80)

    print("ID       :", results["ids"][i])

    print("Metadata :", results["metadatas"][i])

    print("\nDocument :")
    print(results["documents"][i])

    # Embedding
    embedding = results["embeddings"][i]

    print("\nEmbedding dimension:", len(embedding))

    print("First 10 embedding values:")
    print(embedding[:10])