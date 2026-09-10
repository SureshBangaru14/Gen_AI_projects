import chromadb

# Connect to persistent ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Get existing collection OR create it
collection = client.get_or_create_collection(name="transportation")

print("Collection name:", collection.name)
print("Total records:", collection.count())

# Get stored records
results = collection.get(include=["documents", "metadatas"])

# Show records
for i in range(len(results["ids"])):
    print("=" * 50)
    print("ID       :", results["ids"][i])
    print("Document :", results["documents"][i])
    print("Metadata :", results["metadatas"][i])