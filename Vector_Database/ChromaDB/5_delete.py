import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="transportation")

# Delete bus record
collection.delete(ids=["bus_001"])

print("Bus record deleted")


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