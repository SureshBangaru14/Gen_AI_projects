import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="transportation")

collection.upsert(ids=["bus_001"], 
                  documents=["Bus carries 50 passengers on road"],
                  metadatas=[
                                {
                                    "vehicle": "bus",
                                    "passengers": 50,
                                    "location": "road"
                                }
                            ]
                        )

print("Bus record upserted successfully")




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