import chromadb

# Connect to persistent ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Get existing collection OR create it
collection = client.get_or_create_collection(name="transportation")

print("Collection name:", collection.name)
print("Total records:", collection.count())

result = collection.get(
    where={
        "fuel": "petrol"})

print(result)




result = collection.get(
    where={
        "$and": [
            {"vehicle": "bus"},
            {"passengers": 50}
        ]
    }
)


print(result)