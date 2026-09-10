import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="transportation")

documents = [
    "Car runs on petrol",
    "Bus carries passengers",
    "Bicycle runs without fuel",
    "Boat travels on water",
    "Plane flies in the sky"
]

ids = [
    "transport_001",
    "transport_002",
    "transport_003",
    "transport_004",
    "transport_005"
]

metadatas = [
    {"vehicle": "car", "fuel": "petrol"},
    {"vehicle": "bus", "purpose": "passenger_transport"},
    {"vehicle": "bicycle", "fuel": "none"},
    {"vehicle": "boat", "medium": "water"},
    {"vehicle": "plane", "medium": "sky"}
]

collection.add(ids=ids, documents=documents, metadatas=metadatas)

print("Total documents:", collection.count())