import numpy as np

vectors = np.array([
    [1.0, 2.0],
    [1.1, 2.1],
    [5.0, 6.0],
    [5.1, 6.1]
], dtype="float32")


# create Faiss index


import faiss

dimension = 2

index = faiss.IndexFlatL2(dimension)

print(index)

index.add(vectors)

print(index.ntotal)


query = np.array([
		    [1.05, 2.05]
		], dtype="float32")

distances, indices = index.search(query, 2)

print("distances : ",distances)
print("indices : ",indices)