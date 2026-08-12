"""

        JSON Q&A
            ↓
        Load JSON
            ↓
        Extract Q + A
            ↓
        Combine Q + A
            ↓
        Light Text Preprocessing
            ↓
        OpenAI Embedding Model
            ↓
        Document Embeddings
            ↓
                    USER QUERY
                            ↓
                    Query Preprocessing
                            ↓
                    OpenAI Embedding
                            ↓
                    Query Vector
                            ↓
                ┌────────┴─────────┐
                ↓                  ↓
        Cosine Similarity    Euclidean Distance
                ↓                  ↓
            Higher = Better     Lower = Better
                ↓                  ↓
                └────────┬─────────┘
                            ↓
                    Rank Documents
                            ↓
                        Top-K
                            ↓
                    Relevant Q&A

"""



import json
import re
import os
import numpy as np

from pathlib import Path

from openai import OpenAI

from sklearn.metrics.pairwise import (cosine_similarity, euclidean_distances)



# ============================================================
# OpenAI Client
# ============================================================

client = OpenAI(api_key=api_key)

# ============================================================
# Load JSON Data
# ============================================================

def load_policy_data(path: str):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


BASE_DIR = Path(__file__).resolve().parent.parent

print("BASE_DIR:", BASE_DIR)

DATA_PATH = BASE_DIR / "data" / "hr_policies.json"


# ============================================================
# Load Data
# ============================================================

raw_data = load_policy_data(DATA_PATH)

print("Number of records:",len(raw_data))
print(raw_data)


# ============================================================
# Extract + Combine Question + Answer
# ============================================================

documents = []

for item in raw_data:

    question = item["question"]
    answer = item["answer"]

    combine_data = (
        question + " " + answer
    )

    documents.append(combine_data)


print("\n============================================================")
print("DOCUMENTS")
print("============================================================")

for index, document in enumerate(documents, start=1):

    print(f"\nDocument {index}:")
    print(document)
    
    
    
# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Light preprocessing for OpenAI embeddings.

    1. Lowercase
    2. Remove unwanted punctuation
    3. Preserve values like 9:00
    """

    # --------------------------------------------------------
    # Lowercase
    # --------------------------------------------------------

    text = text.lower()


    # --------------------------------------------------------
    # Remove unwanted punctuation
    # --------------------------------------------------------

    text = re.sub(r"[^a-zA-Z0-9:\s]", " ", text)
    
    # --------------------------------------------------------
    # Remove extra spaces
    # --------------------------------------------------------

    text = re.sub(r"\s+", " ", text).strip()


    return text



# ============================================================
# Preprocess Documents
# ============================================================

clean_documents = []

for document in documents:

    clean_data = preprocess_text(document)

    clean_documents.append(clean_data)


print("\n============================================================")
print("CLEAN DOCUMENTS")
print("============================================================")

for index, document in enumerate(clean_documents, start=1):

    print(f"\nDocument {index}:")
    print(document)
    
    
# ============================================================
# OPENAI EMBEDDING FUNCTION
# ============================================================

def create_embeddings(texts):

    response = client.embeddings.create(model="text-embedding-3-large", input=texts)

    embeddings = [item.embedding for item in response.data]

    return np.array(embeddings, dtype=np.float32)



# ============================================================
# DOCUMENT EMBEDDINGS
# ============================================================

document_embeddings = create_embeddings(clean_documents)


print("\n============================================================")
print("DOCUMENT EMBEDDINGS")
print("============================================================")

print("Shape:",document_embeddings.shape)
print(document_embeddings)



# ============================================================
# FIRST DOCUMENT EMBEDDING
# ============================================================

print("\n============================================================")
print("FIRST DOCUMENT EMBEDDING")
print("============================================================")

print(document_embeddings[0])

print("Vector Dimensions:",len(document_embeddings[0]))



# ============================================================
# USER QUERY
# ============================================================

user_query = ("How many vacation days can employees take?")


print("\n============================================================")
print("USER QUERY")
print("============================================================")

print(user_query)


# ============================================================
# QUERY PREPROCESSING
# ============================================================

clean_query = preprocess_text(user_query)


print("\n============================================================")
print("CLEAN QUERY")
print("============================================================")

print(clean_query)



# ============================================================
# QUERY EMBEDDING
# ============================================================

query_embedding = create_embeddings([clean_query])


print("\n============================================================")
print("QUERY EMBEDDING")
print("============================================================")

print("Shape:",query_embedding.shape)




# ============================================================
# COSINE SIMILARITY
# ============================================================

cosine_scores = cosine_similarity(query_embedding, document_embeddings)


print("\n============================================================")
print("COSINE SIMILARITY")
print("============================================================")

for index, score in enumerate(cosine_scores[0]):

    print(f"Document {index + 1}: "f"{score:.4f}")


# ============================================================
# EUCLIDEAN DISTANCE
# ============================================================

euclidean_scores = euclidean_distances(query_embedding, document_embeddings)


print("\n============================================================")
print("EUCLIDEAN DISTANCE")
print("============================================================")

for index, distance in enumerate(euclidean_scores[0]):

    print(f"Document {index + 1}: "f"{distance:.4f}")
    
    
# ============================================================
# RANK DOCUMENTS - COSINE
#
# Higher = Better
# ============================================================

cosine_ranked_indices = np.argsort(cosine_scores[0])[::-1]


print("\n============================================================")
print("COSINE RANKING")
print("============================================================")

for rank, index in enumerate(cosine_ranked_indices, start=1):

    print(f"Rank {rank} | "f"Score "f"{cosine_scores[0][index]:.4f} | "f"{raw_data[index]['question']}")
    
# ============================================================
# RANK DOCUMENTS - EUCLIDEAN
#
# Lower = Better
# ============================================================

euclidean_ranked_indices = np.argsort(euclidean_scores[0])


print("\n============================================================")
print("EUCLIDEAN RANKING")
print("============================================================")

for rank, index in enumerate(euclidean_ranked_indices, start=1):

    print(f"Rank {rank} | "f"Distance "f"{euclidean_scores[0][index]:.4f} | "f"{raw_data[index]['question']}")

# ============================================================
# TOP-K
# ============================================================

TOP_K = 3

# ============================================================
# TOP-K COSINE RESULTS
# ============================================================

top_cosine_indices = (cosine_ranked_indices[:TOP_K])


print("\n============================================================")
print(f"TOP-{TOP_K} COSINE RESULTS")
print("============================================================")

for rank, index in enumerate(top_cosine_indices, start=1):

    print(f"\nRank: {rank}")

    print("Similarity Score:", round(cosine_scores[0][index], 4))

    print("Question:", raw_data[index]["question"])

    print("Answer:", raw_data[index]["answer"])
    
    
# ============================================================
# TOP-K EUCLIDEAN RESULTS
# ============================================================

top_euclidean_indices = (euclidean_ranked_indices[:TOP_K])


print("\n============================================================")
print(f"TOP-{TOP_K} EUCLIDEAN RESULTS")
print("============================================================")

for rank, index in enumerate(top_euclidean_indices, start=1):

    print(f"\nRank: {rank}")

    print("Distance:",round(euclidean_scores[0][index],4))

    print("Question:",raw_data[index]["question"])

    print("Answer:",raw_data[index]["answer"])
    
    
# ============================================================
# MOST RELEVANT Q&A
# ============================================================

best_index = cosine_ranked_indices[0]


print("\n============================================================")
print("MOST RELEVANT Q&A - COSINE")
print("============================================================")

print("Similarity Score:",round(cosine_scores[0][best_index], 4))

print("Question:", raw_data[best_index]["question"])

print("Answer:",raw_data[best_index]["answer"])


# ============================================================
# MOST RELEVANT Q&A - EUCLIDEAN
# ============================================================

best_euclidean_index = euclidean_ranked_indices[0]


print("\n============================================================")
print("MOST RELEVANT Q&A - EUCLIDEAN")
print("============================================================")

print("Distance:", round(euclidean_scores[0][best_euclidean_index],4))

print("Question:",raw_data[best_euclidean_index]["question"])

print("Answer:",raw_data[best_euclidean_index]["answer"])
