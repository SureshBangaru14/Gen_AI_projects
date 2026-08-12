"""

For SentenceTransformer, keep the same coding style and architecture. 
The main difference from Word2Vec is that SentenceTransformer directly generates a vector for the complete sentence/document, so you don't need to manually average word vectors.


    JSON Q&A
        ↓
    Load JSON
        ↓
    Extract Q + A
        ↓
    Combine Q + A
        ↓
    Text Preprocessing
        ↓
    Tokenization
        ↓
    Lowercase
        ↓
    Stop Words
        ↓
    SentenceTransformer
        ↓
    Sentence / Document Embedding
        ↓
                  USER QUERY
                         ↓
                 Query Preprocessing
                         ↓
                 Query Embedding
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
import numpy as np

from pathlib import Path

from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS)

from sklearn.metrics.pairwise import (cosine_similarity, euclidean_distances)

from sentence_transformers import SentenceTransformer



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

print("Number of records:", len(raw_data))
# print("Number of records: \n", raw_data)

# ============================================================
# Extract + Combine Question + Answer
# ============================================================

documents = []

for item in raw_data:

    question = item["question"]
    answer = item["answer"]

    combine_data = question + " " + answer

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
    Text preprocessing:

    1. Lowercase
    2. Remove unwanted punctuation
    3. Tokenization
    4. Optional stop word removal
    """

    # --------------------------------------------------------
    # 1. Lowercase
    # --------------------------------------------------------

    text = text.lower()


    # --------------------------------------------------------
    # 2. Remove unwanted punctuation
    #
    # Preserve time values:
    # 9:00
    # 5:00
    # --------------------------------------------------------

    text = re.sub(r"[^a-zA-Z0-9:\s]", " ", text)
    
    # --------------------------------------------------------
    # 3. Tokenization
    # --------------------------------------------------------

    tokens = text.split()


    # --------------------------------------------------------
    # 4. Remove Stop Words
    # --------------------------------------------------------

    # IMPORTANT:
    # For SentenceTransformer, normally DON'T remove stop words.
    #
    # SentenceTransformer is trained on natural language.
    #
    # Keep the original sentence as much as possible.
    #

    # tokens = [
    #     token
    #     for token in tokens
    #     if token not in ENGLISH_STOP_WORDS
    # ]


    # --------------------------------------------------------
    # 5. Join tokens
    # --------------------------------------------------------

    return " ".join(tokens)



# ============================================================
# Preprocess All Documents
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
# SENTENCE TRANSFORMER
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

print("\n============================================================")
print("MODEL")
print("============================================================")

print("Model loaded successfully")


# ============================================================
# DOCUMENT EMBEDDINGS
# ============================================================

document_embeddings = model.encode(clean_documents, convert_to_numpy=True)


print("\n============================================================")
print("DOCUMENT EMBEDDINGS")
print("============================================================")

print("Shape:",document_embeddings.shape)
print(document_embeddings)


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

query_embedding = model.encode([clean_query], convert_to_numpy=True)


print("\n============================================================")
print("QUERY EMBEDDING")
print("============================================================")

print("Shape:", query_embedding.shape)


print(query_embedding[0])


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
print("cosine_scores",cosine_scores)

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

    print("Question:",raw_data[index]["question"])

    print("Answer:",raw_data[index]["answer"])
    
    
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