"""
            JSON Q&A Data
                ↓
            Load JSON
                ↓
            Extract Question + Answer
                ↓
            Combine Q + A into Text
                ↓
            Text Preprocessing
                ↓
            Tokenization
                ↓
            Lowercasing
                ↓
            Remove Stop Words
                ↓
            CountVectorizer
                ↓
            Build Vocabulary
                ↓
            Document-Term Matrix
                │
                │
                │             USER SIDE
                │                 ↓
                │             User Query
                │                 ↓
                │         Preprocess Query
                │                 ↓
                │            Query Vector
                │                 │
                └─────────┬───────┘
                          ↓
                ┌─────────────────────┐
                │Similarity / Distance│
                └──────────┬──────────┘
                           ↓
                 ┌─────────┴─────────┐
                 ↓                   ↓
            Cosine Similarity     Euclidean Distance
                 ↓                   ↓
            Similarity Score      Distance Score
                 ↓                   ↓
            Higher = Better       Lower = Better
                 └─────────┬─────────┘
                           ↓
                      Rank Documents
                           ↓
                    Top-K Documents
                           ↓
                    Most Relevant Q&A
"""

import json
import pandas as pd
from pathlib import Path
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from sklearn.metrics.pairwise import (cosine_similarity, euclidean_distances)



def load_policy_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)
DATA_PATH = BASE_DIR / "data" / "hr_policies.json"

# ============================================================
# Load Data
# ============================================================
raw_data = load_policy_data(DATA_PATH)
# print(raw_data)



# ============================================================
# Combine question + answer
# ============================================================
documents = []

for item in raw_data:
    question = item["question"]
    answer = item["answer"]

    combine_data = question + " " + answer

    documents.append(combine_data)
    
# print(documents)



# ============================================================
# TEXT PREPROCESSING
# ============================================================


def preprocess_text(text):
    """
    Text preprocessing:
    1. Lowercase
    2. Remove punctuation
    3. Tokenization
    4. Remove stop words
    """

    # -------------------------
    # Lowercasing
    # -------------------------

    text = text.lower()

    # -------------------------
    # Remove punctuation
    # -------------------------

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    
    # text = re.sub(r"(?<!\d):|:(?!\d)", " ", text)

    # -------------------------
    # Tokenization
    # -------------------------

    # tokens = text.split()

    # -------------------------
    # Remove stop words
    # -------------------------

    # tokens = [ token for token in tokens if token not in ENGLISH_STOP_WORDS ]

    # -------------------------
    # Join tokens
    # -------------------------

    # return " ".join(tokens)
    return text

clean_documents = []
for doc in documents:
    clean_data = preprocess_text(doc)
    clean_documents.append(clean_data)

# print(clean_documents)


# ============================================================
# COUNT VECTORIZER
# ============================================================

vectorizer = CountVectorizer()



# ============================================================
# Build Vocabulary + Document-Term Matrix
# ============================================================

document_term_matrix = vectorizer.fit_transform(clean_documents)


# ============================================================
# Vocabulary
# ============================================================

vocabulary = vectorizer.vocabulary_


print("\n============================================================")
print("VOCABULARY")
print("============================================================")

# print(vocabulary)


# ============================================================
# Document-Term Matrix
# ============================================================

print("\n============================================================")
print("DOCUMENT-TERM MATRIX")
print("============================================================")

print("Shape:", document_term_matrix.shape)

# print(document_term_matrix.toarray())



feature_names = vectorizer.get_feature_names_out()


print("\n============================================================")
print("FEATURE NAMES")
print("============================================================")

# print(feature_names)




# ============================================================
# USER QUERY
# ============================================================

user_query = "How many vacation days can employees take?"


print("\n============================================================")
print("USER QUERY")
print("============================================================")

print(user_query)


# ============================================================
# Query Preprocessing
# ============================================================

clean_query = preprocess_text(user_query)


print("\n============================================================")
print("CLEAN QUERY")
print("============================================================")

print(clean_query)



# ============================================================
# Query Vector
# ============================================================

query_vector = vectorizer.transform(
    [clean_query]
)


print("\n============================================================")
print("QUERY VECTOR")
print("============================================================")

# print(query_vector.toarray())



# ============================================================
# COSINE SIMILARITY
# ============================================================

cosine_scores = cosine_similarity(query_vector, document_term_matrix)


print("\n============================================================")
print("COSINE SIMILARITY")
print("============================================================")

for index, score in enumerate(cosine_scores[0]):

    print(f"Document {index + 1}: "f"{score:.4f}")
    
    
    
# ============================================================
# EUCLIDEAN DISTANCE
# ============================================================

euclidean_scores = euclidean_distances(query_vector, document_term_matrix)


print("\n============================================================")
print("EUCLIDEAN DISTANCE")
print("============================================================")

for index, distance in enumerate(euclidean_scores[0]):

    print(f"Document {index + 1}: "f"{distance:.4f}")
    
    
    
# ============================================================
# RANK DOCUMENTS - COSINE
#
# Higher score = Better
# ============================================================

cosine_ranked_indices = np.argsort(cosine_scores[0])[::-1]


print("\n============================================================")
print("COSINE RANKING")
print("============================================================")

for rank, index in enumerate(cosine_ranked_indices,start=1):

    print(f"Rank {rank} | "f"Score {cosine_scores[0][index]:.4f} | "f"{raw_data[index]['question']}")
    
    
    
# ============================================================
# RANK DOCUMENTS - EUCLIDEAN
#
# Lower distance = Better
# ============================================================

euclidean_ranked_indices = np.argsort(euclidean_scores[0])


print("\n============================================================")
print("EUCLIDEAN RANKING")
print("============================================================")

for rank, index in enumerate(euclidean_ranked_indices,start=1):

    print(f"Rank {rank} | "f"Distance {euclidean_scores[0][index]:.4f} | "f"{raw_data[index]['question']}")
    

# ============================================================
# TOP-K RESULTS
# ============================================================

TOP_K = 3


# ============================================================
# TOP-K COSINE
# ============================================================

top_cosine_indices = cosine_ranked_indices[:TOP_K]


print("\n============================================================")
print(f"TOP-{TOP_K} COSINE RESULTS")
print("============================================================")

for rank, index in enumerate(top_cosine_indices,start=1):

    print(f"\nRank: {rank}")

    print(f"Similarity Score: "f"{cosine_scores[0][index]:.4f}")

    print(f"Question: "f"{raw_data[index]['question']}")

    print(f"Answer: "f"{raw_data[index]['answer']}")


# ============================================================
# TOP-K EUCLIDEAN
# ============================================================

top_euclidean_indices = euclidean_ranked_indices[:TOP_K]


print("\n============================================================")
print(f"TOP-{TOP_K} EUCLIDEAN RESULTS")
print("============================================================")

for rank, index in enumerate(top_euclidean_indices,start=1):

    print(f"\nRank: {rank}")

    print(f"Distance: "f"{euclidean_scores[0][index]:.4f}")

    print(f"Question: "f"{raw_data[index]['question']}")

    print(f"Answer: "f"{raw_data[index]['answer']}")
    
    
    
# ============================================================
# MOST RELEVANT Q&A - COSINE
# ============================================================

best_cosine_index = cosine_ranked_indices[0]


print("\n============================================================")
print("MOST RELEVANT Q&A - COSINE")
print("============================================================")

print("Similarity Score:",round(cosine_scores[0][best_cosine_index],4))

print("Question:",raw_data[best_cosine_index]["question"])

print("Answer:",raw_data[best_cosine_index]["answer"])


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