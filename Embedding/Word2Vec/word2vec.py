"""
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
            Word2Vec
                ↓
            Build Word Vocabulary
                ↓
            Train Word Embeddings
                ↓
            Word → Vector
                ↓
                ↓       USER QUERY
                ↓               ↓
                ↓       Query Preprocessing
                ↓               ↓
                ↓           Query Tokens
                ↓               ↓
                ↓   Convert Words → Vectors
                ↓               ↓
                ↓   Query / Document Vector
                ↓               ↓
                ↓----->┌────────┴─────────┐
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
                            
                            
Word2Vec, the architecture changes slightly because Word2Vec creates a vector for each word, unlike CountVectorizer and TfidfVectorizer, 
which create vectors for entire documents.  
              
"""

import json
import pandas as pd
from pathlib import Path
import re
import numpy as np
from sklearn.feature_extraction.text import  ENGLISH_STOP_WORDS
from gensim.models import Word2Vec

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

     # --------------------------------------------------------
    # 3. Tokenization
    # --------------------------------------------------------

    tokens = text.split()


    # --------------------------------------------------------
    # 4. Remove Stop Words
    # --------------------------------------------------------

    tokens = [
        token
        for token in tokens
        if token not in ENGLISH_STOP_WORDS
    ]


    return tokens


# ============================================================
# Preprocess All Documents
# ============================================================

tokenized_documents = []

for document in documents:

    tokens = preprocess_text(document)

    tokenized_documents.append(tokens)

print("====tokenized_documents==== ")
print(tokenized_documents)
print("\n============================================================")
print("TOKENIZED DOCUMENTS")
print("============================================================")

for index, tokens in enumerate(tokenized_documents, start=1):

    print(f"Document {index}:")
    print(tokens)

# ============================================================
# WORD2VEC
# ============================================================

word2vec_model = Word2Vec(sentences=tokenized_documents, vector_size=100,
                          window=5, min_count=1, workers=4, sg=1, seed=42)


# ============================================================
# WORD VOCABULARY
# ============================================================

word_vocabulary = (word2vec_model.wv.key_to_index)


print("\n============================================================")
print("WORD2VEC VOCABULARY")
print("============================================================")

print(word_vocabulary)

# ============================================================
# DOCUMENT VECTOR
# ============================================================

def document_vector(tokens,model):

    vectors = []

    for token in tokens:

        if token in model.wv:

            vectors.append(
                model.wv[token]
            )

    if not vectors:

        return np.zeros(
            model.wv.vector_size
        )

    return np.mean(vectors,axis=0)

# ============================================================
# CREATE DOCUMENT VECTORS
# ============================================================

document_vectors = []

for tokens in tokenized_documents:

    vector = document_vector(tokens, word2vec_model)

    document_vectors.append(vector)


document_vectors = np.array(document_vectors)


print("\n============================================================")
print("DOCUMENT VECTORS")
print("============================================================")

print("Shape:",document_vectors.shape)


# ============================================================
# USER QUERY
# ============================================================

user_query = "How many vacation days can employees take?"


print("\n============================================================")
print("USER QUERY")
print("============================================================")

print(user_query)


# ============================================================
# QUERY PREPROCESSING
# ============================================================

query_tokens = preprocess_text(user_query)


print("\n============================================================")
print("QUERY TOKENS")
print("============================================================")

print(query_tokens)



# ============================================================
# QUERY VECTOR
# ============================================================

query_vector = document_vector(query_tokens,word2vec_model)


query_vector = query_vector.reshape(1,-1)


print("\n============================================================")
print("QUERY VECTOR")
print("============================================================")

print("Shape:",query_vector.shape)

print(query_vector)


# ============================================================
# COSINE SIMILARITY
# ============================================================

cosine_scores = cosine_similarity(query_vector, document_vectors)


print("\n============================================================")
print("COSINE SIMILARITY")
print("============================================================")

for index, score in enumerate(cosine_scores[0]):

    print(f"Document {index + 1}: "f"{score:.4f}")
    
    
    
# ============================================================
# EUCLIDEAN DISTANCE
# ============================================================

euclidean_scores = euclidean_distances(query_vector, document_vectors)


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