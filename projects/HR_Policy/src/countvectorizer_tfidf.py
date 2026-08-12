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


import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS

from sklearn.metrics.pairwise import (cosine_similarity, euclidean_distances)

class CountVectorizerTfidfEmbedding:
    
    def __init__(self, raw_data, similarity_metric, embedding_method, top_k, user_query = None):
            
            self.raw_data = raw_data
            self.similarity_metric = similarity_metric
            self.embedding_method = embedding_method
            self.top_k = top_k
            self.user_query = user_query
            
    def generate_search_engine(self):
    
        # ============================================================
        # Combine question + answer
        # ============================================================
        documents = []

        for item in self.raw_data:
            question = item["question"]
            answer = item["answer"]

            combine_data = question + " " + answer

            documents.append(combine_data)
            
        # ============================================================
        # TEXT PREPROCESSING
        # ============================================================
            
        clean_documents = []
        for doc in documents:
            clean_data = self.preprocess_text(doc)
            clean_documents.append(clean_data)

        # ============================================================
        # COUNT VECTORIZER
        # ============================================================

        if self.embedding_method == "CountVectorizer":
            vectorizer = CountVectorizer()
        
        else:
            vectorizer = TfidfVectorizer()
        
        # ============================================================
        # Build Vocabulary + Document-Term Matrix
        # ============================================================

        document_term_matrix = vectorizer.fit_transform(clean_documents)
        # print("Shape:", document_term_matrix.shape)

        # print(document_term_matrix.toarray())


        # ============================================================
        # Vocabulary
        # ============================================================

        vocabulary = vectorizer.vocabulary_

        # print(vocabulary)
        
        feature_names = vectorizer.get_feature_names_out()

        # print(feature_names)
        
        
        # ============================================================
        # USER QUERY
        # ============================================================

        user_query = self.user_query


        # ============================================================
        # Query Preprocessing
        # ============================================================

        clean_query = self.preprocess_text(user_query)

        # ============================================================
        # Query Vector
        # ============================================================

        query_vector = vectorizer.transform([clean_query])
        
        similarity_Score, Answer = self.similarity_check(query_vector, document_term_matrix, self.similarity_metric, self.top_k)


            
        return similarity_Score, Answer
        


    


    def preprocess_text(self, text):
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
    
    def similarity_check(self,query_vector, document_term_matrix, similarity_metric, top_k):
        
        if similarity_metric.lower() =="cosine_similarity":
            
            # ============================================================
            # COSINE SIMILARITY
            # ============================================================

            cosine_scores = cosine_similarity(query_vector, document_term_matrix)
            
            # ============================================================
            # RANK DOCUMENTS - COSINE
            #
            # Higher score = Better
            # ============================================================

            cosine_ranked_indices = np.argsort(cosine_scores[0])[::-1]
            
            # ============================================================
            # TOP-K RESULTS
            # ============================================================

            top_cosine_indices = cosine_ranked_indices[:top_k]
            
            # ============================================================
            # MOST RELEVANT Q&A - COSINE
            # ============================================================

            best_cosine_index = cosine_ranked_indices[0]
            
            similarity_Score = round(cosine_scores[0][best_cosine_index],4)
            
            Answer = self.raw_data[best_cosine_index]["answer"]
            
            return similarity_Score, Answer
        
        elif similarity_metric.lower() =="euclidean_distance":
                    
                    # ============================================================
                    # EUCLIDEAN DISTANCE
                    # ============================================================

                    euclidean_scores = euclidean_distances(query_vector, document_term_matrix)
                    
                    # ============================================================
                    # RANK DOCUMENTS - EUCLIDEAN
                    #
                    # Lower distance = Better
                    # ============================================================

                    euclidean_ranked_indices = np.argsort(euclidean_scores[0])
                    
                    # ============================================================
                    # TOP-K EUCLIDEAN
                    # ============================================================

                    top_euclidean_indices = euclidean_ranked_indices[:top_k]
        
                    
                    # ============================================================
                    # MOST RELEVANT Q&A - EUCLIDEAN
                    # ============================================================

                    best_euclidean_index = euclidean_ranked_indices[0]
                    
                    similarity_Score = round(euclidean_scores[0][best_euclidean_index],4)
                    
                    Answer = self.raw_data[best_euclidean_index]["answer"]
                    
                    return similarity_Score, Answer

