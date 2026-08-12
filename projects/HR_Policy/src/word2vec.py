# Word2Vec Embedding Search Engine

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
Lowercasing
↓
Remove Punctuation
↓
Tokenization
↓
Remove Stop Words
↓
Word2Vec
↓
Build Word Vocabulary
↓
Learn Word Vectors
↓
Create Document Vectors
│
│
│             USER SIDE
│                 ↓
│             User Query
│                 ↓
│         Preprocess Query
│                 ↓
│            Query Tokens
│                 ↓
│            Query Vector
│                 │
└─────────┬───────┘
          ↓
┌─────────────────────┐
│Similarity / Distance│
└──────────┬──────────┘
           ↓
    ┌──────┴──────────┐
    ↓                 ↓
Cosine Similarity     Euclidean Distance
    ↓                 ↓
Similarity Score      Distance Score
    ↓                 ↓
Higher = Better       Lower = Better
    └──────┬──────────┘
           ↓
      Rank Documents
           ↓
        Top-K
           ↓
   Most Relevant Q&A"""
   
   
   
import re
import numpy as np

from gensim.models import Word2Vec

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances
)


class Word2VecProcess:

    def __init__(
        self,
        raw_data,
        similarity_metric,
        embedding_method,
        top_k,
        user_query=None
    ):

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

        tokenized_documents = []

        for doc in documents:

            clean_data = self.preprocess_text(doc)

            tokenized_documents.append(clean_data)


        # ============================================================
        # WORD2VEC
        # ============================================================

        if self.embedding_method.lower() == "word2vec":

            word2vec_model = Word2Vec(
                sentences=tokenized_documents,
                vector_size=100,
                window=5,
                min_count=1,
                workers=4,
                sg=1,
                seed=42
            )

        else:

            raise ValueError(
                f"Unsupported embedding method: "
                f"{self.embedding_method}"
            )


        # ============================================================
        # WORD VOCABULARY
        # ============================================================

        word_vocabulary = word2vec_model.wv.key_to_index


        # ============================================================
        # CREATE DOCUMENT VECTORS
        # ============================================================

        document_vectors = []

        for tokens in tokenized_documents:

            vector = self.document_vector(
                tokens,
                word2vec_model
            )

            document_vectors.append(vector)


        # Convert list into NumPy array

        document_vectors = np.array(document_vectors)


        # ============================================================
        # USER QUERY
        # ============================================================

        user_query = self.user_query


        # ============================================================
        # QUERY PREPROCESSING
        # ============================================================

        query_tokens = self.preprocess_text(user_query)


        # ============================================================
        # QUERY VECTOR
        # ============================================================

        query_vector = self.document_vector(
            query_tokens,
            word2vec_model
        )


        # sklearn expects:
        #
        # (number_of_documents, vector_size)
        #
        # Query should be:
        #
        # (1, vector_size)

        query_vector = query_vector.reshape(1, -1)


        # ============================================================
        # SIMILARITY CHECK
        # ============================================================

        similarity_score, answer = self.similarity_check(
            query_vector,
            document_vectors,
            self.similarity_metric,
            self.top_k
        )


        return similarity_score, answer


    # ================================================================
    # TEXT PREPROCESSING
    # ================================================================

    def preprocess_text(self, text):

        """
        Text preprocessing:

        1. Lowercase
        2. Remove punctuation
        3. Tokenization
        4. Remove stop words

        Returns:
            List of tokens
        """

        # -------------------------
        # Lowercasing
        # -------------------------

        text = text.lower()


        # -------------------------
        # Remove punctuation
        # -------------------------

        text = re.sub(
            r"[^a-zA-Z0-9\s]",
            " ",
            text
        )


        # -------------------------
        # Tokenization
        # -------------------------

        tokens = text.split()


        # -------------------------
        # Remove Stop Words
        # -------------------------

        tokens = [
            token
            for token in tokens
            if token not in ENGLISH_STOP_WORDS
        ]


        # -------------------------
        # Return tokens
        # -------------------------

        return tokens


    # ================================================================
    # DOCUMENT VECTOR
    # ================================================================

    def document_vector(self, tokens, model):

        vectors = []

        for token in tokens:

            if token in model.wv:

                vectors.append(
                    model.wv[token]
                )


        # ============================================================
        # No known words
        # ============================================================

        if not vectors:

            return np.zeros(
                model.wv.vector_size
            )


        # ============================================================
        # Mean of Word Vectors
        # ============================================================

        return np.mean(
            vectors,
            axis=0
        )


    # ================================================================
    # SIMILARITY CHECK
    # ================================================================

    def similarity_check(
        self,
        query_vector,
        document_vectors,
        similarity_metric,
        top_k
    ):

        # ============================================================
        # COSINE SIMILARITY
        # ============================================================

        if similarity_metric.lower() == "cosine_similarity":

            cosine_scores = cosine_similarity(
                query_vector,
                document_vectors
            )


            # ========================================================
            # RANK DOCUMENTS - COSINE
            #
            # Higher score = Better
            # ========================================================

            cosine_ranked_indices = np.argsort(
                cosine_scores[0]
            )[::-1]


            # ========================================================
            # TOP-K RESULTS
            # ========================================================

            top_cosine_indices = (
                cosine_ranked_indices[:top_k]
            )


            # ========================================================
            # MOST RELEVANT Q&A - COSINE
            # ========================================================

            best_cosine_index = (
                cosine_ranked_indices[0]
            )


            similarity_score = round(
                cosine_scores[0][best_cosine_index],
                4
            )


            answer = self.raw_data[
                best_cosine_index
            ]["answer"]


            return similarity_score, answer


        # ============================================================
        # EUCLIDEAN DISTANCE
        # ============================================================

        elif similarity_metric.lower() == "euclidean_distance":

            euclidean_scores = euclidean_distances(
                query_vector,
                document_vectors
            )


            # ========================================================
            # RANK DOCUMENTS - EUCLIDEAN
            #
            # Lower distance = Better
            # ========================================================

            euclidean_ranked_indices = np.argsort(
                euclidean_scores[0]
            )


            # ========================================================
            # TOP-K RESULTS
            # ========================================================

            top_euclidean_indices = (
                euclidean_ranked_indices[:top_k]
            )


            # ========================================================
            # MOST RELEVANT Q&A - EUCLIDEAN
            # ========================================================

            best_euclidean_index = (
                euclidean_ranked_indices[0]
            )


            similarity_score = round(
                euclidean_scores[0][best_euclidean_index],
                4
            )


            answer = self.raw_data[
                best_euclidean_index
            ]["answer"]


            return similarity_score, answer


        # ============================================================
        # INVALID SIMILARITY METRIC
        # ============================================================

        else:

            raise ValueError(
                "Unsupported similarity metric. "
                "Use 'cosine_similarity' or "
                "'euclidean_distance'."
            )
