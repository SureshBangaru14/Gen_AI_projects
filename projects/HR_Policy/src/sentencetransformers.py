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


import re
import numpy as np

from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances
)

from sentence_transformers import SentenceTransformer


class SentenceTransformerProcess:

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

        clean_documents = []

        for doc in documents:

            clean_data = self.preprocess_text(doc)

            clean_documents.append(clean_data)


        # ============================================================
        # SENTENCE TRANSFORMER
        # ============================================================

        if self.embedding_method.lower() == "sentencetransformer":

            model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

        else:

            raise ValueError(
                f"Unsupported embedding method: "
                f"{self.embedding_method}"
            )


        # ============================================================
        # DOCUMENT EMBEDDINGS
        # ============================================================

        document_embeddings = model.encode(
            clean_documents,
            convert_to_numpy=True
        )


        # ============================================================
        # USER QUERY
        # ============================================================

        user_query = self.user_query


        # ============================================================
        # QUERY PREPROCESSING
        # ============================================================

        clean_query = self.preprocess_text(
            user_query
        )


        # ============================================================
        # QUERY EMBEDDING
        # ============================================================

        query_embedding = model.encode(
            [clean_query],
            convert_to_numpy=True
        )


        # ============================================================
        # SIMILARITY CHECK
        # ============================================================

        similarity_score, answer = self.similarity_check(
            query_embedding,
            document_embeddings,
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
        2. Remove unwanted punctuation
        3. Tokenization
        4. Join tokens

        IMPORTANT:
        SentenceTransformer generally works better
        when natural language is preserved.
        """

        # ------------------------------------------------------------
        # Lowercase
        # ------------------------------------------------------------

        text = text.lower()


        # ------------------------------------------------------------
        # Remove unwanted punctuation
        #
        # Preserve time values such as:
        # 9:00
        # 5:00
        # ------------------------------------------------------------

        text = re.sub(
            r"[^a-zA-Z0-9:\s]",
            " ",
            text
        )


        # ------------------------------------------------------------
        # Tokenization
        # ------------------------------------------------------------

        tokens = text.split()


        # ------------------------------------------------------------
        # Join tokens
        # ------------------------------------------------------------

        return " ".join(tokens)


    # ================================================================
    # SIMILARITY CHECK
    # ================================================================

    def similarity_check(
        self,
        query_embedding,
        document_embeddings,
        similarity_metric,
        top_k
    ):

        # ============================================================
        # COSINE SIMILARITY
        # ============================================================

        if similarity_metric.lower() == "cosine_similarity":

            cosine_scores = cosine_similarity(
                query_embedding,
                document_embeddings
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
            # MOST RELEVANT Q&A
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
                query_embedding,
                document_embeddings
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
            # MOST RELEVANT Q&A
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