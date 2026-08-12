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

import re
import numpy as np

from openai import OpenAI

from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances
)


class OpenAIProcess:

    def __init__(self, raw_data, similarity_metric, embedding_method, top_k, user_query=None, openai_api_key=None):

        self.raw_data = raw_data
        self.similarity_metric = similarity_metric
        self.embedding_method = embedding_method
        self.top_k = top_k
        self.user_query = user_query
        self.open_api_key = openai_api_key

        # ============================================================
        # OpenAI Client
        # ============================================================

        self.client = OpenAI(api_key=self.open_api_key)


    # ================================================================
    # GENERATE SEARCH ENGINE
    # ================================================================

    def generate_search_engine(self):

        # ============================================================
        # Combine Question + Answer
        # ============================================================

        documents = []

        for item in self.raw_data:

            question = item["question"]
            answer = item["answer"]

            combine_data = (
                question + " " + answer
            )

            documents.append(combine_data)


        # ============================================================
        # TEXT PREPROCESSING
        # ============================================================

        clean_documents = []

        for document in documents:

            clean_data = self.preprocess_text(
                document
            )

            clean_documents.append(clean_data)


        # ============================================================
        # OPENAI EMBEDDING
        # ============================================================

        if self.embedding_method.lower() == "openai":

            document_embeddings = (
                self.create_embeddings(
                    clean_documents
                )
            )

        else:

            raise ValueError(
                f"Unsupported embedding method: "
                f"{self.embedding_method}"
            )


        # ============================================================
        # USER QUERY
        # ============================================================

        user_query = self.user_query

        if not user_query:

            raise ValueError(
                "user_query cannot be None or empty"
            )


        # ============================================================
        # QUERY PREPROCESSING
        # ============================================================

        clean_query = self.preprocess_text(
            user_query
        )


        # ============================================================
        # QUERY EMBEDDING
        # ============================================================

        query_embedding = self.create_embeddings(
            [clean_query]
        )


        # ============================================================
        # SIMILARITY CHECK
        # ============================================================

        similarity_score, answer = (self.similarity_check(query_embedding, document_embeddings, self.similarity_metric, self.top_k))


        return similarity_score, answer


    # ================================================================
    # TEXT PREPROCESSING
    # ================================================================

    def preprocess_text(self, text):

        """
        Light preprocessing for OpenAI embeddings.

        1. Lowercase
        2. Remove unwanted punctuation
        3. Preserve values like 9:00
        4. Remove extra spaces
        """

        # ------------------------------------------------------------
        # Lowercase
        # ------------------------------------------------------------

        text = text.lower()


        # ------------------------------------------------------------
        # Remove unwanted punctuation
        #
        # Preserve values like:
        # 9:00
        # 5:00
        # ------------------------------------------------------------

        text = re.sub(
            r"[^a-zA-Z0-9:\s]",
            " ",
            text
        )


        # ------------------------------------------------------------
        # Remove extra spaces
        # ------------------------------------------------------------

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()


        return text


    # ================================================================
    # CREATE OPENAI EMBEDDINGS
    # ================================================================

    def create_embeddings(self, texts):

        response = self.client.embeddings.create(

            model="text-embedding-3-large",

            input=texts
        )


        embeddings = [
            item.embedding
            for item in response.data
        ]


        return np.array(
            embeddings,
            dtype=np.float32
        )


    # ================================================================
    # SIMILARITY CHECK
    # ================================================================

    def similarity_check(self, query_embedding, document_embeddings, similarity_metric, top_k):

        # ============================================================
        # COSINE SIMILARITY
        # ============================================================

        if similarity_metric.lower() == "cosine_similarity":

            cosine_scores = cosine_similarity(query_embedding, document_embeddings)


            # ========================================================
            # RANK DOCUMENTS - COSINE
            #
            # Higher score = Better
            # ========================================================

            cosine_ranked_indices = np.argsort(cosine_scores[0])[::-1]


            # ========================================================
            # TOP-K RESULTS
            # ========================================================

            top_cosine_indices = (cosine_ranked_indices[:top_k])


            # ========================================================
            # MOST RELEVANT Q&A - COSINE
            # ========================================================

            best_cosine_index = (cosine_ranked_indices[0])


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