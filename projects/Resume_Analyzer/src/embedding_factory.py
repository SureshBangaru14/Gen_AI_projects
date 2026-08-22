# ============================================================
# EMBEDDING FACTORY
# ============================================================

from typing import List

import numpy as np


class BaseEmbedding:

    # ========================================================
    # EMBED DOCUMENTS
    # ========================================================

    def embed_documents(
        self,
        documents: List[str]
    ):

        raise NotImplementedError


    # ========================================================
    # EMBED QUERY
    # ========================================================

    def embed_query(
        self,
        query: str
    ):

        raise NotImplementedError


    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        documents
    ):

        return self


# ============================================================
# OPENAI EMBEDDINGS
# ============================================================

class OpenAIEmbedding(BaseEmbedding):

    def __init__(
        self,
        api_key,
        model_name="text-embedding-3-small"
    ):

        if not api_key:

            raise ValueError(
                "OpenAI API key is required."
            )


        from openai import OpenAI


        self.client = OpenAI(
            api_key=api_key
        )

        self.model_name = model_name


    # ========================================================
    # DOCUMENT EMBEDDINGS
    # ========================================================

    def embed_documents(
        self,
        documents
    ):

        if not documents:

            return []


        response = (
            self.client.embeddings.create(

                model=self.model_name,

                input=documents

            )
        )


        return [

            item.embedding

            for item
            in response.data

        ]


    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def embed_query(
        self,
        query
    ):

        response = (
            self.client.embeddings.create(

                model=self.model_name,

                input=query

            )
        )


        return response.data[
            0
        ].embedding


# ============================================================
# SENTENCE TRANSFORMER / SBERT
# ============================================================

class SentenceTransformerEmbedding(
    BaseEmbedding
):

    def __init__(
        self,
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    ):

        from sentence_transformers import (
            SentenceTransformer
        )


        self.model = (
            SentenceTransformer(
                model_name
            )
        )


        self.model_name = model_name


    # ========================================================
    # DOCUMENTS
    # ========================================================

    def embed_documents(
        self,
        documents
    ):

        if not documents:

            return []


        embeddings = (
            self.model.encode(

                documents,

                normalize_embeddings=True,

                show_progress_bar=False

            )
        )


        return embeddings.tolist()


    # ========================================================
    # QUERY
    # ========================================================

    def embed_query(
        self,
        query
    ):

        embedding = (
            self.model.encode(

                query,

                normalize_embeddings=True

            )
        )


        return embedding.tolist()


# ============================================================
# MULTILINGUAL EMBEDDING
# ============================================================

class MultilingualEmbedding(
    SentenceTransformerEmbedding
):

    def __init__(
        self,
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    ):

        super().__init__(
            model_name=model_name
        )


# ============================================================
# MODERN TRANSFORMER EMBEDDING
# ============================================================

class TransformerEmbedding(
    SentenceTransformerEmbedding
):

    def __init__(
        self,
        model_name=(
            "sentence-transformers/"
            "all-mpnet-base-v2"
        )
    ):

        super().__init__(
            model_name=model_name
        )


# ============================================================
# DOMAIN-SPECIFIC EMBEDDING
# ============================================================

class DomainSpecificEmbedding(
    SentenceTransformerEmbedding
):

    def __init__(
        self,
        model_name
    ):

        if not model_name:

            raise ValueError(

                "A domain-specific model "
                "name is required."

            )


        super().__init__(
            model_name=model_name
        )


# ============================================================
# TF-IDF
# ============================================================

class TFIDFEmbedding(
    BaseEmbedding
):

    def __init__(self):

        from sklearn.feature_extraction.text import (
            TfidfVectorizer
        )


        self.vectorizer = (
            TfidfVectorizer(

                lowercase=True,

                ngram_range=(1, 2),

                max_features=50000

            )
        )


        self.is_fitted = False


    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        documents
    ):

        self.vectorizer.fit(
            documents
        )


        self.is_fitted = True


        return self


    # ========================================================
    # DOCUMENTS
    # ========================================================

    def embed_documents(
        self,
        documents
    ):

        if not self.is_fitted:

            self.fit(
                documents
            )


        matrix = (
            self.vectorizer.transform(
                documents
            )
        )


        return matrix.toarray().tolist()


    # ========================================================
    # QUERY
    # ========================================================

    def embed_query(
        self,
        query
    ):

        if not self.is_fitted:

            raise ValueError(

                "TF-IDF must be fitted before "
                "embedding a query."

            )


        vector = (
            self.vectorizer.transform(
                [query]
            )
        )


        return vector.toarray()[
            0
        ].tolist()


# ============================================================
# WORD2VEC
# ============================================================

class Word2VecEmbedding(
    BaseEmbedding
):

    def __init__(
        self,
        vector_size=300,
        window=5,
        min_count=1,
        workers=4
    ):

        from gensim.models import (
            Word2Vec
        )


        self.Word2Vec = Word2Vec

        self.vector_size = vector_size

        self.window = window

        self.min_count = min_count

        self.workers = workers

        self.model = None


    # ========================================================
    # TOKENIZE
    # ========================================================

    def tokenize(
        self,
        text
    ):

        return text.lower().split()


    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        documents
    ):

        tokenized_documents = [

            self.tokenize(
                document
            )

            for document
            in documents

        ]


        self.model = self.Word2Vec(

            sentences=
                tokenized_documents,

            vector_size=
                self.vector_size,

            window=
                self.window,

            min_count=
                self.min_count,

            workers=
                self.workers

        )


        return self


    # ========================================================
    # DOCUMENT VECTOR
    # ========================================================

    def document_vector(
        self,
        document
    ):

        tokens = (
            self.tokenize(
                document
            )
        )


        vectors = [

            self.model.wv[token]

            for token
            in tokens

            if token
            in self.model.wv

        ]


        if not vectors:

            return [

                0.0

                for _ in range(
                    self.vector_size
                )

            ]


        vector = np.mean(
            vectors,
            axis=0
        )


        return vector.tolist()


    # ========================================================
    # DOCUMENTS
    # ========================================================

    def embed_documents(
        self,
        documents
    ):

        if self.model is None:

            self.fit(
                documents
            )


        return [

            self.document_vector(
                document
            )

            for document
            in documents

        ]


    # ========================================================
    # QUERY
    # ========================================================

    def embed_query(
        self,
        query
    ):

        if self.model is None:

            raise ValueError(

                "Word2Vec must be fitted "
                "before embedding query."

            )


        return self.document_vector(
            query
        )


# ============================================================
# SPARSE EMBEDDING
# ============================================================

class SparseEmbedding(
    TFIDFEmbedding
):

    pass


# ============================================================
# HYBRID DENSE + SPARSE
# ============================================================

class HybridDenseSparseEmbedding(
    BaseEmbedding
):

    def __init__(
        self,
        dense_embedding,
        sparse_embedding,
        dense_weight=0.7,
        sparse_weight=0.3
    ):

        self.dense_embedding = (
            dense_embedding
        )

        self.sparse_embedding = (
            sparse_embedding
        )

        self.dense_weight = (
            dense_weight
        )

        self.sparse_weight = (
            sparse_weight
        )


    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        documents
    ):

        self.dense_embedding.fit(
            documents
        )


        self.sparse_embedding.fit(
            documents
        )


        return self


    # ========================================================
    # NORMALIZE
    # ========================================================

    @staticmethod
    def normalize(
        vector
    ):

        vector = np.asarray(
            vector,
            dtype=float
        )


        norm = np.linalg.norm(
            vector
        )


        if norm == 0:

            return vector


        return vector / norm


    # ========================================================
    # DOCUMENTS
    # ========================================================

    def embed_documents(
        self,
        documents
    ):

        dense_vectors = (
            self.dense_embedding
            .embed_documents(
                documents
            )
        )


        sparse_vectors = (
            self.sparse_embedding
            .embed_documents(
                documents
            )
        )


        results = []


        for (

            dense,
            sparse

        ) in zip(

            dense_vectors,
            sparse_vectors

        ):

            dense = self.normalize(
                dense
            )


            sparse = self.normalize(
                sparse
            )


            # ------------------------------------------------
            # Hybrid vector
            # ------------------------------------------------

            vector = np.concatenate(

                [

                    dense
                    *
                    self.dense_weight,

                    sparse
                    *
                    self.sparse_weight

                ]

            )


            results.append(
                vector.tolist()
            )


        return results


    # ========================================================
    # QUERY
    # ========================================================

    def embed_query(
        self,
        query
    ):

        dense = (
            self.dense_embedding
            .embed_query(
                query
            )
        )


        sparse = (
            self.sparse_embedding
            .embed_query(
                query
            )
        )


        dense = self.normalize(
            dense
        )


        sparse = self.normalize(
            sparse
        )


        vector = np.concatenate(

            [

                dense
                *
                self.dense_weight,

                sparse
                *
                self.sparse_weight

            ]

        )


        return vector.tolist()


# ============================================================
# LATE INTERACTION EMBEDDING
# ============================================================

class LateInteractionEmbedding(
    SentenceTransformerEmbedding
):

    """
    Practical placeholder for late-interaction retrieval.

    A true ColBERT implementation requires token-level
    embeddings and MaxSim scoring rather than a single
    pooled document vector.

    This class therefore provides a transformer interface
    while RetrievalEngine should use a dedicated ColBERT
    implementation when the Late-Interaction method is
    selected.
    """

    pass


# ============================================================
# MULTIMODAL EMBEDDING
# ============================================================

class MultimodalEmbedding(
    BaseEmbedding
):

    def __init__(
        self,
        model_name=None
    ):

        self.model_name = model_name


    def embed_documents(
        self,
        documents
    ):

        raise NotImplementedError(

            "Multimodal embeddings require a "
            "multimodal model implementation. "
            "Use OpenAI or a dedicated multimodal "
            "embedding model."

        )


    def embed_query(
        self,
        query
    ):

        raise NotImplementedError(

            "Multimodal embeddings require a "
            "multimodal model implementation."

        )


# ============================================================
# EMBEDDING FACTORY
# ============================================================

class EmbeddingFactory:

    # ========================================================
    # CREATE
    # ========================================================

    @staticmethod
    def create(
        embedding_method,
        api_key=None,
        model_name=None
    ):

        method = (
            embedding_method
            .strip()
            .lower()
        )


        # ====================================================
        # OPENAI
        # ====================================================

        if method == (
            "openai embeddings"
        ):

            return OpenAIEmbedding(

                api_key=
                    api_key,

                model_name=
                    model_name
                    or
                    "text-embedding-3-small"

            )


        # ====================================================
        # TRANSFORMER
        # ====================================================

        if method in [

            "transformer-based embeddings",

            "transformer embeddings"

        ]:

            return TransformerEmbedding(

                model_name=
                    model_name
                    or
                    "sentence-transformers/"
                    "all-mpnet-base-v2"

            )


        # ====================================================
        # SBERT
        # ====================================================

        if method in [

            "sentence-bert (sbert)",

            "sbert",

            "sentence-transformer",

            "sentence-transformers"

        ]:

            return SentenceTransformerEmbedding(

                model_name=
                    model_name
                    or
                    "sentence-transformers/"
                    "all-MiniLM-L6-v2"

            )


        # ====================================================
        # MODERN TEXT EMBEDDINGS
        # ====================================================

        if method in [

            "other modern text embedding models",

            "modern text embeddings"

        ]:

            return SentenceTransformerEmbedding(

                model_name=
                    model_name
                    or
                    "sentence-transformers/"
                    "all-mpnet-base-v2"

            )


        # ====================================================
        # MULTILINGUAL
        # ====================================================

        if method in [

            "multilingual embeddings",

            "multilingual"

        ]:

            return MultilingualEmbedding(

                model_name=
                    model_name
                    or
                    "sentence-transformers/"
                    "paraphrase-multilingual-MiniLM-L12-v2"

            )


        # ====================================================
        # DOMAIN SPECIFIC
        # ====================================================

        if method in [

            "domain-specific embeddings",

            "domain specific"

        ]:

            if not model_name:

                raise ValueError(

                    "Provide a domain-specific "
                    "embedding model."

                )


            return DomainSpecificEmbedding(

                model_name=model_name

            )


        # ====================================================
        # SPARSE
        # ====================================================

        if method in [

            "sparse embeddings",

            "sparse"

        ]:

            return SparseEmbedding()


        # ====================================================
        # HYBRID
        # ====================================================

        if method in [

            "hybrid dense + sparse embeddings",

            "hybrid embeddings"

        ]:

            if not api_key:

                raise ValueError(

                    "OpenAI API key is required "
                    "for hybrid dense + sparse "
                    "embeddings."

                )


            dense = OpenAIEmbedding(

                api_key=api_key,

                model_name=
                    model_name
                    or
                    "text-embedding-3-small"

            )


            sparse = SparseEmbedding()


            return HybridDenseSparseEmbedding(

                dense_embedding=dense,

                sparse_embedding=sparse,

                dense_weight=0.7,

                sparse_weight=0.3

            )


        # ====================================================
        # LATE INTERACTION
        # ====================================================

        if method in [

            "late-interaction embeddings",

            "late interaction",

            "colbert"

        ]:

            return LateInteractionEmbedding(

                model_name=
                    model_name
                    or
                    "sentence-transformers/"
                    "all-MiniLM-L6-v2"

            )


        # ====================================================
        # MULTIMODAL
        # ====================================================

        if method in [

            "multimodal embeddings",

            "multimodal"

        ]:

            return MultimodalEmbedding(

                model_name=
                    model_name

            )


        # ====================================================
        # TF-IDF
        # ====================================================

        if method in [

            "tf-idf",

            "tfidf"

        ]:

            return TFIDFEmbedding()


        # ====================================================
        # WORD2VEC
        # ====================================================

        if method in [

            "word2vec",

            "word2vec embeddings"

        ]:

            return Word2VecEmbedding()


        # ====================================================
        # UNKNOWN
        # ====================================================

        raise ValueError(

            f"Unsupported embedding method: "
            f"{embedding_method}"

        )