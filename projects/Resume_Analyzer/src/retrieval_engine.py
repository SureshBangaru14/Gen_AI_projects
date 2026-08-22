# src/retrieval_engine.py

import math
import re

import numpy as np

from src.chroma_db import ChromaDBManager
from src.embedding_factory import EmbeddingFactory


class RetrievalEngine:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        embedding_method,
        embedding_model,
        similarity_method,
        api_key=None,
        top_k=5,
        chroma_persist_directory="chroma_db",
        chroma_collection_name="resume_collection"
    ):

        self.embedding_method = (
            embedding_method
        )

        self.embedding_model = (
            embedding_model
        )

        self.similarity_method = (
            similarity_method
        )

        self.api_key = api_key

        self.top_k = int(top_k)

        self.chroma_persist_directory = (
            chroma_persist_directory
        )

        self.chroma_collection_name = (
            chroma_collection_name
        )

        # --------------------------------------------------------
        # Retrieval depth
        #
        # Top-K in the application represents FINAL RESUME rows.
        # Vector retrieval works on CHUNKS. When resume_chunks are
        # available, retrieve enough chunks to allow every uploaded
        # resume to participate in the final scoring stage.
        # --------------------------------------------------------
        self.final_top_k = max(1, int(top_k))

        self.embedding_manager = (
            EmbeddingFactory.create(

                embedding_method=
                    embedding_method,

                api_key=
                    api_key,

                model_name=
                    embedding_model

            )
        )


    # ========================================================
    # EFFECTIVE RETRIEVAL K
    # ========================================================

    def _get_retrieval_top_k(self, resume_chunks=None):
        """
        Determine how many chunk-level results should be retrieved.

        The UI Top-K is the number of final resume rows. Therefore,
        when the complete resume chunk map is available, retrieve
        enough chunks for all resumes. DocumentProcess then groups
        chunks by resume and scores every uploaded resume.
        """
        if not resume_chunks:
            return self.final_top_k

        total_chunks = sum(
            len(chunks or [])
            for chunks in resume_chunks.values()
        )

        if total_chunks <= 0:
            return self.final_top_k

        return max(
            self.final_top_k,
            total_chunks
        )

    # ========================================================
    # MAIN SEARCH
    # ========================================================

    def search(
        self,
        jd_text,
        resume_chunks=None,
        vector_data=None
    ):

        if not jd_text:

            return []

        # --------------------------------------------------------
        # Use chunk-level retrieval depth large enough to include
        # every uploaded resume. Final Top-K is applied later by
        # DocumentProcess after resume-level scoring.
        # --------------------------------------------------------
        self.top_k = self._get_retrieval_top_k(
            resume_chunks
        )

        method = (
            self.similarity_method
            .strip()
            .lower()
        )


        # ====================================================
        # COSINE
        # ====================================================

        if method == "cosine similarity":

            return self.cosine_search(
                jd_text
            )


        # ====================================================
        # DOT PRODUCT
        # ====================================================

        if method in [

            "dot product / inner product",

            "dot product",

            "inner product"

        ]:

            return self.dot_product_search(
                jd_text
            )


        # ====================================================
        # BM25
        # ====================================================

        if method == "bm25":

            return self.bm25_search(
                jd_text,
                resume_chunks
            )


        # ====================================================
        # HYBRID
        # ====================================================

        if method == "hybrid search":

            return self.hybrid_search(
                jd_text,
                resume_chunks
            )


        # ====================================================
        # CROSS ENCODER
        # ====================================================

        if method == "cross-encoder reranking":

            return self.cross_encoder_search(
                jd_text,
                resume_chunks
            )


        # ====================================================
        # ANN
        # ====================================================

        if method in [

            "ann (approximate nearest neighbor)",

            "ann",

            "approximate nearest neighbor"

        ]:

            return self.ann_search(
                jd_text
            )


        # ====================================================
        # DENSE
        # ====================================================

        if method == "dense vector retrieval":

            return self.dense_search(
                jd_text
            )


        # ====================================================
        # SPARSE
        # ====================================================

        if method == "sparse vector retrieval":

            return self.bm25_search(
                jd_text,
                resume_chunks
            )


        # ====================================================
        # DENSE + SPARSE
        # ====================================================

        if method == "dense + sparse hybrid retrieval":

            return self.hybrid_search(
                jd_text,
                resume_chunks
            )


        # ====================================================
        # MMR
        # ====================================================

        if method == "mmr (maximal marginal relevance)":

            return self.mmr_search(
                jd_text
            )


        # ====================================================
        # L2
        # ====================================================

        if method == "euclidean distance (l2)":

            return self.l2_search(
                jd_text
            )


        # ====================================================
        # COLBERT
        # ====================================================

        if method in [

            "colbert / late-interaction retrieval",

            "colbert",

            "late-interaction retrieval"

        ]:

            return self.colbert_search(
                jd_text,
                resume_chunks
            )


        raise ValueError(

            f"Unsupported similarity method: "
            f"{self.similarity_method}"

        )


    # ========================================================
    # CHROMA
    # ========================================================

    def get_chroma(self):

        return ChromaDBManager(

            persist_directory=
                self.chroma_persist_directory,

            collection_name=
                self.chroma_collection_name

        )


    # ========================================================
    # QUERY VECTOR
    # ========================================================

    def embed_query(
        self,
        query
    ):

        return (
            self.embedding_manager
            .embed_query(
                query
            )
        )


    # ========================================================
    # COSINE
    # ========================================================

    def cosine_search(
        self,
        query
    ):

        query_vector = (
            self.embed_query(
                query
            )
        )

        db = self.get_chroma()

        result = db.query(

            query_embeddings=
                query_vector,

            n_results=
                self.top_k

        )

        return self.convert_chroma_results(
            result
        )


    # ========================================================
    # DENSE
    # ========================================================

    def dense_search(
        self,
        query
    ):

        return self.cosine_search(
            query
        )


    # ========================================================
    # ANN
    # ========================================================

    def ann_search(
        self,
        query
    ):

        # Chroma's vector index performs ANN-style
        # nearest-neighbor retrieval internally.

        return self.cosine_search(
            query
        )


    # ========================================================
    # DOT PRODUCT
    # ========================================================

    def dot_product_search(
        self,
        query
    ):

        query_vector = np.asarray(

            self.embed_query(
                query
            ),

            dtype=float

        )

        db = self.get_chroma()

        data = db.get(

            include=[
                "documents",
                "metadatas",
                "embeddings"
            ]

        )

        documents = (
            data.get(
                "documents",
                []
            )
        )

        metadatas = (
            data.get(
                "metadatas",
                []
            )
        )

        embeddings = (
            data.get(
                "embeddings",
                []
            )
        )

        scored = []

        for (

            document,
            metadata,
            embedding

        ) in zip(

            documents,
            metadatas,
            embeddings

        ):

            vector = np.asarray(
                embedding,
                dtype=float
            )

            score = float(
                np.dot(
                    query_vector,
                    vector
                )
            )

            scored.append({

                "document":
                    document,

                "metadata":
                    metadata,

                "score":
                    score

            })


        scored.sort(

            key=lambda x:
                x["score"],

            reverse=True

        )


        return self.group_results(
            scored[
                :self.top_k
            ]
        )


    # ========================================================
    # L2
    # ========================================================

    def l2_search(
        self,
        query
    ):

        query_vector = np.asarray(

            self.embed_query(
                query
            ),

            dtype=float

        )

        db = self.get_chroma()

        data = db.get(

            include=[
                "documents",
                "metadatas",
                "embeddings"
            ]

        )

        scored = []

        for (

            document,
            metadata,
            embedding

        ) in zip(

            data.get(
                "documents",
                []
            ),

            data.get(
                "metadatas",
                []
            ),

            data.get(
                "embeddings",
                []
            )

        ):

            vector = np.asarray(
                embedding,
                dtype=float
            )

            distance = float(
                np.linalg.norm(
                    query_vector
                    -
                    vector
                )
            )

            score = (
                1.0
                /
                (
                    1.0
                    +
                    distance
                )
            )

            scored.append({

                "document":
                    document,

                "metadata":
                    metadata,

                "score":
                    score

            })


        scored.sort(

            key=lambda x:
                x["score"],

            reverse=True

        )


        return self.group_results(
            scored[
                :self.top_k
            ]
        )


    # ========================================================
    # BM25
    # ========================================================

    def bm25_search(
        self,
        query,
        resume_chunks=None
    ):

        if not resume_chunks:

            return []


        documents = []

        metadata = []


        for (

            file_name,
            chunks

        ) in resume_chunks.items():

            for chunk in chunks:

                documents.append(
                    chunk["text"]
                )

                metadata.append({

                    "file_name":
                        file_name,

                    "chunk_id":
                        chunk["chunk_id"]

                })


        if not documents:

            return []


        try:

            from rank_bm25 import (
                BM25Okapi
            )

        except ImportError:

            raise ImportError(

                "rank-bm25 is required for BM25. "
                "Install with: pip install rank-bm25"

            )


        tokenized_documents = [

            self.tokenize(
                document
            )

            for document
            in documents

        ]


        bm25 = BM25Okapi(
            tokenized_documents
        )


        query_tokens = (
            self.tokenize(
                query
            )
        )


        scores = bm25.get_scores(
            query_tokens
        )


        ranked = []


        for (

            document,
            meta,
            score

        ) in zip(

            documents,
            metadata,
            scores

        ):

            ranked.append({

                "document":
                    document,

                "metadata":
                    meta,

                "score":
                    float(score)

            })


        ranked.sort(

            key=lambda x:
                x["score"],

            reverse=True

        )


        return self.group_results(
            ranked[
                :self.top_k
            ]
        )


    # ========================================================
    # HYBRID
    # ========================================================

    def hybrid_search(
        self,
        query,
        resume_chunks=None
    ):

        dense_results = (
            self.cosine_search(
                query
            )
        )

        sparse_results = (
            self.bm25_search(
                query,
                resume_chunks
            )
        )


        dense_scores = {}

        sparse_scores = {}


        for result in dense_results:

            key = (
                result["file_name"]
            )

            dense_scores[
                key
            ] = result["score"]


        for result in sparse_results:

            key = (
                result["file_name"]
            )

            sparse_scores[
                key
            ] = result["score"]


        all_files = set(
            dense_scores
        ) | set(
            sparse_scores
        )


        results = []


        max_sparse = max(

            sparse_scores.values(),

            default=1.0

        )


        for file_name in all_files:

            dense_score = (
                dense_scores.get(
                    file_name,
                    0.0
                )
            )

            sparse_score = (
                sparse_scores.get(
                    file_name,
                    0.0
                )
            )


            if max_sparse > 0:

                sparse_score = (
                    sparse_score
                    /
                    max_sparse
                )


            final_score = (

                0.7
                *
                dense_score

                +

                0.3
                *
                sparse_score

            )


            results.append({

                "file_name":
                    file_name,

                "score":
                    final_score,

                "chunks":
                    self.get_resume_chunks(
                        file_name,
                        resume_chunks
                    ),

                "metadata": {}

            })


        results.sort(

            key=lambda x:
                x["score"],

            reverse=True

        )


        return results[
            :self.top_k
        ]


    # ========================================================
    # CROSS ENCODER
    # ========================================================

    def cross_encoder_search(
        self,
        query,
        resume_chunks=None
    ):

        if not resume_chunks:

            return []


        try:

            from sentence_transformers import (
                CrossEncoder
            )

        except ImportError:

            raise ImportError(

                "sentence-transformers is required "
                "for CrossEncoder."

            )


        model = CrossEncoder(

            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"

        )


        candidates = []


        for (

            file_name,
            chunks

        ) in resume_chunks.items():

            for chunk in chunks:

                candidates.append({

                    "file_name":
                        file_name,

                    "text":
                        chunk["text"],

                    "chunk_id":
                        chunk["chunk_id"]

                })


        pairs = [

            [
                query,
                candidate["text"]
            ]

            for candidate
            in candidates

        ]


        scores = model.predict(
            pairs
        )


        ranked = []


        for candidate, score in zip(

            candidates,
            scores

        ):

            ranked.append({

                "file_name":
                    candidate[
                        "file_name"
                    ],

                "score":
                    float(score),

                "chunks": [

                    {

                        "text":
                            candidate[
                                "text"
                            ],

                        "chunk_id":
                            candidate[
                                "chunk_id"
                            ]

                    }

                ],

                "metadata": {}

            })


        ranked.sort(

            key=lambda x:
                x["score"],

            reverse=True

        )


        return ranked[
            :self.top_k
        ]


    # ========================================================
    # MMR
    # ========================================================

    def mmr_search(
        self,
        query
    ):

        query_vector = np.asarray(

            self.embed_query(
                query
            ),

            dtype=float

        )

        db = self.get_chroma()

        data = db.get(

            include=[
                "documents",
                "metadatas",
                "embeddings"
            ]

        )

        documents = data.get(
            "documents",
            []
        )

        metadatas = data.get(
            "metadatas",
            []
        )

        embeddings = np.asarray(

            data.get(
                "embeddings",
                []
            ),

            dtype=float

        )


        if len(
            embeddings
        ) == 0:

            return []


        embeddings = self.normalize_matrix(
            embeddings
        )

        query_vector = self.normalize_vector(
            query_vector
        )


        relevance = (
            embeddings
            @
            query_vector
        )


        selected = []

        remaining = list(
            range(
                len(documents)
            )
        )


        lambda_value = 0.7


        while (

            remaining

            and

            len(selected)
            <
            self.top_k

        ):

            best_index = None

            best_score = -float(
                "inf"
            )


            for index in remaining:

                relevance_score = (
                    relevance[index]
                )


                if not selected:

                    diversity_penalty = 0

                else:

                    similarity = max(

                        np.dot(

                            embeddings[index],

                            embeddings[
                                selected_index
                            ]

                        )

                        for selected_index
                        in selected

                    )

                    diversity_penalty = (
                        similarity
                    )


                mmr_score = (

                    lambda_value
                    *
                    relevance_score

                    -

                    (
                        1
                        -
                        lambda_value
                    )
                    *
                    diversity_penalty

                )


                if mmr_score > best_score:

                    best_score = (
                        mmr_score
                    )

                    best_index = index


            selected.append(
                best_index
            )

            remaining.remove(
                best_index
            )


        results = []


        for index in selected:

            results.append({

                "document":
                    documents[index],

                "metadata":
                    metadatas[index],

                "score":
                    float(
                        relevance[index]
                    )

            })


        return self.group_results(
            results
        )


    # ========================================================
    # COLBERT
    # ========================================================

    def colbert_search(
        self,
        query,
        resume_chunks=None
    ):

        raise NotImplementedError(

            "True ColBERT / late-interaction retrieval "
            "requires token-level embeddings and MaxSim "
            "scoring. It should not be treated as ordinary "
            "single-vector cosine retrieval."

        )


    # ========================================================
    # GET RESUME CHUNKS
    # ========================================================

    @staticmethod
    def get_resume_chunks(
        file_name,
        resume_chunks
    ):

        if not resume_chunks:

            return []


        return resume_chunks.get(
            file_name,
            []
        )


    # ========================================================
    # GROUP CHROMA RESULTS
    # ========================================================

    def convert_chroma_results(
        self,
        result
    ):

        ids = result.get(
            "ids",
            [[]]
        )[0]

        documents = result.get(
            "documents",
            [[]]
        )[0]

        metadatas = result.get(
            "metadatas",
            [[]]
        )[0]

        distances = result.get(
            "distances",
            [[]]
        )[0]


        grouped = {}


        for index, document in enumerate(
            documents
        ):

            metadata = (

                metadatas[index]

                if index < len(metadatas)

                else {}

            )


            distance = (

                distances[index]

                if index < len(distances)

                else 0

            )


            # Chroma cosine distance:
            # similarity ≈ 1 - distance

            score = max(

                -1.0,

                min(

                    1.0,

                    1.0
                    -
                    float(distance)

                )

            )


            file_name = metadata.get(
                "file_name",
                "Unknown"
            )


            if file_name not in grouped:

                grouped[
                    file_name
                ] = {

                    "file_name":
                        file_name,

                    "score":
                        score,

                    "chunks": [],

                    "metadata": metadata

                }


            grouped[
                file_name
            ]["chunks"].append({

                "id":
                    ids[index]
                    if index < len(ids)
                    else "",

                "text":
                    document,

                "score":
                    score,

                "metadata":
                    metadata

            })


            grouped[
                file_name
            ]["score"] = max(

                grouped[
                    file_name
                ]["score"],

                score

            )


        results = list(
            grouped.values()
        )


        results.sort(

            key=lambda x:
                x["score"],

            reverse=True

        )


        return results[
            :self.top_k
        ]


    # ========================================================
    # GROUP GENERIC RESULTS
    # ========================================================

    def group_results(
        self,
        scored_results
    ):

        grouped = {}


        for result in scored_results:

            metadata = result.get(
                "metadata",
                {}
            )


            file_name = metadata.get(
                "file_name",
                result.get(
                    "file_name",
                    "Unknown"
                )
            )


            if file_name not in grouped:

                grouped[
                    file_name
                ] = {

                    "file_name":
                        file_name,

                    "score":
                        float(
                            result.get(
                                "score",
                                0
                            )
                        ),

                    "chunks": [],

                    "metadata":
                        metadata

                }


            grouped[
                file_name
            ]["chunks"].append({

                "text":
                    result.get(
                        "document",
                        ""
                    ),

                "score":
                    result.get(
                        "score",
                        0
                    ),

                "metadata":
                    metadata

            })


            grouped[
                file_name
            ]["score"] = max(

                grouped[
                    file_name
                ]["score"],

                float(
                    result.get(
                        "score",
                        0
                    )
                )

            )


        results = list(
            grouped.values()
        )


        results.sort(

            key=lambda x:
                x["score"],

            reverse=True

        )


        return results[
            :self.top_k
        ]


    # ========================================================
    # TOKENIZE
    # ========================================================

    @staticmethod
    def tokenize(
        text
    ):

        text = str(
            text or ""
        ).lower()


        text = re.sub(

            r"[^a-z0-9+#.\s]",

            " ",

            text

        )


        return text.split()


    # ========================================================
    # NORMALIZE VECTOR
    # ========================================================

    @staticmethod
    def normalize_vector(
        vector
    ):

        norm = np.linalg.norm(
            vector
        )


        if norm == 0:

            return vector


        return vector / norm


    # ========================================================
    # NORMALIZE MATRIX
    # ========================================================

    @staticmethod
    def normalize_matrix(
        matrix
    ):

        norms = np.linalg.norm(

            matrix,

            axis=1,

            keepdims=True

        )


        norms[norms == 0] = 1


        return (
            matrix
            /
            norms
        )