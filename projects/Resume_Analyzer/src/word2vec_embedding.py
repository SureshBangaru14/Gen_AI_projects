import re

import numpy as np

from gensim.models import Word2Vec


class Word2VecEmbedding:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        vector_size=300,
        window=5,
        min_count=1,
        workers=4
    ):

        self.vector_size = (
            vector_size
        )

        self.window = (
            window
        )

        self.min_count = (
            min_count
        )

        self.workers = (
            workers
        )

        self.model = None


    # ========================================================
    # TOKENIZE
    # ========================================================

    def tokenize(
        self,
        text
    ):

        tokens = re.findall(

            r"\b[a-zA-Z0-9+#.]+\b",

            text.lower()

        )


        return tokens


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


        tokenized_documents = [

            tokens

            for tokens
            in tokenized_documents

            if tokens

        ]


        if not tokenized_documents:

            raise ValueError(
                "No valid text available "
                "for Word2Vec training."
            )


        self.model = Word2Vec(

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


    # ========================================================
    # SINGLE TEXT
    # ========================================================

    def embed_text(
        self,
        text
    ):

        if self.model is None:

            raise ValueError(
                "Word2Vec must be fitted before embedding."
            )


        tokens = (
            self.tokenize(
                text
            )
        )


        vectors = []


        for token in tokens:

            if token in self.model.wv:

                vectors.append(

                    self.model.wv[
                        token
                    ]

                )


        if not vectors:

            return [

                0.0

                for _ in range(
                    self.vector_size
                )

            ]


        document_vector = (
            np.mean(
                vectors,
                axis=0
            )
        )


        return (
            document_vector
            .tolist()
        )


    # ========================================================
    # MULTIPLE DOCUMENTS
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

            self.embed_text(
                document
            )

            for document
            in documents

        ]