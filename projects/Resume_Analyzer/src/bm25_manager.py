import re

from rank_bm25 import BM25Okapi


class BM25Manager:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(self):

        self.bm25 = None

        self.documents = []

        self.metadata = []


    # ========================================================
    # TOKENIZE
    # ========================================================

    def tokenize(
        self,
        text
    ):

        if not text:

            return []


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
        documents,
        metadata=None
    ):

        if not documents:

            raise ValueError(
                "BM25 requires at least one document."
            )


        self.documents = list(
            documents
        )


        if metadata is None:

            self.metadata = [

                {}

                for _ in documents

            ]

        else:

            self.metadata = list(
                metadata
            )


        tokenized_documents = [

            self.tokenize(
                document
            )

            for document
            in documents

        ]


        self.bm25 = BM25Okapi(

            tokenized_documents

        )


        return self


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=5
    ):

        if self.bm25 is None:

            raise ValueError(

                "BM25 model has not been fitted."

            )


        query_tokens = (
            self.tokenize(
                query
            )
        )


        if not query_tokens:

            return []


        scores = (
            self.bm25.get_scores(
                query_tokens
            )
        )


        ranked_indices = sorted(

            range(
                len(scores)
            ),

            key=lambda index:
                scores[index],

            reverse=True

        )


        ranked_indices = (
            ranked_indices[:top_k]
        )


        results = []


        for index in ranked_indices:

            results.append(

                {

                    "document":
                        self.documents[
                            index
                        ],

                    "score":
                        float(
                            scores[
                                index
                            ]
                        ),

                    "metadata":
                        self.metadata[
                            index
                        ]

                }

            )


        return results