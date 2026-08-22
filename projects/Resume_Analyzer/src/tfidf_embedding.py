from sklearn.feature_extraction.text import (
    TfidfVectorizer
)


class TFIDFEmbedding:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        max_features=10000
    ):

        self.max_features = (
            max_features
        )


        self.vectorizer = (
            TfidfVectorizer(

                max_features=
                    self.max_features,

                ngram_range=(
                    1,
                    2
                )

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

        if not documents:

            raise ValueError(
                "Documents are required for TF-IDF."
            )


        self.vectorizer.fit(
            documents
        )


        self.is_fitted = True


    # ========================================================
    # SINGLE TEXT
    # ========================================================

    def embed_text(
        self,
        text
    ):

        if not self.is_fitted:

            raise ValueError(
                "TF-IDF must be fitted before embedding."
            )


        vector = (
            self.vectorizer.transform(
                [text]
            )
        )


        return (
            vector
            .toarray()[0]
            .tolist()
        )


    # ========================================================
    # MULTIPLE DOCUMENTS
    # ========================================================

    def embed_documents(
        self,
        documents
    ):

        if not self.is_fitted:

            self.fit(
                documents
            )


        vectors = (
            self.vectorizer.transform(
                documents
            )
        )


        return (
            vectors
            .toarray()
            .tolist()
        )