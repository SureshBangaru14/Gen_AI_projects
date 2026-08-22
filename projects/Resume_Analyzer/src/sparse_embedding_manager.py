from sklearn.feature_extraction.text import (
    TfidfVectorizer
)


class SparseEmbeddingManager:

    def __init__(self):

        self.vectorizer = (
            TfidfVectorizer()
        )

        self.fitted = False


    def fit(
        self,
        texts
    ):

        self.vectorizer.fit(
            texts
        )

        self.fitted = True


    def embed_documents(
        self,
        texts
    ):

        if not self.fitted:

            self.fit(
                texts
            )


        return (
            self.vectorizer.transform(
                texts
            )
        )


    def embed_text(
        self,
        text
    ):

        if not self.fitted:

            raise ValueError(
                "Sparse vectorizer "
                "has not been fitted."
            )


        return (
            self.vectorizer.transform(
                [text]
            )
        )