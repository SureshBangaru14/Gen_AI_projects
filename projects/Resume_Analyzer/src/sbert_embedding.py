from sentence_transformers import (
    SentenceTransformer
)


class SBERTEmbedding:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        self.model_name = (
            model_name
        )


        self.model = (
            SentenceTransformer(
                self.model_name
            )
        )


    # ========================================================
    # SINGLE TEXT
    # ========================================================

    def embed_text(
        self,
        text
    ):

        if not text:

            return []


        embedding = (
            self.model.encode(

                text,

                convert_to_numpy=True,

                normalize_embeddings=True

            )
        )


        return (
            embedding
            .tolist()
        )


    # ========================================================
    # MULTIPLE DOCUMENTS
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

                convert_to_numpy=True,

                normalize_embeddings=True

            )
        )


        return (
            embeddings
            .tolist()
        )