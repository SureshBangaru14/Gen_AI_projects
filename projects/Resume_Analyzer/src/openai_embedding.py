from openai import OpenAI


class OpenAIEmbedding:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        api_key,
        model_name="text-embedding-3-small"
    ):

        if not api_key:

            raise ValueError(
                "OpenAI API key is required."
            )


        self.client = OpenAI(
            api_key=api_key
        )


        self.model_name = (
            model_name
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


        response = (
            self.client.embeddings.create(

                model=
                    self.model_name,

                input=
                    text

            )
        )


        return (
            response
            .data[0]
            .embedding
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


        response = (
            self.client.embeddings.create(

                model=
                    self.model_name,

                input=
                    documents

            )
        )


        return [

            item.embedding

            for item
            in response.data

        ]