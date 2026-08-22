from openai import OpenAI


class OpenAIEmbeddingManager:

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


        self.model_name = model_name


    def embed_text(
        self,
        text
    ):

        response = (
            self.client.embeddings.create(

                model=
                    self.model_name,

                input=
                    text

            )
        )


        return (
            response.data[
                0
            ].embedding
        )


    def embed_documents(
        self,
        texts
    ):

        if not texts:

            return []


        response = (
            self.client.embeddings.create(

                model=
                    self.model_name,

                input=
                    texts

            )
        )


        return [

            item.embedding

            for item in response.data

        ]