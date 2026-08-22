import tiktoken


class TokenCounter:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        model_name="gpt-4o-mini"
    ):

        self.model_name = model_name

        try:

            self.encoder = (
                tiktoken.encoding_for_model(
                    model_name
                )
            )

        except Exception:

            self.encoder = (
                tiktoken.get_encoding(
                    "cl100k_base"
                )
            )


    # ========================================================
    # COUNT TOKENS
    # ========================================================

    def count_tokens(
        self,
        text
    ):

        if not text:

            return 0


        if not isinstance(
            text,
            str
        ):

            text = str(
                text
            )


        return len(

            self.encoder.encode(
                text
            )

        )


    # ========================================================
    # COUNT MULTIPLE TEXTS
    # ========================================================

    def count_multiple(
        self,
        texts
    ):

        return {

            key:
                self.count_tokens(value)

            for key, value
            in texts.items()

        }