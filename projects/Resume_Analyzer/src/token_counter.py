import tiktoken


class TokenCounter:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        model_name
    ):

        self.model_name = model_name


        # ----------------------------------------------------
        # GET MODEL ENCODING
        # ----------------------------------------------------

        try:

            self.encoding = (
                tiktoken.encoding_for_model(
                    model_name
                )
            )

        except KeyError:

            self.encoding = (
                tiktoken.get_encoding(
                    "o200k_base"
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


        tokens = (
            self.encoding.encode(
                text
            )
        )


        return len(
            tokens
        )


    # ========================================================
    # GET TOKENS
    # ========================================================

    def encode(
        self,
        text
    ):

        return (
            self.encoding.encode(
                text
            )
        )


    # ========================================================
    # DECODE TOKENS
    # ========================================================

    def decode(
        self,
        tokens
    ):

        return (
            self.encoding.decode(
                tokens
            )
        )