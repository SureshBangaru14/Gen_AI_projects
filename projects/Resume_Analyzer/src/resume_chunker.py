import tiktoken


class ResumeChunker:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        model_name,
        chunk_size,
        chunk_overlap
    ):

        self.model_name = (
            model_name
        )

        self.chunk_size = (
            int(chunk_size)
        )

        self.chunk_overlap = (
            int(chunk_overlap)
        )


        # ----------------------------------------------------
        # TOKENIZER
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


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if self.chunk_size <= 0:

            raise ValueError(
                "chunk_size must be greater than 0."
            )


        if self.chunk_overlap < 0:

            raise ValueError(
                "chunk_overlap cannot be negative."
            )


        if self.chunk_overlap >= self.chunk_size:

            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )


    # ========================================================
    # TEXT → TOKENS
    # ========================================================

    def text_to_tokens(
        self,
        text
    ):

        return (
            self.encoding.encode(
                text
            )
        )


    # ========================================================
    # TOKENS → TEXT
    # ========================================================

    def tokens_to_text(
        self,
        tokens
    ):

        return (
            self.encoding.decode(
                tokens
            )
        )


    # ========================================================
    # TOKEN CHUNKING
    # ========================================================

    def chunk_text(
        self,
        text
    ):

        if not text:

            return []


        # ----------------------------------------------------
        # TEXT → TOKENS
        # ----------------------------------------------------

        tokens = (
            self.text_to_tokens(
                text
            )
        )


        if not tokens:

            return []


        # ----------------------------------------------------
        # TOTAL TOKENS
        # ----------------------------------------------------

        total_tokens = len(
            tokens
        )


        chunks = []


        start = 0


        # ====================================================
        # LOOP
        # ====================================================

        while start < total_tokens:

            # ------------------------------------------------
            # END
            # ------------------------------------------------

            end = min(
                start + self.chunk_size,
                total_tokens
            )


            # ------------------------------------------------
            # CHUNK TOKENS
            # ------------------------------------------------

            chunk_tokens = (
                tokens[
                    start:end
                ]
            )


            # ------------------------------------------------
            # TOKENS → TEXT
            # ------------------------------------------------

            chunk_text = (
                self.tokens_to_text(
                    chunk_tokens
                )
            )


            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            chunks.append(
                chunk_text
            )


            # ------------------------------------------------
            # LAST CHUNK
            # ------------------------------------------------

            if end >= total_tokens:

                break


            # ------------------------------------------------
            # NEXT START
            # ------------------------------------------------

            start = (
                end
                - self.chunk_overlap
            )


        return chunks