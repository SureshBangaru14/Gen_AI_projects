from langchain_text_splitters import RecursiveCharacterTextSplitter

import re


class ResumeChunker:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        chunk_size=500,
        chunk_overlap=100
    ):

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if chunk_size <= 0:

            raise ValueError(
                "chunk_size must be greater than 0."
            )


        if chunk_overlap < 0:

            raise ValueError(
                "chunk_overlap cannot be negative."
            )


        if chunk_overlap >= chunk_size:

            raise ValueError(
                "chunk_overlap must be smaller "
                "than chunk_size."
            )


        self.chunk_size = int(
            chunk_size
        )

        self.chunk_overlap = int(
            chunk_overlap
        )


        # ====================================================
        # RECURSIVE CHARACTER SPLITTER
        # ====================================================

        self.text_splitter = (
            RecursiveCharacterTextSplitter(

                chunk_size=
                    self.chunk_size,

                chunk_overlap=
                    self.chunk_overlap,

                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    ""
                ],

                length_function=len,

                is_separator_regex=False

            )
        )


    # ========================================================
    # CHUNK TEXT
    # ========================================================

    def chunk_text(
        self,
        text
    ):

        if not text:

            return []


        text = self.clean_text(
            text
        )


        if not text:

            return []


        chunks = (
            self.text_splitter.split_text(
                text
            )
        )


        # ----------------------------------------------------
        # REMOVE EMPTY CHUNKS
        # ----------------------------------------------------

        chunks = [

            chunk.strip()

            for chunk
            in chunks

            if chunk.strip()

        ]


        return chunks


    # ========================================================
    # CLEAN TEXT
    # ========================================================

    def clean_text(
        self,
        text
    ):

        if not text:

            return ""


        # ----------------------------------------------------
        # NORMALIZE WINDOWS LINE BREAKS
        # ----------------------------------------------------

        text = text.replace(
            "\r\n",
            "\n"
        )


        text = text.replace(
            "\r",
            "\n"
        )


        # ----------------------------------------------------
        # REMOVE EXCESS SPACES
        # ----------------------------------------------------

        text = re.sub(

            r"[ \t]+",

            " ",

            text

        )


        # ----------------------------------------------------
        # REMOVE EXCESS NEWLINES
        # ----------------------------------------------------

        text = re.sub(

            r"\n{3,}",

            "\n\n",

            text

        )


        return text.strip()


    # ========================================================
    # TEXT → TOKENS
    #
    # Used by DocumentProcess for chunk token information.
    # ========================================================

    def text_to_tokens(
        self,
        text
    ):

        if not text:

            return []


        # ----------------------------------------------------
        # Approximate tokenization
        #
        # For exact LLM token counts use TokenCounter.
        # ----------------------------------------------------

        tokens = re.findall(

            r"\w+|[^\w\s]",

            text

        )


        return tokens


    # ========================================================
    # CHUNK WITH METADATA
    # ========================================================

    def chunk_with_metadata(
        self,
        text,
        file_name=None
    ):

        chunks = self.chunk_text(
            text
        )


        results = []


        for index, chunk in enumerate(

            chunks,

            start=1

        ):

            results.append(

                {

                    "chunk_id":
                        index,

                    "file_name":
                        file_name,

                    "text":
                        chunk,

                    "token_count":
                        len(
                            self.text_to_tokens(
                                chunk
                            )
                        ),

                    "character_count":
                        len(
                            chunk
                        )

                }

            )


        return results