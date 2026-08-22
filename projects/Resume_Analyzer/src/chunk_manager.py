class ChunkManager:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        context_window,
        output_tokens,
        safety_buffer,
        top_k
    ):

        if context_window is None:

            raise ValueError(
                "context_window is required."
            )


        if output_tokens is None:

            raise ValueError(
                "output_tokens is required."
            )


        if safety_buffer is None:

            raise ValueError(
                "safety_buffer is required."
            )


        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0."
            )


        self.context_window = (
            context_window
        )

        self.output_tokens = (
            output_tokens
        )

        self.safety_buffer = (
            safety_buffer
        )

        self.top_k = top_k


    # ========================================================
    # FIXED TOKENS
    # ========================================================

    def calculate_fixed_tokens(
        self,
        schema_tokens,
        system_prompt_tokens,
        job_description_tokens
    ):

        return (

            schema_tokens

            + system_prompt_tokens

            + job_description_tokens

            + self.output_tokens

            + self.safety_buffer

        )


    # ========================================================
    # RETRIEVAL BUDGET
    # ========================================================

    def calculate_retrieval_budget(
        self,
        schema_tokens,
        system_prompt_tokens,
        job_description_tokens
    ):

        fixed_tokens = (
            self.calculate_fixed_tokens(

                schema_tokens,

                system_prompt_tokens,

                job_description_tokens

            )
        )


        retrieval_budget = (
            self.context_window
            - fixed_tokens
        )


        if retrieval_budget < 0:

            retrieval_budget = 0


        return retrieval_budget


    # ========================================================
    # CHUNK SIZE
    # ========================================================

    def calculate_chunk_size(
        self,
        retrieval_budget
    ):

        if retrieval_budget <= 0:

            raise ValueError(
                "No token budget available for retrieval."
            )


        chunk_size = (
            retrieval_budget
            // self.top_k
        )


        if chunk_size < 100:

            chunk_size = 100


        return chunk_size


    # ========================================================
    # CHUNK OVERLAP
    # ========================================================

    def calculate_chunk_overlap(
        self,
        chunk_size
    ):

        chunk_overlap = int(
            chunk_size * 0.10
        )


        if chunk_overlap < 50:

            chunk_overlap = 50


        if chunk_overlap >= chunk_size:

            chunk_overlap = (
                chunk_size // 5
            )


        return chunk_overlap


    # ========================================================
    # COMPLETE CALCULATION
    # ========================================================

    def calculate(
        self,
        schema_tokens,
        system_prompt_tokens,
        job_description_tokens
    ):

        fixed_tokens = (
            self.calculate_fixed_tokens(

                schema_tokens,

                system_prompt_tokens,

                job_description_tokens

            )
        )


        retrieval_budget = (
            self.calculate_retrieval_budget(

                schema_tokens,

                system_prompt_tokens,

                job_description_tokens

            )
        )


        chunk_size = (
            self.calculate_chunk_size(
                retrieval_budget
            )
        )


        chunk_overlap = (
            self.calculate_chunk_overlap(
                chunk_size
            )
        )


        return {

            "context_window":
                self.context_window,

            "fixed_tokens":
                fixed_tokens,

            "retrieval_budget":
                retrieval_budget,

            "top_k":
                self.top_k,

            "chunk_size":
                chunk_size,

            "chunk_overlap":
                chunk_overlap

        }