class ChunkManager:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        context_window,
        output_tokens,
        safety_buffer,
        top_k=5,
        maximum_chunk_size=1500,
        minimum_chunk_size=100,
        overlap_percentage=15
    ):

        self.context_window = int(
            context_window
        )

        self.output_tokens = int(
            output_tokens
        )

        self.safety_buffer = int(
            safety_buffer
        )

        self.top_k = max(
            1,
            int(top_k)
        )

        self.maximum_chunk_size = int(
            maximum_chunk_size
        )

        self.minimum_chunk_size = int(
            minimum_chunk_size
        )

        self.overlap_percentage = float(
            overlap_percentage
        )


    # ========================================================
    # CALCULATE AVAILABLE TOKENS
    # ========================================================

    def calculate_available_tokens(
        self,
        schema_tokens,
        system_prompt_tokens,
        user_prompt_instruction_tokens,
        job_description_tokens
    ):

        reserved_tokens = (

            int(schema_tokens)

            +

            int(system_prompt_tokens)

            +

            int(user_prompt_instruction_tokens)

            +

            int(job_description_tokens)

            +

            self.output_tokens

            +

            self.safety_buffer

        )


        available_tokens = (

            self.context_window

            -

            reserved_tokens

        )


        return {

            "reserved_tokens":
                reserved_tokens,

            "available_tokens":
                max(
                    0,
                    available_tokens
                )

        }


    # ========================================================
    # CALCULATE CHUNK SIZE
    # ========================================================

    def calculate(
        self,
        schema_tokens=0,
        system_prompt_tokens=0,
        user_prompt_template_tokens=0,
        job_description_tokens=0
    ):

        token_data = (
            self.calculate_available_tokens(

                schema_tokens=
                    schema_tokens,

                system_prompt_tokens=
                    system_prompt_tokens,

                user_prompt_instruction_tokens=
                    user_prompt_template_tokens,

                job_description_tokens=
                    job_description_tokens

            )
        )


        available_tokens = (
            token_data[
                "available_tokens"
            ]
        )


        reserved_tokens = (
            token_data[
                "reserved_tokens"
            ]
        )


        # ====================================================
        # TOKEN BUDGET PER CHUNK
        #
        # We don't want to fill the entire model context
        # with one chunk.
        #
        # Top-K controls how much retrieval context we expect.
        # ====================================================

        chunk_size = int(

            available_tokens
            /
            self.top_k

        )


        # ====================================================
        # MAXIMUM LIMIT
        # ====================================================

        chunk_size = min(

            chunk_size,

            self.maximum_chunk_size

        )


        # ====================================================
        # MINIMUM LIMIT
        # ====================================================

        chunk_size = max(

            chunk_size,

            self.minimum_chunk_size

        )


        # ====================================================
        # OVERLAP
        # ====================================================

        chunk_overlap = int(

            chunk_size
            *
            self.overlap_percentage
            /
            100

        )


        # ====================================================
        # SAFETY CHECK
        # ====================================================

        if chunk_overlap >= chunk_size:

            chunk_overlap = int(

                chunk_size
                *
                0.10

            )


        chunk_overlap = max(

            1,

            chunk_overlap

        )


        return {

            "context_window":
                self.context_window,

            "output_tokens":
                self.output_tokens,

            "safety_buffer":
                self.safety_buffer,

            "schema_tokens":
                int(
                    schema_tokens
                ),

            "system_prompt_tokens":
                int(
                    system_prompt_tokens
                ),

            "user_prompt_instruction_tokens":
                int(
                    user_prompt_template_tokens
                ),

            "job_description_tokens":
                int(
                    job_description_tokens
                ),

            "reserved_tokens":
                reserved_tokens,

            "available_tokens":
                available_tokens,

            "chunk_size":
                chunk_size,

            "chunk_overlap":
                chunk_overlap,

            "top_k":
                self.top_k,

            "overlap_percentage":
                self.overlap_percentage

        }


    # ========================================================
    # CALCULATE FOR A SPECIFIC RESUME
    # ========================================================

    def calculate_for_resume(
        self,
        schema_tokens,
        system_prompt_tokens,
        user_prompt_tokens,
        job_description_tokens,
        resume_tokens
    ):

        # ====================================================
        # TOTAL INPUT
        # ====================================================

        total_input_tokens = (

            int(schema_tokens)

            +

            int(system_prompt_tokens)

            +

            int(user_prompt_tokens)

        )


        # ====================================================
        # TOTAL REQUEST
        # ====================================================

        total_request_tokens = (

            total_input_tokens

            +

            self.output_tokens

            +

            self.safety_buffer

        )


        # ====================================================
        # AVAILABLE
        # ====================================================

        available_tokens = (

            self.context_window

            -

            total_input_tokens

            -

            self.output_tokens

            -

            self.safety_buffer

        )


        available_tokens = max(

            0,

            available_tokens

        )


        # ====================================================
        # RESUME FIT
        # ====================================================

        resume_fits = (

            int(resume_tokens)

            <=

            available_tokens

        )


        # ====================================================
        # CONTEXT USAGE
        # ====================================================

        context_usage_percentage = (

            total_request_tokens

            /

            self.context_window

            *

            100

        )


        return {

            "resume_tokens":
                int(resume_tokens),

            "total_input_tokens":
                total_input_tokens,

            "output_tokens":
                self.output_tokens,

            "safety_buffer":
                self.safety_buffer,

            "total_request_tokens":
                total_request_tokens,

            "available_tokens":
                available_tokens,

            "resume_fits":
                resume_fits,

            "context_usage_percentage":
                round(
                    context_usage_percentage,
                    2
                )

        }