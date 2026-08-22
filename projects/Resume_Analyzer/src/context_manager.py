class ContextManager:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        context_window,
        output_tokens,
        safety_buffer
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


        self.context_window = (
            context_window
        )

        self.output_tokens = (
            output_tokens
        )

        self.safety_buffer = (
            safety_buffer
        )


    # ========================================================
    # TOTAL TOKENS
    # ========================================================

    def calculate_total_tokens(
        self,
        schema_tokens,
        system_prompt_tokens,
        user_prompt_tokens
    ):

        return (

            schema_tokens

            + system_prompt_tokens

            + user_prompt_tokens

            + self.output_tokens

            + self.safety_buffer

        )


    # ========================================================
    # AVAILABLE TOKENS
    # ========================================================

    def calculate_available_tokens(
        self,
        total_tokens
    ):

        available_tokens = (
            self.context_window
            - total_tokens
        )


        if available_tokens < 0:

            available_tokens = 0


        return available_tokens


    # ========================================================
    # CONTEXT CHECK
    # ========================================================

    def check_context_window(
        self,
        total_tokens
    ):

        return (
            total_tokens
            <= self.context_window
        )


    # ========================================================
    # COMPLETE CALCULATION
    # ========================================================

    def calculate(
        self,
        schema_tokens,
        system_prompt_tokens,
        user_prompt_tokens,
        job_description_tokens,
        resume_tokens
    ):

        total_tokens = (
            self.calculate_total_tokens(

                schema_tokens,

                system_prompt_tokens,

                user_prompt_tokens

            )
        )


        available_tokens = (
            self.calculate_available_tokens(
                total_tokens
            )
        )


        fits_context = (
            self.check_context_window(
                total_tokens
            )
        )


        return {

            "context_window":
                self.context_window,

            "schema_tokens":
                schema_tokens,

            "system_prompt_tokens":
                system_prompt_tokens,

            "user_prompt_tokens":
                user_prompt_tokens,

            "job_description_tokens":
                job_description_tokens,

            "resume_tokens":
                resume_tokens,

            "output_tokens":
                self.output_tokens,

            "safety_buffer":
                self.safety_buffer,

            "total_tokens":
                total_tokens,

            "available_tokens":
                available_tokens,

            "fits_context":
                fits_context

        }