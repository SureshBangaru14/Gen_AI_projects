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

        self.context_window = int(
            context_window
        )

        self.output_tokens = int(
            output_tokens
        )

        self.safety_buffer = int(
            safety_buffer
        )


    # ========================================================
    # AVAILABLE INPUT TOKENS
    # ========================================================

    def available_input_tokens(self):

        available = (

            self.context_window

            -

            self.output_tokens

            -

            self.safety_buffer

        )


        return max(
            0,
            available
        )


    # ========================================================
    # CALCULATE
    # ========================================================

    def calculate(
        self,
        schema_tokens=0,
        system_prompt_tokens=0,
        user_prompt_tokens=0,
        job_description_tokens=0,
        resume_tokens=0
    ):

        fixed_tokens = (

            schema_tokens

            +

            system_prompt_tokens

        )


        available_input = (
            self.available_input_tokens()
        )


        total_input = (

            fixed_tokens

            +

            user_prompt_tokens

        )


        remaining_tokens = (

            available_input

            -

            total_input

        )


        # ====================================================
        # CONTEXT USAGE
        # ====================================================

        total_context_used = (

            fixed_tokens

            +

            user_prompt_tokens

            +

            self.output_tokens

            +

            self.safety_buffer

        )


        context_usage_percent = (

            total_context_used

            /

            self.context_window

            *

            100

        )


        # ====================================================
        # RESUME FIT
        # ====================================================

        resume_fits_directly = (

            resume_tokens
            <=
            max(
                0,
                remaining_tokens
            )

        )


        return {

            "context_window":
                self.context_window,

            "output_tokens":
                self.output_tokens,

            "safety_buffer":
                self.safety_buffer,

            "available_input_tokens":
                available_input,

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

            "total_context_used":
                total_context_used,

            "remaining_tokens":
                max(
                    0,
                    remaining_tokens
                ),

            "context_usage_percent":
                round(
                    context_usage_percent,
                    2
                ),

            "resume_fits_directly":
                resume_fits_directly

        }