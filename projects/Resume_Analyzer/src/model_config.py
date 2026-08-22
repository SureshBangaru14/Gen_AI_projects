class ModelConfig:

    MODELS = {

        "gpt-4o-mini": {
            "context_window": 128000,
            "default_output_tokens": 4096
        },

        "gpt-4o": {
            "context_window": 128000,
            "default_output_tokens": 4096
        }

    }

    # ========================================================
    # GET MODEL CONFIG
    # ========================================================

    @classmethod
    def get_model_config(
        cls,
        model_name
    ):

        if model_name in cls.MODELS:

            return cls.MODELS[
                model_name
            ]


        # ----------------------------------------------------
        # Safe default
        # ----------------------------------------------------

        return {

            "context_window": 128000,

            "default_output_tokens": 4096

        }