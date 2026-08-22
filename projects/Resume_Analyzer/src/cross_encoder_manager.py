from sentence_transformers import CrossEncoder


class CrossEncoderManager:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        model_name=(
            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"
        )
    ):

        self.model_name = (
            model_name
        )


        self.model = (
            CrossEncoder(
                self.model_name
            )
        )


    # ========================================================
    # RERANK
    # ========================================================

    def rerank(
        self,
        query,
        documents,
        top_k=10
    ):

        if not query:

            return []


        if not documents:

            return []


        # ====================================================
        # CREATE QUERY-DOCUMENT PAIRS
        # ====================================================

        pairs = [

            (
                query,
                document
            )

            for document
            in documents

        ]


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        scores = (
            self.model.predict(
                pairs
            )
        )


        # ====================================================
        # BUILD RESULTS
        # ====================================================

        results = []


        for (

            document,

            score

        ) in zip(

            documents,

            scores

        ):

            results.append(

                {

                    "document":
                        document,

                    "score":
                        float(score)

                }

            )


        # ====================================================
        # SORT
        # ====================================================

        results.sort(

            key=lambda item:
                item["score"],

            reverse=True

        )


        # ====================================================
        # TOP K
        # ====================================================

        return results[
            :top_k
        ]