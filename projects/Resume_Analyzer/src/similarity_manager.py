class SimilarityManager:

    # ========================================================
    # COSINE
    # ========================================================

    @staticmethod
    def cosine(
        distance
    ):

        similarity = (
            1.0 - distance
        )


        return max(
            0.0,
            min(
                similarity,
                1.0
            )
        )


    # ========================================================
    # INNER PRODUCT
    # ========================================================

    @staticmethod
    def inner_product(
        distance
    ):

        similarity = (
            1.0 - distance
        )


        return max(
            0.0,
            min(
                similarity,
                1.0
            )
        )


    # ========================================================
    # L2
    # ========================================================

    @staticmethod
    def l2(
        distance
    ):

        similarity = (
            1.0
            /
            (1.0 + distance)
        )


        return max(
            0.0,
            min(
                similarity,
                1.0
            )
        )


    # ========================================================
    # CALCULATE
    # ========================================================

    @classmethod
    def calculate_score(
        cls,
        distance,
        method
    ):

        if method == (
            "Cosine Similarity"
        ):

            similarity = (
                cls.cosine(
                    distance
                )
            )


        elif method == (
            "Dot Product / Inner Product"
        ):

            similarity = (
                cls.inner_product(
                    distance
                )
            )


        elif method == (
            "Euclidean Distance (L2)"
        ):

            similarity = (
                cls.l2(
                    distance
                )
            )


        else:

            similarity = (
                cls.cosine(
                    distance
                )
            )


        percentage = round(
            similarity * 100,
            2
        )


        return {

            "distance":
                round(
                    distance,
                    6
                ),

            "similarity":
                round(
                    similarity,
                    6
                ),

            "percentage":
                percentage

        }