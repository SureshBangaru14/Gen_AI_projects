class HybridSearch:

    # ========================================================
    # WEIGHTED FUSION
    # ========================================================

    @staticmethod
    def weighted_fusion(
        dense_scores,
        sparse_scores,
        dense_weight=0.70,
        sparse_weight=0.30
    ):

        if len(
            dense_scores
        ) != len(
            sparse_scores
        ):

            raise ValueError(

                "Dense and sparse score lists "
                "must have the same length."

            )


        if dense_weight < 0:

            raise ValueError(
                "Dense weight cannot be negative."
            )


        if sparse_weight < 0:

            raise ValueError(
                "Sparse weight cannot be negative."
            )


        total_weight = (

            dense_weight
            +
            sparse_weight

        )


        if total_weight == 0:

            raise ValueError(

                "At least one search weight "
                "must be greater than zero."

            )


        # ====================================================
        # NORMALIZE WEIGHTS
        # ====================================================

        dense_weight = (

            dense_weight
            /
            total_weight

        )


        sparse_weight = (

            sparse_weight
            /
            total_weight

        )


        # ====================================================
        # FUSION
        # ====================================================

        hybrid_scores = []


        for (

            dense_score,

            sparse_score

        ) in zip(

            dense_scores,

            sparse_scores

        ):

            score = (

                dense_weight
                *
                float(dense_score)

                +

                sparse_weight
                *
                float(sparse_score)

            )


            hybrid_scores.append(
                float(score)
            )


        return hybrid_scores


    # ========================================================
    # RECIPROCAL RANK FUSION
    # ========================================================

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        k=60
    ):

        scores = {}


        # ====================================================
        # DENSE RESULTS
        # ====================================================

        for rank, result in enumerate(

            dense_results,

            start=1

        ):

            document_id = (
                result["id"]
            )


            scores.setdefault(
                document_id,
                0.0
            )


            scores[
                document_id
            ] += (

                1.0
                /
                (
                    k
                    +
                    rank
                )

            )


        # ====================================================
        # SPARSE RESULTS
        # ====================================================

        for rank, result in enumerate(

            sparse_results,

            start=1

        ):

            document_id = (
                result["id"]
            )


            scores.setdefault(
                document_id,
                0.0
            )


            scores[
                document_id
            ] += (

                1.0
                /
                (
                    k
                    +
                    rank
                )

            )


        # ====================================================
        # SORT
        # ====================================================

        ranked = sorted(

            scores.items(),

            key=lambda item:
                item[1],

            reverse=True

        )


        return ranked