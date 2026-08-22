import math
import re


class ScoreCalculator:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        weights=None
    ):

        # ----------------------------------------------------
        # Default weights
        # ----------------------------------------------------
        #
        # These are configurable.
        # They are NOT tied to one retrieval method.
        #
        # ----------------------------------------------------

        self.weights = weights or {

            "skill_match": 0.40,

            "experience_match": 0.25,

            "responsibility_match": 0.20,

            "education_match": 0.05,

            "project_relevance": 0.10

        }


        self.validate_weights()


    # ========================================================
    # VALIDATE WEIGHTS
    # ========================================================

    def validate_weights(self):

        total = sum(
            self.weights.values()
        )


        if not math.isclose(
            total,
            1.0,
            rel_tol=1e-6
        ):

            raise ValueError(

                f"Score weights must add up to 1.0. "
                f"Current total: {total}"

            )


        for name, weight in self.weights.items():

            if weight < 0:

                raise ValueError(

                    f"Weight cannot be negative: {name}"

                )


    # ========================================================
    # CLAMP SCORE
    # ========================================================

    @staticmethod
    def clamp(
        value,
        minimum=0.0,
        maximum=100.0
    ):

        return max(

            minimum,

            min(
                maximum,
                float(value)
            )

        )


    # ========================================================
    # NORMALIZE RETRIEVAL SCORE
    # ========================================================

    def normalize_retrieval_score(
        self,
        score,
        method=None
    ):

        if score is None:

            return 0.0


        score = float(score)


        # ----------------------------------------------------
        # Cosine
        # ----------------------------------------------------

        if method == "Cosine Similarity":

            # cosine can be [-1, 1]

            normalized = (

                (score + 1.0)
                /
                2.0

            )


            return self.clamp(

                normalized * 100

            )


        # ----------------------------------------------------
        # Dot Product
        # ----------------------------------------------------

        if method == "Dot Product / Inner Product":

            # A raw dot product does not have a universal
            # 0-100 interpretation.
            #
            # Do not pretend it does.
            #
            # This bounded transformation is only a
            # ranking-oriented approximation.

            normalized = (

                1.0
                /
                (
                    1.0
                    +
                    math.exp(
                        -score
                    )
                )

            )


            return self.clamp(

                normalized * 100

            )


        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        if method == "BM25":

            # BM25 has no fixed upper bound.
            #
            # Therefore this method should preferably use
            # rank/relative normalization before being used
            # as a final percentage.
            #
            # For an individual score, sigmoid is used.

            normalized = (

                1.0
                /
                (
                    1.0
                    +
                    math.exp(
                        -score
                    )
                )

            )


            return self.clamp(

                normalized * 100

            )


        # ----------------------------------------------------
        # L2
        # ----------------------------------------------------

        if method == "Euclidean Distance (L2)":

            normalized = (

                1.0
                /
                (
                    1.0
                    +
                    score
                )

            )


            return self.clamp(

                normalized * 100

            )


        # ----------------------------------------------------
        # Already percentage
        # ----------------------------------------------------

        if 0 <= score <= 100:

            return self.clamp(
                score
            )


        return 0.0


    # ========================================================
    # SKILL MATCH
    # ========================================================

    def calculate_skill_match(
        self,
        required_skills,
        candidate_skills
    ):

        required = {

            self.normalize_skill(
                skill
            )

            for skill
            in (required_skills or [])

            if skill

        }


        candidate = {

            self.normalize_skill(
                skill
            )

            for skill
            in (candidate_skills or [])

            if skill

        }


        if not required:

            return 100.0


        matched = (
            required.intersection(
                candidate
            )
        )


        return (

            len(matched)

            /

            len(required)

            *

            100

        )


    # ========================================================
    # EXPERIENCE MATCH
    # ========================================================

    def calculate_experience_match(
        self,
        required_years,
        candidate_years
    ):

        try:

            required_years = float(
                required_years or 0
            )

            candidate_years = float(
                candidate_years or 0
            )

        except (

            TypeError,
            ValueError

        ):

            return 0.0


        if required_years <= 0:

            return 100.0


        score = (

            candidate_years
            /
            required_years
            *
            100

        )


        return self.clamp(
            score
        )


    # ========================================================
    # RESPONSIBILITY MATCH
    # ========================================================

    def calculate_responsibility_match(
        self,
        required_responsibilities,
        candidate_responsibilities
    ):

        return self.calculate_text_overlap(

            required_responsibilities,

            candidate_responsibilities

        )


    # ========================================================
    # EDUCATION MATCH
    # ========================================================

    def calculate_education_match(
        self,
        required_education,
        candidate_education
    ):

        if not required_education:

            return 100.0


        if not candidate_education:

            return 0.0


        required_text = (
            self.normalize_text(
                required_education
            )
        )


        candidate_text = (
            self.normalize_text(
                candidate_education
            )
        )


        required_words = set(
            required_text.split()
        )


        candidate_words = set(
            candidate_text.split()
        )


        if not required_words:

            return 100.0


        overlap = (

            required_words
            &
            candidate_words

        )


        return (

            len(overlap)
            /
            len(required_words)
            *
            100

        )


    # ========================================================
    # PROJECT RELEVANCE
    # ========================================================

    def calculate_project_relevance(
        self,
        required_skills,
        project_text
    ):

        if not project_text:

            return 0.0


        project_text = (
            self.normalize_text(
                project_text
            )
        )


        if not required_skills:

            return 100.0


        matched = 0


        for skill in required_skills:

            normalized_skill = (
                self.normalize_skill(
                    skill
                )
            )


            if normalized_skill in project_text:

                matched += 1


        return (

            matched
            /
            len(required_skills)
            *
            100

        )


    # ========================================================
    # TEXT OVERLAP
    # ========================================================

    def calculate_text_overlap(
        self,
        required_text,
        candidate_text
    ):

        if isinstance(
            required_text,
            list
        ):

            required_text = " ".join(

                str(item)

                for item
                in required_text

            )


        if isinstance(
            candidate_text,
            list
        ):

            candidate_text = " ".join(

                str(item)

                for item
                in candidate_text

            )


        required_words = set(

            self.tokenize(
                required_text
            )

        )


        candidate_words = set(

            self.tokenize(
                candidate_text
            )

        )


        if not required_words:

            return 100.0


        overlap = (

            required_words
            &
            candidate_words

        )


        return (

            len(overlap)
            /
            len(required_words)
            *
            100

        )


    # ========================================================
    # FINAL MATCH SCORE
    # ========================================================

    def calculate_final_score(
        self,
        skill_match=0,
        experience_match=0,
        responsibility_match=0,
        education_match=0,
        project_relevance=0,
        retrieval_score=None,
        retrieval_weight=0.0
    ):

        # ====================================================
        # BASE SCORE
        # ====================================================

        base_score = (

            skill_match
            *
            self.weights[
                "skill_match"
            ]

            +

            experience_match
            *
            self.weights[
                "experience_match"
            ]

            +

            responsibility_match
            *
            self.weights[
                "responsibility_match"
            ]

            +

            education_match
            *
            self.weights[
                "education_match"
            ]

            +

            project_relevance
            *
            self.weights[
                "project_relevance"
            ]

        )


        base_score = self.clamp(
            base_score
        )


        # ====================================================
        # OPTIONAL RETRIEVAL COMPONENT
        # ====================================================

        if (

            retrieval_score is not None

            and

            retrieval_weight > 0

        ):

            retrieval_weight = self.clamp(

                retrieval_weight,

                0,

                1

            ) / 100


            base_weight = (
                1.0
                -
                retrieval_weight
            )


            final_score = (

                base_score
                *
                base_weight

                +

                float(retrieval_score)
                *
                retrieval_weight

            )

        else:

            final_score = base_score


        return round(

            self.clamp(
                final_score
            ),

            2

        )


    # ========================================================
    # COMPLETE SCORE
    # ========================================================

    def calculate(
        self,
        required_skills=None,
        candidate_skills=None,

        required_years=0,
        candidate_years=0,

        required_responsibilities=None,
        candidate_responsibilities=None,

        required_education="",
        candidate_education="",

        project_text="",

        retrieval_score=None,
        retrieval_method=None
    ):

        # ====================================================
        # SKILLS
        # ====================================================

        skill_match = (
            self.calculate_skill_match(

                required_skills,

                candidate_skills

            )
        )


        # ====================================================
        # EXPERIENCE
        # ====================================================

        experience_match = (
            self.calculate_experience_match(

                required_years,

                candidate_years

            )
        )


        # ====================================================
        # RESPONSIBILITIES
        # ====================================================

        responsibility_match = (
            self.calculate_responsibility_match(

                required_responsibilities,

                candidate_responsibilities

            )
        )


        # ====================================================
        # EDUCATION
        # ====================================================

        education_match = (
            self.calculate_education_match(

                required_education,

                candidate_education

            )
        )


        # ====================================================
        # PROJECT
        # ====================================================

        project_relevance = (
            self.calculate_project_relevance(

                required_skills,

                project_text

            )
        )


        # ====================================================
        # RETRIEVAL SCORE
        # ====================================================

        normalized_retrieval_score = None


        if retrieval_score is not None:

            normalized_retrieval_score = (
                self.normalize_retrieval_score(

                    retrieval_score,

                    retrieval_method

                )
            )


        # ====================================================
        # FINAL
        # ====================================================

        final_score = (
            self.calculate_final_score(

                skill_match=
                    skill_match,

                experience_match=
                    experience_match,

                responsibility_match=
                    responsibility_match,

                education_match=
                    education_match,

                project_relevance=
                    project_relevance,

                retrieval_score=
                    normalized_retrieval_score,

                retrieval_weight=0

            )
        )


        return {

            "overall_match_percentage":
                final_score,

            "component_scores": {

                "skill_match":
                    round(
                        skill_match,
                        2
                    ),

                "experience_match":
                    round(
                        experience_match,
                        2
                    ),

                "responsibility_match":
                    round(
                        responsibility_match,
                        2
                    ),

                "education_match":
                    round(
                        education_match,
                        2
                    ),

                "project_relevance":
                    round(
                        project_relevance,
                        2
                    )

            },

            "retrieval_score":
                normalized_retrieval_score,

            "retrieval_method":
                retrieval_method,

            "weights":
                self.weights

        }


    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    @staticmethod
    def normalize_text(
        text
    ):

        if text is None:

            return ""


        text = str(
            text
        ).lower()


        text = re.sub(

            r"[^a-z0-9+#.\s]",

            " ",

            text

        )


        text = re.sub(

            r"\s+",

            " ",

            text

        )


        return text.strip()


    # ========================================================
    # NORMALIZE SKILL
    # ========================================================

    @staticmethod
    def normalize_skill(
        skill
    ):

        return ScoreCalculator.normalize_text(
            skill
        )


    # ========================================================
    # TOKENIZE
    # ========================================================

    @staticmethod
    def tokenize(
        text
    ):

        if text is None:

            return []


        text = ScoreCalculator.normalize_text(
            text
        )


        return text.split()