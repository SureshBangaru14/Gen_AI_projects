# ============================================================

# src/score_calculator.py

# ============================================================

from typing import Any, Dict, List, Optional

import re



class ScoreCalculator:

    # ========================================================

    # DEFAULT WEIGHTS

    # ========================================================

    DEFAULT_WEIGHTS = {

        "skills": 0.60,

        "experience": 0.20,

        "responsibilities": 0.15,

        "education": 0.05,

    }

    # ========================================================

    # CONSTRUCTOR

    # ========================================================

    def __init__(

        self,

        skill_weight=0.60,

        experience_weight=0.20,

        responsibility_weight=0.15,

        education_weight=0.05,

    ):

        self.weights = {

            "skills": float(skill_weight),

            "experience": float(experience_weight),

            "responsibilities": float(

                responsibility_weight

            ),

            "education": float(

                education_weight

            ),

        }

        self._validate_weights()

    # ========================================================

    # VALIDATE WEIGHTS

    # ========================================================

    def _validate_weights(self):

        total = sum(

            self.weights.values()

        )

        if abs(total - 1.0) > 0.001:

            raise ValueError(

                "Score weights must total 100%. "

                f"Current total = {total * 100:.2f}%"

            )

    # ========================================================

    # MAIN CALCULATION

    # ========================================================

    def calculate(

        self,

        required_skills=None,

        candidate_skills=None,

        required_years=None,

        candidate_years=None,

        required_responsibilities=None,

        candidate_responsibilities=None,

        required_education=None,

        candidate_education=None,

        retrieval_score=None,

    ):

        required_skills = (

            required_skills or []

        )

        candidate_skills = (

            candidate_skills or []

        )

        required_responsibilities = (

            required_responsibilities or []

        )

        candidate_responsibilities = (

            candidate_responsibilities or []

        )

        # ----------------------------------------------------

        # SKILLS

        # ----------------------------------------------------

        skill_result = (

            self.calculate_skill_score(

                required_skills,

                candidate_skills

            )

        )

        # ----------------------------------------------------

        # EXPERIENCE

        # ----------------------------------------------------

        experience_score = (

            self.calculate_experience_score(

                required_years,

                candidate_years

            )

        )

        # ----------------------------------------------------

        # RESPONSIBILITIES

        # ----------------------------------------------------

        responsibility_result = (

            self.calculate_responsibility_score(

                required_responsibilities,

                candidate_responsibilities

            )

        )

        # ----------------------------------------------------

        # EDUCATION

        # ----------------------------------------------------

        education_score = (

            self.calculate_education_score(

                required_education,

                candidate_education

            )

        )

        # ----------------------------------------------------

        # FINAL MATCH

        # ----------------------------------------------------

        overall_match_percentage = (

            skill_result["score"]

            * self.weights["skills"]

            +

            experience_score

            * self.weights["experience"]

            +

            responsibility_result["score"]

            * self.weights["responsibilities"]

            +

            education_score

            * self.weights["education"]

        )

        overall_match_percentage = (

            self.clamp(

                overall_match_percentage

            )

        )

        # ----------------------------------------------------

        # RETURN

        # ----------------------------------------------------

        return {

            # Main value used by app.py

            "overall_match_percentage":

                round(

                    overall_match_percentage,

                    2

                ),

            # Backward compatibility

            "match_percentage":

                round(

                    overall_match_percentage,

                    2

                ),

            # Retrieval remains separate

            "retrieval_score":

                retrieval_score,

            "component_scores": {

                "skills":

                    round(

                        skill_result["score"],

                        2

                    ),

                "experience":

                    round(

                        experience_score,

                        2

                    ),

                "responsibilities":

                    round(

                        responsibility_result["score"],

                        2

                    ),

                "education":

                    round(

                        education_score,

                        2

                    )

            },

            "skill_score":

                round(

                    skill_result["score"],

                    2

                ),

            "experience_score":

                round(

                    experience_score,

                    2

                ),

            "responsibility_score":

                round(

                    responsibility_result["score"],

                    2

                ),

            "education_score":

                round(

                    education_score,

                    2

                ),

            "skill_coverage":

                skill_result["coverage"],

            "responsibility_coverage":

                responsibility_result[

                    "coverage"

                ],

            "matched_required_skills":

                skill_result[

                    "matched_required_skills"

                ],

            "missing_required_skills":

                skill_result[

                    "missing_required_skills"

                ],

            "additional_candidate_skills":

                skill_result[

                    "additional_candidate_skills"

                ],

            "matched_skills":

                skill_result[

                    "matched_required_skills"

                ],

            "missing_skills":

                skill_result[

                    "missing_required_skills"

                ],

            "weights":

                self.weights

        }

    # ========================================================

    # SKILL SCORE

    # ========================================================

    def calculate_skill_score(

        self,

        required_skills,

        candidate_skills

    ):

        required = self.normalize_list(

            required_skills

        )

        candidate = self.normalize_list(

            candidate_skills

        )

        # ----------------------------------------------------

        # NO REQUIRED SKILLS

        # ----------------------------------------------------

        if not required:

            return {

                "score": 0.0,

                "coverage": 0.0,

                "matched_required_skills": [],

                "missing_required_skills": [],

                "additional_candidate_skills":

                    candidate

            }

        matched = []

        missing = []

        # ----------------------------------------------------

        # MATCH REQUIRED SKILLS

        # ----------------------------------------------------

        for required_skill in required:

            found = False

            for candidate_skill in candidate:

                if self.skills_match(

                    required_skill,

                    candidate_skill

                ):

                    found = True

                    break

            if found:

                matched.append(

                    required_skill

                )

            else:

                missing.append(

                    required_skill

                )

        # ----------------------------------------------------

        # COVERAGE

        # ----------------------------------------------------

        coverage = (

            len(matched)

            /

            len(required)

        ) * 100

        # ----------------------------------------------------

        # ADDITIONAL SKILLS

        # ----------------------------------------------------

        additional = []

        for candidate_skill in candidate:

            is_required = False

            for required_skill in required:

                if self.skills_match(

                    required_skill,

                    candidate_skill

                ):

                    is_required = True

                    break

            if not is_required:

                additional.append(

                    candidate_skill

                )

        return {

            "score":

                self.clamp(

                    coverage

                ),

            "coverage":

                round(

                    coverage,

                    2

                ),

            "matched_required_skills":

                matched,

            "missing_required_skills":

                missing,

            "additional_candidate_skills":

                additional

        }

    # ========================================================

    # SKILL MATCH

    # ========================================================

    @staticmethod

    def skills_match(

        required_skill,

        candidate_skill

    ):

        required = (

            ScoreCalculator.normalize_skill(

                required_skill

            )

        )

        candidate = (

            ScoreCalculator.normalize_skill(

                candidate_skill

            )

        )

        if not required or not candidate:

            return False

        # Exact

        if required == candidate:

            return True

        # ----------------------------------------------------

        # ALIASES

        # ----------------------------------------------------

        aliases = {

            "rag": {

                "rag",

                "retrieval augmented generation",

                "retrieval-augmented generation"

            },

            "llm": {

                "llm",

                "large language model",

                "large language models"

            },

            "generative ai": {

                "generative ai",

                "genai",

                "gen ai",

                "generative artificial intelligence"

            },

            "embeddings": {

                "embedding",

                "embeddings",

                "text embeddings"

            },

            "vector databases": {

                "vector database",

                "vector databases",

                "vector db",

                "vector dbs"

            },

            "semantic search": {

                "semantic search",

                "semantic retrieval"

            },

            "prompt engineering": {

                "prompt engineering",

                "prompt design"

            },

            "langchain": {

                "langchain"

            },

            "langgraph": {

                "langgraph"

            },

            "llamaindex": {

                "llamaindex",

                "llama index"

            },

            "python": {

                "python",

                "python programming"

            },

            "machine learning": {

                "machine learning",

                "ml"

            },

            "deep learning": {

                "deep learning",

                "dl"

            },

            "natural language processing": {

                "natural language processing",

                "nlp"

            }

        }

        for values in aliases.values():

            if (

                required in values

                and

                candidate in values

            ):

                return True

        # ----------------------------------------------------

        # CONSERVATIVE SUBSTRING

        # ----------------------------------------------------

        if (

            len(required) >= 4

            and

            len(candidate) >= 4

        ):

            if (

                required in candidate

                or

                candidate in required

            ):

                return True

        return False

    # ========================================================

    # EXPERIENCE

    # ========================================================

    @staticmethod

    def calculate_experience_score(

        required_years,

        candidate_years

    ):

        if (

            required_years is None

            or

            float(required_years or 0) <= 0

        ):

            return 100.0

        if candidate_years is None:

            return 0.0

        required = float(

            required_years

        )

        candidate = float(

            candidate_years

        )

        if candidate >= required:

            return 100.0

        return ScoreCalculator.clamp(

            (

                candidate

                /

                required

            ) * 100

        )

    # ========================================================

    # RESPONSIBILITY SCORE

    # ========================================================

    def calculate_responsibility_score(

        self,

        required_responsibilities,

        candidate_responsibilities

    ):

        required = self.normalize_list(

            required_responsibilities

        )

        candidate = self.normalize_list(

            candidate_responsibilities

        )

        if not required:

            return {

                "score": 0.0,

                "coverage": 0.0,

                "matched": [],

                "missing": []

            }

        matched = []

        missing = []

        for requirement in required:

            found = False

            for candidate_item in candidate:

                if self.text_match(

                    requirement,

                    candidate_item

                ):

                    found = True

                    break

            if found:

                matched.append(

                    requirement

                )

            else:

                missing.append(

                    requirement

                )

        coverage = (

            len(matched)

            /

            len(required)

        ) * 100

        return {

            "score":

                self.clamp(

                    coverage

                ),

            "coverage":

                round(

                    coverage,

                    2

                ),

            "matched":

                matched,

            "missing":

                missing

        }

    # ========================================================

    # TEXT MATCH

    # ========================================================

    @staticmethod

    def text_match(

        text_a,

        text_b

    ):

        a = (

            ScoreCalculator.normalize_text(

                text_a

            )

        )

        b = (

            ScoreCalculator.normalize_text(

                text_b

            )

        )

        if not a or not b:

            return False

        if a == b:

            return True

        tokens_a = set(

            a.split()

        )

        tokens_b = set(

            b.split()

        )

        if not tokens_a or not tokens_b:

            return False

        overlap = (

            tokens_a

            &

            tokens_b

        )

        ratio_a = (

            len(overlap)

            /

            len(tokens_a)

        )

        ratio_b = (

            len(overlap)

            /

            len(tokens_b)

        )

        return (

            ratio_a >= 0.5

            or

            ratio_b >= 0.5

        )

    # ========================================================

    # EDUCATION

    # ========================================================

    @staticmethod

    def calculate_education_score(

        required_education,

        candidate_education

    ):

        required = (

            ScoreCalculator.normalize_text(

                required_education or ""

            )

        )

        candidate = (

            ScoreCalculator.normalize_text(

                candidate_education or ""

            )

        )

        if not required:

            return 100.0

        if not candidate:

            return 0.0

        if required in candidate:

            return 100.0

        if candidate in required:

            return 100.0

        degree_levels = {

            "high school": 1,

            "diploma": 2,

            "associate": 3,

            "bachelor": 4,

            "bachelors": 4,

            "b.tech": 4,

            "btech": 4,

            "be": 4,

            "master": 5,

            "masters": 5,

            "m.tech": 5,

            "mtech": 5,

            "me": 5,

            "mba": 5,

            "phd": 6,

            "doctorate": 6

        }

        required_level = None

        candidate_level = None

        for degree, level in degree_levels.items():

            if degree in required:

                required_level = level

                break

        for degree, level in degree_levels.items():

            if degree in candidate:

                candidate_level = level

                break

        if (

            required_level is not None

            and

            candidate_level is not None

        ):

            if candidate_level >= required_level:

                return 100.0

            if candidate_level == (

                required_level - 1

            ):

                return 50.0

            return 0.0

        return 0.0

    # ========================================================

    # NORMALIZE LIST

    # ========================================================

    @staticmethod

    def normalize_list(values):

        if not values:

            return []

        if isinstance(

            values,

            str

        ):

            values = [

                values

            ]

        result = []

        seen = set()

        for value in values:

            if value is None:

                continue

            value = str(

                value

            ).strip()

            if not value:

                continue

            key = (

                ScoreCalculator.normalize_skill(

                    value

                )

            )

            if key not in seen:

                result.append(

                    value

                )

                seen.add(

                    key

                )

        return result

    # ========================================================

    # NORMALIZE SKILL

    # ========================================================

    @staticmethod

    def normalize_skill(

        value

    ):

        if value is None:

            return ""

        value = str(

            value

        ).lower().strip()

        value = value.replace(

            "&",

            " and "

        )

        value = re.sub(

            r"[^a-z0-9+#.\-/ ]+",

            " ",

            value

        )

        value = re.sub(

            r"\s+",

            " ",

            value

        )

        return value.strip()

    # ========================================================

    # NORMALIZE TEXT

    # ========================================================

    @staticmethod

    def normalize_text(

        value

    ):

        if value is None:

            return ""

        value = str(

            value

        ).lower().strip()

        value = re.sub(

            r"\s+",

            " ",

            value

        )

        return value

    # ========================================================

    # CLAMP

    # ========================================================

    @staticmethod

    def clamp(

        value

    ):

        try:

            value = float(

                value

            )

        except (

            TypeError,

            ValueError

        ):

            return 0.0

        return max(

            0.0,

            min(

                100.0,

                value

            )

        )