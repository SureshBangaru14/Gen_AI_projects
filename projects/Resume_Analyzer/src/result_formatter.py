# ============================================================
# RESULT FORMATTER
# ============================================================

from typing import Any, Dict, List


class ResultFormatter:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(self):

        pass


    # ========================================================
    # FORMAT ALL RESULTS
    # ========================================================

    def format_results(
        self,
        resume_results
    ):

        if not resume_results:

            return []


        formatted_results = []


        for result in resume_results:

            formatted = (
                self.format_single_result(
                    result
                )
            )


            formatted_results.append(
                formatted
            )


        # ====================================================
        # SORT BY MATCH %
        # ====================================================

        formatted_results.sort(

            key=lambda item:
                item.get(
                    "match_percentage",
                    0
                ),

            reverse=True

        )


        # ====================================================
        # ADD RANK
        # ====================================================

        for index, result in enumerate(

            formatted_results,

            start=1

        ):

            result["rank"] = index


        return formatted_results


    # ========================================================
    # FORMAT SINGLE RESULT
    # ========================================================

    def format_single_result(
        self,
        result
    ):

        # ====================================================
        # BASIC INFORMATION
        # ====================================================

        file_name = (

            result.get(
                "file_name"
            )

            or

            result.get(
                "resume_name"
            )

            or

            "Unknown Resume"

        )


        candidate_name = (

            result.get(
                "candidate_name"
            )

            or

            result.get(
                "analysis",
                {}
            ).get(
                "candidate_name",
                ""
            )

        )


        # ====================================================
        # MATCH %
        # ====================================================

        match_percentage = (

            result.get(
                "match_percentage"
            )

            or

            result.get(
                "overall_match_percentage"
            )

            or

            result.get(
                "score",
                0
            )

        )


        match_percentage = (
            self.safe_float(
                match_percentage
            )
        )


        match_percentage = max(

            0.0,

            min(
                100.0,
                match_percentage
            )

        )


        # ====================================================
        # ANALYSIS
        # ====================================================

        analysis = result.get(

            "analysis",

            {}

        )


        if not isinstance(
            analysis,
            dict
        ):

            analysis = {}


        # ====================================================
        # COMPONENT SCORES
        # ====================================================

        component_scores = (

            result.get(
                "component_scores"
            )

            or

            analysis.get(
                "component_scores",
                {}
            )

        )


        if not isinstance(
            component_scores,
            dict
        ):

            component_scores = {}


        # ====================================================
        # FINAL RESULT
        # ====================================================

        return {

            # ------------------------------------------------
            # Ranking
            # ------------------------------------------------

            "rank":
                result.get(
                    "rank",
                    0
                ),

            # ------------------------------------------------
            # Resume
            # ------------------------------------------------

            "file_name":
                file_name,

            "candidate_name":
                candidate_name,

            # ------------------------------------------------
            # Final Score
            # ------------------------------------------------

            "match_percentage":
                round(
                    match_percentage,
                    2
                ),

            # ------------------------------------------------
            # Retrieval
            # ------------------------------------------------

            "retrieval_score":
                self.safe_float_or_none(

                    result.get(
                        "retrieval_score"
                    )

                ),

            "retrieval_method":
                result.get(
                    "retrieval_method",
                    ""
                ),

            # ------------------------------------------------
            # Components
            # ------------------------------------------------

            "component_scores":
                component_scores,

            # ------------------------------------------------
            # Skills
            # ------------------------------------------------

            "required_skills":
                self.ensure_list(

                    result.get(
                        "required_skills"
                    )

                    or

                    analysis.get(
                        "required_skills"
                    )

                ),

            "candidate_skills":
                self.ensure_list(

                    result.get(
                        "candidate_skills"
                    )

                    or

                    analysis.get(
                        "candidate_skills"
                    )

                ),

            "matched_skills":
                self.ensure_list(

                    result.get(
                        "matched_skills"
                    )

                    or

                    analysis.get(
                        "matched_skills"
                    )

                ),

            "missing_skills":
                self.ensure_list(

                    result.get(
                        "missing_skills"
                    )

                    or

                    analysis.get(
                        "missing_skills"
                    )

                ),

            # ------------------------------------------------
            # Experience
            # ------------------------------------------------

            "required_years":
                self.safe_float(

                    result.get(
                        "required_years"
                    )

                    or

                    analysis.get(
                        "required_years",
                        0
                    )

                ),

            "candidate_years":
                self.safe_float(

                    result.get(
                        "candidate_years"
                    )

                    or

                    analysis.get(
                        "candidate_years",
                        0
                    )

                ),

            "experience_match":
                result.get(

                    "experience_match"

                )

                or

                analysis.get(
                    "experience_match",
                    ""
                ),

            # ------------------------------------------------
            # Responsibilities
            # ------------------------------------------------

            "required_responsibilities":
                self.ensure_list(

                    result.get(
                        "required_responsibilities"
                    )

                    or

                    analysis.get(
                        "required_responsibilities"
                    )

                ),

            "candidate_responsibilities":
                self.ensure_list(

                    result.get(
                        "candidate_responsibilities"
                    )

                    or

                    analysis.get(
                        "candidate_responsibilities"
                    )

                ),

            "responsibility_match":
                result.get(

                    "responsibility_match"

                )

                or

                analysis.get(
                    "responsibility_match",
                    ""
                ),

            # ------------------------------------------------
            # Education
            # ------------------------------------------------

            "required_education":
                result.get(

                    "required_education"

                )

                or

                analysis.get(
                    "required_education",
                    ""
                ),

            "candidate_education":
                result.get(

                    "candidate_education"

                )

                or

                analysis.get(
                    "candidate_education",
                    ""
                ),

            "education_match":
                result.get(

                    "education_match"

                )

                or

                analysis.get(
                    "education_match",
                    ""
                ),

            # ------------------------------------------------
            # Projects
            # ------------------------------------------------

            "relevant_projects":
                self.ensure_list(

                    result.get(
                        "relevant_projects"
                    )

                    or

                    analysis.get(
                        "relevant_projects"
                    )

                ),

            "project_relevance":
                result.get(

                    "project_relevance"

                )

                or

                analysis.get(
                    "project_relevance",
                    ""
                ),

            # ------------------------------------------------
            # Analysis
            # ------------------------------------------------

            "strengths":
                self.ensure_list(

                    result.get(
                        "strengths"
                    )

                    or

                    analysis.get(
                        "strengths"
                    )

                ),

            "skill_gaps":
                self.ensure_list(

                    result.get(
                        "skill_gaps"
                    )

                    or

                    analysis.get(
                        "skill_gaps"
                    )

                ),

            "recommendations":
                self.ensure_list(

                    result.get(
                        "recommendations"
                    )

                    or

                    analysis.get(
                        "recommendations"
                    )

                ),

            "summary":
                result.get(

                    "summary"

                )

                or

                analysis.get(
                    "summary",
                    ""
                ),

            # ------------------------------------------------
            # Original data
            # ------------------------------------------------

            "metadata":
                result.get(
                    "metadata",
                    {}
                )

        }


    # ========================================================
    # STREAMLIT TABLE
    # ========================================================

    def create_table_data(
        self,
        formatted_results
    ):

        table_data = []


        for result in formatted_results:

            table_data.append(

                {

                    "Rank":
                        result.get(
                            "rank",
                            0
                        ),

                    "Resume":
                        result.get(
                            "file_name",
                            ""
                        ),

                    "Candidate":
                        result.get(
                            "candidate_name",
                            ""
                        ),

                    "Match %":
                        result.get(
                            "match_percentage",
                            0
                        ),

                    "Retrieval":
                        result.get(
                            "retrieval_method",
                            ""
                        )

                }

            )


        return table_data


    # ========================================================
    # DETAIL DATA
    # ========================================================

    def create_detail_data(
        self,
        result
    ):

        if not result:

            return {}


        return {

            "Candidate":
                result.get(
                    "candidate_name",
                    ""
                ),

            "Resume":
                result.get(
                    "file_name",
                    ""
                ),

            "Match Percentage":
                result.get(
                    "match_percentage",
                    0
                ),

            "Matched Skills":
                result.get(
                    "matched_skills",
                    []
                ),

            "Missing Skills":
                result.get(
                    "missing_skills",
                    []
                ),

            "Strengths":
                result.get(
                    "strengths",
                    []
                ),

            "Skill Gaps":
                result.get(
                    "skill_gaps",
                    []
                ),

            "Recommendations":
                result.get(
                    "recommendations",
                    []
                ),

            "Summary":
                result.get(
                    "summary",
                    ""
                )

        }


    # ========================================================
    # FIND RESULT BY FILE
    # ========================================================

    def find_by_file_name(
        self,
        results,
        file_name
    ):

        for result in results:

            if (

                result.get(
                    "file_name"
                )

                ==

                file_name

            ):

                return result


        return None


    # ========================================================
    # FIND RESULT BY RANK
    # ========================================================

    def find_by_rank(
        self,
        results,
        rank
    ):

        for result in results:

            if (

                result.get(
                    "rank"
                )

                ==

                rank

            ):

                return result


        return None


    # ========================================================
    # SAFE FLOAT
    # ========================================================

    @staticmethod
    def safe_float(
        value
    ):

        if value is None:

            return 0.0


        try:

            return float(
                value
            )

        except (

            TypeError,
            ValueError

        ):

            return 0.0


    # ========================================================
    # SAFE FLOAT OR NONE
    # ========================================================

    @staticmethod
    def safe_float_or_none(
        value
    ):

        if value is None:

            return None


        try:

            return float(
                value
            )

        except (

            TypeError,
            ValueError

        ):

            return None


    # ========================================================
    # ENSURE LIST
    # ========================================================

    @staticmethod
    def ensure_list(
        value
    ):

        if value is None:

            return []


        if isinstance(
            value,
            list
        ):

            return value


        if isinstance(
            value,
            tuple
        ):

            return list(
                value
            )


        if isinstance(
            value,
            str
        ):

            if not value.strip():

                return []


            return [
                value.strip()
            ]


        return [
            value
        ]