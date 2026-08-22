# ============================================================
# src/candidate_analyzer.py
# ============================================================

import json
import re

from typing import Any, Dict, List, Optional

from src.score_calculator import ScoreCalculator


class CandidateAnalyzer:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        openai_api_key=None,
        model="gpt-4o-mini",
        score_calculator=None
    ):

        self.openai_api_key = (
            openai_api_key
        )

        self.model = model

        self.score_calculator = (
            score_calculator
            or
            ScoreCalculator()
        )

        self.client = None

        # ----------------------------------------------------
        # OpenAI
        # ----------------------------------------------------

        if self.openai_api_key:

            try:

                from openai import OpenAI

                self.client = OpenAI(
                    api_key=
                        self.openai_api_key
                )

            except Exception:

                self.client = None

    # ========================================================
    # MAIN ANALYZE
    # ========================================================

    def analyze(
        self,
        resume_text,
        jd_text,
        retrieval_score=None,
        file_name=None,
        candidate_name=None
    ):

        resume_text = (
            resume_text or ""
        ).strip()

        jd_text = (
            jd_text or ""
        ).strip()

        if not resume_text:

            raise ValueError(
                "Resume text is empty."
            )

        if not jd_text:

            raise ValueError(
                "Job Description text is empty."
            )

        # ----------------------------------------------------
        # JD
        # ----------------------------------------------------

        jd_requirements = (
            self.extract_jd_requirements(
                jd_text
            )
        )

        # ----------------------------------------------------
        # RESUME
        # ----------------------------------------------------

        resume_information = (
            self.extract_resume_information(
                resume_text
            )
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score_result = (
            self.score_calculator.calculate(

                required_skills=
                    jd_requirements.get(
                        "required_skills",
                        []
                    ),

                candidate_skills=
                    resume_information.get(
                        "skills",
                        []
                    ),

                required_years=
                    jd_requirements.get(
                        "required_years"
                    ),

                candidate_years=
                    resume_information.get(
                        "years_of_experience"
                    ),

                required_responsibilities=
                    jd_requirements.get(
                        "responsibilities",
                        []
                    ),

                candidate_responsibilities=
                    resume_information.get(
                        "responsibilities",
                        []
                    ),

                required_education=
                    jd_requirements.get(
                        "education"
                    ),

                candidate_education=
                    resume_information.get(
                        "education"
                    ),

                retrieval_score=
                    retrieval_score
            )
        )

        # ----------------------------------------------------
        # PROJECTS
        # ----------------------------------------------------

        projects = (
            resume_information.get(
                "projects",
                []
            )
        )

        relevant_projects = (
            self.find_relevant_projects(
                projects,
                jd_requirements
            )
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {

            "file_name":
                file_name or "",

            "candidate_name":
                candidate_name
                or
                resume_information.get(
                    "candidate_name",
                    "Unknown"
                ),

            # ------------------------------------------------
            # SCORE
            # ------------------------------------------------

            "match_percentage":
                score_result[
                    "match_percentage"
                ],

            "overall_match_percentage":
                score_result[
                    "overall_match_percentage"
                ],

            # IMPORTANT:
            # Retrieval score remains separate.
            "retrieval_score":
                retrieval_score,

            # ------------------------------------------------
            # COMPONENTS
            # ------------------------------------------------

            "component_scores":
                score_result[
                    "component_scores"
                ],

            "skill_score":
                score_result[
                    "skill_score"
                ],

            "experience_score":
                score_result[
                    "experience_score"
                ],

            "responsibility_score":
                score_result[
                    "responsibility_score"
                ],

            "education_score":
                score_result[
                    "education_score"
                ],

            # ------------------------------------------------
            # JD SKILLS
            # ------------------------------------------------

            "required_skills":
                jd_requirements.get(
                    "required_skills",
                    []
                ),

            "preferred_skills":
                jd_requirements.get(
                    "preferred_skills",
                    []
                ),

            # ------------------------------------------------
            # RESUME SKILLS
            # ------------------------------------------------

            "candidate_skills":
                resume_information.get(
                    "skills",
                    []
                ),

            # ------------------------------------------------
            # MATCHED
            # ------------------------------------------------

            "matched_required_skills":
                score_result[
                    "matched_required_skills"
                ],

            "matched_skills":
                score_result[
                    "matched_required_skills"
                ],

            # ------------------------------------------------
            # MISSING
            # ------------------------------------------------

            "missing_required_skills":
                score_result[
                    "missing_required_skills"
                ],

            "missing_skills":
                score_result[
                    "missing_required_skills"
                ],

            # ------------------------------------------------
            # ADDITIONAL
            # ------------------------------------------------

            "additional_candidate_skills":
                score_result[
                    "additional_candidate_skills"
                ],

            # ------------------------------------------------
            # EXPERIENCE
            # ------------------------------------------------

            "required_years":
                jd_requirements.get(
                    "required_years"
                ),

            "candidate_years":
                resume_information.get(
                    "years_of_experience"
                ),

            # ------------------------------------------------
            # RESPONSIBILITIES
            # ------------------------------------------------

            "required_responsibilities":
                jd_requirements.get(
                    "responsibilities",
                    []
                ),

            "candidate_responsibilities":
                resume_information.get(
                    "responsibilities",
                    []
                ),

            "responsibility_match":
                score_result[
                    "responsibility_coverage"
                ],

            # ------------------------------------------------
            # EDUCATION
            # ------------------------------------------------

            "required_education":
                jd_requirements.get(
                    "education"
                ),

            "candidate_education":
                resume_information.get(
                    "education"
                ),

            # ------------------------------------------------
            # PROJECTS
            # ------------------------------------------------

            "relevant_projects":
                relevant_projects,

            # ------------------------------------------------
            # ANALYSIS
            # ------------------------------------------------

            "strengths":
                self.generate_strengths(
                    score_result
                ),

            "skill_gaps":
                self.generate_skill_gaps(
                    score_result
                ),

            "recommendations":
                self.generate_recommendations(
                    score_result
                ),

            "summary":
                self.generate_summary(
                    score_result
                ),

            # ------------------------------------------------
            # RAW STRUCTURED DATA
            # ------------------------------------------------

            "jd_requirements":
                jd_requirements,

            "resume_information":
                resume_information
        }

    # ========================================================
    # JD EXTRACTION
    # ========================================================

    def extract_jd_requirements(
        self,
        jd_text
    ):

        if self.client:

            try:

                return self.extract_jd_with_llm(
                    jd_text
                )

            except Exception:

                pass

        return self.extract_jd_locally(
            jd_text
        )

    # ========================================================
    # JD LLM
    # ========================================================

    def extract_jd_with_llm(
        self,
        jd_text
    ):

        system_prompt = """
You are an expert Job Description parser.

Return ONLY valid JSON.

Schema:

{
    "required_skills": [],
    "preferred_skills": [],
    "required_years": null,
    "responsibilities": [],
    "education": null,
    "job_title": null,
    "domain": null
}

Rules:

1. Extract only skills actually required by the JD.
2. Do not invent skills.
3. Do not mark unrelated resume skills as JD skills.
4. Git is NOT a match unless Git is required by the JD.
5. required_years must be numeric or null.
6. responsibilities must be a list.
7. education must be string or null.
"""

        response = (
            self.client
            .chat
            .completions
            .create(

                model=self.model,

                temperature=0,

                response_format={
                    "type":
                        "json_object"
                },

                messages=[

                    {
                        "role":
                            "system",

                        "content":
                            system_prompt
                    },

                    {
                        "role":
                            "user",

                        "content":
                            jd_text
                    }
                ]
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        result = json.loads(
            content
        )

        return self.normalize_jd_result(
            result
        )

    # ========================================================
    # LOCAL JD
    # ========================================================

    def extract_jd_locally(
        self,
        jd_text
    ):

        lower_text = (
            jd_text.lower()
        )

        known_skills = [

            "Python",
            "Java",
            "C++",
            "JavaScript",
            "TypeScript",
            "SQL",

            "Machine Learning",
            "Deep Learning",
            "NLP",

            "Generative AI",
            "GenAI",
            "LLM",
            "LLM applications",

            "RAG",
            "Retrieval Augmented Generation",

            "Embeddings",

            "Vector Database",
            "Vector Databases",

            "Semantic Search",

            "Prompt Engineering",

            "LangChain",
            "LangGraph",
            "LlamaIndex",

            "ChromaDB",
            "Pinecone",
            "FAISS",

            "OpenAI",
            "Hugging Face",
            "Transformers",

            "PyTorch",
            "TensorFlow",

            "Docker",
            "Kubernetes",

            "Git",
            "GitHub",

            "FastAPI",
            "Flask",
            "Streamlit",

            "REST API",

            "AWS",
            "Azure",
            "GCP"
        ]

        required_skills = []

        for skill in known_skills:

            if skill.lower() in lower_text:

                required_skills.append(
                    skill
                )

        return {

            "required_skills":
                self.unique_list(
                    required_skills
                ),

            "preferred_skills": [],

            "required_years":
                self.extract_years(
                    jd_text
                ),

            "responsibilities":
                self.extract_responsibilities(
                    jd_text
                ),

            "education":
                self.extract_education(
                    jd_text
                ),

            "job_title":
                self.extract_job_title(
                    jd_text
                ),

            "domain":
                None
        }

    # ========================================================
    # RESUME EXTRACTION
    # ========================================================

    def extract_resume_information(
        self,
        resume_text
    ):

        if self.client:

            try:

                return self.extract_resume_with_llm(
                    resume_text
                )

            except Exception:

                pass

        return self.extract_resume_locally(
            resume_text
        )

    # ========================================================
    # RESUME LLM
    # ========================================================

    def extract_resume_with_llm(
        self,
        resume_text
    ):

        system_prompt = """
You are an expert Resume parser.

Return ONLY valid JSON.

Schema:

{
    "candidate_name": null,
    "skills": [],
    "years_of_experience": null,
    "responsibilities": [],
    "education": null,
    "projects": []
}

Rules:

1. Extract only information actually present.
2. Do not invent skills.
3. Do not infer technologies.
4. Preserve technology names.
"""

        response = (
            self.client
            .chat
            .completions
            .create(

                model=self.model,

                temperature=0,

                response_format={
                    "type":
                        "json_object"
                },

                messages=[

                    {
                        "role":
                            "system",

                        "content":
                            system_prompt
                    },

                    {
                        "role":
                            "user",

                        "content":
                            resume_text
                    }
                ]
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        result = json.loads(
            content
        )

        return self.normalize_resume_result(
            result
        )

    # ========================================================
    # LOCAL RESUME
    # ========================================================

    def extract_resume_locally(
        self,
        resume_text
    ):

        lower_text = (
            resume_text.lower()
        )

        known_skills = [

            "Python",
            "Java",
            "C++",
            "JavaScript",
            "TypeScript",
            "SQL",

            "Machine Learning",
            "Deep Learning",
            "NLP",

            "Generative AI",
            "GenAI",
            "LLM",
            "LLM applications",

            "RAG",
            "Retrieval Augmented Generation",

            "Embeddings",

            "Vector Database",
            "Vector Databases",

            "Semantic Search",

            "Prompt Engineering",

            "LangChain",
            "LangGraph",
            "LlamaIndex",

            "ChromaDB",
            "Pinecone",
            "FAISS",

            "OpenAI",
            "Hugging Face",
            "Transformers",

            "PyTorch",
            "TensorFlow",

            "Docker",
            "Kubernetes",

            "Git",
            "GitHub",

            "FastAPI",
            "Flask",
            "Streamlit",

            "REST API",

            "AWS",
            "Azure",
            "GCP"
        ]

        skills = []

        for skill in known_skills:

            if skill.lower() in lower_text:

                skills.append(
                    skill
                )

        return {

            "candidate_name":
                self.extract_candidate_name(
                    resume_text
                ),

            "skills":
                self.unique_list(
                    skills
                ),

            "years_of_experience":
                self.extract_years(
                    resume_text
                ),

            "responsibilities":
                self.extract_responsibilities(
                    resume_text
                ),

            "education":
                self.extract_education(
                    resume_text
                ),

            "projects":
                self.extract_projects(
                    resume_text
                )
        }

    # ========================================================
    # STRENGTHS
    # ========================================================

    def generate_strengths(
        self,
        score_result
    ):

        strengths = []

        matched = (
            score_result.get(
                "matched_required_skills",
                []
            )
        )

        if matched:

            strengths.append(
                "Matches required skills: "
                +
                ", ".join(
                    matched
                )
            )

        if (
            score_result.get(
                "experience_score",
                0
            )
            >=
            100
        ):

            strengths.append(
                "Meets or exceeds the "
                "required experience."
            )

        if (
            score_result.get(
                "responsibility_score",
                0
            )
            >=
            70
        ):

            strengths.append(
                "Strong alignment with "
                "required responsibilities."
            )

        if (
            score_result.get(
                "education_score",
                0
            )
            >=
            100
        ):

            strengths.append(
                "Education requirement is satisfied."
            )

        if not strengths:

            strengths.append(
                "Limited direct alignment "
                "with the current JD."
            )

        return strengths

    # ========================================================
    # SKILL GAPS
    # ========================================================

    def generate_skill_gaps(
        self,
        score_result
    ):

        return list(
            score_result.get(
                "missing_required_skills",
                []
            )
        )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def generate_recommendations(
        self,
        score_result
    ):

        recommendations = []

        missing = (
            score_result.get(
                "missing_required_skills",
                []
            )
        )

        if missing:

            recommendations.append(
                "Develop the missing required skills: "
                +
                ", ".join(
                    missing
                )
            )

        if (
            score_result.get(
                "experience_score",
                0
            )
            < 100
        ):

            recommendations.append(
                "Gain additional relevant experience."
            )

        if (
            score_result.get(
                "responsibility_score",
                0
            )
            < 70
        ):

            recommendations.append(
                "Highlight experience that maps "
                "directly to the JD responsibilities."
            )

        return recommendations

    # ========================================================
    # SUMMARY
    # ========================================================

    def generate_summary(
        self,
        score_result
    ):

        match_percentage = (
            score_result.get(
                "match_percentage",
                0
            )
        )

        matched = (
            score_result.get(
                "matched_required_skills",
                []
            )
        )

        missing = (
            score_result.get(
                "missing_required_skills",
                []
            )
        )

        return (

            f"Overall Resume-to-JD match is "
            f"{match_percentage:.2f}%. "

            f"{len(matched)} required skill(s) "
            f"matched and "

            f"{len(missing)} required skill(s) "
            f"are missing."

        )

    # ========================================================
    # PROJECTS
    # ========================================================

    def find_relevant_projects(
        self,
        projects,
        jd_requirements
    ):

        if not projects:

            return []

        required_skills = [

            str(skill).lower()

            for skill
            in jd_requirements.get(
                "required_skills",
                []
            )
        ]

        relevant = []

        for project in projects:

            project_lower = (
                str(project).lower()
            )

            for skill in required_skills:

                if skill in project_lower:

                    relevant.append(
                        project
                    )

                    break

        return self.unique_list(
            relevant
        )

    # ========================================================
    # YEARS
    # ========================================================

    @staticmethod
    def extract_years(
        text
    ):

        patterns = [

            r"(\d+(?:\.\d+)?)\s*\+?\s*years",

            r"(\d+(?:\.\d+)?)\s*\+?\s*yrs",

            r"minimum\s+(\d+(?:\.\d+)?)",

            r"at\s+least\s+(\d+(?:\.\d+)?)"
        ]

        values = []

        for pattern in patterns:

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            for value in matches:

                try:

                    values.append(
                        float(value)
                    )

                except Exception:

                    pass

        if not values:

            return None

        return max(
            values
        )

    # ========================================================
    # EDUCATION
    # ========================================================

    @staticmethod
    def extract_education(
        text
    ):

        patterns = [

            "PhD",
            "Doctorate",
            "Master's",
            "Masters",
            "M.Tech",
            "MTech",
            "MBA",
            "Bachelor's",
            "Bachelors",
            "B.Tech",
            "BTech",
            "BE",
            "Diploma"
        ]

        found = []

        for education in patterns:

            if re.search(
                re.escape(education),
                text,
                flags=re.IGNORECASE
            ):

                found.append(
                    education
                )

        if not found:

            return None

        return ", ".join(
            found
        )

    # ========================================================
    # RESPONSIBILITIES
    # ========================================================

    @staticmethod
    def extract_responsibilities(
        text
    ):

        lines = text.splitlines()

        responsibilities = []

        capture = False

        headers = [

            "responsibilities",

            "roles and responsibilities",

            "what you will do",

            "what you'll do",

            "key responsibilities",

            "job responsibilities"
        ]

        for line in lines:

            clean = line.strip()

            if not clean:

                continue

            lower = clean.lower()

            if any(
                header in lower
                for header in headers
            ):

                capture = True
                continue

            if capture:

                if (
                    clean.startswith("-")
                    or
                    clean.startswith("•")
                    or
                    re.match(
                        r"^\d+[.)]",
                        clean
                    )
                ):

                    clean = re.sub(
                        r"^[-•]\s*",
                        "",
                        clean
                    )

                    clean = re.sub(
                        r"^\d+[.)]\s*",
                        "",
                        clean
                    )

                    if clean:

                        responsibilities.append(
                            clean
                        )

                elif (
                    responsibilities
                    and
                    len(clean) < 80
                    and
                    clean.endswith(":")
                ):

                    capture = False

        return responsibilities[:30]

    # ========================================================
    # PROJECTS
    # ========================================================

    @staticmethod
    def extract_projects(
        text
    ):

        lines = text.splitlines()

        projects = []

        capture = False

        for line in lines:

            clean = line.strip()

            if not clean:

                continue

            lower = clean.lower()

            if (
                "projects" in lower
                and
                len(clean) < 80
            ):

                capture = True
                continue

            if capture:

                if (
                    clean.startswith("-")
                    or
                    clean.startswith("•")
                    or
                    re.match(
                        r"^\d+[.)]",
                        clean
                    )
                ):

                    clean = re.sub(
                        r"^[-•]\s*",
                        "",
                        clean
                    )

                    clean = re.sub(
                        r"^\d+[.)]\s*",
                        "",
                        clean
                    )

                    projects.append(
                        clean
                    )

        return projects[:30]

    # ========================================================
    # CANDIDATE NAME
    # ========================================================

    @staticmethod
    def extract_candidate_name(
        text
    ):

        lines = [

            line.strip()

            for line
            in text.splitlines()

            if line.strip()
        ]

        if not lines:

            return "Unknown"

        invalid = {

            "resume",
            "curriculum vitae",
            "cv",
            "profile",
            "summary",
            "experience",
            "skills",
            "education"
        }

        if (
            lines[0].lower()
            not in invalid
            and
            len(lines[0]) <= 80
        ):

            return lines[0]

        return "Unknown"

    # ========================================================
    # JOB TITLE
    # ========================================================

    @staticmethod
    def extract_job_title(
        text
    ):

        patterns = [

            r"job title\s*:\s*(.+)",

            r"position\s*:\s*(.+)",

            r"role\s*:\s*(.+)"
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            if match:

                return match.group(
                    1
                ).strip()

        return None

    # ========================================================
    # NORMALIZE JD
    # ========================================================

    def normalize_jd_result(
        self,
        result
    ):

        return {

            "required_skills":
                self.normalize_list(
                    result.get(
                        "required_skills",
                        []
                    )
                ),

            "preferred_skills":
                self.normalize_list(
                    result.get(
                        "preferred_skills",
                        []
                    )
                ),

            "required_years":
                self.safe_float(
                    result.get(
                        "required_years"
                    )
                ),

            "responsibilities":
                self.normalize_list(
                    result.get(
                        "responsibilities",
                        []
                    )
                ),

            "education":
                self.safe_string(
                    result.get(
                        "education"
                    )
                ),

            "job_title":
                self.safe_string(
                    result.get(
                        "job_title"
                    )
                ),

            "domain":
                self.safe_string(
                    result.get(
                        "domain"
                    )
                )
        }

    # ========================================================
    # NORMALIZE RESUME
    # ========================================================

    def normalize_resume_result(
        self,
        result
    ):

        return {

            "candidate_name":
                self.safe_string(
                    result.get(
                        "candidate_name"
                    )
                ),

            "skills":
                self.normalize_list(
                    result.get(
                        "skills",
                        []
                    )
                ),

            "years_of_experience":
                self.safe_float(
                    result.get(
                        "years_of_experience"
                    )
                ),

            "responsibilities":
                self.normalize_list(
                    result.get(
                        "responsibilities",
                        []
                    )
                ),

            "education":
                self.safe_string(
                    result.get(
                        "education"
                    )
                ),

            "projects":
                self.normalize_list(
                    result.get(
                        "projects",
                        []
                    )
                )
        }

    # ========================================================
    # NORMALIZE LIST
    # ========================================================

    @staticmethod
    def normalize_list(
        values
    ):

        if not values:

            return []

        if isinstance(
            values,
            str
        ):

            values = [
                values
            ]

        return CandidateAnalyzer.unique_list(
            values
        )

    # ========================================================
    # UNIQUE
    # ========================================================

    @staticmethod
    def unique_list(
        values
    ):

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

            key = value.lower()

            if key not in seen:

                result.append(
                    value
                )

                seen.add(
                    key
                )

        return result

    # ========================================================
    # SAFE STRING
    # ========================================================

    @staticmethod
    def safe_string(
        value
    ):

        if value is None:

            return None

        value = str(
            value
        ).strip()

        return value or None

    # ========================================================
    # SAFE FLOAT
    # ========================================================

    @staticmethod
    def safe_float(
        value
    ):

        if value is None:

            return None

        try:

            return float(
                value
            )

        except Exception:

            return None