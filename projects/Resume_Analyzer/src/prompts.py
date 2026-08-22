# ============================================================
# RESPONSE SCHEMA
# ============================================================

RESPONSE_SCHEMA = {
    "candidate_name": "string",
    "email": "string",
    "phone": "string",

    "overall_match_percentage": "number",

    "skills_match": {
        "matched_skills": [
            "string"
        ],

        "missing_skills": [
            "string"
        ]
    },

    "experience": {
        "required_years": "number",
        "candidate_years": "number",
        "meets_requirement": "boolean"
    },

    "education": {
        "required": "string",
        "candidate": "string",
        "matches": "boolean"
    },

    "strengths": [
        "string"
    ],

    "skill_gaps": [
        "string"
    ],

    "recommendations": [
        "string"
    ],

    "summary": "string"
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert AI Resume Analyzer and Recruiter.

Your task is to compare a candidate resume against a
Job Description and produce an objective candidate
matching analysis.

Follow these rules:

1. Analyze only the information available in the
   provided Job Description and Resume.

2. Do not invent candidate information.

3. Identify skills explicitly mentioned in the resume.

4. Identify skills required by the Job Description.

5. Separate matched skills from missing skills.

6. Compare required experience with candidate experience.

7. Compare required education with candidate education.

8. Consider technical skills, frameworks, tools,
   responsibilities, domain experience and projects.

9. Calculate an overall matching percentage between
   0 and 100.

10. The matching percentage must represent relevance
    between the resume and the Job Description.

11. Provide strengths.

12. Provide skill gaps.

13. Provide practical recommendations.

14. If information is unavailable, return an empty
    value rather than guessing.

15. Return the result using the requested response schema.

Be precise, consistent and evidence-based.
"""


# ============================================================
# USER PROMPT TEMPLATE
# ============================================================

USER_PROMPT = """
Analyze the following candidate against the Job Description.

============================================================
JOB DESCRIPTION
============================================================

{job_description}


============================================================
CANDIDATE RESUME
============================================================

{resume_text}


============================================================
ANALYSIS REQUIREMENTS
============================================================

Analyze the candidate using the following dimensions:

1. Technical Skills
2. Programming Languages
3. Frameworks
4. Libraries
5. AI / ML Skills
6. Generative AI Skills
7. RAG Skills
8. LLM Skills
9. Vector Database Experience
10. Cloud / DevOps Skills
11. Tools
12. Domain Experience
13. Projects
14. Professional Experience
15. Education
16. Certifications
17. Required Years of Experience
18. Responsibilities
19. Missing Skills
20. Overall Job Relevance


============================================================
MATCHING SCORE
============================================================

Calculate an overall matching percentage from 0 to 100.

Consider:

- Required skills
- Preferred skills
- Experience
- Responsibilities
- Technical stack
- Domain relevance
- Education
- Projects
- Certifications


============================================================
OUTPUT
============================================================

Return the result according to the provided response schema.

Do not invent information.
"""


# ============================================================
# PROMPT BUILDER
# ============================================================

class PromptBuilder:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        system_prompt=None,
        user_prompt_template=None,
        response_schema=None
    ):

        self.system_prompt = (
            system_prompt
            or
            SYSTEM_PROMPT
        )

        self.user_prompt_template = (
            user_prompt_template
            or
            USER_PROMPT
        )

        self.response_schema = (
            response_schema
            or
            RESPONSE_SCHEMA
        )


    # ========================================================
    # BUILD USER PROMPT
    # ========================================================

    def build_user_prompt(
        self,
        job_description,
        resume_text
    ):

        if not job_description:

            raise ValueError(
                "Job Description cannot be empty."
            )


        if not resume_text:

            raise ValueError(
                "Resume text cannot be empty."
            )


        try:

            prompt = (
                self.user_prompt_template.format(

                    job_description=
                        job_description,

                    resume_text=
                        resume_text

                )
            )

        except KeyError as error:

            raise ValueError(

                f"Invalid placeholder in "
                f"user prompt: {error}"

            ) from error


        return prompt


    # ========================================================
    # GET SYSTEM PROMPT
    # ========================================================

    def get_system_prompt(self):

        return self.system_prompt


    # ========================================================
    # GET RESPONSE SCHEMA
    # ========================================================

    def get_response_schema(self):

        return self.response_schema


    # ========================================================
    # BUILD COMPLETE PROMPT
    # ========================================================

    def build(
        self,
        job_description,
        resume_text
    ):

        return {

            "system_prompt":
                self.get_system_prompt(),

            "user_prompt":
                self.build_user_prompt(

                    job_description=
                        job_description,

                    resume_text=
                        resume_text

                ),

            "response_schema":
                self.get_response_schema()

        }