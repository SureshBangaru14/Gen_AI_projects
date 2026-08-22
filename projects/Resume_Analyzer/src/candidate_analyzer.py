import json
import re


class CandidateAnalyzer:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        api_key,
        model_name="gpt-4o-mini",
        prompt_builder=None
    ):

        if not api_key:

            raise ValueError(
                "OpenAI API key is required "
                "for CandidateAnalyzer."
            )


        from openai import OpenAI


        self.client = OpenAI(
            api_key=api_key
        )


        self.model_name = model_name

        self.prompt_builder = (
            prompt_builder
        )


    # ========================================================
    # ANALYZE CANDIDATE
    # ========================================================

    def analyze(
        self,
        job_description,
        resume_text,
        retrieved_chunks=None
    ):

        if not job_description:

            raise ValueError(
                "Job Description cannot be empty."
            )


        if not resume_text:

            raise ValueError(
                "Resume text cannot be empty."
            )


        # ====================================================
        # BUILD CONTEXT
        # ====================================================

        retrieved_context = (
            self.build_retrieved_context(
                retrieved_chunks
            )
        )


        # ====================================================
        # SYSTEM PROMPT
        # ====================================================

        system_prompt = self.build_system_prompt()


        # ====================================================
        # USER PROMPT
        # ====================================================

        user_prompt = self.build_user_prompt(

            job_description=
                job_description,

            resume_text=
                resume_text,

            retrieved_context=
                retrieved_context

        )


        # ====================================================
        # OPENAI REQUEST
        # ====================================================

        response = (
            self.client.chat.completions.create(

                model=self.model_name,

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
                            user_prompt

                    }

                ],

                temperature=0,

                response_format={
                    "type":
                        "json_object"
                }

            )
        )


        # ====================================================
        # GET RESPONSE
        # ====================================================

        content = (
            response
            .choices[0]
            .message
            .content
        )


        # ====================================================
        # PARSE JSON
        # ====================================================

        result = (
            self.parse_json(
                content
            )
        )


        # ====================================================
        # NORMALIZE RESULT
        # ====================================================

        result = (
            self.normalize_result(
                result
            )
        )


        return result


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    def build_system_prompt(self):

        return """
You are an expert technical recruiter and resume
analysis system.

Your job is to compare a candidate's resume against
a Job Description.

Analyze only evidence available in the supplied data.

Never invent skills, experience, education, projects,
certifications, employers or technologies.

Extract structured information that can be used by a
separate scoring engine.

You must return valid JSON.

Important:

- Do not calculate the final recruitment score.
- Do not treat cosine similarity as a match percentage.
- Do not treat BM25 score as a match percentage.
- Identify required skills.
- Identify candidate skills.
- Identify matched skills.
- Identify missing skills.
- Extract required experience.
- Extract candidate experience.
- Compare responsibilities.
- Compare education.
- Identify relevant projects.
- Identify strengths.
- Identify skill gaps.
- Provide recommendations.

If information is unavailable, use an empty value.

Return JSON only.
"""


    # ========================================================
    # USER PROMPT
    # ========================================================

    def build_user_prompt(
        self,
        job_description,
        resume_text,
        retrieved_context=""
    ):

        prompt = f"""
Analyze the following candidate.

============================================================
JOB DESCRIPTION
============================================================

{job_description}


============================================================
FULL RESUME
============================================================

{resume_text}


============================================================
RETRIEVED RELEVANT RESUME INFORMATION
============================================================

{retrieved_context}


============================================================
REQUIRED ANALYSIS
============================================================

Extract the following:

1. Candidate name

2. Email

3. Phone

4. Required technical skills

5. Candidate technical skills

6. Matched skills

7. Missing skills

8. Required years of experience

9. Candidate years of experience

10. Experience match

11. Required responsibilities

12. Candidate responsibilities

13. Responsibility match

14. Required education

15. Candidate education

16. Education match

17. Relevant projects

18. Project relevance

19. Strengths

20. Skill gaps

21. Recommendations

22. Summary


============================================================
IMPORTANT
============================================================

Do not invent information.

If the Job Description says "3+ years Python"
and the resume does not provide experience information,
do not assume that the candidate has 3 years.

Use null, empty string or empty array when information
is unavailable.

Return JSON only.
"""

        return prompt


    # ========================================================
    # RETRIEVED CONTEXT
    # ========================================================

    def build_retrieved_context(
        self,
        retrieved_chunks
    ):

        if not retrieved_chunks:

            return ""


        context_parts = []


        for index, item in enumerate(

            retrieved_chunks,

            start=1

        ):

            # ------------------------------------------------
            # Dictionary result
            # ------------------------------------------------

            if isinstance(
                item,
                dict
            ):

                text = (

                    item.get(
                        "text"
                    )

                    or

                    item.get(
                        "document"
                    )

                    or

                    item.get(
                        "content"
                    )

                    or

                    ""

                )


                score = item.get(
                    "score"
                )


                if score is not None:

                    context_parts.append(

                        f"[Chunk {index} | "
                        f"Score: {score}]\n"
                        f"{text}"

                    )

                else:

                    context_parts.append(

                        f"[Chunk {index}]\n"
                        f"{text}"

                    )


            # ------------------------------------------------
            # String result
            # ------------------------------------------------

            else:

                context_parts.append(

                    f"[Chunk {index}]\n"
                    f"{str(item)}"

                )


        return "\n\n".join(
            context_parts
        )


    # ========================================================
    # PARSE JSON
    # ========================================================

    def parse_json(
        self,
        content
    ):

        if not content:

            raise ValueError(
                "LLM returned an empty response."
            )


        content = content.strip()


        # ====================================================
        # DIRECT JSON
        # ====================================================

        try:

            return json.loads(
                content
            )

        except json.JSONDecodeError:

            pass


        # ====================================================
        # JSON CODE BLOCK
        # ====================================================

        match = re.search(

            r"```json\s*(.*?)\s*```",

            content,

            re.DOTALL

        )


        if match:

            try:

                return json.loads(

                    match.group(1)

                )

            except json.JSONDecodeError:

                pass


        # ====================================================
        # GENERIC CODE BLOCK
        # ====================================================

        match = re.search(

            r"```\s*(.*?)\s*```",

            content,

            re.DOTALL

        )


        if match:

            try:

                return json.loads(

                    match.group(1)

                )

            except json.JSONDecodeError:

                pass


        # ====================================================
        # FIRST JSON OBJECT
        # ====================================================

        start = content.find(
            "{"
        )


        end = content.rfind(
            "}"
        )


        if (

            start != -1

            and

            end != -1

            and

            end > start

        ):

            json_text = content[
                start:
                end + 1
            ]


            try:

                return json.loads(
                    json_text
                )

            except json.JSONDecodeError:

                pass


        raise ValueError(

            "Unable to parse the LLM response "
            "as valid JSON."

        )


    # ========================================================
    # NORMALIZE RESULT
    # ========================================================

    def normalize_result(
        self,
        result
    ):

        if not isinstance(
            result,
            dict
        ):

            result = {}


        # ====================================================
        # BASIC INFORMATION
        # ====================================================

        normalized = {

            "candidate_name":
                result.get(
                    "candidate_name",
                    ""
                ),

            "email":
                result.get(
                    "email",
                    ""
                ),

            "phone":
                result.get(
                    "phone",
                    ""
                ),

            "required_skills":
                self.ensure_list(

                    result.get(
                        "required_skills",
                        []
                    )

                ),

            "candidate_skills":
                self.ensure_list(

                    result.get(
                        "candidate_skills",
                        []
                    )

                ),

            "matched_skills":
                self.ensure_list(

                    result.get(
                        "matched_skills",
                        []
                    )

                ),

            "missing_skills":
                self.ensure_list(

                    result.get(
                        "missing_skills",
                        []
                    )

                ),

            "required_years":
                self.to_number(

                    result.get(
                        "required_years",
                        0
                    )

                ),

            "candidate_years":
                self.to_number(

                    result.get(
                        "candidate_years",
                        0
                    )

                ),

            "experience_match":
                result.get(
                    "experience_match",
                    ""
                ),

            "required_responsibilities":
                self.ensure_list(

                    result.get(
                        "required_responsibilities",
                        []
                    )

                ),

            "candidate_responsibilities":
                self.ensure_list(

                    result.get(
                        "candidate_responsibilities",
                        []
                    )

                ),

            "responsibility_match":
                result.get(
                    "responsibility_match",
                    ""
                ),

            "required_education":
                result.get(
                    "required_education",
                    ""
                ),

            "candidate_education":
                result.get(
                    "candidate_education",
                    ""
                ),

            "education_match":
                result.get(
                    "education_match",
                    ""
                ),

            "relevant_projects":
                self.ensure_list(

                    result.get(
                        "relevant_projects",
                        []
                    )

                ),

            "project_relevance":
                result.get(
                    "project_relevance",
                    ""
                ),

            "strengths":
                self.ensure_list(

                    result.get(
                        "strengths",
                        []
                    )

                ),

            "skill_gaps":
                self.ensure_list(

                    result.get(
                        "skill_gaps",
                        []
                    )

                ),

            "recommendations":
                self.ensure_list(

                    result.get(
                        "recommendations",
                        []
                    )

                ),

            "summary":
                result.get(
                    "summary",
                    ""
                )

        }


        return normalized


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


    # ========================================================
    # NUMBER CONVERSION
    # ========================================================

    @staticmethod
    def to_number(
        value
    ):

        if value is None:

            return 0


        if isinstance(
            value,
            (
                int,
                float
            )
        ):

            return value


        try:

            value = str(
                value
            )


            match = re.search(

                r"\d+(?:\.\d+)?",

                value

            )


            if match:

                return float(
                    match.group()
                )


        except Exception:

            pass


        return 0