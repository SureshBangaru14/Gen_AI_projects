# ============================================================
# RESPONSE SCHEMA
# ============================================================

RESPONSE_SCHEMA = """
{
    "candidate_name": "",
    "match_percentage": 0,
    "strengths": [],
    "skill_gaps": [],
    "recommendations": []
}
"""


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert AI Resume Analyzer.

Your task is to compare a candidate resume
against a Job Description.

Identify:

1. Candidate strengths
2. Skill gaps
3. Matching skills
4. Missing skills
5. Recommendations

Calculate a meaningful matching percentage.

Return the result strictly according
to the provided response schema.
"""


# ============================================================
# USER PROMPT
# ============================================================

USER_PROMPT = """
Analyze the following candidate resume
against the provided Job Description.

JOB DESCRIPTION:

{job_description}


CANDIDATE RESUME:

{resume_text}


Return the analysis according
to the required response schema.
"""