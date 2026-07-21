import streamlit as st
import requests
import pandas as pd

from io import BytesIO

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="AI ATS Matcher",
    layout="wide"
)

st.title("AI Resume ATS Matcher")

# -------------------------
# API KEY (Optional)
# -------------------------

api_key = st.text_input(
    "OpenAI API Key (Optional)",
    type="password",
    help="Leave blank to use the OPENAI_API_KEY configured on the FastAPI server."
)

# -------------------------
# Job Description
# -------------------------

jd_type = st.radio(
    "Job Description",
    [
        "Text",
        "File"
    ]
)

jd_text = None
jd_file = None

if jd_type == "Text":

    jd_text = st.text_area(
        "Enter Job Description"
    )

else:

    jd_file = st.file_uploader(
        "Upload Job Description",
        type=[
            "pdf",
            "txt"
        ]
    )

# -------------------------
# Resume Upload
# -------------------------

upload_type = st.radio(
    "Resume Upload",
    [
        "Single PDF",
        "ZIP Folder"
    ]
)

resume = None
resume_zip = None

if upload_type == "Single PDF":

    resume = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )

else:

    resume_zip = st.file_uploader(
        "Upload Resume ZIP",
        type=["zip"]
    )

# -------------------------
# Submit
# -------------------------

if st.button("Check ATS"):

    if not resume and not resume_zip:
        st.warning("Please upload a Resume PDF or ZIP file.")
        st.stop()

    if jd_type == "Text" and not jd_text:
        st.warning("Please enter the Job Description.")
        st.stop()

    if jd_type == "File" and not jd_file:
        st.warning("Please upload the Job Description file.")
        st.stop()

    files = []

    # Resume PDF
    if resume:
        files.append(
            (
                "resume",
                (
                    resume.name,
                    resume.getvalue(),
                    "application/pdf"
                )
            )
        )

    # Resume ZIP
    if resume_zip:
        files.append(
            (
                "resume_zip",
                (
                    resume_zip.name,
                    resume_zip.getvalue(),
                    "application/zip"
                )
            )
        )

    # JD File
    if jd_file:
        files.append(
            (
                "jd_file",
                (
                    jd_file.name,
                    jd_file.getvalue(),
                    jd_file.type
                )
            )
        )

    data = {}

    # JD Text
    if jd_text:
        data["jd_text"] = jd_text

    # Optional API Key
    # If left blank, FastAPI will use OPENAI_API_KEY from .env
    if api_key.strip():
        data["api_key"] = api_key.strip()

    with st.spinner("Matching resumes..."):

        try:

            response = requests.post(
                                    "http://127.0.0.1:8001/match",
                                    files=files,
                                    data=data
                                )

        except requests.exceptions.RequestException as e:
            st.error(f"Unable to connect to API.\n\n{e}")
            st.stop()

    if response.status_code != 200:
        st.error(response.text)
        st.stop()

    output = response.json()

    st.success(
        f"Processed {output['total_resume']} resume(s)"
    )

    df = pd.DataFrame(output["results"])

    st.subheader("ATS Results")

    st.dataframe(
        df,
        use_container_width=True
    )

    # -------------------------
    # Download Excel
    # -------------------------

    excel = BytesIO()

    with pd.ExcelWriter(
        excel,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="ATS Results"
        )

    excel.seek(0)

    st.download_button(
        label="📥 Download Excel Report",
        data=excel,
        file_name="ATS_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )