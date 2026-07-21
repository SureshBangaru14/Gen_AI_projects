# import streamlit as st
# import requests
# import pandas as pd

# from io import BytesIO



# st.set_page_config(

#     page_title="AI ATS Matcher",

#     layout="wide"

# )



# st.title(
#     "AI Resume ATS Matcher"
# )



# # -------------------------
# # API KEY
# # -------------------------

# api_key = st.text_input(

#     "OpenAI API Key",

#     type="password"

# )



# # -------------------------
# # JD
# # -------------------------

# jd_type = st.radio(

#     "Job Description",

#     [
#         "Text",
#         "File"
#     ]

# )


# jd_text = None

# jd_file = None



# if jd_type == "Text":


#     jd_text = st.text_area(
#         "Enter JD"
#     )


# else:


#     jd_file = st.file_uploader(

#         "Upload JD",

#         type=[
#             "pdf",
#             "txt"
#         ]

#     )



# # -------------------------
# # Resume
# # -------------------------

# upload_type = st.radio(

#     "Resume Upload",

#     [
#         "Single PDF",
#         "ZIP Folder"
#     ]

# )


# resume = None

# resume_zip = None



# if upload_type == "Single PDF":


#     resume = st.file_uploader(

#         "Upload Resume PDF",

#         type=["pdf"]

#     )


# else:


#     resume_zip = st.file_uploader(

#         "Upload Resume Folder ZIP",

#         type=["zip"]

#     )



# # -------------------------
# # Submit
# # -------------------------

# if st.button(
#     "Check ATS"
# ):


#     if not resume and not resume_zip:

#         st.warning(
#             "Upload resume PDF or ZIP"
#         )

#         st.stop()



#     files = []



#     if resume:


#         files.append(

#             (

#                 "resume",

#                 (

#                     resume.name,

#                     resume.getvalue(),

#                     "application/pdf"

#                 )

#             )

#         )



#     if resume_zip:


#         files.append(

#             (

#                 "resume_zip",

#                 (

#                     resume_zip.name,

#                     resume_zip.getvalue(),

#                     "application/zip"

#                 )

#             )

#         )



#     if jd_file:


#         files.append(

#             (

#                 "jd_file",

#                 (

#                     jd_file.name,

#                     jd_file.getvalue()

#                 )

#             )

#         )



#     data = {}



#     if jd_text:

#         data["jd_text"] = jd_text



#     if api_key:

#         data["api_key"] = api_key



#     response = requests.post(

#         "http://127.0.0.1:8000/match",

#         files=files,

#         data=data

#     )



#     if response.status_code != 200:

#         st.error(
#             response.text
#         )

#         st.stop()



#     output = response.json()



#     df = pd.DataFrame(

#         output["results"]

#     )



#     st.subheader(
#         "ATS Results"
#     )


#     st.dataframe(

#         df,

#         use_container_width=True

#     )



#     # Excel Download

#     excel = BytesIO()



#     with pd.ExcelWriter(

#         excel,

#         engine="openpyxl"

#     ) as writer:


#         df.to_excel(

#             writer,

#             index=False,

#             sheet_name="ATS"

#         )



#     excel.seek(0)



#     st.download_button(

#         "Download Excel Report",

#         excel,

#         "ATS_Report.xlsx",

#         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

#     )




################################ streamlit cloud purpose ################################

import streamlit as st
import requests
import pandas as pd

from io import BytesIO


# ==================================================
# FASTAPI BACKEND URL
# ==================================================

API_URL = "https://your-fastapi-url.onrender.com"


# ==================================================
# STREAMLIT CONFIG
# ==================================================

st.set_page_config(
    page_title="AI ATS Matcher",
    layout="wide"
)


st.title(
    "AI Resume ATS Matcher"
)



# ==================================================
# OPENAI API KEY
# ==================================================

api_key = st.text_input(
    "OpenAI API Key",
    type="password"
)



# ==================================================
# JOB DESCRIPTION
# ==================================================

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



# ==================================================
# RESUME UPLOAD
# ==================================================

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
        type=[
            "pdf"
        ]
    )


else:

    resume_zip = st.file_uploader(
        "Upload Resume ZIP",
        type=[
            "zip"
        ]
    )



# ==================================================
# SUBMIT
# ==================================================

if st.button(
    "Check ATS"
):


    if not resume and not resume_zip:

        st.warning(
            "Please upload resume PDF or ZIP folder"
        )

        st.stop()



    files = []



    # -------------------------------
    # Single Resume PDF
    # -------------------------------

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



    # -------------------------------
    # Resume ZIP
    # -------------------------------

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



    # -------------------------------
    # JD File
    # -------------------------------

    if jd_file:

        files.append(

            (
                "jd_file",

                (
                    jd_file.name,
                    jd_file.getvalue()
                )

            )

        )



    # -------------------------------
    # Form Data
    # -------------------------------

    data = {}


    if jd_text:

        data["jd_text"] = jd_text



    if api_key:

        data["api_key"] = api_key



    # ==================================================
    # SEND REQUEST TO FASTAPI
    # ==================================================

    try:

        with st.spinner(
            "Analyzing resumes..."
        ):


            response = requests.post(

                f"{API_URL}/match",

                files=files,

                data=data,

                timeout=180

            )


    except requests.exceptions.ConnectionError:


        st.error(
            "FastAPI connection failed. Check your API URL."
        )

        st.stop()



    except requests.exceptions.Timeout:


        st.error(
            "FastAPI request timed out."
        )

        st.stop()



    # ==================================================
    # ERROR RESPONSE
    # ==================================================

    if response.status_code != 200:


        st.error(
            response.text
        )

        st.stop()



    # ==================================================
    # DISPLAY RESULT
    # ==================================================

    output = response.json()


    results = output.get(
        "results",
        []
    )


    if not results:

        st.warning(
            "No ATS results found"
        )

        st.stop()



    df = pd.DataFrame(
        results
    )



    st.subheader(
        "ATS Results"
    )


    st.dataframe(

        df,

        use_container_width=True

    )



    # ==================================================
    # DOWNLOAD EXCEL
    # ==================================================

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

        label="Download Excel Report",

        data=excel,

        file_name="ATS_Report.xlsx",

        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    )