# from fastapi import (
#     FastAPI,
#     UploadFile,
#     File,
#     Form,
#     HTTPException
# )

# from dotenv import load_dotenv

# import os
# import tempfile
# import zipfile
# import asyncio


# from app.file_reader import FileReader
# from app.text_cleaner import TextCleaner
# from app.matcher import ResumeJDMatcher



# # ---------------------------------
# # Load Environment
# # ---------------------------------

# load_dotenv(
#     "/home/suresh/Gen_AI_Practice/.env"
# )



# app = FastAPI()



# reader = FileReader()

# cleaner = TextCleaner()



# # ---------------------------------
# # Workload Control
# # ---------------------------------

# WORKLOAD = int(

#     os.getenv(
#         "WORKLOAD",
#         5
#     )

# )


# semaphore = asyncio.Semaphore(
#     WORKLOAD
# )



# # ---------------------------------
# # API Key
# # ---------------------------------

# def get_api_key(
#     api_key
# ):


#     if api_key:

#         return api_key



#     env_key = os.getenv(
#         "OPENAI_API_KEY"
#     )


#     if env_key:

#         return env_key



#     raise HTTPException(

#         status_code=400,

#         detail="OpenAI API key missing"

#     )



# # ---------------------------------
# # Read PDF
# # ---------------------------------

# async def read_pdf(

#     pdf_path

# ):


#     text = reader.read_file(

#         pdf_path

#     )


#     text = cleaner.clean_text(

#         text

#     )


#     return text



# # ---------------------------------
# # Process One Resume
# # ---------------------------------

# async def process_resume(

#     pdf_path,

#     filename,

#     matcher,

#     jd_text

# ):


#     async with semaphore:


#         resume_text = await read_pdf(

#             pdf_path

#         )


#         result = matcher.match(

#             resume_text,

#             jd_text

#         )


#         result["resume_name"] = filename


#         return result




# # ---------------------------------
# # Match API
# # ---------------------------------

# @app.post("/match")
# async def match_resume(

#     resume: UploadFile = File(None),

#     resume_zip: UploadFile = File(None),

#     jd_file: UploadFile = File(None),

#     jd_text: str = Form(None),

#     api_key: str = Form(None)

# ):


#     # -----------------------------
#     # API KEY
#     # -----------------------------

#     final_key = get_api_key(

#         api_key

#     )


#     matcher = ResumeJDMatcher(

#         final_key

#     )



#     # -----------------------------
#     # Job Description
#     # -----------------------------

#     if jd_file:


#         extension = os.path.splitext(

#             jd_file.filename

#         )[1]



#         with tempfile.NamedTemporaryFile(

#             delete=False,

#             suffix=extension

#         ) as temp:


#             temp.write(

#                 await jd_file.read()

#             )


#             jd_path = temp.name



#         jd_text = reader.read_file(

#             jd_path

#         )



#     if not jd_text:


#         raise HTTPException(

#             status_code=400,

#             detail="Job Description required"

#         )



#     jd_text = cleaner.clean_text(

#         jd_text

#     )



#     results = []



#     # -----------------------------
#     # Single PDF
#     # -----------------------------

#     if resume:



#         with tempfile.NamedTemporaryFile(

#             delete=False,

#             suffix=".pdf"

#         ) as temp:


#             temp.write(

#                 await resume.read()

#             )


#             pdf_path = temp.name



#         result = await process_resume(

#             pdf_path,

#             resume.filename,

#             matcher,

#             jd_text

#         )


#         results.append(

#             result

#         )



#     # -----------------------------
#     # ZIP Folder
#     # -----------------------------

#     elif resume_zip:



#         with tempfile.TemporaryDirectory() as folder:



#             zip_path = os.path.join(

#                 folder,

#                 resume_zip.filename

#             )



#             with open(

#                 zip_path,

#                 "wb"

#             ) as f:


#                 f.write(

#                     await resume_zip.read()

#                 )



#             with zipfile.ZipFile(

#                 zip_path,

#                 "r"

#             ) as zip_ref:


#                 zip_ref.extractall(

#                     folder

#                 )



#             tasks = []



#             for root, dirs, files in os.walk(folder):


#                 for file in files:


#                     if file.lower().endswith(

#                         ".pdf"

#                     ):


#                         pdf_path = os.path.join(

#                             root,

#                             file

#                         )


#                         tasks.append(

#                             process_resume(

#                                 pdf_path,

#                                 file,

#                                 matcher,

#                                 jd_text

#                             )

#                         )



#             results = await asyncio.gather(

#                 *tasks

#             )



#     else:


#         raise HTTPException(

#             status_code=400,

#             detail="Upload resume PDF or ZIP folder"

#         )



#     return {


#         "workload":

#         WORKLOAD,


#         "total_resume":

#         len(results),


#         "results":

#         results

#     }




################################ streamlit cloud purpose ################################

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from dotenv import load_dotenv

import os
import tempfile
import zipfile
import asyncio
from pathlib import Path


from app.file_reader import FileReader
from app.text_cleaner import TextCleaner
from app.matcher import ResumeJDMatcher


# ---------------------------------
# Load Environment
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")


app = FastAPI()


reader = FileReader()

cleaner = TextCleaner()


# ---------------------------------
# Workload Control
# ---------------------------------

WORKLOAD = int(
    os.getenv(
        "WORKLOAD",
        5
    )
)


semaphore = asyncio.Semaphore(
    WORKLOAD
)


# ---------------------------------
# API Key
# ---------------------------------

def get_api_key(api_key):

    if api_key:
        return api_key


    env_key = os.getenv(
        "OPENAI_API_KEY"
    )


    if env_key:
        return env_key


    raise HTTPException(
        status_code=400,
        detail="OpenAI API key missing"
    )


# ---------------------------------
# Read PDF
# ---------------------------------

async def read_pdf(pdf_path):

    text = reader.read_file(
        pdf_path
    )

    text = cleaner.clean_text(
        text
    )

    return text


# ---------------------------------
# Process Resume
# ---------------------------------

async def process_resume(
    pdf_path,
    filename,
    matcher,
    jd_text
):

    async with semaphore:

        resume_text = await read_pdf(
            pdf_path
        )

        result = matcher.match(
            resume_text,
            jd_text
        )

        result["resume_name"] = filename

        return result



# ---------------------------------
# Match API
# ---------------------------------

@app.post("/match")
async def match_resume(

    resume: UploadFile = File(None),

    resume_zip: UploadFile = File(None),

    jd_file: UploadFile = File(None),

    jd_text: str = Form(None),

    api_key: str = Form(None)

):


    final_key = get_api_key(
        api_key
    )


    matcher = ResumeJDMatcher(
        final_key
    )


    # -----------------------------
    # Job Description
    # -----------------------------

    if jd_file:

        extension = os.path.splitext(
            jd_file.filename
        )[1]


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp:

            temp.write(
                await jd_file.read()
            )

            jd_path = temp.name


        jd_text = reader.read_file(
            jd_path
        )


    if not jd_text:

        raise HTTPException(
            status_code=400,
            detail="Job Description required"
        )


    jd_text = cleaner.clean_text(
        jd_text
    )


    results = []


    # -----------------------------
    # Single PDF
    # -----------------------------

    if resume:


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp:


            temp.write(
                await resume.read()
            )


            pdf_path = temp.name



        result = await process_resume(
            pdf_path,
            resume.filename,
            matcher,
            jd_text
        )


        results.append(
            result
        )


    # -----------------------------
    # ZIP Resume
    # -----------------------------

    elif resume_zip:


        with tempfile.TemporaryDirectory() as folder:


            zip_path = os.path.join(
                folder,
                resume_zip.filename
            )


            with open(
                zip_path,
                "wb"
            ) as f:

                f.write(
                    await resume_zip.read()
                )


            with zipfile.ZipFile(
                zip_path,
                "r"
            ) as zip_ref:

                zip_ref.extractall(
                    folder
                )


            tasks = []


            for root, dirs, files in os.walk(folder):

                for file in files:


                    if file.lower().endswith(".pdf"):


                        pdf_path = os.path.join(
                            root,
                            file
                        )


                        tasks.append(
                            process_resume(
                                pdf_path,
                                file,
                                matcher,
                                jd_text
                            )
                        )


            results = await asyncio.gather(
                *tasks
            )


    else:

        raise HTTPException(
            status_code=400,
            detail="Upload resume PDF or ZIP folder"
        )


    return {

        "workload": WORKLOAD,

        "total_resume": len(results),

        "results": results

    }