import tempfile
import os

from src.ocr_extractor import OCRExtractor
from src.jd_processor import JDProcessor


class DocumentProcess:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        resume_files=None,
        jd_input_method=None,
        jd_file_name=None
    ):

        self.resume_files = resume_files

        self.jd_input_method = jd_input_method

        self.jd_file_name = jd_file_name


    # ========================================================
    # PROCESS MULTIPLE RESUMES
    # ========================================================

    def process_resume(self):

        resume_text_map = {}


        for resume_file in self.resume_files:

            # ------------------------------------------------
            # PROCESS PDF + OCR
            # ------------------------------------------------

            resume_data = self.process_resume_pdf(
                resume_file
            )


            # ------------------------------------------------
            # COMBINE ALL PAGE TEXT
            # ------------------------------------------------

            full_resume_text = (
                self.get_full_resume_text(
                    resume_data
                )
            )


            # ------------------------------------------------
            # STORE USING ORIGINAL FILE NAME
            # ------------------------------------------------

            resume_text_map[
                resume_data["file_name"]
            ] = full_resume_text


        # ----------------------------------------------------
        # RETURN ALL RESUME FULL TEXT
        # ----------------------------------------------------

        return resume_text_map


    # ========================================================
    # PROCESS SINGLE RESUME PDF
    # ========================================================

    def process_resume_pdf(
        self,
        resume_file
    ):

        # ----------------------------------------------------
        # GET UPLOADED FILE BYTES
        # ----------------------------------------------------

        file_bytes = (
            resume_file.getvalue()
        )


        # ----------------------------------------------------
        # ORIGINAL FILE NAME
        # ----------------------------------------------------

        original_file_name = (
            resume_file.name
        )


        temp_pdf_path = None


        try:

            # ------------------------------------------------
            # CREATE TEMPORARY PDF
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    file_bytes
                )

                temp_pdf_path = (
                    temp_file.name
                )


            # ------------------------------------------------
            # OCR PDF
            # ------------------------------------------------

            resume_data = OCRExtractor().extract_pdf(
                temp_pdf_path,
                original_file_name
            )


            return resume_data


        finally:

            # ------------------------------------------------
            # DELETE TEMPORARY PDF
            # ------------------------------------------------

            if (
                temp_pdf_path
                and os.path.exists(
                    temp_pdf_path
                )
            ):

                os.remove(
                    temp_pdf_path
                )


    # ========================================================
    # GET FULL RESUME TEXT
    # ========================================================

    def get_full_resume_text(
        self,
        resume_data
    ):

        page_texts = []


        # ----------------------------------------------------
        # LOOP THROUGH ALL PAGES
        # ----------------------------------------------------

        for page in resume_data["data"]:

            page_text = (
                page["page_data"]
            )


            # ------------------------------------------------
            # CLEAN TEXT
            # ------------------------------------------------

            page_text = (
                page_text.strip()
            )


            # ------------------------------------------------
            # IGNORE EMPTY PAGE
            # ------------------------------------------------

            if page_text:

                page_texts.append(
                    page_text
                )


        # ----------------------------------------------------
        # COMBINE ALL PAGES
        # ----------------------------------------------------

        full_resume_text = (
            "\n\n".join(
                page_texts
            )
        )


        return full_resume_text


    # ========================================================
    # PROCESS JOB DESCRIPTION
    # ========================================================

    def process_jd(self):

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if self.jd_input_method == "Upload PDF":

            jd_text = JDProcessor(
                input_method="Upload PDF",
                pdf_file=self.jd_file_name
            ).process()


            return jd_text


        # ----------------------------------------------------
        # DOCX
        # ----------------------------------------------------

        elif self.jd_input_method == "Upload DOCX":

            jd_text = JDProcessor(
                input_method="Upload DOCX",
                docx_file=self.jd_file_name
            ).process()


            return jd_text


        # ----------------------------------------------------
        # PASTE TEXT
        # ----------------------------------------------------

        elif self.jd_input_method == "Paste Text":

            jd_text = JDProcessor(
                input_method="Paste Text",
                pasted_text=self.jd_file_name
            ).process()


            return jd_text


        # ----------------------------------------------------
        # INVALID METHOD
        # ----------------------------------------------------

        raise ValueError(
            "Invalid Job Description input method."
        )