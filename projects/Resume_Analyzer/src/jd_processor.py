from docx import Document

import tempfile
import os

from src.ocr_extractor import OCRExtractor


class JDProcessor:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        input_method,
        pdf_file=None,
        docx_file=None,
        pasted_text=None
    ):

        self.input_method = input_method

        self.pdf_file = pdf_file

        self.docx_file = docx_file

        self.pasted_text = pasted_text


    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process(self):

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if self.input_method == "Upload PDF":

            return self.process_pdf()


        # ----------------------------------------------------
        # DOCX
        # ----------------------------------------------------

        elif self.input_method == "Upload DOCX":

            return self.process_docx()


        # ----------------------------------------------------
        # PASTE TEXT
        # ----------------------------------------------------

        elif self.input_method == "Paste Text":

            return self.process_text()


        # ----------------------------------------------------
        # INVALID
        # ----------------------------------------------------

        raise ValueError(
            "Invalid Job Description input method."
        )


    # ========================================================
    # PROCESS JD PDF
    # ========================================================

    def process_pdf(self):

        if self.pdf_file is None:

            raise ValueError(
                "Job Description PDF is missing."
            )


        file_bytes = (
            self.pdf_file.getvalue()
        )


        original_file_name = (
            self.pdf_file.name
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
            # OCR
            # ------------------------------------------------

            jd_data = OCRExtractor().extract_pdf(
                temp_pdf_path,
                original_file_name
            )


            # ------------------------------------------------
            # COMBINE PAGE TEXT
            # ------------------------------------------------

            page_texts = []


            for page in jd_data["data"]:

                page_text = (
                    page["page_data"]
                    .strip()
                )


                if page_text:

                    page_texts.append(
                        page_text
                    )


            # ------------------------------------------------
            # FULL JD TEXT
            # ------------------------------------------------

            jd_text = "\n\n".join(
                page_texts
            )


            return jd_text


        finally:

            # ------------------------------------------------
            # DELETE TEMP FILE
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
    # PROCESS JD DOCX
    # ========================================================

    def process_docx(self):

        if self.docx_file is None:

            raise ValueError(
                "Job Description DOCX is missing."
            )


        file_bytes = (
            self.docx_file.getvalue()
        )


        temp_docx_path = None


        try:

            # ------------------------------------------------
            # CREATE TEMPORARY DOCX
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".docx"
            ) as temp_file:

                temp_file.write(
                    file_bytes
                )

                temp_docx_path = (
                    temp_file.name
                )


            # ------------------------------------------------
            # OPEN DOCX
            # ------------------------------------------------

            document = Document(
                temp_docx_path
            )


            paragraphs = []


            # ------------------------------------------------
            # EXTRACT PARAGRAPHS
            # ------------------------------------------------

            for paragraph in document.paragraphs:

                text = (
                    paragraph.text
                    .strip()
                )


                if text:

                    paragraphs.append(
                        text
                    )


            # ------------------------------------------------
            # COMBINE TEXT
            # ------------------------------------------------

            jd_text = "\n\n".join(
                paragraphs
            )


            return jd_text


        finally:

            # ------------------------------------------------
            # DELETE TEMP FILE
            # ------------------------------------------------

            if (
                temp_docx_path
                and os.path.exists(
                    temp_docx_path
                )
            ):

                os.remove(
                    temp_docx_path
                )


    # ========================================================
    # PROCESS PASTED JD TEXT
    # ========================================================

    def process_text(self):

        if self.pasted_text is None:

            raise ValueError(
                "Job Description text is missing."
            )


        jd_text = (
            self.pasted_text
            .strip()
        )


        if not jd_text:

            raise ValueError(
                "Job Description text is empty."
            )


        return jd_text