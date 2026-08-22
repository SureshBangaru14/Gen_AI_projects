import tempfile
import os

from pdf2image import convert_from_path
import pytesseract

from docx import Document


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

        self.input_method = (
            input_method
        )

        self.pdf_file = (
            pdf_file
        )

        self.docx_file = (
            docx_file
        )

        self.pasted_text = (
            pasted_text
        )


    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process(self):

        if self.input_method == "Upload PDF":

            return (
                self.process_pdf()
            )


        elif self.input_method == "Upload DOCX":

            return (
                self.process_docx()
            )


        elif self.input_method == "Paste Text":

            return (
                self.process_pasted_text()
            )


        raise ValueError(
            "Invalid Job Description input method."
        )


    # ========================================================
    # PROCESS PDF
    # ========================================================

    def process_pdf(self):

        file_bytes = (
            self.pdf_file.getvalue()
        )


        temp_pdf_path = None


        try:

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


            images = (
                convert_from_path(
                    temp_pdf_path,
                    dpi=300
                )
            )


            page_texts = []


            for image in images:

                text = (
                    pytesseract.image_to_string(
                        image
                    )
                )


                text = (
                    text.strip()
                )


                if text:

                    page_texts.append(
                        text
                    )


            return "\n\n".join(
                page_texts
            )


        finally:

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
    # PROCESS DOCX
    # ========================================================

    def process_docx(self):

        file_bytes = (
            self.docx_file.getvalue()
        )


        temp_docx_path = None


        try:

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


            document = Document(
                temp_docx_path
            )


            paragraphs = []


            for paragraph in document.paragraphs:

                text = (
                    paragraph.text.strip()
                )


                if text:

                    paragraphs.append(
                        text
                    )


            return "\n\n".join(
                paragraphs
            )


        finally:

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
    # PROCESS PASTED TEXT
    # ========================================================

    def process_pasted_text(self):

        if not self.pasted_text:

            return ""


        return (
            self.pasted_text.strip()
        )