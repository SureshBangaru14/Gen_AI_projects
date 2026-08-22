from pdf2image import convert_from_path

import pytesseract

from pathlib import Path


class OCRExtractor:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(self):

        pass


    # ========================================================
    # PDF → IMAGES
    # ========================================================

    def pdf_to_images(
        self,
        pdf_path
    ):

        images = convert_from_path(
            pdf_path,
            dpi=300
        )


        return images


    # ========================================================
    # IMAGE → TEXT
    # ========================================================

    def image_to_text(
        self,
        image
    ):

        text = pytesseract.image_to_string(
            image
        )


        return text


    # ========================================================
    # PDF → OCR JSON
    # ========================================================

    def extract_pdf(
        self,
        pdf_path,
        original_file_name
    ):

        pdf_path = Path(
            pdf_path
        )


        # ----------------------------------------------------
        # PDF → IMAGES
        # ----------------------------------------------------

        images = self.pdf_to_images(
            pdf_path
        )


        # ----------------------------------------------------
        # PAGE DATA
        # ----------------------------------------------------

        pages = []


        # ----------------------------------------------------
        # PROCESS EVERY PAGE
        # ----------------------------------------------------

        for page_number, image in enumerate(
            images,
            start=1
        ):

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            text = self.image_to_text(
                image
            )


            # ------------------------------------------------
            # CLEAN TEXT
            # ------------------------------------------------

            text = text.strip()


            # ------------------------------------------------
            # STORE PAGE
            # ------------------------------------------------

            pages.append(
                {
                    "page_no": str(
                        page_number
                    ),

                    "page_data": text
                }
            )


        # ----------------------------------------------------
        # RETURN COMPLETE RESULT
        # ----------------------------------------------------

        return {

            "file_name": original_file_name,

            "file_type": "pdf",

            "No_pages": str(
                len(pages)
            ),

            "data": pages

        }