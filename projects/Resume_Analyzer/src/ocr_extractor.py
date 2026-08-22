from pdf2image import convert_from_path

import pytesseract

from pathlib import Path


class OCRExtractor:

    def __init__(self):

        pass


    # ========================================================
    # PDF → IMAGES
    # ========================================================

    def pdf_to_images(
        self,
        pdf_path
    ):

        return convert_from_path(
            pdf_path,
            dpi=300
        )


    # ========================================================
    # IMAGE → TEXT
    # ========================================================

    def image_to_text(
        self,
        image
    ):

        return (
            pytesseract
            .image_to_string(
                image
            )
        )


    # ========================================================
    # PDF → OCR
    # ========================================================

    def extract_pdf(
        self,
        pdf_path,
        original_file_name
    ):

        pdf_path = Path(
            pdf_path
        )


        images = (
            self.pdf_to_images(
                pdf_path
            )
        )


        pages = []


        for (
            page_number,
            image
        ) in enumerate(
            images,
            start=1
        ):

            text = (
                self.image_to_text(
                    image
                )
            )


            pages.append(

                {

                    "page_no":
                        str(
                            page_number
                        ),

                    "page_data":
                        text.strip()

                }

            )


        return {

            "file_name":
                original_file_name,

            "file_type":
                "pdf",

            "No_pages":
                str(
                    len(pages)
                ),

            "data":
                pages

        }