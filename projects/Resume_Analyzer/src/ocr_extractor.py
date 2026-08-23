from pathlib import Path
import shutil
import subprocess

from pdf2image import convert_from_path
import pytesseract


class OCRExtractor:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(self):

        self.poppler_path = (
            self.find_poppler_path()
        )


    # ========================================================
    # FIND POPPLER
    # ========================================================

    def find_poppler_path(self):

        pdfinfo_path = shutil.which(
            "pdfinfo"
        )

        if pdfinfo_path:

            return str(
                Path(
                    pdfinfo_path
                ).parent
            )

        possible_paths = [

            "/usr/bin",

            "/usr/local/bin",

            "/opt/homebrew/bin",

            "/usr/bin/poppler-utils",

            "/usr/local/bin/poppler-utils"

        ]

        for path in possible_paths:

            pdfinfo = (
                Path(path)
                /
                "pdfinfo"
            )

            if pdfinfo.exists():

                return path

        return None


    # ========================================================
    # CHECK POPPLER
    # ========================================================

    def check_poppler(self):

        pdfinfo_path = shutil.which(
            "pdfinfo"
        )

        if pdfinfo_path:

            return pdfinfo_path

        if self.poppler_path:

            pdfinfo = (
                Path(
                    self.poppler_path
                )
                /
                "pdfinfo"
            )

            if pdfinfo.exists():

                return str(
                    pdfinfo
                )

        raise RuntimeError(
            "Poppler is not installed or "
            "pdfinfo is not available in PATH. "
            "Make sure packages.txt contains "
            "'poppler-utils'."
        )


    # ========================================================
    # GET POPPLER VERSION
    # ========================================================

    def get_poppler_version(self):

        pdfinfo_path = (
            self.check_poppler()
        )

        try:

            result = subprocess.run(

                [
                    pdfinfo_path,
                    "-v"
                ],

                capture_output=True,

                text=True,

                timeout=10

            )

            return (
                result.stderr
                or
                result.stdout
            ).strip()

        except Exception as error:

            return (
                f"Unable to determine Poppler version: "
                f"{error}"
            )


    # ========================================================
    # PDF INFO
    # ========================================================

    def get_pdf_info(
        self,
        pdf_path
    ):

        pdf_path = Path(
            pdf_path
        )

        if not pdf_path.exists():

            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        pdfinfo_path = (
            self.check_poppler()
        )

        try:

            result = subprocess.run(

                [
                    pdfinfo_path,
                    str(pdf_path)
                ],

                capture_output=True,

                text=True,

                timeout=30

            )

        except Exception as error:

            raise RuntimeError(
                "Failed to execute Poppler pdfinfo. "
                f"Poppler: {pdfinfo_path}. "
                f"Error: {error}"
            ) from error

        if result.returncode != 0:

            raise RuntimeError(
                "Poppler could not read the PDF. "
                f"PDF: {pdf_path.name}. "
                f"pdfinfo error: "
                f"{result.stderr.strip()}"
            )

        return result.stdout


    # ========================================================
    # PDF → IMAGES
    # ========================================================

    def pdf_to_images(
        self,
        pdf_path
    ):

        pdf_path = Path(
            pdf_path
        )

        if not pdf_path.exists():

            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":

            raise ValueError(
                f"File is not a PDF: {pdf_path}"
            )

        self.check_poppler()

        try:

            if self.poppler_path:

                return convert_from_path(

                    str(pdf_path),

                    dpi=300,

                    poppler_path=
                        self.poppler_path

                )

            return convert_from_path(

                str(pdf_path),

                dpi=300

            )

        except Exception as error:

            raise RuntimeError(

                "Unable to convert PDF to images. "

                f"PDF: {pdf_path.name}. "

                f"Poppler path: "
                f"{self.poppler_path}. "

                f"Original error: {error}"

            ) from error


    # ========================================================
    # IMAGE → TEXT
    # ========================================================

    def image_to_text(
        self,
        image
    ):

        try:

            return (
                pytesseract.image_to_string(
                    image
                )
            )

        except Exception as error:

            raise RuntimeError(

                "Tesseract OCR failed. "

                "Make sure tesseract-ocr is installed. "

                f"Original error: {error}"

            ) from error


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

        if not pdf_path.exists():

            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":

            raise ValueError(
                f"File is not a PDF: {pdf_path}"
            )

        # ----------------------------------------------------
        # Verify Poppler
        # ----------------------------------------------------

        self.check_poppler()

        # ----------------------------------------------------
        # Verify PDF using pdfinfo
        # ----------------------------------------------------

        self.get_pdf_info(
            pdf_path
        )

        # ----------------------------------------------------
        # Convert PDF
        # ----------------------------------------------------

        images = (
            self.pdf_to_images(
                pdf_path
            )
        )

        pages = []

        # ----------------------------------------------------
        # OCR every page
        # ----------------------------------------------------

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

            pages.append({

                "page_no":
                    str(
                        page_number
                    ),

                "page_data":
                    text.strip()

            })

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

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