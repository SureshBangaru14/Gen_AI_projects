import os
import shutil
import subprocess
import tempfile

from pdf2image import convert_from_path
from rapidocr_onnxruntime import RapidOCR


class DOCXProcessor:

    def __init__(self, docx_file):

        self.docx_file = docx_file

        self.ocr = RapidOCR()

    # ========================================================
    # CREATE TEMP DOCX
    # ========================================================

    def create_temp_docx(self):

        temp_dir = tempfile.mkdtemp(prefix="docx_processing_")

        docx_path = os.path.join(temp_dir, self.docx_file.name)

        with open(docx_path, "wb") as file:

            file.write(self.docx_file.getvalue())

        return docx_path, temp_dir

    # ========================================================
    # DOCX → PDF
    #
    # IMPORTANT:
    #
    # DOCX page 1
    #       ↓
    # PDF page 1
    #
    # DOCX page 2
    #       ↓
    # PDF page 2
    #
    # DOCX page 3
    #       ↓
    # PDF page 3
    #
    # LibreOffice renders the DOCX instead of rebuilding
    # the document from extracted text.
    # ========================================================

    def convert_docx_to_pdf(self, docx_path):

        pdf_dir = tempfile.mkdtemp(prefix="docx_pdf_")

        # ----------------------------------------------------
        # Separate LibreOffice profile
        #
        # This avoids:
        #
        # - locked profile
        # - readonly database
        # - existing LibreOffice process problems
        # ----------------------------------------------------

        lo_profile = tempfile.mkdtemp(prefix="libreoffice_profile_")

        try:

            command = [

                "libreoffice",

                "--headless",

                # Isolated LibreOffice profile
                f"-env:UserInstallation=file://{lo_profile}",

                # Conversion
                "--convert-to",
                "pdf:writer_pdf_Export",

                # Output directory
                "--outdir",
                pdf_dir,

                # Input DOCX
                docx_path
            ]

            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)

            # ------------------------------------------------
            # Check LibreOffice status
            # ------------------------------------------------

            if result.returncode != 0:

                raise RuntimeError(

                    "LibreOffice conversion failed.\n\n"

                    f"STDOUT:\n"
                    f"{result.stdout}\n\n"

                    f"STDERR:\n"
                    f"{result.stderr}"
                )

            # ------------------------------------------------
            # Expected PDF name
            # ------------------------------------------------

            pdf_name = (os.path.splitext(os.path.basename(docx_path))[0] + ".pdf")

            pdf_path = os.path.join(pdf_dir, pdf_name)

            # ------------------------------------------------
            # Verify PDF exists
            # ------------------------------------------------

            if not os.path.exists(pdf_path):

                raise FileNotFoundError(

                    "PDF was not created.\n\n"

                    f"Expected:\n"
                    f"{pdf_path}\n\n"

                    f"LibreOffice output:\n"
                    f"{result.stdout}\n"

                    f"{result.stderr}"
                )

            # ------------------------------------------------
            # Verify PDF is not empty
            # ------------------------------------------------

            pdf_size = os.path.getsize(pdf_path)

            if pdf_size == 0:

                raise RuntimeError("Generated PDF is empty.")

            # ------------------------------------------------
            # Get PDF page count
            #
            # Using pdfinfo instead of modifying pages.
            # ------------------------------------------------

            page_count = self.get_pdf_page_count(pdf_path)

            if page_count == 0:

                raise RuntimeError("Generated PDF contains zero pages.")

            return (pdf_path, pdf_dir, lo_profile, page_count)

        except Exception:

            shutil.rmtree(pdf_dir, ignore_errors=True)

            shutil.rmtree(lo_profile, ignore_errors=True)

            raise

    # ========================================================
    # GET PDF PAGE COUNT
    # ========================================================

    def get_pdf_page_count(self, pdf_path):

        try:

            result = subprocess.run(

                [
                    "pdfinfo",
                    pdf_path
                ],

                stdout=subprocess.PIPE,

                stderr=subprocess.PIPE,

                text=True,

                timeout=30
            )

            if result.returncode != 0:

                raise RuntimeError(result.stderr)

            for line in result.stdout.splitlines():

                line = line.strip()

                if line.startswith("Pages:"):

                    return int(line.split(":")[1].strip())

            raise RuntimeError(
                "Unable to determine PDF page count."
            )

        except FileNotFoundError:

            raise RuntimeError(

                "pdfinfo is not installed.\n\n"

                "Install it using:\n"

                "sudo apt install poppler-utils"
            )

    # ========================================================
    # PDF → IMAGES
    # ========================================================

    def convert_pdf_to_images(self, pdf_path):

        images = convert_from_path(pdf_path, dpi=250, fmt="png")

        return images

    # ========================================================
    # RAPIDOCR
    # ========================================================

    def extract_text_from_image(self, image):

        result, _ = self.ocr(image)

        page_text = []

        if result:

            for item in result:

                # OCR result:
                #
                # item[0] = bounding box
                # item[1] = text
                # item[2] = confidence

                text = item[1]

                confidence = float(item[2])

                if confidence >= 0.30:

                    page_text.append(text)

        return "\n".join(page_text)

    # ========================================================
    # PROCESS DOCX
    # ========================================================

    def process_docx(self):

        docx_path = None

        docx_temp_dir = None

        pdf_path = None

        pdf_dir = None

        lo_profile = None

        try:

            # =================================================
            # STEP 1
            #
            # Streamlit UploadedFile
            #             ↓
            # Temporary DOCX
            # =================================================

            (
                docx_path,
                docx_temp_dir
            ) = self.create_temp_docx()

            # =================================================
            # STEP 2
            #
            # DOCX
            #   ↓
            # LibreOffice Writer Rendering
            #   ↓
            # PDF
            # =================================================

            (
                pdf_path,
                pdf_dir,
                lo_profile,
                pdf_page_count
            ) = self.convert_docx_to_pdf(
                docx_path
            )

            # =================================================
            # STEP 3
            #
            # PDF → Images
            #
            # IMPORTANT:
            #
            # Every PDF page becomes ONE image.
            #
            # PDF Page 1 → image[0]
            # PDF Page 2 → image[1]
            # PDF Page 3 → image[2]
            # =================================================

            images = self.convert_pdf_to_images(pdf_path)

            # =================================================
            # PAGE COUNT VALIDATION
            # =================================================

            image_page_count = len(images)

            if image_page_count != pdf_page_count:

                raise RuntimeError(

                    "PDF page/image count mismatch.\n"

                    f"PDF pages   : "
                    f"{pdf_page_count}\n"

                    f"Image pages : "
                    f"{image_page_count}"
                )

            # =================================================
            # STEP 4
            #
            # PAGE-WISE OCR
            # =================================================

            extracted_data = []

            for page_no, image in enumerate(images, start=1):

                # ---------------------------------------------
                # OCR current page ONLY
                # ---------------------------------------------

                page_text = (self.extract_text_from_image(image))

                # ---------------------------------------------
                # Store page-wise result
                # ---------------------------------------------

                extracted_data.append({

                    "page_no": page_no,

                    "page_data": page_text

                })

            # =================================================
            # FINAL VALIDATION
            # =================================================

            if len(extracted_data) != pdf_page_count:

                raise RuntimeError(

                    "Page extraction count mismatch.\n"

                    f"Expected : {pdf_page_count}\n"

                    f"Extracted: "
                    f"{len(extracted_data)}"
                )

            # =================================================
            # RETURN
            # =================================================

            return extracted_data

        finally:

            # =================================================
            # CLEANUP TEMP DOCX
            # =================================================

            if docx_temp_dir:

                shutil.rmtree(docx_temp_dir, ignore_errors=True)

            # =================================================
            # CLEANUP TEMP PDF
            # =================================================

            if pdf_dir:

                shutil.rmtree(pdf_dir, ignore_errors=True)

            # =================================================
            # CLEANUP LIBREOFFICE PROFILE
            # =================================================

            if lo_profile:

                shutil.rmtree(lo_profile, ignore_errors=True)