import os
import tempfile

from pdf2image import convert_from_path
from rapidocr_onnxruntime import RapidOCR


class PDFProcessor:

    def __init__(self, pdf_file):

        self.pdf_file = pdf_file

        self.ocr = RapidOCR()

    # ========================================================
    # CREATE TEMP PDF
    # ========================================================

    def create_temp_pdf(self):

        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)

        temp_file.write(self.pdf_file.getvalue())

        temp_file.close()

        return temp_file.name

    # ========================================================
    # PDF → IMAGES
    # ========================================================

    def convert_pdf_to_images(self, pdf_path):

        images = convert_from_path(pdf_path,dpi=300)

        return images

    # ========================================================
    # OCR
    # ========================================================

    def extract_text_from_image(self, image):

        result, elapsed = self.ocr(image)

        page_text = []

        if result:

            for item in result:

                text = item[1]

                confidence = float(item[2])

                if confidence >= 0.30:

                    page_text.append(text)

        return "\n".join(page_text)

    # ========================================================
    # PROCESS PDF
    # ========================================================

    def process_pdf(self):

        temp_pdf_path = None

        try:

            temp_pdf_path = self.create_temp_pdf()

            images = self.convert_pdf_to_images(temp_pdf_path)

            extracted_data = []

            for page_no, image in enumerate(images, start=1):

                page_text = self.extract_text_from_image(image)

                extracted_data.append({
                    "page_no": page_no,
                    "page_data": page_text
                })

            return extracted_data

        finally:

            if (temp_pdf_path and os.path.exists(temp_pdf_path)):

                os.remove(temp_pdf_path)