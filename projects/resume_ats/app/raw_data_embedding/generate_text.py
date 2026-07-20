from pdf2image import convert_from_path
import pytesseract


class GenerateText:

    @staticmethod
    def extract_with_ocr(pdf_path):
        images = convert_from_path(pdf_path, dpi=300)

        text = []

        for image in images:
            page_text = pytesseract.image_to_string(
                image,
                lang="eng",
                config="--oem 3 --psm 6"
            )
            text.append(page_text)

        return "\n".join(text)