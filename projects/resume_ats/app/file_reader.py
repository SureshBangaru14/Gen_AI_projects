from pathlib import Path

from pdf2image import convert_from_path
import pytesseract


class FileReader:


    def extract_with_ocr(self, pdf_path):

        images = convert_from_path(
            pdf_path,
            dpi=300
        )

        text = []


        for page_no, image in enumerate(images):

            page_text = pytesseract.image_to_string(

                image,

                lang="eng",

                config="--oem 3 --psm 6"

            )


            text.append(page_text)


            print(
                f"OCR page {page_no + 1}/{len(images)} completed"
            )


        return "\n".join(text)



    def read_txt(self, file_path):

        with open(

            file_path,

            "r",

            encoding="utf-8"

        ) as file:

            return file.read()



    def read_file(self, file_path):

        extension = Path(
            file_path
        ).suffix.lower()



        if extension == ".pdf":

            return self.extract_with_ocr(
                file_path
            )


        elif extension == ".txt":

            return self.read_txt(
                file_path
            )


        else:

            raise ValueError(
                "Only PDF and TXT supported"
            )