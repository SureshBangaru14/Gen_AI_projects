from docx import Document
import tempfile
import os
from src.ocr_extractor import OCRExtractor


class JDProcessor:
    
        def __init__(self):
            
            self.ocr_extractor = OCRExtractor()
            
        # ========================================================
        # MAIN PROCESSOR
        # ========================================================

        def process(self, input_method, pdf_file=None, docx_file=None, pasted_text=None):

            if input_method == "Upload PDF":

                return self.process_pdf(pdf_file)

            elif input_method == "Upload DOCX":

                return self.process_docx(docx_file)

            elif input_method == "Paste Text":

                return self.process_text(pasted_text)

            raise ValueError("Invalid Job Description input method.")
        
            
        def process_pdf(self, uploaded_file):
            
            file_bytes = uploaded_file.getvalue()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                
                temp_file.write(file_bytes)
                
            temp_pdf_path = temp_file.name
            
            jd_data = self.ocr_extractor.extract_pdf(temp_pdf_path, uploaded_file.name)
            os.remove(temp_pdf_path)
            jd_text = "\n\n".join(page["page_data"] for page in jd_data["data"] if page["page_data"].strip())

            return jd_text
        
        def process_docx(self, uploaded_file):
            
            file_bytes = uploaded_file.getvalue()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                
                temp_file.write(file_bytes)

            temp_docx_path = temp_file.name
            
            document = Document(temp_docx_path)

            paragraphs = []

            for paragraph in document.paragraphs:

                text = paragraph.text.strip()

                if text:

                    paragraphs.append(text)

            jd_text = "\n\n".join(paragraphs)
            os.remove(temp_docx_path)

            return jd_text
        
        def process_text(self, text):
            
            return text.strip()
        