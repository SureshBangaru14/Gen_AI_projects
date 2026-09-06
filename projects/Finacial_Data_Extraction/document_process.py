import os

from src.data_reader.pdf_reader import PDFProcessor
from src.data_reader.docx_reader import DOCXProcessor
from src.data_reader.xlsx_reader import XLSXProcessor
from src.data_reader.csv_reader import CSVProcessor

from src.document_normalizer import DocumentNormalizer


class DocumentProcess:

    def __init__(
        self,
        input_file,
        financial_doc_type = None,
        openai_api_key = None,
        embedding_method=None,
        embedding_model=None
    ):

        self.input_file = input_file
        self.financial_doc_type = financial_doc_type
        self.openai_api_key = openai_api_key
        self.embedding_method = embedding_method
        self.embedding_model = embedding_model

    def process(self):

        # ====================================================
        # IMPORTANT
        # Streamlit UploadedFile → use .name
        # ====================================================

        extension = os.path.splitext(self.input_file.name)[1].lower()
        print(f"File extension: {extension}")

        # ====================================================
        # PDF
        # ====================================================

        if extension == ".pdf":

            extracted_data = PDFProcessor(pdf_file=self.input_file).process_pdf()
            file_type = "pdf"

        # ====================================================
        # DOCX
        # ====================================================

        elif extension == ".docx":

            extracted_data = DOCXProcessor(docx_file=self.input_file).process_docx()
            file_type = "docx"

        # ====================================================
        # XLSX
        # ====================================================

        elif extension == ".xlsx":

            extracted_data = XLSXProcessor(xlsx_file=self.input_file).process_xlsx()
            file_type = "xlsx"

        # ====================================================
        # CSV
        # ====================================================

        elif extension == ".csv":
            
            extracted_data = CSVProcessor(csv_file=self.input_file).process_csv()
            file_type = "csv"

        # ====================================================
        # UNSUPPORTED
        # ====================================================

        else:

            raise ValueError(f"Unsupported file type: {extension}")
        
        
        # ====================================================
        # NORMALIZATION
        # ====================================================

        normalized_data = DocumentNormalizer(file_name=self.input_file.name, file_type=file_type, extracted_data=extracted_data).normalize()

        return normalized_data
