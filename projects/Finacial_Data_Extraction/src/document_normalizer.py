class DocumentNormalizer:

    def __init__(self, file_name, file_type, extracted_data):

        self.file_name = file_name
        self.file_type = file_type
        self.extracted_data = extracted_data

    # ========================================================
    # NORMALIZE DOCUMENT
    # ========================================================

    def normalize(self):

        # ----------------------------------------------------
        # PDF / DOCX
        # ----------------------------------------------------

        if self.file_type in ["pdf", "docx"]:

            pages = []

            for page in self.extracted_data:

                page_no = page.get("page_no")

                page_data = page.get("page_data", "")

                # Clean text
                page_data = self.clean_text(page_data)

                pages.append({

                    "page_no": page_no,

                    "page_data": page_data

                })

            return {

                "file_name": self.file_name,

                "file_type": self.file_type,

                "total_pages": len(pages),

                "pages": pages

            }

        # ----------------------------------------------------
        # XLSX
        # ----------------------------------------------------

        elif self.file_type == "xlsx":

            sheets = []

            for sheet in self.extracted_data:

                sheets.append({

                    "sheet_name": sheet.get(
                        "sheet_name",
                        ""
                    ),

                    "sheet_data": self.clean_text(
                        sheet.get(
                            "sheet_data",
                            ""
                        )
                    )

                })

            return {

                "file_name": self.file_name,

                "file_type": self.file_type,

                "total_sheets": len(sheets),

                "sheets": sheets

            }

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        elif self.file_type == "csv":

            return {

                "file_name": self.file_name,

                "file_type": self.file_type,

                "data": self.clean_text(
                    self.extracted_data
                )

            }

        # ----------------------------------------------------
        # Unsupported
        # ----------------------------------------------------

        else:

            raise ValueError(
                f"Unsupported file type: {self.file_type}"
            )

    # ========================================================
    # CLEAN TEXT
    # ========================================================

    def clean_text(self, text):

        if not text:
            return ""

        lines = []

        for line in str(text).splitlines():

            # Remove spaces at beginning/end
            line = line.strip()

            # Keep non-empty lines
            if line:
                lines.append(line)

        # Preserve line-by-line structure
        return "\n".join(lines)