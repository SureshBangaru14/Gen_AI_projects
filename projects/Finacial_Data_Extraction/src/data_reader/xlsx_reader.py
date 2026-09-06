import pandas as pd


class XLSXProcessor:

    def __init__(self, xlsx_file):

        self.xlsx_file = xlsx_file

    # ========================================================
    # PROCESS XLSX
    # ========================================================

    def process_xlsx(self):

        # Read all Excel sheets
        sheets = pd.read_excel(self.xlsx_file, sheet_name=None)

        extracted_data = []

        # ----------------------------------------------------
        # PROCESS EACH SHEET
        # ----------------------------------------------------

        for sheet_name, dataframe in sheets.items():

            # Replace NaN with empty string
            dataframe = dataframe.fillna("")

            # Convert dataframe to text
            sheet_text = dataframe.to_string(index=False)

            extracted_data.append({

                "sheet_name": sheet_name,

                "sheet_data": sheet_text})

        return extracted_data