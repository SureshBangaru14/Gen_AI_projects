import pandas as pd


class CSVProcessor:

    def __init__(self, csv_file):

        self.csv_file = csv_file

    # ========================================================
    # PROCESS CSV
    # ========================================================

    def process_csv(self):

        dataframe = pd.read_csv(self.csv_file)

        # Replace NaN
        dataframe = dataframe.fillna("")

        # Convert CSV into text
        text = dataframe.to_string(index=False)

        return text