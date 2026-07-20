import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import re
import unicodedata


class ResumeTextCleaner:

    def __init__(self, env_path):
        self.env_path = env_path
        self.base_path = None
        self.df = None

        self.load_environment()


    def load_environment(self):
        """
        Load input file path from .env
        """

        load_dotenv(self.env_path)

        file_path = os.getenv("RESUME_RAW_DATA_EXCEL")

        if not file_path:
            raise ValueError(
                "RESUME_RAW_DATA_EXCEL is not set in the .env file."
            )

        self.base_path = Path(file_path)

        if not self.base_path.exists():
            raise FileNotFoundError(
                f"File does not exist: {self.base_path}"
            )


    def read_file(self):
        """
        Read input file.

        Supported formats:
        - .xlsx
        - .xls
        - .csv
        """

        extension = self.base_path.suffix.lower()

        if extension == ".xlsx":

            self.df = pd.read_excel(
                self.base_path,
                engine="openpyxl"
            )

        elif extension == ".xls":

            self.df = pd.read_excel(
                self.base_path,
                engine="xlrd"
            )

        elif extension == ".csv":

            self.df = pd.read_csv(
                self.base_path
            )

        else:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                "Supported: .xlsx, .xls, .csv"
            )


        print("\nFile loaded successfully")
        print("Columns:")
        print(self.df.columns.tolist())


    @staticmethod
    def clean_resume_text(text):
        """
        Clean resume text.

        Keeps:
        - alphabets
        - numbers
        - spaces
        - new lines
        - @
        - .
        - +
        - -

        Removes:
        - Unicode symbols
        - emojis
        - brackets
        - unwanted special characters
        """

        if not isinstance(text, str):
            return ""


        # Unicode normalization
        text = unicodedata.normalize(
            "NFKC",
            text
        )


        cleaned_lines = []


        # Process each line separately
        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue


            # Keep only:
            # A-Z a-z 0-9 @ . + - space
            line = re.sub(
                r"[^A-Za-z0-9@.\+\- ]+",
                " ",
                line
            )


            # Remove extra spaces
            line = re.sub(
                r"\s+",
                " ",
                line
            ).strip()


            if line:
                cleaned_lines.append(line)


        # Preserve line breaks
        return "\n".join(cleaned_lines)



    def clean_column(self):
        """
        Clean Extracted Text column
        """

        if "Extracted Text" not in self.df.columns:

            raise ValueError(
                "Column 'Extracted Text' not found"
            )


        self.df["Extracted Text"] = (
            self.df["Extracted Text"]
            .apply(self.clean_resume_text)
        )


        print("\nText cleaning completed")



    def save_file(self):
        """
        Save cleaned data as Excel
        """

        output_file = self.base_path.with_name(
            f"{self.base_path.stem}_cleaned.xlsx"
        )


        self.df.to_excel(
            output_file,
            index=False,
            engine="openpyxl"
        )


        print("\nSaved successfully")
        print(f"Output file: {output_file}")



    def run(self):
        """
        Complete preprocessing pipeline
        """

        self.read_file()

        self.clean_column()

        self.save_file()



if __name__ == "__main__":

    env_file = "/home/suresh/Gen_AI_Practice/.env"

    cleaner = ResumeTextCleaner(
        env_file
    )

    cleaner.run()