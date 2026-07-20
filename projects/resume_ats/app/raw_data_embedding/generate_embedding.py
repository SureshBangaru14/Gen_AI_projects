import os
import json
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


class ResumeEmbeddingGenerator:

    def __init__(self, env_path, workers=10):

        self.env_path = env_path
        self.workers = workers

        self.file_path = None
        self.df = None

        self.client = None

        self.load_environment()
        self.initialize_openai()


    def load_environment(self):
        """
        Load environment variables
        """

        load_dotenv(self.env_path)

        file_path = os.getenv(
            "RESUME_CLEANED_DATA_EXCEL"
        )

        if not file_path:
            raise ValueError(
                "RESUME_CLEANED_FILE missing in .env"
            )

        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"File not found: {self.file_path}"
            )


    def initialize_openai(self):
        """
        Initialize OpenAI client
        """

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY missing in .env"
            )

        self.client = OpenAI(
            api_key=api_key
        )


    def read_file(self):
        """
        Read input file
        Supports:
        .xlsx
        .xls
        .csv
        """

        extension = self.file_path.suffix.lower()


        if extension == ".xlsx":

            self.df = pd.read_excel(
                self.file_path,
                engine="openpyxl"
            )


        elif extension == ".xls":

            self.df = pd.read_excel(
                self.file_path,
                engine="xlrd"
            )


        elif extension == ".csv":

            self.df = pd.read_csv(
                self.file_path
            )


        else:

            raise ValueError(
                f"Unsupported file type: {extension}"
            )


        print("\nFile loaded successfully")
        print(
            "Columns:",
            self.df.columns.tolist()
        )

        print(
            "Total resumes:",
            len(self.df)
        )



    def generate_embedding(self, text):
        """
        Generate OpenAI embedding
        """

        try:

            if not isinstance(text, str):
                return []


            if not text.strip():
                return []


            response = self.client.embeddings.create(

                model="text-embedding-3-small",

                input=text

            )


            return response.data[0].embedding


        except Exception as e:

            print(
                "Embedding error:",
                e
            )

            return []



    def embedding_worker(
        self,
        index,
        text
    ):

        vector = self.generate_embedding(
            text
        )

        return index, vector



    def create_embeddings(self):
        """
        Generate embeddings using parallel workers
        """

        if "Extracted Text" not in self.df.columns:

            raise ValueError(
                "Extracted Text column missing"
            )


        texts = (
            self.df["Extracted Text"]
            .fillna("")
            .tolist()
        )


        total = len(texts)


        embeddings = [
            []
            for _ in range(total)
        ]


        print("\n================================")
        print(
            f"Starting embedding generation"
        )
        print(
            f"Workers: {self.workers}"
        )
        print(
            f"Total rows: {total}"
        )
        print("================================\n")



        with ThreadPoolExecutor(
            max_workers=self.workers
        ) as executor:


            futures = []


            for index, text in enumerate(texts):

                future = executor.submit(

                    self.embedding_worker,

                    index,

                    text

                )

                futures.append(
                    future
                )


            completed = 0


            for future in as_completed(futures):

                index, vector = future.result()


                embeddings[index] = vector


                completed += 1


                dimension = (
                    len(vector)
                    if vector
                    else 0
                )


                print(
                    f"Embedding completed "
                    f"| Row {index + 1}/{total} "
                    f"| Completed {completed}/{total} "
                    f"| Dimension {dimension}"
                )



        self.df["Embedding"] = embeddings



        print("\n================================")
        print("All embeddings generated")
        print(
            f"Total embeddings: {len(embeddings)}"
        )
        print(
            "Model: text-embedding-3-small"
        )
        print(
            "Vector dimension: 1536"
        )
        print("================================\n")



    def save_excel(self):
        """
        Save Excel embedding file
        """

        output_file = self.file_path.with_name(

            f"{self.file_path.stem}_embeddings.xlsx"

        )


        excel_df = self.df.copy()


        excel_df["Embedding"] = (

            excel_df["Embedding"]

            .apply(json.dumps)

        )


        excel_df.to_excel(

            output_file,

            index=False,

            engine="openpyxl"

        )


        print(
            "Excel saved:"
        )

        print(
            output_file
        )



    def save_pickle(self):
        """
        Save pickle file
        """

        output_file = self.file_path.with_name(

            f"{self.file_path.stem}_embeddings.pkl"

        )


        pickle_data = {

            "embeddings":
                self.df["Embedding"].tolist(),


            "metadata":
                self.df[
                    [
                        "Category",
                        "File Name",
                        "Extracted Text"
                    ]
                ].to_dict(
                    orient="records"
                )

        }


        with open(
            output_file,
            "wb"
        ) as f:

            pickle.dump(
                pickle_data,
                f
            )


        print(
            "Pickle saved:"
        )

        print(
            output_file
        )



    def run(self):

        self.read_file()

        self.create_embeddings()

        self.save_excel()

        self.save_pickle()



if __name__ == "__main__":

    generator = ResumeEmbeddingGenerator(

        env_path="/home/suresh/Gen_AI_Practice/.env",

        workers=10

    )


    generator.run()