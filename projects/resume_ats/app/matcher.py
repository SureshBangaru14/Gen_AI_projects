import numpy as np
from openai import OpenAI


class ResumeJDMatcher:


    def __init__(self, api_key):

        self.client = OpenAI(
            api_key=api_key
        )


    def generate_embedding(self, text):

        response = self.client.embeddings.create(

            model="text-embedding-3-small",

            input=text

        )


        vector = np.array(

            response.data[0].embedding,

            dtype="float32"

        )


        vector = vector / np.linalg.norm(vector)


        return vector



    def similarity(

        self,

        resume_vector,

        jd_vector

    ):

        score = np.dot(

            resume_vector,

            jd_vector

        )


        return round(

            float(score) * 100,

            2

        )



    def match(

        self,

        resume_text,

        jd_text

    ):


        resume_vector = self.generate_embedding(

            resume_text

        )


        jd_vector = self.generate_embedding(

            jd_text

        )


        score = self.similarity(

            resume_vector,

            jd_vector

        )


        # Match classification

        if score >= 85:

            match_type = "Strong Match"

            status = "Match"



        elif score >= 70:

            match_type = "Good Match"

            status = "Match"



        elif score >= 50:

            match_type = "Weak Match"

            status = "Partial Match"



        else:

            match_type = "Not Match"

            status = "Not Match"



        return {

            "score": score,

            "status": status,

            "match_type": match_type

        }