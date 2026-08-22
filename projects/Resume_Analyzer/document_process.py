import tempfile
import os

from src.ocr_extractor import OCRExtractor
from src.jd_processor import JDProcessor
from src.token_counter import TokenCounter
from src.context_manager import ContextManager
from src.chunk_manager import ChunkManager
from src.resume_chunker import ResumeChunker

from src.prompts import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    USER_PROMPT
)


class DocumentProcess:

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        resume_files=None,
        jd_input_method=None,
        jd_file_name=None,
        openai_api_key=None,
        generation_model=None,
        embedding_model="text-embedding-3-small",
        top_k=5
    ):

        self.resume_files = resume_files

        self.jd_input_method = jd_input_method

        self.jd_file_name = jd_file_name

        self.openai_api_key = openai_api_key

        self.generation_model = generation_model

        self.embedding_model = embedding_model

        self.top_k = top_k


        # ----------------------------------------------------
        # These values should later come from model config.
        # ----------------------------------------------------

        self.context_window = 8192

        self.output_tokens = 1000

        self.safety_buffer = 300


    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process(self):

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        self.validate_configuration()


        # ----------------------------------------------------
        # RESUMES
        # ----------------------------------------------------

        resume_text_map = (
            self.process_resume()
        )


        # ----------------------------------------------------
        # JOB DESCRIPTION
        # ----------------------------------------------------

        jd_text = (
            self.process_jd()
        )


        # ----------------------------------------------------
        # TOKEN COUNT
        # ----------------------------------------------------

        token_data = (
            self.calculate_tokens(
                resume_text_map,
                jd_text
            )
        )


        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        context_data = (
            self.calculate_context(
                token_data
            )
        )


        # ----------------------------------------------------
        # CHUNK CONFIG
        # ----------------------------------------------------

        chunk_data = (
            self.calculate_chunking(
                token_data
            )
        )


        # ----------------------------------------------------
        # ACTUAL CHUNKS
        # ----------------------------------------------------

        resume_chunks = (
            self.process_resume_chunks(
                resume_text_map,
                chunk_data
            )
        )


        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        return {

            "resume_text":
                resume_text_map,

            "job_description":
                jd_text,

            "token_data":
                token_data,

            "context_data":
                context_data,

            "chunk_data":
                chunk_data,

            "resume_chunks":
                resume_chunks

        }


    # ========================================================
    # VALIDATE CONFIGURATION
    # ========================================================

    def validate_configuration(self):

        if not self.openai_api_key:

            raise ValueError(
                "OpenAI API key is required."
            )


        if not self.generation_model:

            raise ValueError(
                "Generation model is required."
            )


        if self.top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0."
            )


    # ========================================================
    # PROCESS MULTIPLE RESUMES
    # ========================================================

    def process_resume(self):

        resume_text_map = {}


        for resume_file in self.resume_files:

            resume_data = (
                self.process_resume_pdf(
                    resume_file
                )
            )


            full_resume_text = (
                self.get_full_resume_text(
                    resume_data
                )
            )


            resume_text_map[
                resume_data["file_name"]
            ] = full_resume_text


        return resume_text_map


    # ========================================================
    # PROCESS SINGLE RESUME PDF
    # ========================================================

    def process_resume_pdf(
        self,
        resume_file
    ):

        file_bytes = (
            resume_file.getvalue()
        )


        original_file_name = (
            resume_file.name
        )


        temp_pdf_path = None


        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    file_bytes
                )

                temp_pdf_path = (
                    temp_file.name
                )


            return (
                OCRExtractor().extract_pdf(
                    temp_pdf_path,
                    original_file_name
                )
            )


        finally:

            if (
                temp_pdf_path
                and os.path.exists(
                    temp_pdf_path
                )
            ):

                os.remove(
                    temp_pdf_path
                )


    # ========================================================
    # COMBINE PAGE TEXT
    # ========================================================

    def get_full_resume_text(
        self,
        resume_data
    ):

        page_texts = []


        for page in resume_data["data"]:

            page_text = (
                page["page_data"].strip()
            )


            if page_text:

                page_texts.append(
                    page_text
                )


        return "\n\n".join(
            page_texts
        )


    # ========================================================
    # PROCESS JOB DESCRIPTION
    # ========================================================

    def process_jd(self):

        if self.jd_input_method == "Upload PDF":

            return (
                JDProcessor(
                    input_method="Upload PDF",
                    pdf_file=self.jd_file_name
                ).process()
            )


        elif self.jd_input_method == "Upload DOCX":

            return (
                JDProcessor(
                    input_method="Upload DOCX",
                    docx_file=self.jd_file_name
                ).process()
            )


        elif self.jd_input_method == "Paste Text":

            return (
                JDProcessor(
                    input_method="Paste Text",
                    pasted_text=self.jd_file_name
                ).process()
            )


        raise ValueError(
            "Invalid Job Description input method."
        )


    # ========================================================
    # TOKEN COUNT
    # ========================================================

    def calculate_tokens(
        self,
        resume_text_map,
        jd_text
    ):

        token_counter = TokenCounter(
            self.generation_model
        )


        # ----------------------------------------------------
        # SCHEMA
        # ----------------------------------------------------

        schema_tokens = (
            token_counter.count_tokens(
                RESPONSE_SCHEMA
            )
        )


        # ----------------------------------------------------
        # SYSTEM PROMPT
        # ----------------------------------------------------

        system_prompt_tokens = (
            token_counter.count_tokens(
                SYSTEM_PROMPT
            )
        )


        # ----------------------------------------------------
        # JD
        # ----------------------------------------------------

        jd_tokens = (
            token_counter.count_tokens(
                jd_text
            )
        )


        resume_token_map = {}

        user_prompt_token_map = {}

        user_prompt_map = {}


        # ====================================================
        # EACH RESUME
        # ====================================================

        for file_name, resume_text in resume_text_map.items():

            # ------------------------------------------------
            # ACTUAL USER PROMPT
            # ------------------------------------------------

            user_prompt = (
                USER_PROMPT.format(
                    job_description=jd_text,
                    resume_text=resume_text
                )
            )


            # ------------------------------------------------
            # USER PROMPT TOKENS
            # ------------------------------------------------

            user_prompt_tokens = (
                token_counter.count_tokens(
                    user_prompt
                )
            )


            # ------------------------------------------------
            # RESUME TOKENS
            # ------------------------------------------------

            resume_tokens = (
                token_counter.count_tokens(
                    resume_text
                )
            )


            resume_token_map[
                file_name
            ] = resume_tokens


            user_prompt_token_map[
                file_name
            ] = user_prompt_tokens


            user_prompt_map[
                file_name
            ] = user_prompt


        return {

            "schema":
                RESPONSE_SCHEMA,

            "system_prompt":
                SYSTEM_PROMPT,

            "user_prompt":
                user_prompt_map,

            "schema_tokens":
                schema_tokens,

            "system_prompt_tokens":
                system_prompt_tokens,

            "user_prompt_tokens":
                user_prompt_token_map,

            "job_description_tokens":
                jd_tokens,

            "resume_tokens":
                resume_token_map,

            "generation_model":
                self.generation_model

        }


    # ========================================================
    # CONTEXT WINDOW
    # ========================================================

    def calculate_context(
        self,
        token_data
    ):

        context_manager = ContextManager(

            context_window=
                self.context_window,

            output_tokens=
                self.output_tokens,

            safety_buffer=
                self.safety_buffer

        )


        context_data = {}


        for file_name in token_data[
            "resume_tokens"
        ]:

            context_data[
                file_name
            ] = context_manager.calculate(

                schema_tokens=
                    token_data[
                        "schema_tokens"
                    ],

                system_prompt_tokens=
                    token_data[
                        "system_prompt_tokens"
                    ],

                user_prompt_tokens=
                    token_data[
                        "user_prompt_tokens"
                    ][file_name],

                job_description_tokens=
                    token_data[
                        "job_description_tokens"
                    ],

                resume_tokens=
                    token_data[
                        "resume_tokens"
                    ][file_name]

            )


        return context_data


    # ========================================================
    # DYNAMIC CHUNK SIZE
    # ========================================================

    def calculate_chunking(
        self,
        token_data
    ):

        chunk_manager = ChunkManager(

            context_window=
                self.context_window,

            output_tokens=
                self.output_tokens,

            safety_buffer=
                self.safety_buffer,

            top_k=
                self.top_k

        )


        chunk_data = (
            chunk_manager.calculate(

                schema_tokens=
                    token_data[
                        "schema_tokens"
                    ],

                system_prompt_tokens=
                    token_data[
                        "system_prompt_tokens"
                    ],

                job_description_tokens=
                    token_data[
                        "job_description_tokens"
                    ]

            )
        )


        chunk_data[
            "generation_model"
        ] = self.generation_model


        chunk_data[
            "embedding_model"
        ] = self.embedding_model


        return chunk_data


    # ========================================================
    # ACTUAL RESUME CHUNKING
    # ========================================================

    def process_resume_chunks(
        self,
        resume_text_map,
        chunk_data
    ):

        chunk_size = (
            chunk_data[
                "chunk_size"
            ]
        )


        chunk_overlap = (
            chunk_data[
                "chunk_overlap"
            ]
        )


        chunker = ResumeChunker(

            model_name=
                self.generation_model,

            chunk_size=
                chunk_size,

            chunk_overlap=
                chunk_overlap

        )


        resume_chunks = {}


        for file_name, resume_text in resume_text_map.items():

            chunks = (
                chunker.chunk_text(
                    resume_text
                )
            )


            resume_chunks[
                file_name
            ] = []


            for index, chunk in enumerate(
                chunks,
                start=1
            ):

                token_count = len(
                    chunker.text_to_tokens(
                        chunk
                    )
                )


                resume_chunks[
                    file_name
                ].append(
                    {

                        "chunk_id":
                            index,

                        "text":
                            chunk,

                        "token_count":
                            token_count

                    }
                )


        return resume_chunks