# ============================================================
# document_process.py
# ============================================================

import os
import tempfile


from src.ocr_extractor import OCRExtractor
from src.jd_processor import JDProcessor

from src.token_counter import TokenCounter
from src.context_manager import ContextManager
from src.chunk_manager import ChunkManager

from src.resume_chunker import ResumeChunker

from src.model_config import ModelConfig

from src.embedding_factory import EmbeddingFactory

from src.chroma_db import ChromaDBManager

from src.retrieval_engine import RetrievalEngine

from src.prompts import PromptBuilder

from src.candidate_analyzer import CandidateAnalyzer
from src.score_calculator import ScoreCalculator

from src.result_formatter import ResultFormatter


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

        generation_model="gpt-4o-mini",

        embedding_method="OpenAI Embeddings",
        embedding_model="text-embedding-3-small",

        similarity_method="Cosine Similarity",

        top_k=5,

        output_tokens=None,

        safety_buffer_method="Percentage",
        safety_buffer_percent=20,

        fixed_safety_buffer=None,
        maximum_safety_buffer=None,

        chroma_persist_directory="chroma_db",
        chroma_collection_name="resume_collection"
    ):

        self.resume_files = resume_files

        self.jd_input_method = jd_input_method

        self.jd_file_name = jd_file_name

        self.openai_api_key = openai_api_key

        self.generation_model = generation_model

        self.embedding_method = embedding_method

        self.embedding_model = embedding_model

        self.similarity_method = similarity_method

        self.top_k = int(
            top_k
        )

        self.output_tokens = output_tokens

        self.safety_buffer_method = (
            safety_buffer_method
        )

        self.safety_buffer_percent = (
            safety_buffer_percent
        )

        self.fixed_safety_buffer = (
            fixed_safety_buffer
        )

        self.maximum_safety_buffer = (
            maximum_safety_buffer
        )

        self.chroma_persist_directory = (
            chroma_persist_directory
        )

        self.chroma_collection_name = (
            chroma_collection_name
        )

        # ----------------------------------------------------
        # PROMPTS
        # ----------------------------------------------------

        self.prompt_builder = (
            PromptBuilder()
        )

        # ----------------------------------------------------
        # MODEL CONFIG
        # ----------------------------------------------------

        model_config = (
            ModelConfig.get_model_config(
                self.generation_model
            )
        )

        self.context_window = int(
            model_config[
                "context_window"
            ]
        )

        if self.output_tokens is None:

            self.output_tokens = int(
                model_config[
                    "default_output_tokens"
                ]
            )

        # ----------------------------------------------------
        # SAFETY BUFFER
        # ----------------------------------------------------

        self.safety_buffer = (
            self.calculate_safety_buffer()
        )

    # ========================================================
    # MAIN PROCESS
    # ========================================================

    def process(self):

        self.validate_configuration()

        # ----------------------------------------------------
        # 1. RESUME OCR
        # ----------------------------------------------------

        resume_data = (
            self.process_resume()
        )

        # ----------------------------------------------------
        # 2. FULL RESUME TEXT
        # ----------------------------------------------------

        resume_text_map = (
            self.process_resume_text(
                resume_data
            )
        )

        # ----------------------------------------------------
        # 3. JOB DESCRIPTION
        # ----------------------------------------------------

        jd_text = (
            self.process_jd()
        )

        # ----------------------------------------------------
        # 4. TOKEN CALCULATION
        # ----------------------------------------------------

        token_data = (
            self.calculate_tokens(
                resume_text_map,
                jd_text
            )
        )

        # ----------------------------------------------------
        # 5. CONTEXT
        # ----------------------------------------------------

        context_data = (
            self.calculate_context(
                token_data
            )
        )

        # ----------------------------------------------------
        # 6. CHUNKING
        # ----------------------------------------------------

        chunk_data = (
            self.calculate_chunking(
                token_data
            )
        )

        # ----------------------------------------------------
        # 7. RESUME CHUNKS
        # ----------------------------------------------------

        resume_chunks = (
            self.process_resume_chunks(
                resume_text_map,
                chunk_data
            )
        )

        # ----------------------------------------------------
        # 8. EMBEDDINGS + CHROMA
        # ----------------------------------------------------

        vector_data = (
            self.prepare_vector_database(
                resume_chunks
            )
        )

        # ----------------------------------------------------
        # 9. RETRIEVAL + ANALYSIS + SCORING
        # ----------------------------------------------------

        matching_results = (
            self.search_resumes(
                jd_text,
                resume_text_map,
                resume_chunks,
                vector_data
            )
        )

        # ----------------------------------------------------
        # FINAL
        # ----------------------------------------------------

        return {

            "resume_data":
                resume_data,

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
                resume_chunks,

            "vector_data":
                vector_data,

            "matching_results":
                matching_results
        }

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_configuration(self):

        if not self.resume_files:

            raise ValueError(
                "Please upload at least one resume."
            )

        if not self.jd_file_name:

            raise ValueError(
                "Job Description is required."
            )

        if self.top_k < 1:

            raise ValueError(
                "Top-K must be at least 1."
            )

        # ----------------------------------------------------
        # BUFFER
        # ----------------------------------------------------

        if (
            self.safety_buffer_method
            ==
            "Percentage"
        ):

            allowed = [
                10,
                20,
                30,
                40,
                50,
                60
            ]

            if (
                self.safety_buffer_percent
                not in allowed
            ):

                raise ValueError(
                    "Safety buffer must be "
                    "10%, 20%, 30%, 40%, 50%, or 60%."
                )

        elif (
            self.safety_buffer_method
            ==
            "Fixed Tokens"
        ):

            if self.fixed_safety_buffer is None:

                raise ValueError(
                    "Fixed safety buffer is required."
                )

        elif (
            self.safety_buffer_method
            ==
            "Hybrid"
        ):

            if (
                self.safety_buffer_percent
                not in [
                    10,
                    20,
                    30,
                    40,
                    50,
                    60
                ]
            ):

                raise ValueError(
                    "Hybrid safety buffer must be "
                    "10%, 20%, 30%, 40%, 50%, or 60%."
                )

        else:

            raise ValueError(
                "Invalid safety buffer method."
            )

        # ----------------------------------------------------
        # OPENAI
        # ----------------------------------------------------

        if (
            self.embedding_method
            ==
            "OpenAI Embeddings"
        ):

            if not self.openai_api_key:

                raise ValueError(
                    "OpenAI API key is required "
                    "for OpenAI embeddings."
                )

    # ========================================================
    # SAFETY BUFFER
    # ========================================================

    def calculate_safety_buffer(self):

        if (
            self.safety_buffer_method
            ==
            "Percentage"
        ):

            return int(

                self.context_window
                *
                self.safety_buffer_percent
                /
                100

            )

        if (
            self.safety_buffer_method
            ==
            "Fixed Tokens"
        ):

            return int(
                self.fixed_safety_buffer
            )

        if (
            self.safety_buffer_method
            ==
            "Hybrid"
        ):

            percentage_buffer = int(

                self.context_window
                *
                self.safety_buffer_percent
                /
                100

            )

            if (
                self.maximum_safety_buffer
                is None
            ):

                return percentage_buffer

            return min(

                percentage_buffer,

                int(
                    self.maximum_safety_buffer
                )
            )

        raise ValueError(
            "Invalid safety buffer method."
        )

    # ========================================================
    # PROCESS RESUMES
    # ========================================================

    def process_resume(self):

        resume_data_map = {}

        for resume_file in self.resume_files:

            try:

                resume_data = (
                    self.process_resume_pdf(
                        resume_file
                    )
                )

                file_name = (
                    resume_data[
                        "file_name"
                    ]
                )

                resume_data_map[
                    file_name
                ] = resume_data

            except Exception as error:

                raise RuntimeError(

                    f"Failed to process "
                    f"{resume_file.name}: "
                    f"{error}"

                ) from error

        return resume_data_map

    # ========================================================
    # SINGLE RESUME
    # ========================================================

    def process_resume_pdf(
        self,
        resume_file
    ):

        temp_pdf_path = None

        try:

            file_bytes = (
                resume_file.getvalue()
            )

            original_file_name = (
                resume_file.name
            )

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

            extractor = OCRExtractor()

            return extractor.extract_pdf(

                temp_pdf_path,

                original_file_name
            )

        finally:

            if (
                temp_pdf_path
                and
                os.path.exists(
                    temp_pdf_path
                )
            ):

                try:

                    os.remove(
                        temp_pdf_path
                    )

                except Exception:

                    pass

    # ========================================================
    # FULL RESUME TEXT
    # ========================================================

    def get_full_resume_text(
        self,
        resume_data
    ):

        page_texts = []

        for page in resume_data.get(
            "data",
            []
        ):

            page_text = page.get(
                "page_data",
                ""
            )

            if page_text:

                page_texts.append(
                    page_text.strip()
                )

        return "\n\n".join(
            page_texts
        )

    # ========================================================
    # RESUME TEXT MAP
    # ========================================================

    def process_resume_text(
        self,
        resume_data_map
    ):

        resume_text_map = {}

        for (
            file_name,
            resume_data
        ) in resume_data_map.items():

            full_text = (
                self.get_full_resume_text(
                    resume_data
                )
            )

            # Preserve complete resume
            resume_data[
                "full_text"
            ] = full_text

            resume_text_map[
                file_name
            ] = full_text

        return resume_text_map

    # ========================================================
    # JOB DESCRIPTION
    # ========================================================

    def process_jd(self):

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if (
            self.jd_input_method
            ==
            "Upload PDF"
        ):

            processor = JDProcessor(

                input_method=
                    "Upload PDF",

                pdf_file=
                    self.jd_file_name
            )

            return processor.process()

        # ----------------------------------------------------
        # DOCX
        # ----------------------------------------------------

        if (
            self.jd_input_method
            ==
            "Upload DOCX"
        ):

            processor = JDProcessor(

                input_method=
                    "Upload DOCX",

                docx_file=
                    self.jd_file_name
            )

            return processor.process()

        # ----------------------------------------------------
        # TXT
        # ----------------------------------------------------

        if (
            self.jd_input_method
            ==
            "Upload TXT"
        ):

            processor = JDProcessor(

                input_method=
                    "Upload TXT",

                txt_file=
                    self.jd_file_name
            )

            return processor.process()

        # ----------------------------------------------------
        # PASTE
        # ----------------------------------------------------

        if (
            self.jd_input_method
            ==
            "Paste Text"
        ):

            processor = JDProcessor(

                input_method=
                    "Paste Text",

                pasted_text=
                    self.jd_file_name
            )

            return processor.process()

        raise ValueError(

            "Unsupported JD method: "
            +
            str(
                self.jd_input_method
            )
        )

    # ========================================================
    # TOKEN CALCULATION
    # ========================================================

    def calculate_tokens(
        self,
        resume_text_map,
        jd_text
    ):

        token_counter = TokenCounter(
            self.generation_model
        )

        schema = (
            self.prompt_builder
            .get_response_schema()
        )

        system_prompt = (
            self.prompt_builder
            .get_system_prompt()
        )

        schema_tokens = (
            token_counter.count_tokens(
                str(schema)
            )
        )

        system_prompt_tokens = (
            token_counter.count_tokens(
                system_prompt
            )
        )

        jd_tokens = (
            token_counter.count_tokens(
                jd_text
            )
        )

        resume_tokens = {}

        user_prompt_tokens = {}

        prompt_data_map = {}

        for (
            file_name,
            resume_text
        ) in resume_text_map.items():

            prompt_data = (
                self.prompt_builder.build(

                    job_description=
                        jd_text,

                    resume_text=
                        resume_text
                )
            )

            prompt_data_map[
                file_name
            ] = prompt_data

            user_prompt = (
                prompt_data[
                    "user_prompt"
                ]
            )

            resume_tokens[
                file_name
            ] = token_counter.count_tokens(
                resume_text
            )

            user_prompt_tokens[
                file_name
            ] = token_counter.count_tokens(
                user_prompt
            )

        return {

            "schema_tokens":
                schema_tokens,

            "system_prompt_tokens":
                system_prompt_tokens,

            "job_description_tokens":
                jd_tokens,

            "resume_tokens":
                resume_tokens,

            "user_prompt_tokens":
                user_prompt_tokens,

            "prompt_data":
                prompt_data_map
        }

    # ========================================================
    # CONTEXT
    # ========================================================

    def calculate_context(
        self,
        token_data
    ):

        manager = ContextManager(

            context_window=
                self.context_window,

            output_tokens=
                self.output_tokens,

            safety_buffer=
                self.safety_buffer
        )

        context_data = {}

        for (
            file_name,
            resume_tokens
        ) in token_data[
            "resume_tokens"
        ].items():

            context_data[
                file_name
            ] = manager.calculate(

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
                    resume_tokens
            )

        return context_data

    # ========================================================
    # DYNAMIC CHUNKING
    # ========================================================

    def calculate_chunking(
        self,
        token_data
    ):

        token_counter = TokenCounter(
            self.generation_model
        )

        user_prompt_template = (
            self.prompt_builder
            .user_prompt_template
        )

        user_prompt_instruction = (

            user_prompt_template

            .replace(
                "{job_description}",
                ""
            )

            .replace(
                "{resume_text}",
                ""
            )
        )

        instruction_tokens = (
            token_counter.count_tokens(
                user_prompt_instruction
            )
        )

        manager = ChunkManager(

            context_window=
                self.context_window,

            output_tokens=
                self.output_tokens,

            safety_buffer=
                self.safety_buffer,

            top_k=
                self.top_k,

            maximum_chunk_size=1500,

            minimum_chunk_size=100,

            overlap_percentage=15
        )

        chunk_data = manager.calculate(

            schema_tokens=
                token_data[
                    "schema_tokens"
                ],

            system_prompt_tokens=
                token_data[
                    "system_prompt_tokens"
                ],

            user_prompt_template_tokens=
                instruction_tokens,

            job_description_tokens=
                token_data[
                    "job_description_tokens"
                ]
        )

        chunk_data[
            "embedding_method"
        ] = self.embedding_method

        chunk_data[
            "embedding_model"
        ] = self.embedding_model

        chunk_data[
            "similarity_method"
        ] = self.similarity_method

        chunk_data[
            "safety_buffer_method"
        ] = self.safety_buffer_method

        chunk_data[
            "safety_buffer_percent"
        ] = self.safety_buffer_percent

        chunk_data[
            "safety_buffer_tokens"
        ] = self.safety_buffer

        return chunk_data

    # ========================================================
    # CHUNK RESUMES
    # ========================================================

    def process_resume_chunks(
        self,
        resume_text_map,
        chunk_data
    ):

        chunker = ResumeChunker(

            chunk_size=
                int(
                    chunk_data[
                        "chunk_size"
                    ]
                ),

            chunk_overlap=
                int(
                    chunk_data[
                        "chunk_overlap"
                    ]
                )
        )

        resume_chunks = {}

        for (
            file_name,
            resume_text
        ) in resume_text_map.items():

            chunks = chunker.chunk_text(
                resume_text
            )

            resume_chunks[
                file_name
            ] = []

            for (
                index,
                chunk_text
            ) in enumerate(
                chunks,
                start=1
            ):

                if not chunk_text.strip():

                    continue

                resume_chunks[
                    file_name
                ].append({

                    "chunk_id":
                        index,

                    "file_name":
                        file_name,

                    "text":
                        chunk_text.strip(),

                    "character_count":
                        len(
                            chunk_text
                        )
                })

        return resume_chunks

    # ========================================================
    # VECTOR DATABASE
    # ========================================================

    def prepare_vector_database(
        self,
        resume_chunks
    ):

        embedding_manager = (
            EmbeddingFactory.create(

                embedding_method=
                    self.embedding_method,

                api_key=
                    self.openai_api_key,

                model_name=
                    self.embedding_model
            )
        )

        documents = []

        metadatas = []

        ids = []

        for (
            file_name,
            chunks
        ) in resume_chunks.items():

            for chunk in chunks:

                documents.append(
                    chunk["text"]
                )

                metadatas.append({

                    "file_name":
                        file_name,

                    "chunk_id":
                        str(
                            chunk[
                                "chunk_id"
                            ]
                        )
                })

                safe_name = (

                    file_name

                    .replace(
                        " ",
                        "_"
                    )

                    .replace(
                        "/",
                        "_"
                    )

                    .replace(
                        "\\",
                        "_"
                    )
                )

                ids.append(

                    f"{safe_name}"
                    f"__chunk_"
                    f"{chunk['chunk_id']}"

                )

        if not documents:

            raise ValueError(
                "No resume chunks were created."
            )

        # ----------------------------------------------------
        # SPARSE MODELS
        # ----------------------------------------------------

        if hasattr(
            embedding_manager,
            "fit"
        ):

            embedding_manager.fit(
                documents
            )

        # ----------------------------------------------------
        # EMBEDDINGS
        # ----------------------------------------------------

        embeddings = (
            embedding_manager
            .embed_documents(
                documents
            )
        )

        # ----------------------------------------------------
        # CHROMA
        # ----------------------------------------------------

        chroma_db = ChromaDBManager(

            persist_directory=
                self.chroma_persist_directory,

            collection_name=
                self.chroma_collection_name
        )

        chroma_db.clear()

        chroma_db.add_documents(

            documents=
                documents,

            embeddings=
                embeddings,

            metadatas=
                metadatas,

            ids=
                ids
        )

        embedding_dimension = 0

        if embeddings:

            try:

                embedding_dimension = len(
                    embeddings[0]
                )

            except Exception:

                pass

        return {

            "documents":
                documents,

            "embeddings":
                embeddings,

            "metadatas":
                metadatas,

            "ids":
                ids,

            "embedding_dimension":
                embedding_dimension,

            "total_chunks":
                len(
                    documents
                )
        }

    # ========================================================
    # SEARCH + ANALYSIS + SCORE
    # ========================================================

    def search_resumes(
        self,
        jd_text,
        resume_text_map,
        resume_chunks,
        vector_data
    ):

        # ----------------------------------------------------
        # RETRIEVAL ENGINE
        # ----------------------------------------------------

        retrieval_engine = RetrievalEngine(

            embedding_method=
                self.embedding_method,

            embedding_model=
                self.embedding_model,

            similarity_method=
                self.similarity_method,

            api_key=
                self.openai_api_key,

            top_k=
                self.top_k,

            chroma_persist_directory=
                self.chroma_persist_directory,

            chroma_collection_name=
                self.chroma_collection_name
        )

        # ----------------------------------------------------
        # RETRIEVE
        # ----------------------------------------------------

        retrieval_results = (
            retrieval_engine.search(

                jd_text=
                    jd_text,

                resume_chunks=
                    resume_chunks,

                vector_data=
                    vector_data
            )
        )

        # ----------------------------------------------------
        # SCORE CALCULATOR
        # ----------------------------------------------------

        score_calculator = (
            ScoreCalculator()
        )

        # ----------------------------------------------------
        # CANDIDATE ANALYZER
        #
        # FIXED:
        #
        # OLD:
        # api_key=
        # model_name=
        # prompt_builder=
        #
        # NEW:
        # openai_api_key=
        # model=
        # score_calculator=
        # ----------------------------------------------------

        candidate_analyzer = CandidateAnalyzer(

            openai_api_key=
                self.openai_api_key,

            model=
                self.generation_model,

            score_calculator=
                score_calculator
        )

        # ----------------------------------------------------
        # FORMATTER
        # ----------------------------------------------------

        formatter = (
            ResultFormatter()
        )

        final_results = []

        # ====================================================
        # EACH RETRIEVED RESUME
        # ====================================================

        for retrieval_result in retrieval_results:

            file_name = (
                retrieval_result.get(
                    "file_name"
                )
            )

            if not file_name:

                continue

            # ------------------------------------------------
            # COMPLETE RESUME
            # ------------------------------------------------

            resume_text = (
                resume_text_map.get(
                    file_name,
                    ""
                )
            )

            if not resume_text:

                continue

            # ------------------------------------------------
            # RETRIEVAL SCORE
            # ------------------------------------------------

            retrieval_score = (
                retrieval_result.get(
                    "score",

                    retrieval_result.get(
                        "retrieval_score",
                        0.0
                    )
                )
            )

            # ------------------------------------------------
            # CANDIDATE ANALYSIS
            #
            # FIXED SIGNATURE
            # ------------------------------------------------

            analysis = (
                candidate_analyzer.analyze(

                    resume_text=
                        resume_text,

                    jd_text=
                        jd_text,

                    retrieval_score=
                        retrieval_score,

                    file_name=
                        file_name,

                    candidate_name=
                        retrieval_result.get(
                            "candidate_name"
                        )
                )
            )

            # ------------------------------------------------
            # IMPORTANT
            #
            # CandidateAnalyzer already calculates the
            # REAL Match % using ScoreCalculator.
            #
            # We DO NOT calculate:
            #
            # retrieval_score * 100
            #
            # ------------------------------------------------

            final_match_percentage = (
                analysis.get(
                    "match_percentage",
                    0.0
                )
            )

            try:

                final_match_percentage = float(
                    final_match_percentage
                )

            except (
                TypeError,
                ValueError
            ):

                final_match_percentage = 0.0

            final_match_percentage = max(

                0.0,

                min(
                    100.0,
                    final_match_percentage
                )
            )

            # ------------------------------------------------
            # RETRIEVED CHUNKS
            # ------------------------------------------------

            retrieved_chunks = (
                retrieval_result.get(
                    "chunks",
                    []
                )
            )

            # ------------------------------------------------
            # FINAL RESULT
            # ------------------------------------------------

            final_result = {

                "file_name":
                    file_name,

                "candidate_name":
                    analysis.get(
                        "candidate_name",
                        os.path.splitext(
                            file_name
                        )[0]
                    ),

                # REAL MATCH %
                "match_percentage":
                    round(
                        final_match_percentage,
                        2
                    ),

                # KEEP SEPARATE
                "retrieval_score":
                    retrieval_score,

                "retrieval_method":
                    self.similarity_method,

                "component_scores":
                    analysis.get(
                        "component_scores",
                        {}
                    ),

                "required_skills":
                    analysis.get(
                        "required_skills",
                        []
                    ),

                "candidate_skills":
                    analysis.get(
                        "candidate_skills",
                        []
                    ),

                "matched_required_skills":
                    analysis.get(
                        "matched_required_skills",
                        []
                    ),

                "missing_required_skills":
                    analysis.get(
                        "missing_required_skills",
                        []
                    ),

                "additional_candidate_skills":
                    analysis.get(
                        "additional_candidate_skills",
                        []
                    ),

                "matched_skills":
                    analysis.get(
                        "matched_skills",
                        []
                    ),

                "missing_skills":
                    analysis.get(
                        "missing_skills",
                        []
                    ),

                "required_years":
                    analysis.get(
                        "required_years"
                    ),

                "candidate_years":
                    analysis.get(
                        "candidate_years"
                    ),

                "required_responsibilities":
                    analysis.get(
                        "required_responsibilities",
                        []
                    ),

                "candidate_responsibilities":
                    analysis.get(
                        "candidate_responsibilities",
                        []
                    ),

                "responsibility_match":
                    analysis.get(
                        "responsibility_match",
                        0
                    ),

                "required_education":
                    analysis.get(
                        "required_education"
                    ),

                "candidate_education":
                    analysis.get(
                        "candidate_education"
                    ),

                "relevant_projects":
                    analysis.get(
                        "relevant_projects",
                        []
                    ),

                "strengths":
                    analysis.get(
                        "strengths",
                        []
                    ),

                "skill_gaps":
                    analysis.get(
                        "skill_gaps",
                        []
                    ),

                "recommendations":
                    analysis.get(
                        "recommendations",
                        []
                    ),

                "summary":
                    analysis.get(
                        "summary",
                        ""
                    ),

                "retrieved_chunks":
                    retrieved_chunks
            }

            # ------------------------------------------------
            # ADD RAW ANALYSIS
            # ------------------------------------------------

            final_result.update(
                analysis
            )

            # ------------------------------------------------
            # PROTECT IMPORTANT VALUES
            # ------------------------------------------------

            final_result[
                "match_percentage"
            ] = round(
                final_match_percentage,
                2
            )

            final_result[
                "retrieval_score"
            ] = retrieval_score

            final_result[
                "retrieval_method"
            ] = self.similarity_method

            final_results.append(
                final_result
            )

        # ====================================================
        # SORT BY REAL MATCH %
        # ====================================================

        final_results.sort(

            key=lambda item: float(
                item.get(
                    "match_percentage",
                    0
                )
            ),

            reverse=True
        )

        # ====================================================
        # RANK
        # ====================================================

        for (
            rank,
            result
        ) in enumerate(
            final_results,
            start=1
        ):

            result[
                "rank"
            ] = rank

        # ====================================================
        # FORMAT
        # ====================================================

        return formatter.format_results(
            final_results
        )