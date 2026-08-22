# ============================================================

# document_process.py

# ============================================================

import os

import tempfile

import re



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

        chunking_method="Recursive Character Text Splitting",

        chunk_size=None,

        chunk_overlap=None,

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

        # ----------------------------------------------------

        # CHUNKING CONFIGURATION

        # ----------------------------------------------------

        self.chunking_method = (

            chunking_method

        )

        self.chunk_size = (

            int(chunk_size)

            if chunk_size is not None

            else None

        )

        self.chunk_overlap = (

            int(chunk_overlap)

            if chunk_overlap is not None

            else None

        )

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

    # JD SKILL REQUIREMENT EXTRACTION

    # ========================================================

    # Common skills/technologies used by resume/JD matching.

    # This vocabulary is intentionally explicit so a JD without a

    # "Required Skills" heading can still produce deterministic

    # required-skill candidates from its Summary, Responsibilities,

    # and Technology Stack sections.

    JD_SKILL_VOCABULARY = [

        "Python", "SQL", "NumPy", "Pandas", "Scikit-learn",

        "Machine Learning", "Data Science", "Data Preprocessing",

        "Feature Engineering", "Feature Selection", "Exploratory Data Analysis",

        "EDA", "Model Selection", "Model Training", "Model Evaluation",

        "Hyperparameter Tuning", "Supervised Learning", "Unsupervised Learning",

        "Regression", "Classification", "Clustering", "Prediction",

        "Decision Trees", "Random Forest", "XGBoost", "Statistics", "Probability",

        "Performance Metrics", "Data Visualization", "Matplotlib", "Seaborn",

        "TensorFlow", "PyTorch", "Deep Learning", "NLP", "Computer Vision",

        "Time Series", "ML Pipelines", "REST APIs", "REST API",

        "Model Deployment", "Model Monitoring", "Model Retraining", "FastAPI",

        "Flask", "Docker", "AWS", "Azure", "GCP", "MLOps", "MLflow",

        "CI/CD", "Kubernetes", "Git", "Generative AI", "LLMs", "LLM",

        "Prompt Engineering", "RAG", "Embeddings", "Vector Databases",

        "Semantic Search", "LangChain", "LlamaIndex", "FAISS", "Pinecone",

        "Weaviate", "Milvus", "Transformers", "Tokenization", "Model Monitoring"

    ]

    OPTIONAL_SECTION_NAMES = [

        "good to have", "good-to-have", "nice to have", "nice-to-have",

        "preferred skills", "preferred qualifications", "preferred requirements",

        "bonus", "bonus skills", "desirable", "desired skills", "optional"

    ]

    REQUIRED_SECTION_NAMES = [

        "required skills", "required skill", "required qualifications",

        "required qualification", "required requirements", "requirements",

        "must have", "must-have", "mandatory skills", "mandatory requirements"

    ]

    FALLBACK_REQUIRED_SECTION_NAMES = [

        "job summary", "summary", "key responsibilities", "responsibilities",

        "key responsibility", "technology stack", "technical skills",

        "technical skill", "experience", "job description"

    ]

    def _normalize_skill(self, value):

        value = str(value or "").lower().strip()

        value = re.sub(r"[^a-z0-9+#./-]+", " ", value)

        value = re.sub(r"\s+", " ", value).strip()

        aliases = {

            "scikit learn": "scikit-learn",

            "sklearn": "scikit-learn",

            "rest api": "rest apis",

            "llm": "llms",

            "vector database": "vector databases",

            "vector db": "vector databases",

            "ml pipeline": "ml pipelines",

            "eda": "exploratory data analysis",

        }

        return aliases.get(value, value)

    def _unique_skills(self, skills):

        result = []

        seen = set()

        for skill in skills or []:

            clean = str(skill or "").strip()

            if not clean:

                continue

            key = self._normalize_skill(clean)

            if key and key not in seen:

                seen.add(key)

                result.append(clean)

        return result

    def _find_section(self, text, heading_names):

        if not text:

            return ""

        names = sorted(

            [re.escape(x) for x in heading_names],

            key=len,

            reverse=True

        )

        if not names:

            return ""

        pattern = re.compile(

            r"(?im)^\s*(?:#+\s*)?(?:" + "|".join(names) + r")\s*:?\s*$"

        )

        matches = list(pattern.finditer(text))

        if not matches:

            return ""

        start = matches[0].end()

        next_heading = re.search(r"(?im)^\s*#{0,6}\s*[A-Za-z][A-Za-z0-9 &/()+.-]{2,80}\s*:?[ **\t**]*$", text[start:])

        end = start + next_heading.start() if next_heading else len(text)

        return text[start:end].strip()

    def _extract_vocab_skills(self, text):

        if not text:

            return []

        lower_text = text.lower()

        found = []

        for skill in sorted(self.JD_SKILL_VOCABULARY, key=len, reverse=True):

            # Flexible whitespace for multi-word skills.

            words = skill.lower().split()

            pattern = r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b"

            if re.search(pattern, lower_text, flags=re.IGNORECASE):

                found.append(skill)

        return self._unique_skills(found)

    def extract_jd_skill_requirements(self, jd_text):

        """Return required/optional skills with a heading-aware fallback.

        If an explicit Required Skills section exists, it is authoritative.

        If it does not exist, required skills are inferred from the JD Summary,

        Responsibilities, Technology Stack and other non-optional sections.

        Good-to-have/preferred sections are kept separate and never become

        required merely because Required Skills is absent.

        """

        required_section = self._find_section(

            jd_text, self.REQUIRED_SECTION_NAMES

        )

        optional_section = self._find_section(

            jd_text, self.OPTIONAL_SECTION_NAMES

        )

        explicit_required = bool(required_section.strip())

        if explicit_required:

            required_skills = self._extract_vocab_skills(required_section)

        else:

            fallback_parts = []

            for name in self.FALLBACK_REQUIRED_SECTION_NAMES:

                section = self._find_section(jd_text, [name])

                if section:

                    fallback_parts.append(section)

            # Technology Stack is especially useful when a JD has no

            # Required Skills heading.

            fallback_text = "\n".join(fallback_parts)

            required_skills = self._extract_vocab_skills(fallback_text)

            # Final fallback: use the whole JD, excluding an explicitly

            # detected optional section.

            if not required_skills:

                non_optional_text = jd_text

                if optional_section:

                    non_optional_text = non_optional_text.replace(

                        optional_section, ""

                    )

                required_skills = self._extract_vocab_skills(

                    non_optional_text

                )

        good_to_have_skills = self._extract_vocab_skills(

            optional_section

        )

        # Never classify an optional skill as required when headings overlap.

        optional_keys = {

            self._normalize_skill(x)

            for x in good_to_have_skills

        }

        required_skills = [

            x for x in required_skills

            if self._normalize_skill(x) not in optional_keys

        ]

        return {

            "required_skills": self._unique_skills(required_skills),

            "good_to_have_skills": self._unique_skills(good_to_have_skills),

            "has_explicit_required_skills": explicit_required,

            "required_skill_source": (

                "Required Skills section"

                if explicit_required

                else "JD fallback: Summary/Responsibilities/Technology Stack"

            )

        }

    def _skill_matches(self, required_skills, candidate_skills):

        candidate_keys = {

            self._normalize_skill(x)

            for x in candidate_skills or []

            if str(x or "").strip()

        }

        matched = []

        missing = []

        for skill in required_skills or []:

            key = self._normalize_skill(skill)

            if key in candidate_keys:

                matched.append(skill)

            else:

                # Handle common substring/alias cases such as LLM/LLMs,

                # REST API/REST APIs and Scikit-Learn/Scikit Learn.

                found = any(

                    key == candidate_key

                    or key in candidate_key

                    or candidate_key in key

                    for candidate_key in candidate_keys

                )

                if found:

                    matched.append(skill)

                else:

                    missing.append(skill)

        return matched, missing

    def _apply_jd_skill_policy(self, jd_text, analysis):

        policy = self.extract_jd_skill_requirements(jd_text)

        candidate_skills = analysis.get("candidate_skills", []) or []

        matched_required, missing_required = self._skill_matches(

            policy["required_skills"], candidate_skills

        )

        matched_optional, missing_optional = self._skill_matches(

            policy["good_to_have_skills"], candidate_skills

        )

        analysis["required_skills"] = policy["required_skills"]

        analysis["good_to_have_skills"] = policy["good_to_have_skills"]

        analysis["matched_required_skills"] = matched_required

        analysis["missing_required_skills"] = missing_required

        analysis["matched_good_to_have_skills"] = matched_optional

        analysis["missing_good_to_have_skills"] = missing_optional

        analysis["has_explicit_required_skills"] = policy[

            "has_explicit_required_skills"

        ]

        analysis["required_skill_source"] = policy[

            "required_skill_source"

        ]

        # Existing UI fields continue to work.

        analysis["matched_skills"] = matched_required

        analysis["missing_skills"] = missing_required

        analysis["skill_gaps"] = missing_required

        return analysis

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

        # ADD PER-RESUME CONTEXT DETAILS

        # ----------------------------------------------------

        for item in matching_results:

            file_name = item.get("file_name", "")

            item["context_details"] = context_data.get(

                file_name,

                {}

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

        # CHUNKING

        # ----------------------------------------------------

        allowed_chunking_methods = [

            "Recursive Character Text Splitting",

            "Fixed-Size Chunking",

            "Token-Based Chunking",

            "Semantic Chunking",

            "Structure-Based Chunking",

            "Markdown-Based Chunking",

            "HTML-Based Chunking",

            "Sentence-Based Chunking",

            "Paragraph-Based Chunking",

            "Document-Section Chunking",

            "Parent-Child Chunking",

            "Hierarchical Chunking"

        ]

        if (

            self.chunking_method

            not in allowed_chunking_methods

        ):

            raise ValueError(

                "Invalid chunking method: "

                + str(self.chunking_method)

            )

        if self.chunk_size is not None:

            if self.chunk_size < 100:

                raise ValueError(

                    "Chunk size must be at least 100."

                )

        if self.chunk_overlap is not None:

            if self.chunk_overlap < 0:

                raise ValueError(

                    "Chunk overlap cannot be negative."

                )

            if (

                self.chunk_size is not None

                and

                self.chunk_overlap >= self.chunk_size

            ):

                raise ValueError(

                    "Chunk overlap must be smaller "

                    "than chunk size."

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

        # ----------------------------------------------------

        # SELECTED CHUNKING METHOD

        # ----------------------------------------------------

        chunk_data[

            "chunking_method"

        ] = self.chunking_method

        # If app.py supplied an explicit chunk size/overlap,

        # use those values. Otherwise keep the existing

        # dynamic ChunkManager calculation.

        if self.chunk_size is not None:

            chunk_data[

                "chunk_size"

            ] = self.chunk_size

        if self.chunk_overlap is not None:

            chunk_data[

                "chunk_overlap"

            ] = self.chunk_overlap

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

        # ----------------------------------------------------

        # SELECTED CHUNKING METHOD

        # ----------------------------------------------------

        selected_chunking_method = (

            chunk_data.get(

                "chunking_method",

                self.chunking_method

            )

        )

        selected_chunk_size = int(

            chunk_data.get(

                "chunk_size",

                self.chunk_size

                if self.chunk_size is not None

                else 500

            )

        )

        selected_chunk_overlap = int(

            chunk_data.get(

                "chunk_overlap",

                self.chunk_overlap

                if self.chunk_overlap is not None

                else 100

            )

        )

        # Safety check for overlap.

        if selected_chunk_overlap >= selected_chunk_size:

            selected_chunk_overlap = max(

                0,

                selected_chunk_size - 1

            )

        # ----------------------------------------------------

        # RESUME CHUNKER

        # ----------------------------------------------------

        chunker = ResumeChunker(

            chunking_method=

                selected_chunking_method,

            chunk_size=

                selected_chunk_size,

            chunk_overlap=

                selected_chunk_overlap

        )

        resume_chunks = {}

        # ----------------------------------------------------

        # EACH RESUME

        # ----------------------------------------------------

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

            # ------------------------------------------------

            # SUPPORT BOTH:

            #

            # 1. Old ResumeChunker:

            #    ["text 1", "text 2"]

            #

            # 2. New ResumeChunker:

            #    [

            #       {

            #          "chunk_id": 1,

            #          "text": "...",

            #          "metadata": {...}

            #       }

            #    ]

            # ------------------------------------------------

            for (

                index,

                chunk_item

            ) in enumerate(

                chunks,

                start=1

            ):

                if isinstance(

                    chunk_item,

                    dict

                ):

                    chunk_text = str(

                        chunk_item.get(

                            "text",

                            ""

                        )

                    ).strip()

                    chunk_id = (

                        chunk_item.get(

                            "chunk_id",

                            index

                        )

                    )

                    metadata = (

                        chunk_item.get(

                            "metadata",

                            {}

                        )

                    )

                    character_count = (

                        chunk_item.get(

                            "character_count",

                            len(chunk_text)

                        )

                    )

                else:

                    chunk_text = str(

                        chunk_item

                    ).strip()

                    chunk_id = index

                    metadata = {}

                    character_count = (

                        len(chunk_text)

                    )

                if not chunk_text:

                    continue

                # Ensure selected method is always

                # available in metadata.

                metadata = dict(

                    metadata

                    if isinstance(

                        metadata,

                        dict

                    )

                    else {}

                )

                metadata[

                    "chunking_method"

                ] = selected_chunking_method

                resume_chunks[

                    file_name

                ].append({

                    "chunk_id":

                        chunk_id,

                    "file_name":

                        file_name,

                    "text":

                        chunk_text,

                    "character_count":

                        character_count,

                    "chunking_method":

                        selected_chunking_method,

                    "metadata":

                        metadata

                })

        # ----------------------------------------------------

        # UPDATE CHUNK DATA

        # ----------------------------------------------------

        chunk_data[

            "chunking_method"

        ] = selected_chunking_method

        chunk_data[

            "chunk_size"

        ] = selected_chunk_size

        chunk_data[

            "chunk_overlap"

        ] = selected_chunk_overlap

        total_chunks = sum(

            len(chunks)

            for chunks

            in resume_chunks.values()

        )

        chunk_data[

            "total_resume_chunks"

        ] = total_chunks

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

                        "\\\\",

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

        # Top-K in the UI means FINAL RESUME rows.

        # Vector search operates on chunks, so retrieve all chunks

        # first; otherwise multiple chunks from the same resume can

        # consume the Top-K budget and hide other resumes.

        retrieval_top_k = max(

            self.top_k,

            int(

                vector_data.get(

                    "total_chunks",

                    self.top_k

                )

            )

        )

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

                retrieval_top_k,

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

        # BUILD ONE RETRIEVAL RECORD PER RESUME

        # ====================================================

        retrieval_by_file = {}

        for retrieval_result in retrieval_results:

            file_name = retrieval_result.get(

                "file_name"

            )

            if not file_name:

                continue

            current_score = retrieval_result.get(

                "score",

                retrieval_result.get(

                    "retrieval_score",

                    0.0

                )

            )

            try:

                current_score = float(current_score)

            except (TypeError, ValueError):

                current_score = 0.0

            existing = retrieval_by_file.get(

                file_name

            )

            if existing is None:

                retrieval_by_file[file_name] = {

                    "file_name": file_name,

                    "candidate_name": retrieval_result.get(

                        "candidate_name"

                    ),

                    "score": current_score,

                    "chunks": retrieval_result.get(

                        "chunks",

                        []

                    )

                }

            else:

                if current_score > float(

                    existing.get("score", 0.0) or 0.0

                ):

                    existing["score"] = current_score

                    existing["candidate_name"] = retrieval_result.get(

                        "candidate_name",

                        existing.get("candidate_name")

                    )

                existing["chunks"] = (

                    existing.get("chunks", [])

                    + retrieval_result.get("chunks", [])

                )

        # ====================================================

        # ANALYZE EVERY UPLOADED RESUME

        # ====================================================

        # Top-K is applied AFTER all resumes are scored.

        for file_name, resume_text in resume_text_map.items():

            if not resume_text:

                continue

            retrieval_result = retrieval_by_file.get(

                file_name,

                {

                    "file_name": file_name,

                    "candidate_name": None,

                    "score": 0.0,

                    "chunks": []

                }

            )

            # ------------------------------------------------

            # COMPLETE RESUME

            # ------------------------------------------------

            # ------------------------------------------------

            # RETRIEVAL SCORE

            # ------------------------------------------------

            retrieval_score = retrieval_result.get(

                "score",

                0.0

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

            # JD SKILL POLICY

            # ------------------------------------------------

            # If the JD has an explicit Required Skills section,

            # use it. Otherwise infer required skills from the

            # Summary/Responsibilities/Technology Stack and keep

            # Good-to-Have skills separate.

            analysis = self._apply_jd_skill_policy(

                jd_text,

                analysis

            )

            # ------------------------------------------------

            # RE-CALCULATE SCORE AFTER JD SKILL POLICY

            # ----------------------------------------------------

            # CandidateAnalyzer may have extracted its own skill list.

            # When Required Skills is absent, the fallback policy above

            # changes the required skill set, so the final percentage must

            # be recalculated using that normalized set.

            score_result = score_calculator.calculate(

                required_skills=

                    analysis.get(

                        "required_skills",

                        []

                    ),

                candidate_skills=

                    analysis.get(

                        "candidate_skills",

                        []

                    ),

                required_years=

                    analysis.get(

                        "required_years",

                        0

                    ),

                candidate_years=

                    analysis.get(

                        "candidate_years",

                        0

                    ),

                required_responsibilities=

                    analysis.get(

                        "required_responsibilities",

                        []

                    ),

                candidate_responsibilities=

                    analysis.get(

                        "candidate_responsibilities",

                        []

                    ),

                required_education=

                    analysis.get(

                        "required_education",

                        ""

                    ),

                candidate_education=

                    analysis.get(

                        "candidate_education",

                        ""

                    ),

                retrieval_score=

                    retrieval_score

            )

            analysis[

                "match_percentage"

            ] = score_result.get(

                "overall_match_percentage",

                0.0

            )

            analysis[

                "component_scores"

            ] = score_result.get(

                "component_scores",

                {}

            )

            # Keep the explicit matched/missing lists from the

            # normalized JD policy rather than replacing them with

            # a generic LLM-derived list.

            analysis[

                "matched_required_skills"

            ] = analysis.get(

                "matched_required_skills",

                []

            )

            analysis[

                "missing_required_skills"

            ] = analysis.get(

                "missing_required_skills",

                []

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

                "good_to_have_skills":

                    analysis.get(

                        "good_to_have_skills",

                        []

                    ),

                "matched_good_to_have_skills":

                    analysis.get(

                        "matched_good_to_have_skills",

                        []

                    ),

                "missing_good_to_have_skills":

                    analysis.get(

                        "missing_good_to_have_skills",

                        []

                    ),

                "has_explicit_required_skills":

                    analysis.get(

                        "has_explicit_required_skills",

                        False

                    ),

                "required_skill_source":

                    analysis.get(

                        "required_skill_source",

                        "N/A"

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

        # FINAL TOP-K RESUMES

        # ====================================================

        final_results = final_results[:self.top_k]

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