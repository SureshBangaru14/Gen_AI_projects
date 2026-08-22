# ============================================================
# src/resume_chunker.py
# ============================================================

import re


class ResumeChunker:

    # ========================================================
    # SUPPORTED CHUNKING METHODS
    # ========================================================

    CHUNKING_METHODS = [

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

    # ========================================================
    # CONSTRUCTOR
    # ========================================================

    def __init__(
        self,
        chunking_method=
            "Recursive Character Text Splitting",

        chunk_size=500,

        chunk_overlap=100,

        token_chunk_size=300,

        token_overlap=50,

        sentences_per_chunk=5,

        paragraphs_per_chunk=3,

        child_chunk_size=400
    ):

        # ----------------------------------------------------
        # STORE CONFIGURATION
        # ----------------------------------------------------

        self.chunking_method = (
            chunking_method
        )

        self.chunk_size = int(
            chunk_size
        )

        self.chunk_overlap = int(
            chunk_overlap
        )

        self.token_chunk_size = int(
            token_chunk_size
        )

        self.token_overlap = int(
            token_overlap
        )

        self.sentences_per_chunk = int(
            sentences_per_chunk
        )

        self.paragraphs_per_chunk = int(
            paragraphs_per_chunk
        )

        self.child_chunk_size = int(
            child_chunk_size
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        self.validate_configuration()

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_configuration(self):

        if (
            self.chunking_method
            not in self.CHUNKING_METHODS
        ):

            raise ValueError(

                "Invalid chunking method: "
                +
                str(
                    self.chunking_method
                )
                +
                "\n\nSupported methods:\n"
                +
                "\n".join(
                    [
                        f"- {method}"
                        for method
                        in self.CHUNKING_METHODS
                    ]
                )
            )

        if self.chunk_size <= 0:

            raise ValueError(
                "chunk_size must be greater than 0."
            )

        if self.chunk_overlap < 0:

            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if (
            self.chunk_overlap
            >=
            self.chunk_size
        ):

            raise ValueError(

                "chunk_overlap must be "
                "smaller than chunk_size."
            )

        if self.token_chunk_size <= 0:

            raise ValueError(
                "token_chunk_size must be greater than 0."
            )

        if self.token_overlap < 0:

            raise ValueError(
                "token_overlap cannot be negative."
            )

        if (
            self.token_overlap
            >=
            self.token_chunk_size
        ):

            raise ValueError(

                "token_overlap must be "
                "smaller than token_chunk_size."
            )

    # ========================================================
    # MAIN CHUNK FUNCTION
    # ========================================================

    def chunk_text(
        self,
        text
    ):

        if not text:

            return []

        text = self.clean_text(
            text
        )

        if not text:

            return []

        # ====================================================
        # SELECTED METHOD
        # ====================================================

        if (
            self.chunking_method
            ==
            "Recursive Character Text Splitting"
        ):

            return (
                self.recursive_character_chunking(
                    text
                )
            )

        elif (
            self.chunking_method
            ==
            "Fixed-Size Chunking"
        ):

            return (
                self.fixed_size_chunking(
                    text
                )
            )

        elif (
            self.chunking_method
            ==
            "Token-Based Chunking"
        ):

            return (
                self.token_based_chunking(
                    text
                )
            )

        elif (
            self.chunking_method
            ==
            "Semantic Chunking"
        ):

            return (
                self.semantic_chunking(
                    text
                )
            )

        elif (
            self.chunking_method
            ==
            "Structure-Based Chunking"
        ):

            return (
                self.structure_based_chunking(
                    text
                )
            )

        elif (
            self.chunking_method
            ==
            "Markdown-Based Chunking"
        ):

            return (
                self.markdown_based_chunking(
                    text
                )
            )

        elif (
            self.chunking_method
            ==
            "HTML-Based Chunking"
        ):

            return (
                self.html_based_chunking(
                    text
                )
            )

        elif (
            self.chunking_method
            ==
            "Sentence-Based Chunking"
        ):

            return (
                self.sentence_based_chunking(
                    text
                )
            )

        elif (
            self.chunking_method
            ==
            "Paragraph-Based Chunking"
        ):

            return (
                self.paragraph_based_chunking(
                    text
                )
            )

        elif (
            self.chunking_method
            ==
            "Document-Section Chunking"
        ):

            return (
                self.document_section_chunking(
                    text
                )
            )

        elif (
            self.chunking_method
            ==
            "Parent-Child Chunking"
        ):

            return (
                self.parent_child_chunking(
                    text
                )
            )

        elif (
            self.chunking_method
            ==
            "Hierarchical Chunking"
        ):

            return (
                self.hierarchical_chunking(
                    text
                )
            )

        raise ValueError(
            "Unsupported chunking method."
        )

    # ========================================================
    # 01. RECURSIVE CHARACTER CHUNKING
    # ========================================================

    def recursive_character_chunking(
        self,
        text
    ):

        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter
        )

        splitter = (
            RecursiveCharacterTextSplitter(

                chunk_size=
                    self.chunk_size,

                chunk_overlap=
                    self.chunk_overlap,

                separators=[

                    "\n\n",

                    "\n",

                    ". ",

                    " ",

                    ""
                ],

                length_function=len,

                is_separator_regex=False
            )
        )

        return (
            splitter.split_text(
                text
            )
        )

    # ========================================================
    # 02. FIXED-SIZE CHUNKING
    # ========================================================

    def fixed_size_chunking(
        self,
        text
    ):

        chunks = []

        start = 0

        step = (
            self.chunk_size
            -
            self.chunk_overlap
        )

        while start < len(text):

            end = (
                start
                +
                self.chunk_size
            )

            chunk = text[
                start:end
            ].strip()

            if chunk:

                chunks.append(
                    chunk
                )

            start += step

        return chunks

    # ========================================================
    # 03. TOKEN-BASED CHUNKING
    # ========================================================

    def token_based_chunking(
        self,
        text
    ):

        try:

            import tiktoken

            encoding = (
                tiktoken.get_encoding(
                    "cl100k_base"
                )
            )

            tokens = (
                encoding.encode(
                    text
                )
            )

            chunks = []

            step = (
                self.token_chunk_size
                -
                self.token_overlap
            )

            start = 0

            while start < len(tokens):

                end = (
                    start
                    +
                    self.token_chunk_size
                )

                token_chunk = tokens[
                    start:end
                ]

                chunk = (
                    encoding.decode(
                        token_chunk
                    )
                ).strip()

                if chunk:

                    chunks.append(
                        chunk
                    )

                start += step

            return chunks

        except ImportError:

            # Safe fallback
            return (
                self.fixed_size_chunking(
                    text
                )
            )

    # ========================================================
    # 04. SEMANTIC CHUNKING
    # ========================================================

    def semantic_chunking(
        self,
        text
    ):

        try:

            from langchain_experimental.text_splitter import (
                SemanticChunker
            )

            from langchain_openai import (
                OpenAIEmbeddings
            )

            import os

            api_key = os.getenv(
                "OPENAI_API_KEY"
            )

            if not api_key:

                # Fallback instead of crashing
                return (
                    self.sentence_based_chunking(
                        text
                    )
                )

            embeddings = (
                OpenAIEmbeddings(

                    model=
                        "text-embedding-3-small",

                    api_key=
                        api_key
                )
            )

            splitter = (
                SemanticChunker(
                    embeddings
                )
            )

            return (
                splitter.split_text(
                    text
                )
            )

        except Exception:

            # Safe fallback
            return (
                self.sentence_based_chunking(
                    text
                )
            )

    # ========================================================
    # 05. STRUCTURE-BASED CHUNKING
    # ========================================================

    def structure_based_chunking(
        self,
        text
    ):

        section_names = [

            "summary",

            "professional summary",

            "profile",

            "objective",

            "skills",

            "technical skills",

            "technical skill",

            "experience",

            "work experience",

            "professional experience",

            "employment",

            "projects",

            "personal projects",

            "academic projects",

            "education",

            "certifications",

            "achievements",

            "responsibilities",

            "languages",

            "interests",

            "references"
        ]

        sections = []

        current_title = (
            "General"
        )

        current_content = []

        for line in text.splitlines():

            clean_line = line.strip()

            if not clean_line:

                continue

            normalized = (
                clean_line
                .lower()
                .strip(":")
                .strip()
            )

            if (
                normalized
                in section_names
            ):

                if current_content:

                    sections.append({

                        "title":
                            current_title,

                        "content":
                            "\n".join(
                                current_content
                            )
                    })

                current_title = (
                    clean_line
                )

                current_content = []

            else:

                current_content.append(
                    clean_line
                )

        if current_content:

            sections.append({

                "title":
                    current_title,

                "content":
                    "\n".join(
                        current_content
                    )
            })

        chunks = []

        for section in sections:

            section_text = (

                section["title"]

                +
                "\n"

                +
                section["content"]
            )

            if (
                len(section_text)
                <=
                self.chunk_size
            ):

                chunks.append(
                    section_text
                )

            else:

                chunks.extend(
                    self.fixed_size_chunking(
                        section_text
                    )
                )

        return chunks

    # ========================================================
    # 06. MARKDOWN-BASED CHUNKING
    # ========================================================

    def markdown_based_chunking(
        self,
        text
    ):

        sections = re.split(

            r"(?m)(?=^#{1,6}\s+)",

            text
        )

        sections = [

            section.strip()

            for section
            in sections

            if section.strip()
        ]

        final_chunks = []

        for section in sections:

            if (
                len(section)
                <=
                self.chunk_size
            ):

                final_chunks.append(
                    section
                )

            else:

                final_chunks.extend(
                    self.fixed_size_chunking(
                        section
                    )
                )

        return final_chunks

    # ========================================================
    # 07. HTML-BASED CHUNKING
    # ========================================================

    def html_based_chunking(
        self,
        text
    ):

        try:

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(
                text,
                "html.parser"
            )

            elements = (
                soup.find_all(
                    [
                        "h1",
                        "h2",
                        "h3",
                        "h4",
                        "p",
                        "li",
                        "section"
                    ]
                )
            )

            chunks = []

            for element in elements:

                value = (
                    element.get_text(
                        " ",
                        strip=True
                    )
                )

                if value:

                    chunks.append(
                        value
                    )

            if chunks:

                return (
                    self.combine_small_chunks(
                        chunks
                    )
                )

        except Exception:

            pass

        return (
            self.paragraph_based_chunking(
                text
            )
        )

    # ========================================================
    # 08. SENTENCE-BASED CHUNKING
    # ========================================================

    def sentence_based_chunking(
        self,
        text
    ):

        sentences = re.split(

            r"(?<=[.!?])\s+",

            text
        )

        sentences = [

            sentence.strip()

            for sentence
            in sentences

            if sentence.strip()
        ]

        chunks = []

        current_sentences = []

        for sentence in sentences:

            current_sentences.append(
                sentence
            )

            if (
                len(current_sentences)
                >=
                self.sentences_per_chunk
            ):

                chunks.append(

                    " ".join(
                        current_sentences
                    )
                )

                current_sentences = []

        if current_sentences:

            chunks.append(

                " ".join(
                    current_sentences
                )
            )

        return chunks

    # ========================================================
    # 09. PARAGRAPH-BASED CHUNKING
    # ========================================================

    def paragraph_based_chunking(
        self,
        text
    ):

        paragraphs = re.split(

            r"\n\s*\n",

            text
        )

        paragraphs = [

            paragraph.strip()

            for paragraph
            in paragraphs

            if paragraph.strip()
        ]

        chunks = []

        current_paragraphs = []

        for paragraph in paragraphs:

            current_paragraphs.append(
                paragraph
            )

            if (
                len(current_paragraphs)
                >=
                self.paragraphs_per_chunk
            ):

                chunks.append(

                    "\n\n".join(
                        current_paragraphs
                    )
                )

                current_paragraphs = []

        if current_paragraphs:

            chunks.append(

                "\n\n".join(
                    current_paragraphs
                )
            )

        return chunks

    # ========================================================
    # 10. DOCUMENT-SECTION CHUNKING
    # ========================================================

    def document_section_chunking(
        self,
        text
    ):

        pattern = (

            r"(?im)"

            r"(?=^"

            r"(?:"

            r"summary|"

            r"professional summary|"

            r"profile|"

            r"objective|"

            r"skills|"

            r"technical skills|"

            r"experience|"

            r"work experience|"

            r"professional experience|"

            r"projects|"

            r"education|"

            r"certifications|"

            r"achievements|"

            r"responsibilities|"

            r"languages|"

            r"references"

            r")"

            r"\s*:?\s*$)"
        )

        sections = re.split(

            pattern,

            text
        )

        sections = [

            section.strip()

            for section
            in sections

            if section.strip()
        ]

        final_chunks = []

        for section in sections:

            if (
                len(section)
                <=
                self.chunk_size
            ):

                final_chunks.append(
                    section
                )

            else:

                final_chunks.extend(
                    self.fixed_size_chunking(
                        section
                    )
                )

        return final_chunks

    # ========================================================
    # 11. PARENT-CHILD CHUNKING
    # ========================================================

    def parent_child_chunking(
        self,
        text
    ):

        # ----------------------------------------------------
        # Parent chunks
        # ----------------------------------------------------

        parent_chunks = (
            self.fixed_size_chunking(
                text
            )
        )

        results = []

        for parent_id, parent_text in enumerate(
            parent_chunks,
            start=1
        ):

            # ------------------------------------------------
            # Child chunks
            # ------------------------------------------------

            child_chunks = (
                self.fixed_size_chunking_custom(
                    parent_text,

                    self.child_chunk_size,

                    min(
                        self.chunk_overlap,

                        max(
                            0,
                            self.child_chunk_size - 1
                        )
                    )
                )
            )

            for child_id, child_text in enumerate(
                child_chunks,
                start=1
            ):

                results.append({

                    "text":
                        child_text,

                    "metadata": {

                        "chunk_type":
                            "child",

                        "parent_id":
                            parent_id,

                        "child_id":
                            child_id
                    }
                })

        return results

    # ========================================================
    # 12. HIERARCHICAL CHUNKING
    # ========================================================

    def hierarchical_chunking(
        self,
        text
    ):

        # ----------------------------------------------------
        # LEVEL 1
        # Document Sections
        # ----------------------------------------------------

        sections = (
            self.document_section_chunking(
                text
            )
        )

        results = []

        for section_id, section_text in enumerate(
            sections,
            start=1
        ):

            # ------------------------------------------------
            # LEVEL 2
            # Paragraphs
            # ------------------------------------------------

            paragraphs = re.split(

                r"\n\s*\n",

                section_text
            )

            for paragraph_id, paragraph in enumerate(
                paragraphs,
                start=1
            ):

                paragraph = (
                    paragraph.strip()
                )

                if not paragraph:

                    continue

                # --------------------------------------------
                # LEVEL 3
                # Child chunks
                # --------------------------------------------

                small_chunks = (
                    self.fixed_size_chunking_custom(

                        paragraph,

                        self.child_chunk_size,

                        min(
                            self.chunk_overlap,

                            max(
                                0,
                                self.child_chunk_size - 1
                            )
                        )
                    )
                )

                for chunk_id, chunk_text in enumerate(
                    small_chunks,
                    start=1
                ):

                    results.append({

                        "text":
                            chunk_text,

                        "metadata": {

                            "chunk_type":
                                "hierarchical",

                            "level":
                                3,

                            "section_id":
                                section_id,

                            "paragraph_id":
                                paragraph_id,

                            "chunk_id":
                                chunk_id
                        }
                    })

        return results

    # ========================================================
    # CUSTOM FIXED CHUNK HELPER
    # ========================================================

    def fixed_size_chunking_custom(
        self,
        text,
        chunk_size,
        chunk_overlap
    ):

        if not text:

            return []

        if chunk_size <= 0:

            raise ValueError(
                "chunk_size must be greater than 0."
            )

        if (
            chunk_overlap
            >=
            chunk_size
        ):

            chunk_overlap = (
                chunk_size - 1
            )

        step = (
            chunk_size
            -
            chunk_overlap
        )

        chunks = []

        start = 0

        while start < len(text):

            end = (
                start
                +
                chunk_size
            )

            chunk = text[
                start:end
            ].strip()

            if chunk:

                chunks.append(
                    chunk
                )

            start += max(
                1,
                step
            )

        return chunks

    # ========================================================
    # COMBINE SMALL CHUNKS
    # ========================================================

    def combine_small_chunks(
        self,
        chunks
    ):

        final_chunks = []

        current = ""

        for chunk in chunks:

            chunk = chunk.strip()

            if not chunk:

                continue

            if not current:

                current = chunk

                continue

            combined = (

                current
                +
                "\n"
                +
                chunk
            )

            if (
                len(combined)
                <=
                self.chunk_size
            ):

                current = combined

            else:

                final_chunks.append(
                    current
                )

                current = chunk

        if current:

            final_chunks.append(
                current
            )

        return final_chunks

    # ========================================================
    # CLEAN TEXT
    # ========================================================

    def clean_text(
        self,
        text
    ):

        if not text:

            return ""

        # ----------------------------------------------------
        # WINDOWS LINE BREAKS
        # ----------------------------------------------------

        text = text.replace(
            "\r\n",
            "\n"
        )

        text = text.replace(
            "\r",
            "\n"
        )

        # ----------------------------------------------------
        # NORMALIZE TABS / SPACES
        # ----------------------------------------------------

        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        # ----------------------------------------------------
        # NORMALIZE NEWLINES
        # ----------------------------------------------------

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()

    # ========================================================
    # TEXT → TOKENS
    # ========================================================

    def text_to_tokens(
        self,
        text
    ):

        if not text:

            return []

        tokens = re.findall(

            r"\w+|[^\w\s]",

            text
        )

        return tokens

    # ========================================================
    # CHUNK WITH METADATA
    # ========================================================

    def chunk_with_metadata(
        self,
        text,
        file_name=None
    ):

        chunks = self.chunk_text(
            text
        )

        results = []

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            # ------------------------------------------------
            # Handle normal string chunks
            # ------------------------------------------------

            if isinstance(
                chunk,
                str
            ):

                chunk_text = (
                    chunk
                )

                metadata = {}

            # ------------------------------------------------
            # Handle Parent-Child /
            # Hierarchical chunks
            # ------------------------------------------------

            elif isinstance(
                chunk,
                dict
            ):

                chunk_text = (
                    chunk.get(
                        "text",
                        ""
                    )
                )

                metadata = (
                    chunk.get(
                        "metadata",
                        {}
                    )
                )

            else:

                continue

            chunk_text = (
                chunk_text.strip()
            )

            if not chunk_text:

                continue

            results.append({

                "chunk_id":
                    index,

                "file_name":
                    file_name,

                "text":
                    chunk_text,

                "token_count":
                    len(
                        self.text_to_tokens(
                            chunk_text
                        )
                    ),

                "character_count":
                    len(
                        chunk_text
                    ),

                "chunking_method":
                    self.chunking_method,

                "metadata":
                    metadata
            })

        return results