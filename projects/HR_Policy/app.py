import streamlit as st

from streamlit_mic_recorder import speech_to_text

from document_process import DocumentProcess


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR Policies Q&A Search",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# MAIN PAGE HEADER
# ============================================================

st.title(
    "HR Policies Q&A Search and Similarity Engine"
)

st.write(
    "Compare CountVectorizer, TF-IDF, OpenAI embeddings, "
    "Google embeddings, and sentence transformers."
)


# ============================================================
# CREATE LEFT AND RIGHT SECTIONS
# ============================================================

left_col, right_col = st.columns([1, 3])


# ============================================================
# LEFT SIDE - SETTINGS
# ============================================================

with left_col:

    st.header("Settings")


    # ========================================================
    # EMBEDDING METHOD
    # ========================================================

    st.subheader("Embedding method")

    embedding_method = st.selectbox(
        "Embedding method",
        [
            "CountVectorizer",
            "TF-IDF",
            "Word2Vec",
            "SentenceTransformer",
            "OpenAI"
        ],
        label_visibility="collapsed"
    )


    # ========================================================
    # OPENAI API KEY
    # ========================================================

    openai_api_key = None

    if embedding_method == "OpenAI":

        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-..."
        )


    # ========================================================
    # SIMILARITY METRIC
    # ========================================================

    st.subheader("Similarity metric")

    similarity_metric = st.selectbox(
        "Similarity metric",
        [
            "cosine_similarity",
            "euclidean_distance"
        ]
    )


    # ========================================================
    # TOP MATCHING RESULTS
    # ========================================================

    st.subheader("Top matching results")

    top_k = st.slider(
        "Top matching results",
        min_value=1,
        max_value=10,
        value=1
    )


    # ========================================================
    # USE AI ANSWER GENERATION
    # ========================================================

    use_ai = st.checkbox(
        "Use AI answer generation"
    )


# ============================================================
# RIGHT SIDE - MAIN CONTENT
# ============================================================

with right_col:

    st.header("Ask a question")


    # ========================================================
    # SPEECH → TEXT
    # ========================================================

    spoken_text = speech_to_text(
        language="en"
    )


    # ========================================================
    # STORE SPOKEN QUESTION
    # ========================================================

    if spoken_text:

        st.session_state[
            "spoken_question"
        ] = spoken_text


    # ========================================================
    # TYPE QUESTION
    # ========================================================

    typed_question = st.text_input(
        "Question",
        placeholder="Type your HR policy question...",
        key="typed_question"
    )


    # ========================================================
    # SEARCH BUTTON
    # ========================================================

    search_button = st.button(
        "🔍 Search Engine",
        type="primary"
    )


    # ========================================================
    # SEARCH
    # ========================================================

    if search_button:

        # ====================================================
        # TYPED QUESTION GETS PRIORITY
        # ====================================================

        if typed_question.strip():

            question = typed_question.strip()


        # ====================================================
        # OTHERWISE USE SPOKEN QUESTION
        # ====================================================

        elif st.session_state.get(
            "spoken_question"
        ):

            question = (
                st.session_state[
                    "spoken_question"
                ].strip()
            )


        # ====================================================
        # NO QUESTION
        # ====================================================

        else:

            question = ""


        # ====================================================
        # QUESTION FOUND
        # ====================================================

        if question:

            print(
                "Question:",
                question
            )


            # =================================================
            # DISPLAY QUESTION
            # =================================================

            st.subheader("Question")

            st.write(question)


            # =================================================
            # OPENAI EMBEDDING
            # =================================================

            if embedding_method == "OpenAI":

                # ---------------------------------------------
                # CHECK API KEY
                # ---------------------------------------------

                if not openai_api_key:

                    st.error(
                        "Please enter your OpenAI API key."
                    )

                    st.stop()


                # ---------------------------------------------
                # DOCUMENT PROCESS
                # ---------------------------------------------

                similarity_Score, model_generated_ans = (DocumentProcess(similarity_metric=similarity_metric, embedding_method=embedding_method, top_k=top_k, question=question, openai_api_key=openai_api_key).process())


            # =================================================
            # OTHER EMBEDDING METHODS
            # =================================================

            else:

                similarity_Score, model_generated_ans = (
                    DocumentProcess(

                        similarity_metric=similarity_metric,

                        embedding_method=embedding_method,

                        top_k=top_k,

                        question=question, openai_api_key =None

                    ).process()
                )


            # =================================================
            # ANSWER
            # =================================================

            st.subheader("Answer")


            # =================================================
            # SIMILARITY SCORE
            # =================================================

            st.write(
                "Similarity / Distance Score:"
            )

            st.write(
                similarity_Score
            )


            # =================================================
            # MODEL ANSWER
            # =================================================

            st.write(
                model_generated_ans
            )


            # =================================================
            # SPEAK SCORE
            # =================================================

            st.components.v1.html(

                f"""
                <script>

                    const answer =
                        {str(similarity_Score)!r};

                    const speech =
                        new SpeechSynthesisUtterance(
                            answer
                        );

                    speech.lang = "en-US";

                    speech.rate = 1.0;

                    speech.pitch = 1.0;

                    speech.volume = 1.0;

                    window.speechSynthesis.cancel();

                    window.speechSynthesis.speak(
                        speech
                    );

                </script>
                """,

                height=50
            )


            # =================================================
            # SPEAK ANSWER
            # =================================================

            st.components.v1.html(

                f"""
                <script>

                    const answer =
                        {str(model_generated_ans)!r};

                    const speech =
                        new SpeechSynthesisUtterance(
                            answer
                        );

                    speech.lang = "en-US";

                    speech.rate = 1.0;

                    speech.pitch = 1.0;

                    speech.volume = 1.0;

                    window.speechSynthesis.cancel();

                    window.speechSynthesis.speak(
                        speech
                    );

                </script>
                """,

                height=50
            )


        # ====================================================
        # NO QUESTION
        # ====================================================

        else:

            st.warning(
                "Please type or speak a question."
            )