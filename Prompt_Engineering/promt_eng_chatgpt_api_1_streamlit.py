import os
import uuid
import requests
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI
from streamlit_lottie import st_lottie


# ============================================================
# Step 1 : Load Environment
# ============================================================

load_dotenv()

default_api_key = os.getenv("OPENAI_API_KEY")


# ============================================================
# Step 2 : Streamlit Config
# ============================================================

st.set_page_config(
    page_title="Suresh Bangaru AI Voice Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# Step 3 : UI CSS
# ============================================================

st.markdown("""
<style>

#MainMenu {display:none;}
footer {display:none;}
header {display:none;}


.stApp{

background:
linear-gradient(
-45deg,
#0f172a,
#1e3a8a,
#4338ca,
#0f766e
);

background-size:400% 400%;

animation:bg 15s ease infinite;

}


@keyframes bg{

0%{
background-position:0% 50%;
}

50%{
background-position:100% 50%;
}

100%{
background-position:0% 50%;
}

}



*{

color:white !important;

}



.glass{

background:
rgba(255,255,255,0.08);

backdrop-filter:
blur(20px);

border-radius:20px;

padding:20px;

border:
1px solid rgba(255,255,255,.2);

}



[data-testid="stChatMessage"]{

background:
rgba(255,255,255,.08);

border-radius:15px;

padding:15px;

margin-bottom:10px;

}



input,textarea{

background:#1f2937 !important;

color:white !important;

}



section[data-testid="stSidebar"]{

background:#111827;

}



.stButton button{

background:#2563eb;

border-radius:10px;

color:white !important;

}


</style>
""", unsafe_allow_html=True)



# ============================================================
# Step 4 : Load Animation
# ============================================================

def load_lottie(url):

    try:

        r = requests.get(url)

        if r.status_code == 200:

            return r.json()

    except:

        return None



animation = load_lottie(
"https://assets10.lottiefiles.com/packages/lf20_x62chJ.json"
)



# ============================================================
# Step 5 : Sidebar
# ============================================================

with st.sidebar:


    st.title("🤖 Voice AI Assistant")


    st.success(
        "GPT-4.1 Mini + Voice"
    )


    st.info(
        "Enter your API key or use .env"
    )


    if st.button(
        "🗑️ Clear Chat"
    ):

        st.session_state.messages=[]

        st.rerun()



# ============================================================
# Step 6 : Header
# ============================================================


c1,c2 = st.columns([3,1])


with c1:

    st.markdown("""
    <div class="glass">

    <h1>
    🤖 AI Voice Assistant
    </h1>

    <p>
    Ask questions and get spoken answers.
    </p>

    </div>
    """,
    unsafe_allow_html=True)



with c2:

    if animation:

        st_lottie(
            animation,
            height=180
        )



# ============================================================
# Step 7 : API Key
# ============================================================


with st.expander(
"🔑 OpenAI API Key (Optional)"
):

    user_key = st.text_input(
        "API Key",
        type="password"
    )



api_key = user_key if user_key else default_api_key



if not api_key:

    st.warning(
    "Add API key"
    )

    st.stop()



# ============================================================
# Step 8 : OpenAI Client
# ============================================================

client = OpenAI(
    api_key=api_key
)



# ============================================================
# Step 9 : Text To Speech
# ============================================================

def generate_voice(text):

    filename = f"voice_{uuid.uuid4()}.mp3"


    speech = client.audio.speech.create(

        model="gpt-4o-mini-tts",

        voice="nova",

        input=text

    )


    speech.stream_to_file(filename)


    return filename



# ============================================================
# Step 10 : Chat Memory
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages=[]



# ============================================================
# Step 11 : Display Chat
# ============================================================

for msg in st.session_state.messages:


    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )



# ============================================================
# Step 12 : User Input
# ============================================================

prompt = st.chat_input(
"Ask something..."
)



# ============================================================
# Step 13 : Generate Answer
# ============================================================


if prompt:


    st.session_state.messages.append(

        {
        "role":"user",
        "content":prompt
        }

    )


    with st.chat_message("user"):

        st.markdown(prompt)



    with st.chat_message("assistant"):


        stream = client.chat.completions.create(

            model="gpt-4.1-mini",

            messages=[

            {
            "role":"system",
            "content":
            "You are a helpful voice assistant."
            }

            ]
            +
            st.session_state.messages,


            stream=True

        )



        answer = st.write_stream(

            chunk.choices[0].delta.content or ""

            for chunk in stream

        )



        # -------------------------
        # Voice Output
        # -------------------------

        try:

            audio = generate_voice(
                answer
            )


            st.audio(
                audio,
                format="audio/mp3"
            )


        except Exception as e:

            st.warning(
            f"Voice error: {e}"
            )



    st.session_state.messages.append(

        {
        "role":"assistant",
        "content":answer
        }

    )