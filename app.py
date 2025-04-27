import streamlit as st
from langchain_community.chat_models import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
import base64

def get_response(user_query, chat_history, vector_store):
    llm = ChatOllama(model="llama3")
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    
    result = chain({"question": user_query, "chat_history": chat_history})
    return result["answer"]

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak now...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.write("Sorry, I couldn't understand that.")
        return None
    except sr.RequestError:
        st.write("Sorry, there was an error with the speech recognition service.")
        return None

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
    return fp.name

def process_documents(uploaded_files):
    text = []
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".docx" or file_extension == ".doc":
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)

        if loader:
            text.extend(loader.load())
            os.remove(temp_file_path)

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    text_chunks = text_splitter.split_documents(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    return vector_store

st.set_page_config(page_title="FINSIGHT SAVANT", page_icon="🤖", layout="wide")

# Sidebar
with st.sidebar:
    st.title("FINSIGHT SAVANT")
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)
    if st.button("Process Documents"):
        if uploaded_files:
            st.session_state.vector_store = process_documents(uploaded_files)
            st.success("Documents processed successfully!")
        else:
            st.warning("Please upload documents first.")
    
    if st.button("New Chat"):
        st.session_state.chat_history = [AIMessage(content="Hi, I’m a FINSIGHT SAVANT. How can I help you?")]
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("")

# Main chat area
main_container = st.container()

with main_container:
    # Initialize chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hi, I’m a FINSIGHT SAVANT. How can I help you?")]

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
            st.markdown(message.content)

    # User input area
    with st.container():
        col1, col2, col3 = st.columns([0.88, 0.04, 0.04])
        with col1:
            user_query = st.text_input("Type your message here...", key="user_input", label_visibility="collapsed")
        with col2:
            speak_button = st.button("🎤")
        with col3:
            send_button = st.button("➤")

    if speak_button:
        st.write("Listening...")
        user_query = speech_to_text()
        if user_query:
            st.write(f"You said: {user_query}")
            st.session_state.speech_input = user_query
            st.experimental_rerun()

    # Check for speech input
    if 'speech_input' in st.session_state:
        user_query = st.session_state.speech_input
        del st.session_state.speech_input

    if send_button or user_query:
        if user_query:
            st.session_state.chat_history.append(HumanMessage(content=user_query))

            with st.chat_message("Human"):
                st.markdown(user_query)

            with st.chat_message("AI"):
                response_container = st.empty()
                if 'vector_store' in st.session_state:
                    response = get_response(user_query, st.session_state.chat_history, st.session_state.vector_store)
                else:
                    response = "Please upload and process documents first."
                response_container.markdown(response)

                # Convert response to speech
                audio_file = text_to_speech(response)
                st.audio(audio_file, format='audio/mp3')
                os.remove(audio_file)  # Delete the temporary audio file

            st.session_state.chat_history.append(AIMessage(content=response))

# CSS to improve the UI
st.markdown("""
<style>
.stTextInput > div > div > input {
    border-radius: 20px;
}
.stButton > button {
    border-radius: 20px;
    height: 2.4em;
    line-height: 1;
    padding: 0.3em -11px;
    margin-top: 1px;
}
.stSidebar {
    background-color: #f0f2f6;
}
div.row-widget.stButton {
    margin-top: 1px;
}
</style>
""", unsafe_allow_html=True) 

# streamlit run app.py

