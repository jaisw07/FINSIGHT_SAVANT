import streamlit as st
import requests
import yfinance as yf
import plotly.graph_objs as go
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

# --- Functions ---

def get_stock_chart(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="7d")  # Last 7 days
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Date', yaxis_title='Price')
        return fig
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# def get_stock_news(query):  
def get_stock_news():  
    try:
        # Replace 'YOUR_NEWS_API_KEY' with your actual NewsAPI key
        api_key = '0edddf919256451199f040709a0d5611'
    #     url = f'https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}'
    #     response = requests.get(url)
    #     data = response.json()
    #     return data.get('articles', [])
    # except Exception as e:
    #     st.error(f"Error fetching news: {e}")
    #     return []
        query = "stock market"  # <-- Fixed query to always fetch stock market news
        url = f'https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}'
        response = requests.get(url)
        data = response.json()
        return data.get('articles', [])
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

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

# --- App Setup ---

st.set_page_config(page_title="FINSIGHT SAVANT", page_icon="🤖", layout="wide")

# Sidebar Navigation
with st.sidebar:
    st.title("FINSIGHT SAVANT")
    page = st.radio("Go to:", ["Chatbot", "Stock Trends", "Stock News"])

# --- Page Routing ---

if page == "Chatbot":
    with st.sidebar:
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

    main_container = st.container()

    with main_container:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [AIMessage(content="Hi, I’m a FINSIGHT SAVANT. How can I help you?")]

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
                st.markdown(message.content)

        # User input area
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
                    os.remove(audio_file)

                st.session_state.chat_history.append(AIMessage(content=response))

elif page == "Stock Trends":
    st.title("📈 Stock Trends")
    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, MSFT)")
    if st.button("Get Stock Chart"):
        if ticker:
            fig = get_stock_chart(ticker)
            if fig:
                st.plotly_chart(fig)
        else:
            st.warning("Please enter a stock ticker symbol.")

elif page == "Stock News":
    # st.title("📰 Stock News")
    # stock_query = st.text_input("Enter Company Name or Keyword for News (e.g., Tesla, Amazon)")
    # if st.button("Get News"):
    #     if stock_query:
    #         articles = get_stock_news(stock_query)
    #         if articles:
    #             for article in articles[:5]:  # Show top 5 articles
    #                 st.subheader(article['title'])
    #                 st.write(article['description'])
    #                 st.markdown(f"[Read more]({article['url']})")
    #                 st.markdown("---")
    #         else:
    #             st.info("No news articles found.")
    #     else:
    #         st.warning("Please enter a query for news.")
    st.title("Latest Stock Market News")

    news_articles = get_stock_news()

    if news_articles:
        for article in news_articles[:10]:  # Limit to top 10 articles
            st.subheader(article.get('title'))
            st.write(article.get('description'))
            st.write(f"[Read more]({article.get('url')})")
            st.markdown("---")
    else:
        st.write("No news articles found.")


# --- CSS ---
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
    background-color: #000000;
}
div.row-widget.stButton {
    margin-top: 1px;
}
</style>
""", unsafe_allow_html=True)