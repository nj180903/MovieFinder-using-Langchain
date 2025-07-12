from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()


# ✅ Read from Streamlit secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

LOG_LEVEL = st.secrets.get("LOG_LEVEL", "INFO")
LOG_TO_FILE = st.secrets.get("LOG_TO_FILE", False)

MAX_RESULTS = int(st.secrets.get("MAX_RESULTS", 10))
DEFAULT_RANKING_FIELD = st.secrets.get("DEFAULT_RANKING_FIELD", "IMDB_Rating")
ENABLE_VECTOR_SEARCH = st.secrets.get("ENABLE_VECTOR_SEARCH", True)

VECTOR_STORE_PERSIST_DIRECTORY = st.secrets.get("VECTOR_STORE_PERSIST_DIRECTORY", "./vector_store")
EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

GRADIO_SERVER_NAME = st.secrets.get("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(st.secrets.get("GRADIO_SERVER_PORT", 7860))
GRADIO_SHARE = st.secrets.get("GRADIO_SHARE", False)

def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key= GOOGLE_API_KEY)