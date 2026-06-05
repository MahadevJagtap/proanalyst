"""Smoke test — verify all imports resolve correctly."""
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit
import chromadb
print("ALL IMPORTS OK")
print(f"  streamlit     : {streamlit.__version__}")
print(f"  chromadb      : {chromadb.__version__}")
