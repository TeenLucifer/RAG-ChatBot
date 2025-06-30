import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.llms.dashscope import DashScope
from utils.dashscope_embedding import DashScopeEmbedding
from utils.doc_handler import process_uploaded_files, build_text_corpus, load_text_corpus, load_multi_modal_corpus
from utils.retrieve_pipline import expand_query, retrieve_documents
from utils.rag_config import RagConfig
from pymilvus import connections, utility
#from st_pages import show_pages_from_config
from st_pages import Page
from llama_index.vector_stores.milvus import MilvusVectorStore
import nest_asyncio

nest_asyncio.apply()

# Custom CSS
st.markdown("""
    <style>
        .stApp { background-color: #f4f4f9; }
        h1 { color: #00FF99; text-align: center; }
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
        .stChatMessage.user { background-color: #e8f0fe; }
        .stChatMessage.assistant { background-color: #d1e7dd; }
        .stButton>button { background-color: #00AAFF; color: white; }
    </style>
""", unsafe_allow_html=True)

if "rag_config" not in st.session_state:
    st.session_state.rag_config = RagConfig()

st.title("🤖 RAG-ChatBot")
st.caption("Advanced RAG System with GraphRAG, Hybrid Retrieval, Neural Reranking and Chat History")