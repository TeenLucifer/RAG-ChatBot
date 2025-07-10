import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.llms.dashscope import DashScope
from utils.dashscope_embedding import DashScopeEmbedding
from utils.doc_handler import RagModal, CorpusManagement, process_uploaded_files, build_text_modal_corpus, load_text_modal_corpus, load_multi_modal_corpus
from utils.retrieve_pipline import expand_query
from utils.rag_config import RagConfig
from pymilvus import connections, utility
from llama_index.vector_stores.milvus import MilvusVectorStore
import nest_asyncio
import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]# 防止torch报错

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


                                                                                    # Manage Session state
if "rag_config" not in st.session_state:
    st.session_state.rag_config = RagConfig()
if "rag_modal" not in st.session_state:
    st.session_state.rag_modal = RagModal.TEXT
if "corpus_management" not in st.session_state:
    st.session_state.corpus_management = CorpusManagement()
if "milvus_connected" not in st.session_state:
    st.session_state.milvus_connected = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_corpus" not in st.session_state:
    st.session_state.loaded_corpus = None
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False

st.title("🤖 RAG-ChatBot")
st.caption("Advanced RAG System with GraphRAG, Hybrid Retrieval, Neural Reranking and Chat History")