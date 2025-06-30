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

                                                                                    # Manage Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

with st.sidebar:                                                                        # 📁 Sidebar
    st.header("📁 已加载知识库")
    # TODO(wangjintao): 显示已加载知识库名
    st.markdown("---")
    st.header("⚙️ RAG 参数设置")

    # 交互控件
    #if st.button("Load Corpus"):
    #    st.session_state.semantic_retriever, st.session_state.keywords_retriever = load_text_corpus(
    #        embed_model=text_embed_model,
    #        milvus_dense_collection_name=milvus_dense_collection_name,
    #        milvus_sparse_collection_name=milvus_sparse_collection_name,
    #        milvus_uri=MILVUS_URI,
    #        semantic_retriever_top_k=5,
    #        keywords_retriever_top_k=5,
    #    )
    #    if st.session_state.semantic_retriever and st.session_state.keywords_retriever:
    #        st.session_state.documents_loaded = True

    st.session_state.rag_enabled = st.checkbox("Enable RAG", value=True)
    st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=True)
    st.session_state.enable_reranking = st.checkbox("Enable Neural Reranking", value=True)
    st.session_state.enable_graph_rag = st.checkbox("Enable GraphRAG", value=True)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    st.session_state.max_contexts = st.slider("Max Contexts", 1, 5, 3)

    if st.button("清除历史对话"):
        st.session_state.messages = []
        st.rerun()

    # 🚀 Footer (Bottom Right in Sidebar) For some Credits :)
    st.sidebar.markdown("""
        <div style="position: absolute; top: 20px; right: 10px; font-size: 12px; color: gray;">
            <b>Developed by:</b> Jintao Wang &copy;All Rights Reserved 2025
        </div>
    """, unsafe_allow_html=True)

# 💬 Chat Interface
st.title("🤖 RAG-ChatBot")
st.caption("Advanced RAG System with GraphRAG, Hybrid Retrieval, Neural Reranking and Chat History")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(placeholder="Ask about your documents...", disabled=not st.session_state.documents_loaded):
    chat_history = "\n".join([msg["content"] for msg in st.session_state.messages[-5:]])  # Last 5 messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # TODO(wangjintao): 待实现多轮对话
        # 🚀 Build context
        context = ""
        response = retrieve_documents(
            llm=st.session_state.rag_config.llm,
            text_embed_model=st.session_state.rag_config.text_embed_model,
            rerank_model=st.session_state.rag_config.rerank_model,
            semantic_retriever=st.session_state.semantic_retriever,
            keywords_retriever=st.session_state.keywords_retriever,
            query=prompt,
            dashscope_api_key=st.session_state.rag_config.dashscope_api_key,
            dashscope_llm_model_name=st.session_state.rag_config.dashscope_llm_model_name,
        )
        # TODO(wangjintao): 待实现溯源
        # 响应流式输出
        for chunk in response.response_gen:
            full_response += chunk
            response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})