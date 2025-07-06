import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.llms.dashscope import DashScope
from utils.dashscope_embedding import DashScopeEmbedding
from utils.doc_handler import process_uploaded_files, build_text_modal_corpus, load_text_modal_corpus, load_multi_modal_corpus
from utils.retrieve_pipline import expand_query, retrieve_text_modal, retrieve_multi_modal
from utils.rag_config import RagConfig
from pymilvus import connections, utility
#from st_pages import show_pages_from_config
from st_pages import Page
from llama_index.vector_stores.milvus import MilvusVectorStore
from utils.doc_handler import RagModal, CorpusManagement
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

with st.sidebar:                                                                        # 📁 Sidebar
    st.header("📁 已加载知识库")
    if "loaded_corpus" in st.session_state and st.session_state.loaded_corpus:
        st.markdown(f"**当前知识库:** {st.session_state.loaded_corpus}")
    st.markdown("---")
    st.header("⚙️ RAG 参数设置")

    checkbos_value = st.checkbox("多模态问答", value=False if RagModal.TEXT == st.session_state.rag_modal else True)
    st.session_state.rag_modal = RagModal.MULTI_MODAL if checkbos_value else RagModal.TEXT

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

if prompt := st.chat_input(placeholder="Ask about your documents...", disabled=not st.session_state.loaded_corpus):
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
        if st.session_state.rag_modal == RagModal.TEXT:
            response = retrieve_text_modal(
                llm=st.session_state.rag_config.llm,
                text_embed_model=st.session_state.rag_config.text_embed_model,
                rerank_model=st.session_state.rag_config.rerank_model,
                semantic_retriever=st.session_state.semantic_retriever,
                keywords_retriever=st.session_state.keywords_retriever,
                query=prompt,
                dashscope_api_key=st.session_state.rag_config.dashscope_api_key,
                dashscope_llm_model_name=st.session_state.rag_config.dashscope_llm_model_name,
            )
        elif st.session_state.rag_modal == RagModal.MULTI_MODAL:
            response = retrieve_multi_modal(
                query=prompt,
                mllm=st.session_state.rag_config.mllm,
                mm_embed_model=st.session_state.rag_config.mm_embed_model,
                rerank_model=st.session_state.rag_config.rerank_model,
                semantic_retriever=st.session_state.semantic_retriever,
                keywords_retriever=st.session_state.keywords_retriever,
                dashscope_api_key=st.session_state.rag_config.dashscope_api_key,
                dashscope_mllm_model_name=st.session_state.rag_config.dashscope_mllm_model_name,
            )
        else:
            pass
        # TODO(wangjintao): 待实现溯源
        # 响应流式输出
        for chunk in response.response_gen:
            full_response += chunk
            response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})