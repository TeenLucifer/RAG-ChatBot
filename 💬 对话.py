import os
import torch
import nest_asyncio
import streamlit as st
from utils.retrieve_pipline import retrieve_text_modal, retrieve_multi_modal, multi_modal_synthesize_response
from utils.doc_handler import RagModal, CorpusManagement
from utils.rag_config import RagConfig

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]# 防止torch报错

nest_asyncio.apply() # 防止异步报错

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

with st.sidebar:                                                                        # 📁 Sidebar
    st.header("📁 已加载知识库")
    if "loaded_corpus" in st.session_state and st.session_state.loaded_corpus:
        st.markdown(f"**当前知识库:** {st.session_state.loaded_corpus}")

    st.markdown("---")

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
st.caption("基于llama-index搭建的RAG系统, 支持数据库存储、意图判断、查询扩写、多模态检索、多路召回")

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
            text_nodes, image_nodes = retrieve_multi_modal(
                query=prompt,
                mllm=st.session_state.rag_config.mllm,
                mm_embed_model=st.session_state.rag_config.mm_embed_model,
                rerank_model=st.session_state.rag_config.rerank_model,
                semantic_retriever=st.session_state.semantic_retriever,
                keywords_retriever=st.session_state.keywords_retriever,
                dashscope_api_key=st.session_state.rag_config.dashscope_api_key,
                dashscope_mllm_model_name=st.session_state.rag_config.dashscope_mllm_model_name,
            )
            response = multi_modal_synthesize_response(
                query=prompt,
                text_nodes=text_nodes,
                image_nodes=image_nodes,
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
        if RagModal.MULTI_MODAL == st.session_state.rag_modal:
            for node in image_nodes:
                st.image(node.node.image_path)
        st.session_state.messages.append({"role": "assistant", "content": full_response})