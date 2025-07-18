import streamlit as st
from utils.doc_handler import RagModal, process_uploaded_files, build_text_modal_corpus, build_multi_modal_corpus, load_text_modal_corpus, load_multi_modal_corpus
from pymilvus import connections

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

# 侧边栏管理
with st.sidebar:
    st.header("⚙️ RAG 参数设置")
    checkbos_value = st.checkbox("多模态问答", value=False if RagModal.TEXT == st.session_state.rag_modal else True)
    st.session_state.rag_modal = RagModal.MULTI_MODAL if checkbos_value else RagModal.TEXT

    st.markdown("---")

    st.header("⚙️ Milvus数据库设置")
    st.text("milvus uri")
    milvus_uri_input, milvus_connect_button = st.columns([3, 1])
    with milvus_uri_input:
        input_milvus_uri = st.text_input(label="milvus uri", value="http://localhost:19530", placeholder="http://localhost:19530", label_visibility="collapsed")
    with milvus_connect_button:
        connect_button = st.button("连接")
    if connect_button:
        # 检查milvus是否连接
        if "rag_config" in st.session_state:
            with st.spinner("Connecting database..."):
                try:
                    connections.connect(
                        alias="default",
                        uri=input_milvus_uri if input_milvus_uri else st.session_state.rag_config.milvus_uri,
                    )
                    st.session_state.milvus_connected = True
                    connections.disconnect("default")
                except Exception as e:
                    st.session_state.milvus_connected = False
    if True == st.session_state.milvus_connected:
        st.success("Milvus数据库已连接!")
    else:
        st.warning("Milvus数据库未连接!")

# 文档解析, 建立知识库
st.markdown("# 建立知识库")
current_corpus_name:str = None
warning_message:str = None
is_build_corpus: bool = False
uploaded_files = st.file_uploader(
    "上传文档 (PDF/DOCX/TXT/Markdown)",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)
corpus_name_input, corpus_build_button = st.columns([4.5, 1])
with corpus_name_input:
    current_corpus_name = st.text_input(label="corpus name", placeholder="输入知识库名称", label_visibility="collapsed")
with corpus_build_button:
    if st.button("创建"):
        if not uploaded_files:
            warning_message = "请上传至少一个文档"
        elif not current_corpus_name:
            warning_message = "请输入知识库名称"
        elif any('\u4e00' <= ch <= '\u9fff' for ch in current_corpus_name):
            warning_message = "知识库名称必须为全英文"
        else:
            is_build_corpus = True
if warning_message:
    st.warning(warning_message)

# 调用文件解析功能
if uploaded_files and current_corpus_name and True == is_build_corpus:
    with st.spinner("正在创建知识库..."):
        # 文档解析+切分
        refined_nodes = process_uploaded_files(uploaded_files=uploaded_files)
        if refined_nodes:
            dense_collection_name, sparse_collection_name, image_collection_name = st.session_state.corpus_management.create_collection_entry(
                current_corpus_name=current_corpus_name,
                modal=st.session_state.rag_modal
            )
            if RagModal.TEXT == st.session_state.rag_modal:
                st.session_state.semantic_retriever, st.session_state.keywords_retriever = build_text_modal_corpus(
                    nodes=refined_nodes,
                    category=current_corpus_name,
                    embed_model=st.session_state.rag_config.text_embed_model,
                    milvus_dense_collection_name=dense_collection_name,
                    milvus_sparse_collection_name=sparse_collection_name,
                    milvus_uri=st.session_state.rag_config.milvus_uri,
                    use_milvus = st.session_state.milvus_connected,
                    semantic_retriever_top_k=5,
                    keywords_retriever_top_k=5,
                )
                st.session_state.loaded_corpus = current_corpus_name
            elif RagModal.MULTI_MODAL == st.session_state.rag_modal:
                st.session_state.semantic_retriever, st.session_state.keywords_retriever = build_multi_modal_corpus(
                    nodes=refined_nodes,
                    category=current_corpus_name,
                    embed_model=st.session_state.rag_config.mm_embed_model,
                    milvus_dense_collection_name=dense_collection_name,
                    milvus_sparse_collection_name=sparse_collection_name,
                    milvus_image_collection_name=image_collection_name,
                    milvus_uri=st.session_state.rag_config.milvus_uri,
                    use_milvus=st.session_state.milvus_connected,
                    semantic_retriever_top_k=5,
                    keywords_retriever_top_k=5,
                    image_retriever_top_k=5,
                )
                st.session_state.loaded_corpus = current_corpus_name
            else:
                pass
            if st.session_state.semantic_retriever and st.session_state.keywords_retriever:
                st.session_state.documents_loaded = True
                if st.session_state.milvus_connected:
                    st.success("知识库创建成功, 已写入数据库!")
                else:
                    st.success("知识库创建成功!")
            else:
                st.error("知识库创建失败!")

st.markdown("---")

# 连接数据库, 加载知识库
selected_corpuses: list[str] = None
warning_message = None
st.markdown("# 加载数据库")
# 显示数据库中已有的知识库
#if "existing_collections" in st.session_state:
# 根据启用的模态筛选显示的知识库列表
existing_corpuses = st.session_state.corpus_management.list_corpuses(st.session_state.rag_modal)
selected_corpuses = st.multiselect(
    "选择知识库",
    existing_corpuses,
    key="collection_selector",
    disabled=not st.session_state.milvus_connected,
    max_selections=1
)

if st.button("加载知识库", disabled=not st.session_state.milvus_connected):
    if not selected_corpuses:
        warning_message = "请选择至少一个知识库"
    else:
        dense_collection_name: str = None
        sparse_collection_name: str = None
        image_collection_name: str = None

        for selected_corpus in selected_corpuses:
            dense_collection_name, sparse_collection_name, image_collection_name = st.session_state.corpus_management.load_collection_entry(selected_corpus_name=selected_corpus)

        if RagModal.TEXT == st.session_state.rag_modal:
            if dense_collection_name and sparse_collection_name:
                st.session_state.semantic_retriever, st.session_state.keywords_retriever = load_text_modal_corpus(
                    embed_model=st.session_state.rag_config.text_embed_model,
                    milvus_dense_collection_name=dense_collection_name,
                    milvus_sparse_collection_name=sparse_collection_name,
                    milvus_uri=st.session_state.rag_config.milvus_uri,
                    semantic_retriever_top_k=5,
                    keywords_retriever_top_k=5,
                )
                st.session_state.loaded_corpus = selected_corpus
                st.rerun()
        elif RagModal.MULTI_MODAL == st.session_state.rag_modal:
            if dense_collection_name and sparse_collection_name and image_collection_name:
                st.session_state.semantic_retriever, st.session_state.keywords_retriever = load_multi_modal_corpus(
                    embed_model=st.session_state.rag_config.mm_embed_model,
                    milvus_dense_collection_name=dense_collection_name,
                    milvus_sparse_collection_name=sparse_collection_name,
                    milvus_image_collection_name=image_collection_name,
                    milvus_uri=st.session_state.rag_config.milvus_uri,
                    semantic_retriever_top_k=5,
                    keywords_retriever_top_k=5,
                    image_retriever_top_k=5,
                )
                st.session_state.loaded_corpus = selected_corpus
                st.rerun()
        else:
            pass

if warning_message:
    st.warning(warning_message, disabled=not st.session_state.milvus_connected)