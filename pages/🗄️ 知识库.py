import streamlit as st
from utils.doc_handler import process_uploaded_files, build_text_corpus, load_text_corpus, load_multi_modal_corpus
from pymilvus import connections, utility

# TODO(wangjintao): 数据库的增删查改(或实现某种持久化存储如json文件)
# 1. 连接数据库时查询是否存在索引collection, 不存在则创建, 存在就读取
# 2. 索引collection用于存储知识库名对应的dense和sparse collection名称
# 3. build知识库时以知识库名为索引, 存储dense和sparse collection名称
# 4. 选择知识库时加载知识库名索引, 在milvus中load对应的dense和sparse collection

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
    st.header("📁 已加载知识库")
    # TODO(wangjintao): 显示已加载知识库名
    st.markdown("---")
    st.header("⚙️ Milvus数据库设置")

    # TODO(wangjintao): 待完善连接逻辑
    st.text("milvus uri")
    milvus_uri_input, milvus_connect_button = st.columns([3, 1])
    with milvus_uri_input:
        st.text_input(label="milvus uri", placeholder="http://localhost:19530", label_visibility="collapsed")
    with milvus_connect_button:
        st.button("连接")

    # 检查milvus是否连接
    if "rag_config" in st.session_state and "existing_collections" not in st.session_state:
        with st.spinner("Connecting database..."):
            try:
                connections.connect(
                    alias="default",
                    uri=st.session_state.rag_config.milvus_uri,
                )
                st.session_state.existing_collections = utility.list_collections()
                connections.disconnect("default")
            except Exception as e:
                pass
    if "existing_collections" in st.session_state:
        st.success("Milvus数据库已连接!")
    else:
        st.error("Milvus数据库连接失败!")

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
            st.session_state.semantic_retriever, st.session_state.keywords_retriever = build_text_corpus(
                nodes=refined_nodes,
                category=current_corpus_name,
                embed_model=st.session_state.rag_config.text_embed_model,
                milvus_dense_collection_name=st.session_state.rag_config.milvus_dense_collection_name,
                milvus_sparse_collection_name=st.session_state.rag_config.milvus_sparse_collection_name,
                milvus_uri=st.session_state.rag_config.milvus_uri,
                use_milvus = False,
                semantic_retriever_top_k=5,
                keywords_retriever_top_k=5,
            )
            if st.session_state.semantic_retriever and st.session_state.keywords_retriever:
                st.session_state.documents_loaded = True
                st.success("知识库创建成功!")
            else:
                st.error("知识库创建失败!")

st.markdown("---")

selected_collections: list[str] = None
warning_message = None
st.markdown("# 加载数据库")
# 显示数据库中已有的知识库
if "existing_collections" in st.session_state:
    selected_collections = st.multiselect(
        "选择知识库",
        st.session_state.existing_collections,
        key="collection_selector"
    )

if st.button("加载知识库"):
    if not selected_collections:
        warning_message = "请选择至少一个知识库"
    else:
        pass

if warning_message:
    st.warning(warning_message)