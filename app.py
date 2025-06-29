import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.llms.dashscope import DashScope
from utils.dashscope_embedding import DashScopeEmbedding
from utils.doc_handler import document_segmentation, build_text_corpus, load_text_corpus, load_multi_modal_corpus
from utils.retrieve_pipline import expand_query, retrieve_documents
from pymilvus import connections, utility

from llama_index.vector_stores.milvus import MilvusVectorStore

# 加载.env文件
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
DASHSCOPE_LLM_MODEL_NAME = os.getenv("DASHSCOPE_LLM_MODEL_NAME")
DASHSCOPE_MLLM_MODEL_NAME = os.getenv("DASHSCOPE_MLLM_MODEL_NAME")
DASHSCOPE_TEXT_EMBED_MODEL_NAME = os.getenv("DASHSCOPE_TEXT_EMBED_MODEL_NAME")
DASHSCOPE_MM_EMBED_MODEL_NAME = os.getenv("DASHSCOPE_MM_EMBED_MODEL_NAME")
MILVUS_URI = os.getenv("MILVUS_URI")
milvus_dense_collection_name  = "RAG_CHATBOT_" + "TEXT_EMBED" + "_DENSE_COLLECTION"
milvus_sparse_collection_name = "RAG_CHATBOT_" + "TEXT_EMBED" + "_SPARSE_COLLECTION"
# 配置大语言模型
llm = DashScope(
    api_key=DASHSCOPE_API_KEY,
    model_name=DASHSCOPE_LLM_MODEL_NAME,
)
# 配置嵌入模型
text_embed_model = DashScopeEmbedding(
    api_key=DASHSCOPE_API_KEY,
    model_name=DASHSCOPE_TEXT_EMBED_MODEL_NAME,
    embed_batch_size=10
)
# 配置重排序模型
rerank_model = DashScopeRerank(
    api_key=DASHSCOPE_API_KEY,
    model="gte-rerank-v2",
    top_n=5
)
#TODO(wangjintao): 连接milvus失败就提示从本地上传文档
#connections.connect(
#    alias="default",
#    uri=MILVUS_URI,
#)
#try:
#    st.session_state.existing_collections = collections = utility.list_collections()
#    connections.disconnect("default")
#except Exception as e:
#    st.error(f"连接失败: {str(e)}")

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
    st.header("📁 Document Management")
    uploaded_file = st.file_uploader(
        "Upload documents (PDF/DOCX/TXT/Markdown)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=False
    )
    
    # 调用文件解析功能
    if uploaded_file and not st.session_state.documents_loaded:
        with st.spinner("Processing documents..."):
            file_names = [file.name for file in uploaded_files]
            for file_name in file_names:
                refined_nodes = document_segmentation(file_name)
            st.session_state.semantic_retriever, st.session_state.keywords_retriever = build_text_corpus(
                nodes=refined_nodes,
                category="RAG_CHATBOT",
                embed_model=text_embed_model,
                milvus_dense_collection_name=milvus_dense_collection_name,
                milvus_sparse_collection_name=milvus_sparse_collection_name,
                milvus_uri=MILVUS_URI,
                use_milvus = False,
                semantic_retriever_top_k=5,
                keywords_retriever_top_k=5,
            )
            st.success("Documents processed!")

    st.markdown("---")
    st.header("⚙️ RAG Settings")

    # 交互控件
    #if st.button("加载语料库"):
    #    print(milvus_dense_collection_name)
    #    print(milvus_sparse_collection_name)
    #    print(MILVUS_URI)

    #    st.session_state.semantic_retriever, st.session_state.keywords_retriever = load_text_corpus(
    #        embed_model=text_embed_model,
    #        milvus_dense_collection_name=milvus_dense_collection_name,
    #        milvus_sparse_collection_name=milvus_sparse_collection_name,
    #        milvus_uri=MILVUS_URI,
    #        semantic_retriever_top_k=5,
    #        keywords_retriever_top_k=5,
    #    )
    #    st.success("语料库加载成功!")
    #if "existing_collections" in st.session_state:
    #selected_collections = st.multiselect(
    #    "选择已有的知识库",
    #    st.session_state.existing_collections,
    #    key="collection_selector"
    #)
    #if not selected_collections:
    #    st.warning("请至少选择一个数据库")
    #else:
    #    st.session_state.selected_collections = selected_collections
    #if "selected_collections" in st.session_state and st.session_state.selected_collections:
    #    if st.button("加载知识库"):
    #        load_text_corpus()

    #if st.button("加载文本知识库"):
    #    load_text_corpus()
    #st.button("加载多模态知识库")
    #st.session_state.rag_enabled = st.checkbox("Enable RAG", value=True)
    #st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=True)
    #st.session_state.enable_reranking = st.checkbox("Enable Neural Reranking", value=True)
    #st.session_state.enable_graph_rag = st.checkbox("Enable GraphRAG", value=True)
    #st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    #st.session_state.max_contexts = st.slider("Max Contexts", 1, 5, 3)
    
    #if st.button("Clear Chat History"):
    #    st.session_state.messages = []
    #    st.rerun()

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

if prompt := st.chat_input("Ask about your documents..."):
    chat_history = "\n".join([msg["content"] for msg in st.session_state.messages[-5:]])  # Last 5 messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # 🚀 Build context
        context = ""
        full_response = retrieve_documents(
            llm=llm,
            text_embed_model=text_embed_model,
            rerank_model=rerank_model,
            semantic_retriever=st.session_state.semantic_retriever,
            keywords_retriever=st.session_state.keywords_retriever,
            query=prompt,
            dashscope_api_key=DASHSCOPE_API_KEY,
            dashscope_base_url=DASHSCOPE_BASE_URL,
        )
        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        #if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
        #    try:
        #        docs = retrieve_documents(prompt, OLLAMA_API_URL, MODEL, chat_history)
        #        context = "\n".join(
        #            f"[Source {i+1}]: {doc.page_content}" 
        #            for i, doc in enumerate(docs)
        #        )
        #    except Exception as e:
        #        st.error(f"Retrieval error: {str(e)}")
        
        # 🚀 Structured Prompt
        #system_prompt = f"""Use the chat history to maintain context:
        #    Chat History:
        #    {chat_history}

        #    Analyze the question and context through these steps:
        #    1. Identify key entities and relationships
        #    2. Check for contradictions between sources
        #    3. Synthesize information from multiple contexts
        #    4. Formulate a structured response

        #    Context:
        #    {context}

        #    Question: {prompt}
        #    Answer:"""

# Stream response
        #response = requests.post(
        #    OLLAMA_API_URL,
        #    json={
        #        "model": MODEL,
        #        "prompt": system_prompt,
        #        "stream": True,
        #        "options": {
        #            "temperature": st.session_state.temperature,  # Use dynamic user-selected value
        #            "num_ctx": 4096
        #        }
        #    },
        #    stream=True
        #)
        #try:
        #    for line in response.iter_lines():
        #        if line:
        #            data = json.loads(line.decode())
        #            token = data.get("response", "")
        #            full_response += token
        #            response_placeholder.markdown(full_response + "▌")
        #            
        #            # Stop if we detect the end token
        #            if data.get("done", False):
        #                break
        #                
        #    response_placeholder.markdown(full_response)
        #    st.session_state.messages.append({"role": "assistant", "content": full_response})
        #    
        #except Exception as e:
        #    st.error(f"Generation error: {str(e)}")
        #    st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error."})