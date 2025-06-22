# -*- coding: utf-8 -*-
import json
import time
from pathlib import Path
from typing import List, Dict
import re
import streamlit as st
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, get_response_synthesizer
from llama_index.core.schema import TextNode
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank  # 新增重排序组件

QA_TEMPLATE = (
    "<|im_start|>system\n"
    "你是一个专业的法律助手，请严格根据以下法律条文回答问题：\n"
    "相关法律条文：\n{context_str}\n<|im_end|>\n"
    "<|im_start|>user\n{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

response_template = PromptTemplate(QA_TEMPLATE)

# ================== Streamlit页面配置 ==================
st.set_page_config(
    page_title="智能劳动法咨询助手",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="auto"
)

def disable_streamlit_watcher():
    """Patch Streamlit to disable file watcher"""
    def _on_script_changed(_):
        return
        
    from streamlit import runtime
    runtime.get_instance()._on_script_changed = _on_script_changed

# ================== 配置区 ==================
class Config:
    EMBED_MODEL_PATH = "/root/Documents/model_modelscope/thomas/text2vec-base-chinese"
    LLM_MODEL_PATH = "/root/Documents/model_modelscope/Qwen/Qwen1.5-1.8B-Chat"
    RERANK_MODEL_PATH = "/root/Documents/model_modelscope/BAAI/bge-reranker-large"  # 新增重排序模型路径

    DATA_DIR = "/root/Documents/demo20-24/rag_docs"
    VECTOR_DB_DIR = "/root/Documents/demo20-24/chroma_db"
    PERSIST_DIR = "/root/Documents/demo20-24/storage"

    COLLECTION_NAME = "chinese_labor_laws"
    TOP_K = 10
    RERANK_TOP_K = 3  # 新增重排序的top_k参数

# ================== 缓存资源初始化 ==================
@st.cache_resource(show_spinner="初始化模型中...")
# ================== 初始化模型 ==================
def init_models():
    """初始化模型并验证"""
    # Embedding模型
    embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBED_MODEL_PATH
    )
    
    # LLM
    # TODO: 这里可以换成在线模型, 不用本地的
    llm = HuggingFaceLLM(
        model_name=Config.LLM_MODEL_PATH,
        tokenizer_name=Config.LLM_MODEL_PATH,
        model_kwargs={
            "trust_remote_code": True
        },
        tokenizer_kwargs={"trust_remote_code": True},
        generate_kwargs={"temperature": 0.3}
    )
    #llm = OpenAILike(
    #    model="/home/cw/llms/Qwen/Qwen1.5-1.8B-Chat",
    #    api_base="http://localhost:8000/v1",
    #    api_key="fake",
    #    context_window=4096,
    #    is_chat_model=True,
    #    is_function_calling_model=False,
    #)

    # 初始化重排序器（新增）
    reranker = SentenceTransformerRerank(
        model=Config.RERANK_MODEL_PATH,
        top_n=Config.RERANK_TOP_K
    )

    Settings.embed_model = embed_model
    Settings.llm = llm

    # 验证模型
    test_embedding = embed_model.get_text_embedding("测试文本")
    print(f"Embedding维度验证：{len(test_embedding)}")
    
    return embed_model, llm, reranker

# ================== 数据处理 ==================
def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    """加载并验证JSON法律文件"""
    json_files = list(Path(data_dir).glob("*.json"))
    assert json_files, f"未找到JSON文件于 {data_dir}"
    
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # 验证数据结构
                if not isinstance(data, list):
                    raise ValueError(f"文件 {json_file.name} 根元素应为列表")
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"文件 {json_file.name} 包含非字典元素")
                    for k, v in item.items():
                        if not isinstance(v, str):
                            raise ValueError(f"文件 {json_file.name} 中键 '{k}' 的值不是字符串")
                all_data.extend({
                    "content": item,
                    "metadata": {"source": json_file.name}
                } for item in data)
            except Exception as e:
                raise RuntimeError(f"加载文件 {json_file} 失败: {str(e)}")
    
    print(f"成功加载 {len(all_data)} 个法律文件条目")
    return all_data

def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    """添加ID稳定性保障"""
    nodes = []
    for entry in raw_data:
        law_dict = entry["content"]
        source_file = entry["metadata"]["source"]
        
        for full_title, content in law_dict.items():
            # 生成稳定ID（避免重复）
            node_id = f"{source_file}::{full_title}"
            
            parts = full_title.split(" ", 1)
            law_name = parts[0] if len(parts) > 0 else "未知法律"
            article = parts[1] if len(parts) > 1 else "未知条款"
            
            node = TextNode(
                text=content,
                id_=node_id,  # 显式设置稳定ID
                metadata={
                    "law_name": law_name,
                    "article": article,
                    "full_title": full_title,
                    "source_file": source_file,
                    "content_type": "legal_article"
                }
            )
            nodes.append(node)
    
    print(f"生成 {len(nodes)} 个文本节点（ID示例：{nodes[0].id_}）")
    return nodes

# ================== 向量存储 ==================
@st.cache_resource(show_spinner="加载知识库中...")
def init_vector_store(nodes: List[TextNode]) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 确保存储上下文正确初始化
    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    # 判断是否需要新建索引
    if chroma_collection.count() == 0 and nodes is not None:
        print(f"创建新索引（{len(nodes)}个节点）...")
        
        # 显式将节点添加到存储上下文
        storage_context.docstore.add_documents(nodes)  
        
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        # 双重持久化保障
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)  # <-- 新增
    else:
        print("加载已有索引...")
        storage_context = StorageContext.from_defaults(
            persist_dir=Config.PERSIST_DIR,
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    # 安全验证
    print("\n存储验证结果：")
    doc_count = len(storage_context.docstore.docs)
    print(f"DocStore记录数：{doc_count}")
    
    if doc_count > 0:
        sample_key = next(iter(storage_context.docstore.docs.keys()))
        print(f"示例节点ID：{sample_key}")
    else:
        print("警告：文档存储为空，请检查节点添加逻辑！")
    
    
    return index

# ================== 界面组件 ==================
def init_chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg.get("cleaned", msg["content"])  # 优先使用清理后的内容
        
        with st.chat_message(role):
            st.markdown(content)
            
            # 如果是助手消息且包含思维链
            if role == "assistant" and msg.get("think"):
                with st.expander("📝 模型思考过程（历史对话）"):
                    for think_content in msg["think"]:
                        st.markdown(f'<span style="color: #808080">{think_content.strip()}</span>',
                                  unsafe_allow_html=True)
            
            # 如果是助手消息且有参考依据（需要保持原有参考依据逻辑）
            if role == "assistant" and "reference_nodes" in msg:
                show_reference_details(msg["reference_nodes"])

def show_reference_details(nodes):
    with st.expander("查看支持依据"):
        for idx, node in enumerate(nodes, 1):
            meta = node.node.metadata
            st.markdown(f"**[{idx}] {meta['full_title']}**")
            st.caption(f"来源文件：{meta['source_file']} | 法律名称：{meta['law_name']}")
            st.markdown(f"相关度：`{node.score:.4f}`")
            # st.info(f"{node.node.text[:300]}...")
            st.info(f"{node.node.text}")

# ================== 主程序 ==================
def main():
    # 禁用 Streamlit 文件热重载
    disable_streamlit_watcher()
    st.title("⚖️ 智能劳动法咨询助手")
    st.markdown("欢迎使用劳动法智能咨询系统，请输入您的问题，我们将基于最新劳动法律法规为您解答。")

    # 初始化会话状态
    if "history" not in st.session_state:
        st.session_state.history = []

    embed_model, llm, reranker = init_models()

    # 仅当需要更新数据时执行
    if not Path(Config.VECTOR_DB_DIR).exists():
        with st.spinner("正在构建知识库..."):
            raw_data = load_and_validate_json_files(Config.DATA_DIR)
            nodes = create_nodes(raw_data)
    else:
        nodes = None  # 已有数据时不加载


    #print("\n初始化向量存储...")
    #start_time = time.time()
    #index = init_vector_store(nodes)
    #print(f"索引加载耗时：{time.time()-start_time:.2f}s")

    # 创建查询引擎
    # TODO: query_engine和retriever的区别是什么
    #query_engine = index.as_query_engine(
    #    similarity_top_k=Config.TOP_K,
    #    # text_qa_template=response_template,
    #    verbose=True
    #)

    index = init_vector_store(nodes)
    retriever = index.as_retriever(
        similarity_top_k=Config.TOP_K,
        vector_store_query_mode="hybrid",
        alpha=0.5,
    )
    response_synthesizer = get_response_synthesizer(
        # text_qa_template=response_template,
        verbose=True
    )

    # 聊天界面
    init_chat_interface()

    if prompt := st.chat_input("请输入劳动法相关问题"):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 处理查询
        with st.spinner("正在分析问题..."):
            start_time = time.time()
            
            # 检索流程
            initial_nodes = retriever.retrieve(prompt)
            reranked_nodes = reranker.postprocess_nodes(initial_nodes, query_str=prompt)
            
            # 过滤节点
            MIN_RERANK_SCORE = 0.4
            filtered_nodes = [node for node in reranked_nodes if node.score > MIN_RERANK_SCORE]
            
            if not filtered_nodes:
                response_text = "⚠️ 未找到相关法律条文，请尝试调整问题描述或咨询专业律师。"
            else:
                # 生成回答
                response = response_synthesizer.synthesize(prompt, nodes=filtered_nodes)
                response_text = response.response
            
            # 显示回答
            with st.chat_message("assistant"):
                # 提取思维链内容并清理响应文本
                think_contents = re.findall(r'<think>(.*?)</think>', response_text, re.DOTALL)
                cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
                
                # 显示清理后的回答
                st.markdown(cleaned_response)
                
                # 如果有思维链内容则显示
                if think_contents:
                    with st.expander("📝 模型思考过程（点击展开）"):
                        for content in think_contents:
                            st.markdown(f'<span style="color: #808080">{content.strip()}</span>', 
                                      unsafe_allow_html=True)
                
                # 显示参考依据（保持原有逻辑）
                show_reference_details(filtered_nodes[:3])

            # 添加助手消息到历史（需要存储原始响应）
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,  # 保留原始响应
                "cleaned": cleaned_response,  # 存储清理后的文本
                "think": think_contents  # 存储思维链内容
            })
    ## 示例查询
    #while True:
    #    question = input("\n请输入劳动法相关问题（输入q退出）: ")
    #    if question.lower() == 'q':
    #        break

    #    # 流程: 检索->重排序->过滤->回答
    #    start_time = time.time()

    #    # 1. 检索
    #    initial_nodes = retriever.retrieve(question)
    #    retrieval_time = time.time() - start_time
    #    
    #    # 2. 重排序
    #    reranked_nodes = reranker.postprocess_nodes(
    #        initial_nodes, 
    #        query_str=question
    #    )
    #    rerank_time = time.time() - start_time - retrieval_time

    #    # 3. 过滤
    #    MIN_RERANK_SCORE = 0.4
    #    # 执行过滤
    #    filtered_nodes = [
    #        node for node in reranked_nodes 
    #        if node.score > MIN_RERANK_SCORE
    #    ]

    #    #一般对模型的回复做限制就从filtered_nodes的返回值下手
    #    print("原始分数样例：",[node.score for node in reranked_nodes[:3]])
    #    print("重排序过滤后的结果：",filtered_nodes)
    #    # 空结果处理
    #    if not filtered_nodes:
    #        print("你的问题未匹配到相关资料！")
    #        continue

    #    # 4. 合成答案（使用过滤后的节点）
    #    response = response_synthesizer.synthesize(
    #        question, 
    #        nodes=filtered_nodes  # 使用过滤后的节点
    #    )
    #    synthesis_time = time.time() - start_time - retrieval_time - rerank_time

    #    # 打印显示
    #    print(f"\n智能助手回答：\n{response.response}")
    #    print("\n支持依据：")
    #    for idx, node in enumerate(reranked_nodes, 1):
    #        # 兼容新版API的分数获取方式
    #        initial_score = node.metadata.get('initial_score', node.score)  # 获取初始分数
    #        rerank_score = node.score  # 重排序后的分数
    #    
    #        meta = node.node.metadata
    #        print(f"\n[{idx}] {meta['full_title']}")
    #        print(f"  来源文件：{meta['source_file']}")
    #        print(f"  法律名称：{meta['law_name']}")
    #        print(f"  初始相关度：{node.node.metadata.get('initial_score', 0):.4f}")  # 安全访问
    #        print(f"  重排序得分：{getattr(node, 'score', 0):.4f}")  # 兼容属性访问
    #        print(f"  条款内容：{node.node.text[:100]}...")
    #    
    #    print(f"\n[性能分析] 检索: {retrieval_time:.2f}s | 重排序: {rerank_time:.2f}s | 合成: {synthesis_time:.2f}s")

if __name__ == "__main__":
    main()