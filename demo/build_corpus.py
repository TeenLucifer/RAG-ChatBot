# 构建语料库: 向量库+字面库
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import time
from pathlib import Path
from typing import List, Tuple
import jieba
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import BaseNode, TextNode, ImageNode, ImageDocument, Document
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter, SimpleNodeParser
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.readers.file import FlatReader
from llama_index.retrievers.bm25 import BM25Retriever
from pymilvus import connections, utility
from dotenv import load_dotenv
from utils.dashscope_embedding import DashScopeEmbedding
from utils.doc_handler import document_segmentation, build_text_corpus, build_multi_modal_corpus

# 加载.env文件
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
DASHSCOPE_LLM_MODEL_NAME = os.getenv("DASHSCOPE_LLM_MODEL_NAME")
DASHSCOPE_TEXT_EMBED_MODEL_NAME = os.getenv("DASHSCOPE_TEXT_EMBED_MODEL_NAME")
DASHSCOPE_MM_EMBED_MODEL_NAME = os.getenv("DASHSCOPE_MM_EMBED_MODEL_NAME")

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_DENSE_COLLECTION_NAME = os.getenv("MILVUS_DENSE_COLLECTION_NAME")
MILVUS_SPARSE_COLLECTION_NAME = os.getenv("MILVUS_SPARSE_COLLECTION_NAME")

MILVUS_URI = "http://" + MILVUS_HOST + ":" + MILVUS_PORT
milvus_uri = MILVUS_URI
category = "TECHNOLOGY"

## 中文分词器
#def bm25_chinese_tokenizer(text: str) -> List[str]:
#    tokens = jieba.cut(text)
#    stop_words = ["的", "了", "是"]
#    filtered_tokens = [
#        token for token in tokens
#        if token not in stop_words and 2 <= len(token) <= 20]
#    return filtered_tokens
#
## 文档切分
#def document_segmentation(
#    doc_path: str,
#    documents: Document = None,
#) -> List[BaseNode]:
#    # 如果不传入Document对象, 则从文件路径加载文档
#    if documents is None:
#        markdown_file_path = doc_path
#        # 加载Markdown文档
#        documents = FlatReader().load_data(Path(markdown_file_path))
#    # 初始化解析器
#    parser = MarkdownNodeParser()
#    # 按标题层级切分
#    nodes = parser.get_nodes_from_documents(documents)
#
#    # 二次切分：中文按照段落进行切分, 公式特殊处理
#    # 针对中文优化的段落切分（按句号/换行符分割，避免截断句子）
#    text_splitter = SentenceSplitter(
#        chunk_size=200,
#        chunk_overlap=10,
#        separator="。",
#        paragraph_separator="\r\n\r\n",
#        secondary_chunking_regex="。！？\n",
#    )
#    refined_nodes = text_splitter.get_nodes_from_documents(nodes)
#    return refined_nodes
#
## 建立语料库
#def build_text_corpus(
#    nodes: List[BaseNode],
#    category: str,
#    embed_model: DashScopeEmbedding,
#    milvus_dense_collection_name: str,
#    milvus_sparse_collection_name: str,
#    milvus_uri: str = "http://localhost:19530",
#    use_milvus: bool = True,
#    semantic_retriever_top_k: int = 10,
#    keywords_retriever_top_k: int = 10,
#) -> Tuple[BaseRetriever, BaseRetriever]:
#    # 检查milvus数据库能否连接上, 若未连接上, 则直接创建向量检索和字面检索库, 不再创建本地存储
#    milvus_connected = False
#    if True == use_milvus:
#        try:
#            connections.connect(uri=milvus_uri)
#            version = utility.get_server_version() # 尝试获取Milvus的版本以确认连接
#            milvus_connected = True
#        except Exception as e:
#            milvus_connected = False
#
#    if True == milvus_connected:
#        # 能连接上数据库, 用milvus建立语义检索和字面检索的向量库(用稀疏量模拟)
#        milvus_dense_collection_name = "RAG_CHATBOT_" + category + "_DENSE_COLLECTION"
#        milvus_sparse_collection_name = "RAG_CHATBOT_" + category + "_SPARSE_COLLECTION"
#        # 语义检索向量库
#        milvus_dense_store = MilvusVectorStore(
#            uri=milvus_uri,
#            dim=1024,  # 向量维度需与嵌入模型匹配
#            collection_name=milvus_dense_collection_name,
#            overwrite=True,
#        )
#        milvus_dense_storage_context = StorageContext.from_defaults(vector_store=milvus_dense_store)
#        milvus_dense_index = VectorStoreIndex(
#            nodes=nodes,
#            embed_model=embed_model,
#            storage_context=milvus_dense_storage_context,
#            show_progress=True
#        )
#        semantic_retriever = milvus_dense_index.as_retriever(
#            similarity_top_k=semantic_retriever_top_k,
#            vector_store_query_mode="default",
#        )
#
#        # 字面检索向量库(用稀疏向量模拟)
#        bm25_function = BM25BuiltInFunction(
#            analyzer_params={
#                "type": "chinese",
#                "tokenizer": "jieba",  # 中文文档需要使用 Jieba 中文分词器
#                "filter": [
#                    {"type": "stop", "stop_words": ["的", "了", "是"]},  # 中文停用词
#                    {"type": "length", "min": 2, "max": 20},           # 过滤超短/超长词
#                ],
#            },
#            enable_match=True,
#        )
#        Settings.embed_model = None # 建立稀疏向量库时需要显式禁用嵌入模型
#        milvus_sparse_store = MilvusVectorStore(
#            uri=milvus_uri,
#            enable_dense=False,  # 不使用稠密向量
#            enable_sparse=True,  # 启用稀疏向量
#            sparse_embedding_function=bm25_function,
#            collection_name=milvus_sparse_collection_name,
#            overwrite=True,
#        )
#        milvus_sparse_storage_context = StorageContext.from_defaults(vector_store=milvus_sparse_store)
#        milvus_sparse_index = VectorStoreIndex(
#            nodes=nodes,
#            storage_context=milvus_sparse_storage_context,
#            show_progress=True,
#        )
#        keywords_retriever = milvus_sparse_index.as_retriever(
#            vector_store_query_mode="sparse",
#            similarity_top_k=keywords_retriever_top_k,
#        )
#    else:
#        # 连接不上数据库直接创建语料库, 不做本地存储
#        vector_retrieve_index = VectorStoreIndex( # 创建向量索引
#            nodes=nodes,
#            embed_model=embed_model,
#            show_progress=True,
#        )
#        semantic_retriever = vector_retrieve_index.as_retriever(
#            similarity_top_k=semantic_retriever_top_k,
#            vector_store_query_mode="default",
#        )
#        keywords_retriever = BM25Retriever.from_defaults( # 创建字面索引
#            nodes=nodes,
#            similarity_top_k=keywords_retriever_top_k,
#            tokenizer=bm25_chinese_tokenizer,
#        )
#    return semantic_retriever, keywords_retriever
#
## 建立多模态语料库
#def build_multi_modal_corpus(
#    nodes: List[BaseNode],
#    category: str,
#    embed_model: DashScopeEmbedding,
#    milvus_dense_collection_name: str,
#    milvus_image_collection_name: str,
#    milvus_sparse_collection_name: str,
#    milvus_uri: str = "http://localhost:19530",
#    use_milvus: bool = True,
#    semantic_retriever_top_k: int = 10,
#    image_retriever_top_k: int = 10,
#    keywords_retriever_top_k: int = 10,
#) -> Tuple[BaseRetriever, BaseRetriever]:
#    # 检查milvus数据库能否连接上, 若未连接上, 则直接创建向量检索和字面检索库, 不再创建本地存储
#    milvus_connected = False
#    if True == use_milvus:
#        try:
#            connections.connect(uri=milvus_uri)
#            version = utility.get_server_version() # 尝试获取Milvus的版本以确认连接
#            milvus_connected = True
#        except Exception as e:
#            milvus_connected = False
#
#    text_nodes = [node for node in nodes if type(node) is TextNode]
#
#    if True == milvus_connected:
#        # 能连接上数据库, 用milvus建立语义检索和字面检索的向量库(用稀疏量模拟)
#
#        # 语义检索向量库
#        milvus_dense_store = MilvusVectorStore(
#            uri=milvus_uri,
#            dim=1024,  # 向量维度需与嵌入模型匹配
#            collection_name=milvus_dense_collection_name,
#            overwrite=True,
#        )
#        milvus_image_store = MilvusVectorStore(
#            uri=milvus_uri,
#            dim=1024,  # 向量维度需与嵌入模型匹配
#            collection_name=milvus_image_collection_name,
#            overwrite=True,
#        )
#        milvus_dense_storage_context = StorageContext.from_defaults(
#            vector_store=milvus_dense_store,
#            image_store=milvus_image_store,
#        )
#        milvus_dense_index = MultiModalVectorStoreIndex(
#            nodes=nodes,
#            embed_model=embed_model,
#            image_embed_model=embed_model,
#            storage_context=milvus_dense_storage_context,
#            show_progress=True
#        )
#        semantic_retriever = milvus_dense_index.as_retriever(
#            similarity_top_k=semantic_retriever_top_k,
#            image_retriever_top_k=image_retriever_top_k,
#            vector_store_query_mode="default",
#        )
#
#        # 字面检索向量库(用稀疏向量模拟)
#        bm25_function = BM25BuiltInFunction(
#            analyzer_params={
#                "type": "chinese",
#                "tokenizer": "jieba",  # 中文文档需要使用 Jieba 中文分词器
#                "filter": [
#                    {"type": "stop", "stop_words": ["的", "了", "是"]},  # 中文停用词
#                    {"type": "length", "min": 2, "max": 20},           # 过滤超短/超长词
#                ],
#            },
#            enable_match=True,
#        )
#        Settings.embed_model = None # 建立稀疏向量库时需要显式禁用嵌入模型
#        milvus_sparse_store = MilvusVectorStore(
#            uri=milvus_uri,
#            enable_dense=False,  # 不使用稠密向量
#            enable_sparse=True,  # 启用稀疏向量
#            sparse_embedding_function=bm25_function,
#            collection_name=milvus_sparse_collection_name,
#            overwrite=True,
#        )
#        milvus_sparse_storage_context = StorageContext.from_defaults(vector_store=milvus_sparse_store)
#        milvus_sparse_index = VectorStoreIndex(
#            nodes=text_nodes,
#            storage_context=milvus_sparse_storage_context,
#            show_progress=True,
#        )
#        keywords_retriever = milvus_sparse_index.as_retriever(
#            vector_store_query_mode="sparse",
#            similarity_top_k=keywords_retriever_top_k,
#        )
#
#    else:
#        # 连接不上数据库直接创建语料库, 不做本地存储
#        vector_retrieve_index = MultiModalVectorStoreIndex(
#            nodes=nodes,
#            embed_model=embed_model,
#            image_embed_model=embed_model,
#            show_progress=True
#        )
#        semantic_retriever = vector_retrieve_index.as_retriever(
#            similarity_top_k=semantic_retriever_top_k,
#            vector_store_query_mode="default",
#        )
#        keywords_retriever = BM25Retriever.from_defaults( # 创建字面索引
#            nodes=text_nodes,
#            similarity_top_k=keywords_retriever_top_k,
#            tokenizer=bm25_chinese_tokenizer,
#        )
#    return semantic_retriever, keywords_retriever

# 测试文本嵌入语料库的demo
def run_text_embedding_demo():
    doc_path = "./converted_docs/uav_swarm_page23-25/uav_swarm_page23-25.md"
    milvus_dense_collection_name  = "RAG_CHATBOT_" + "TEXT_EMBED" + "_DENSE_COLLECTION"
    milvus_sparse_collection_name = "RAG_CHATBOT_" + "TEXT_EMBED" + "_SPARSE_COLLECTION"
    # 配置大语言模型
    llm = DashScope(
        api_key=DASHSCOPE_API_KEY,
        model_name=DASHSCOPE_LLM_MODEL_NAME,
    )
    # 配置文本嵌入模型
    text_embed_model = DashScopeEmbedding(
        api_key=DASHSCOPE_API_KEY,
        model_name=DASHSCOPE_TEXT_EMBED_MODEL_NAME,
        embed_batch_size=10
    )

    # 加载文档
    refined_nodes = document_segmentation(doc_path)
    # 建立文本语料库
    semantic_retriever, keywords_retriever = build_text_corpus(
        nodes=refined_nodes,
        category=category, 
        embed_model=text_embed_model, 
        milvus_dense_collection_name=milvus_dense_collection_name,
        milvus_sparse_collection_name=milvus_sparse_collection_name,
        milvus_uri=milvus_uri,
        use_milvus=True
    )

    # 等待索引构建完成后再测试查询
    time.sleep(2)

    # 语义索引测试
    res_nodes = semantic_retriever.retrieve("吸引-结队-排斥规则是什么")
    print("\n语义检索结果:")
    for node in res_nodes:
        print(node.get_content())

    # 字面索引测试(用稀疏向量模拟)
    res_nodes = keywords_retriever.retrieve("吸引-结队-排斥规则是什么")
    print("\n字面检索结果:")
    for node in res_nodes:
        print(node.get_content())

# 测试多模态嵌入语料库的demo
def run_multi_modal_embedding_demo():
    text_doc_path = "./converted_docs/uav_swarm_page23-25/uav_swarm_page23-25.md"
    image_doc_path = "./converted_docs/uav_swarm_page23-25"
    doc_path = "./converted_docs/uav_swarm_page23-25"
    milvus_dense_collection_name  = "RAG_CHATBOT_" + "MM_EMBED" + "_DENSE_COLLECTION"
    milvus_image_collection_name  = "RAG_CHATBOT_" + "MM_EMBED" + "_IMAGE_COLLECTION"
    milvus_sparse_collection_name = "RAG_CHATBOT_" + "MM_EMBED" + "_SPARSE_COLLECTION"
    # 配置大语言模型
    llm = DashScope(
        api_key=DASHSCOPE_API_KEY,
        model_name=DASHSCOPE_LLM_MODEL_NAME,
    )
    # 配置多模态嵌入模型
    mm_embed_model = DashScopeEmbedding(
        api_key=DASHSCOPE_API_KEY,
        model_name=DASHSCOPE_MM_EMBED_MODEL_NAME,
        embed_batch_size=10
    )

    # 加载文档
    documents = SimpleDirectoryReader(doc_path).load_data()
    parser = SimpleNodeParser.from_defaults(chunk_size=150, chunk_overlap=20)
    image_documents = [image_doc for image_doc in documents if type(image_doc) is ImageDocument]
    text_documents = [text_doc for text_doc in documents if type(text_doc) is Document]
    image_nodes = parser.get_nodes_from_documents(image_documents)
    text_nodes = document_segmentation(None, text_documents) # 需要通过自定义规则切分文档, 因此需要将图片和文本节点分开处理
    # 合并图片和文本节点
    refined_nodes = []
    refined_nodes.extend(image_nodes)
    refined_nodes.extend(text_nodes)
    # 建立多模态语料库
    semantic_retriever, keywords_retriever = build_multi_modal_corpus(
        nodes=refined_nodes,
        category=category, 
        embed_model=mm_embed_model, 
        milvus_dense_collection_name=milvus_dense_collection_name,
        milvus_image_collection_name=milvus_image_collection_name,
        milvus_sparse_collection_name=milvus_sparse_collection_name,
        milvus_uri=milvus_uri,
        use_milvus=True,
        semantic_retriever_top_k=5,
        image_retriever_top_k=5,
        keywords_retriever_top_k=5,
    )

    # 等待索引构建完成后再测试查询
    time.sleep(2)

    # 语义索引测试
    res_nodes = semantic_retriever.retrieve("吸引-结队-排斥规则是什么")
    print("\n语义检索结果:")
    for node in res_nodes:
        if type(node.node) is ImageNode:
            print(node.metadata['file_path'])
        else:
            print(node.get_content())

    # 字面索引测试(用稀疏向量模拟)
    res_nodes = keywords_retriever.retrieve("吸引-结队-排斥规则是什么")
    print("\n字面检索结果:")
    for node in res_nodes:
        print(node.get_content())

if __name__ == "__main__":
    #run_text_embedding_demo()
    run_multi_modal_embedding_demo()