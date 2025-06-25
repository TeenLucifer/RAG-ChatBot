# 构建语料库: 向量库+字面库
import os
import time
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.readers.file import FlatReader
from llama_index.retrievers.bm25 import BM25Retriever
from pymilvus import connections, utility
from nltk.corpus import stopwords
import jieba
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
DASHSCOPE_LLM_MODEL_NAME = os.getenv("DASHSCOPE_LLM_MODEL_NAME")
DASHSCOPE_EMBED_MODEL_NAME = os.getenv("DASHSCOPE_EMBED_MODEL_NAME")

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_DENSE_COLLECTION_NAME = os.getenv("MILVUS_DENSE_COLLECTION_NAME")
MILVUS_SPARSE_COLLECTION_NAME = os.getenv("MILVUS_SPARSE_COLLECTION_NAME")

doc_path = "./converted_docs/uav_swarm_page23-25/uav_swarm_page23-25.md"
MILVUS_URI = "http://" + MILVUS_HOST + ":" + MILVUS_PORT
milvus_uri = MILVUS_URI
category = "TECHNOLOGY"

def document_segmentation(doc_path):
    markdown_file_path = doc_path

    # 加载Markdown文档
    documents = FlatReader().load_data(Path(markdown_file_path))
    # 初始化解析器
    parser = MarkdownNodeParser()
    # 按标题层级切分
    nodes = parser.get_nodes_from_documents(documents)

    # 二次切分：中文按照段落进行切分, 公式特殊处理
    # 针对中文优化的段落切分（按句号/换行符分割，避免截断句子）
    text_splitter = SentenceSplitter(
        chunk_size=200,
        chunk_overlap=10,
        separator="。",
        paragraph_separator="\r\n\r\n",
        secondary_chunking_regex="。！？\n",
    )
    refined_nodes = text_splitter.get_nodes_from_documents(nodes)
    return refined_nodes

# 建立语料库
def build_corpus(
    refined_nodes: List[BaseNode],
    category: str,
    embed_model: DashScopeEmbedding,
    milvus_uri: str = "http://localhost:19530",
) -> Tuple[BaseRetriever, BaseRetriever]:
    # 检查milvus数据库能否连接上, 若未连接上, 则直接创建向量检索和字面检索库, 不再创建本地存储
    milvus_connected = False
    try:
        connections.connect(uri=milvus_uri)
        version = utility.get_server_version() # 尝试获取Milvus的版本以确认连接
        milvus_connected = True
    except Exception as e:
        milvus_connected = False

    if True == milvus_connected:
        # 能连接上数据库, 用milvus建立语义检索和字面检索的向量库(用稀疏量模拟)
        milvus_dense_collection_name = "RAG_CHATBOT_" + category + "_DENSE_COLLECTION"
        milvus_sparse_collection_name = "RAG_CHATBOT_" + category + "_SPARSE_COLLECTION"
        # 语义检索向量库
        milvus_dense_store = MilvusVectorStore(
            uri=milvus_uri,
            dim=1024,  # 向量维度需与嵌入模型匹配
            collection_name=milvus_dense_collection_name,
            overwrite=True,
        )
        milvus_dense_storage_context = StorageContext.from_defaults(vector_store=milvus_dense_store)
        milvus_dense_index = VectorStoreIndex(
            nodes=refined_nodes,
            embed_model=embed_model,
            storage_context=milvus_dense_storage_context,
            show_progress=True
        )
        semantic_retriever = milvus_dense_index.as_retriever(
            similarity_top_k=5,
            vector_store_query_mode="default",
        )

        # 字面检索向量库(用稀疏向量模拟)
        bm25_function = BM25BuiltInFunction(
            analyzer_params={
                "type": "chinese",
                "tokenizer": "jieba",  # 中文文档需要使用 Jieba 中文分词器
                "filter": [
                    {"type": "stop", "stop_words": ["的", "了", "是"]},  # 中文停用词
                    {"type": "length", "min": 2, "max": 20},           # 过滤超短/超长词
                ],
            },
            enable_match=True,
        )
        Settings.embed_model = None # 建立稀疏向量库时需要显式禁用嵌入模型
        milvus_sparse_store = MilvusVectorStore(
            uri=milvus_uri,
            enable_dense=False,  # 不使用稠密向量
            enable_sparse=True,  # 启用稀疏向量
            sparse_embedding_function=bm25_function,
            collection_name=milvus_sparse_collection_name,
            overwrite=True,
        )
        milvus_sparse_storage_context = StorageContext.from_defaults(vector_store=milvus_sparse_store)
        milvus_sparse_index = VectorStoreIndex(
            nodes=refined_nodes,
            storage_context=milvus_sparse_storage_context,
            show_progress=True,
        )
        keyword_retriever = milvus_sparse_index.as_retriever(
            vector_store_query_mode="sparse",
            similarity_top_k=5,
        )
    else:
        # 连接不上数据库直接创建语料库, 不做本地存储
        vector_retrieve_index = VectorStoreIndex( # 创建向量索引
            nodes=refined_nodes,
            embed_model=embed_model,
            show_progress=True,
        )
        semantic_retriever = vector_retrieve_index.as_retriever(
            similarity_top_k=5,
            vector_store_query_mode="default",
        )
        keyword_retriever = BM25Retriever.from_defaults( # 创建字面索引
            nodes=refined_nodes,
            similarity_top_k=5,
            tokenizer=lambda text: [
                token for token in jieba.cut(text)
                if token not in set(stopwords.words('chinese'))
            ],
        )
    return semantic_retriever, keyword_retriever

# 配置大语言模型
llm = DashScope(
    api_key=DASHSCOPE_API_KEY,
    model_name=DASHSCOPE_LLM_MODEL_NAME,
)
# 配置嵌入模型
embed_model = DashScopeEmbedding(
    api_key=DASHSCOPE_API_KEY,
    model_name=DASHSCOPE_EMBED_MODEL_NAME,
    embed_batch_size=10
)

# 加载文档
refined_nodes = document_segmentation(doc_path)

# 建立语料库
semantic_retriever, keyword_retriever = build_corpus(refined_nodes, category, embed_model, milvus_uri)

# 等待索引构建完成后再测试查询
time.sleep(2)

# 稠密向量索引测试
#milvus_dense_retriever = milvus_dense_index.as_retriever(similarity_top_k=5)
res_nodes = semantic_retriever.retrieve("吸引-结队-排斥规则是什么")
print("\n稠密向量(正常向量)检索结果:")
for node in res_nodes:
    print(node.get_content())

# 稀疏向量索引测试(接近字面检索)
#milvus_sparse_retriever = milvus_sparse_index.as_retriever(
#    vector_store_query_mode="sparse",
#    similarity_top_k=5
#)
res_nodes = keyword_retriever.retrieve("吸引-结队-排斥规则是什么")
print("\n稀疏向量(字面)检索结果:")
for node in res_nodes:
    print(node.get_content())