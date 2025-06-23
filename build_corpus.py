# 构建语料库: 向量库+字面库

from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.readers.file import FlatReader
from dotenv import load_dotenv
import os
import textwrap
from llama_index.vector_stores.elasticsearch import AsyncBM25Strategy
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
import time

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
nodes = document_segmentation(doc_path)

# 向量检索的向量库
milvus_dense_store = MilvusVectorStore(
    uri=MILVUS_URI,
    dim=1024,  # 向量维度需与嵌入模型匹配
    collection_name=MILVUS_DENSE_COLLECTION_NAME,
    overwrite=True,
)
milvus_dense_storage_context = StorageContext.from_defaults(vector_store=milvus_dense_store)
milvus_dense_index = VectorStoreIndex(
    nodes=nodes,
    embed_model=embed_model,
    storage_context=milvus_dense_storage_context,
    show_progress=True
)

# 稀疏检索的向量库(接近于字面检索)
Settings.embed_model = None # 显示禁用嵌入模型
milvus_sparse_store = MilvusVectorStore(
    uri=MILVUS_URI,
    enable_dense=False,  # 不使用稠密向量
    enable_sparse=True,  # 启用稀疏向量
    sparse_embedding_function=BM25BuiltInFunction(),
    collection_name=MILVUS_SPARSE_COLLECTION_NAME,
    overwrite=True,
)
milvus_sparse_storage_context = StorageContext.from_defaults(vector_store=milvus_sparse_store)
milvus_sparse_index = VectorStoreIndex(
    nodes=nodes,
    storage_context=milvus_sparse_storage_context,
    show_progress=True,
)

# 等待索引构建完成后再测试查询
time.sleep(2)

# 稠密向量索引测试
milvus_dense_retriever = milvus_dense_index.as_retriever(similarity_top_k=5)
res_nodes = milvus_dense_retriever.retrieve("吸引-结队-排斥规则是什么")
print("\n稠密向量(正常向量)检索结果:")
for node in res_nodes:
    print(node.get_content())

# 稀疏向量索引测试(接近字面检索)
milvus_sparse_retriever = milvus_sparse_index.as_retriever(
    vector_store_query_mode="sparse",
    similarity_top_k=5
)
res_nodes = milvus_sparse_retriever.retrieve("吸引-结队-排斥规则是什么")
print("\n稀疏向量(字面)检索结果:")
for node in res_nodes:
    print(node.get_content())