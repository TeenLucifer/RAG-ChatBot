# 构建语料库: 向量库+字面库

from pathlib import Path
from milvus import default_server
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.readers.file import FlatReader
from dotenv import load_dotenv
import os
import textwrap

# 加载.env文件
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
DASHSCOPE_LLM_MODEL_NAME = os.getenv("DASHSCOPE_LLM_MODEL_NAME")
DASHSCOPE_EMBED_MODEL_NAME = os.getenv("DASHSCOPE_EMBED_MODEL_NAME")

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_VECTOR_COLLECTION_NAME = os.getenv("MILVUS_VECTOR_COLLECTION_NAME")
MILVUS_SPARSE_COLLECTION_NAME = os.getenv("MILVUS_SPARSE_COLLECTION_NAME")

ES_HOST = os.getenv("ES_URL")
ES_PORT = os.getenv("ES_PORT")
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME")

doc_path = "./converted_docs/uav_swarm_page23-25/uav_swarm_page23-25.md"
LOCAL = 1
MILVUS_URI = 'http://' + MILVUS_HOST + ":" + MILVUS_PORT

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

# 初始化Milvus本地服务
default_server.start()

# 向量检索的向量库
milvus_vector_store = MilvusVectorStore(
    uri=MILVUS_URI,
    dim=1024,  # 向量维度需与嵌入模型匹配
    overwrite=True,
    collection_name=MILVUS_VECTOR_COLLECTION_NAME  # 自定义集合名称
)
milvus_vector_storage_context = StorageContext.from_defaults(vector_store=milvus_vector_store)
milvus_vector_index = VectorStoreIndex(
    nodes=nodes,
    embed_model=embed_model,
    storage_context=milvus_vector_storage_context,
    show_progress=True
)

# 字面检索的向量库
milvus_sparse_vector_store = MilvusVectorStore(
    uri=MILVUS_URI,
    dim=1024,
    enable_sparse=True,
    sparse_embedding_function=BM25BuiltInFunction(),
    overwrite=True,
    collection_name=MILVUS_SPARSE_COLLECTION_NAME  # 自定义集合名称
)
milvus_sparse_storage_context = StorageContext.from_defaults(vector_store=milvus_sparse_vector_store)
milvus_sparse_index = VectorStoreIndex(
    nodes=nodes,
    storage_context=milvus_sparse_storage_context,
    embed_model=embed_model,
    show_progress=True
)

# 向量检索查询示例
milvus_vector_query_engine = milvus_vector_index.as_query_engine(
    llm=llm,
    similarity_top_k=5,
)
milvus_vector_response = milvus_vector_query_engine.query("分离-对齐-凝聚是什么？")
print("向量检索结果：")
print(milvus_vector_response)
# 显示向量检索召回的各个结果
if hasattr(milvus_vector_response, "source_nodes"):
    print("\n召回的各个结果：")
    for i, node in enumerate(milvus_vector_response.source_nodes):
        print(f"\n结果 {i+1}:")
        print(textwrap.fill(str(node.node.get_content()), 100))
        print(f"分数: {getattr(node, 'score', '无')}")
else:
    print("未找到召回结果。")

# 字面检索查询示例
milvus_sparse_query_engine = milvus_sparse_index.as_query_engine(
    llm=llm,
    vector_store_query_mode="sparse",
    similarity_top_k=5
)
milvus_sparse_response = milvus_sparse_query_engine.query("分离-对齐-凝聚是什么？")
print("\n字面检索结果：")
print(milvus_sparse_response)
# 显示字面检索召回的各个结果
if hasattr(milvus_sparse_response, "source_nodes"):
    print("\n召回的各个结果：")
    for i, node in enumerate(milvus_sparse_response.source_nodes):
        print(f"\n结果 {i+1}:")
        print(textwrap.fill(str(node.node.get_content()), 100))
        print(f"分数: {getattr(node, 'score', '无')}")
else:
    print("未找到召回结果。")

if 1 == LOCAL:
    # 关闭Milvus服务（开发时可选）
    default_server.stop()