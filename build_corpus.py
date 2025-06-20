from pathlib import Path
import chromadb
from milvus import default_server
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.file import FlatReader

DASHSCOPE_API_KEY = "sk-4934b9ab077448e594033f2c95bc41c8"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
doc_path = "./converted_docs/uav_swarm_page23-25/uav_swarm_page23-25.md"
LOCAL = 1

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
    model_name="qwen-plus"
)
# 配置嵌入模型（替换为您的API Key）
embed_model = DashScopeEmbedding(
    api_key=DASHSCOPE_API_KEY,
    model_name="text-embedding-v4",
    embed_batch_size=10
)

# 加载文档
nodes = document_segmentation(doc_path)

# 构建语料库: 向量库+字面库(milvus+elasticsearch 或 chromadb+whoosh)
if 1 == LOCAL:
    # 初始化Milvus本地服务（生产环境建议使用独立部署）
    default_server.start()
    vector_store = MilvusVectorStore(
        uri="http://localhost:19530",  # Milvus默认端口
        dim=1024,  # 向量维度需与嵌入模型匹配
        collection_name="product_reviews"  # 自定义集合名称
    )
    # TODO(wangjintao): 第一次构建完向量索引后不需要重复构建
    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
        vector_store=vector_store
    )
else:
    # 创建chromadb向量检索库
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("demo")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # 创建索引
    if (0 == chroma_collection.count()):
        # 若集合为空, 创建新索引
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model,
            vector_store=vector_store
        )
    else:
        # 若集合非空, 加载已有索引
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
    # 创建whoosh字面检索库


# 查询示例
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("研究无人机集群的态势感知问题, 首要任务是什么？")
print(response)

if 1 == LOCAL:
    # 关闭Milvus服务（开发时可选）
    default_server.stop()