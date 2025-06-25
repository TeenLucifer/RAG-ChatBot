from typing import TYPE_CHECKING, Tuple, List, Sequence, Union
if TYPE_CHECKING:
    from llama_index.core.bridge.langchain import Embeddings as LCEmbeddings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import BaseNode
from llama_index.core.base.base_retriever import BaseRetriever
from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.models import create_model_dict
from marker.output import text_from_rendered
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.multi_modal_llms.dashscope import DashScopeMultiModal
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.readers.file import FlatReader
from llama_index.retrievers.bm25 import BM25Retriever
from pymilvus import connections, utility
from nltk.corpus import stopwords
import jieba

EmbedType = Union[BaseEmbedding, "LCEmbeddings", str]

# 解析pdf文件为markdown格式
def parse_pdf(
    uploaded_file: str,
    category: str = "TECHNOLOGY"
) -> Document:
    # 配置marker pdf参数
    config = {
        "use_llm": True,
        "llm_service": "marker.services.openai.OpenAIService",
        "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "openai_model": "qwen-vl-plus",
        "openai_api_key": "sk-4934b9ab077448e594033f2c95bc41c8",
        "max_pages": 50  # 限制处理页数
    }
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config=config_parser.generate_config_dict(),
        llm_service=config_parser.get_llm_service()
    )
    # 解析pdf中的文字(markdown格式)和图片内容
    # TODO(wangjintao): 如何把文字和图片同时转换出来
    rendered = converter(uploaded_file)
    doc_text, _, doc_images = text_from_rendered(rendered)

    # 提取文件名和元数据
    filename = uploaded_file.split("/")[-1] if "/" in uploaded_file else uploaded_file
    metadata = {
        "file_name": filename,
        "source": "upload",
        "category": category,
    }
    # 把文字部分转换为llamaindex的document
    document = Document(
        text=doc_text,
        metadata=metadata,
        exclude_llm_metadata_keys=["file_name", "source"],
    )
    return document

# 切分文档
def segment_documents(
    documents: Sequence[Document],
) -> List[BaseNode]:
    # 一次切分: 用llamaindex的markdown解析器按标题层级切分
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    # 二次切分: 中文按照段落进行切分, 公式特殊处理
    # TODO(wangjintao): 公式和图片特殊切分
    text_splitter = SentenceSplitter(
        chunk_size=200,
        chunk_overlap=10,
        separator="。",
        paragraph_separator="\r\n\r\n",# 针对中文优化的段落切分（按句号/换行符分割，避免截断句子）
        secondary_chunking_regex="。！？\n",
    )

# 建立语料库(向量检索和字面检索)
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

def process_documents(
    uploaded_files: List[str],
    category: str,
    embed_model: DashScopeEmbedding,
    mllm: DashScopeMultiModal,
    milvus_uri: str = "http://localhost:19530",
) -> None:
    # 解析pdf文件
    documents = []
    for file in uploaded_files:
        document = parse_pdf(file)
        documents.extend(document)

    # 对文档进行切分
    refined_nodes = segment_documents(documents=documents)

    # 建立语料库
    semantic_retriever, keyword_retriever = build_corpus(
        refined_nodes=refined_nodes,
        category=category,
        embed_model=embed_model,
        milvus_uri=milvus_uri,
    )

    return semantic_retriever, keyword_retriever