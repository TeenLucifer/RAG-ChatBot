import os
import json
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, List, Union
from enum import Enum
if TYPE_CHECKING:
    from llama_index.core.bridge.langchain import Embeddings as LCEmbeddings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import BaseNode, TextNode, Document
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices import MultiModalVectorStoreIndex
from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.models import create_model_dict
from marker.output import text_from_rendered
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.readers.file import FlatReader
from llama_index.retrievers.bm25 import BM25Retriever
from pymilvus import connections, utility
import jieba
from .dashscope_embedding import DashScopeEmbedding
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

EmbedType = Union[BaseEmbedding, "LCEmbeddings", str]

class RagModal(Enum):
    TEXT = "Text"
    MULTI_MODAL = "MultiModal"

class CorpusManagement:
    def __init__(self):
        self.collection_map_path = "collection_map.json"
        self.multi_modal_keyword = "MultiModal"
        self.text_modal_keyword = "Text"
        self.modal_key = "collection_modal"

        self.dense_collection_key = "dense_collection_name"
        self.sparse_collection_key = "sparse_collection_name"
        self.image_collection_key = "image_collection_name"

    def create_collection_entry(self, current_corpus_name: str, modal: RagModal) -> Tuple[str, str, str]:
        # 创建json文件, 存储知识库名称和对应的milvus collection名k
        dense_collection_name = current_corpus_name + "_dense_collection"
        sparse_collection_name = current_corpus_name + "_sparse_collection"
        image_collection_name = current_corpus_name + "_image_collection"
        if RagModal.MULTI_MODAL == modal:
            collection_entry = {
                current_corpus_name: {
                    self.modal_key: self.multi_modal_keyword,
                    self.dense_collection_key: dense_collection_name,
                    self.sparse_collection_key: sparse_collection_name,
                    self.image_collection_key: image_collection_name,
                }
            }
        elif RagModal.TEXT == modal:
            collection_entry = {
                current_corpus_name: {
                    self.modal_key: self.text_modal_keyword,
                    self.dense_collection_key: dense_collection_name,
                    self.sparse_collection_key: sparse_collection_name,
                }
            }
        if not os.path.exists(self.collection_map_path):
            with open(self.collection_map_path, "w", encoding="utf-8") as f:
                json.dump(collection_entry, f, ensure_ascii=False, indent=2)
        else:
            with open(self.collection_map_path, "r+", encoding="utf-8") as f:
                try:
                    # 若已有json数据, 则判断需要添加的collection是否存在, 若存在则更新, 若不存在则添加
                    collection_map = json.load(f)
                    if current_corpus_name not in collection_map:
                        collection_map.update(collection_entry)
                    else:
                        collection_map[current_corpus_name][self.modal_key] = collection_entry[current_corpus_name][self.modal_key]
                        collection_map[current_corpus_name][self.dense_collection_key] = collection_entry[current_corpus_name][self.dense_collection_key]
                        collection_map[current_corpus_name][self.sparse_collection_key] = collection_entry[current_corpus_name][self.sparse_collection_key]
                        if modal == RagModal.MULTI_MODAL:
                            collection_map[current_corpus_name][self.image_collection_key] = collection_entry[self.image_collection_key]
                    f.seek(0)
                    f.truncate()
                    f.write(json.dumps(collection_map, ensure_ascii=False, indent=2))
                    f.flush()
                except json.JSONDecodeError:
                    # 若加载json数据失败, 则清空文件, 添加条例
                    f.seek(0)
                    f.truncate()
                    f.write(json.dumps(collection_entry, ensure_ascii=False, indent=2))
                    f.flush()

        return dense_collection_name, sparse_collection_name, image_collection_name

    def load_collection_entry(self, selected_corpus_name: str):
        dense_collection_name: str = None
        sparse_collection_name: str = None
        image_collection_name: str = None
        if os.path.exists(self.collection_map_path):
            with open(self.collection_map_path, "r", encoding="utf-8") as f:
                collection_map = json.load(f)

            dense_collection_name = collection_map[selected_corpus_name][self.dense_collection_key]
            sparse_collection_name = collection_map[selected_corpus_name][self.sparse_collection_key]
            if self.image_collection_key in collection_map[selected_corpus_name].values():
                image_collection_name = collection_map[selected_corpus_name][self.image_collection_key]
        return dense_collection_name, sparse_collection_name, image_collection_name

    def list_corpuses(self, modal: RagModal) -> List[str]:
        existing_corpuses: list[str] = []
        if os.path.exists(self.collection_map_path):
            with open(self.collection_map_path, "r", encoding="utf-8") as f:
                try:
                    collection_map = json.load(f)
                    # 获取对应模态的知识库名
                    if modal == RagModal.TEXT:
                        existing_corpuses = [key for key, value in collection_map.items() if value[self.modal_key] == self.text_modal_keyword]
                    elif modal == RagModal.MULTI_MODAL:
                        existing_corpuses = [key for key, value in collection_map.items() if value[self.modal_key] == self.multi_modal_keyword]
                    else:
                        pass
                except:
                    pass

        return existing_corpuses

# 处理从前端上传的文件, 需要先输出到服务端本地再读取
def process_uploaded_files(uploaded_files: List[UploadedFile]):
    nodes = []

    if not os.path.exists("temp"):
        os.makedirs("temp")

    for file in uploaded_files:
        try:
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".pdf"):
                pass
            elif file.name.endswith(".docx"):
                pass
            elif file.name.endswith(".txt"):
                pass
            elif file.name.endswith(".md"):
                refined_nodes = document_segmentation(doc_path=file_path)
            else:
                continue
            nodes.extend(refined_nodes)

            os.remove(file_path)
        except Exception as e:
            st.error(f"加载文件 {file.name} 失败: {str(e)}")
            return

        return nodes

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

# 中文分词器
def bm25_chinese_tokenizer(text: str) -> List[str]:
    tokens = jieba.cut(text)
    stop_words = ["的", "了", "是"]
    filtered_tokens = [
        token for token in tokens
        if token not in stop_words and 2 <= len(token) <= 20]
    return filtered_tokens

# 文档切分
def document_segmentation(
    doc_path: str,
    documents: Document = None,
) -> List[BaseNode]:
    # 如果不传入Document对象, 则从文件路径加载文档
    if documents is None:
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
def build_text_corpus(
    nodes: List[BaseNode],
    category: str,
    embed_model: DashScopeEmbedding,
    milvus_dense_collection_name: str,
    milvus_sparse_collection_name: str,
    milvus_uri: str = "http://localhost:19530",
    use_milvus: bool = True,
    semantic_retriever_top_k: int = 10,
    keywords_retriever_top_k: int = 10,
) -> Tuple[BaseRetriever, BaseRetriever]:
    # 检查milvus数据库能否连接上, 若未连接上, 则直接创建向量检索和字面检索库, 不再创建本地存储
    milvus_connected = False
    if True == use_milvus:
        try:
            connections.connect(uri=milvus_uri)
            version = utility.get_server_version() # 尝试获取Milvus的版本以确认连接
            milvus_connected = True
        except Exception as e:
            milvus_connected = False

    if True == milvus_connected:
        # 能连接上数据库, 用milvus建立语义检索和字面检索的向量库(用稀疏量模拟)

        # 语义检索向量库
        milvus_dense_store = MilvusVectorStore(
            uri=milvus_uri,
            dim=1024,  # 向量维度需与嵌入模型匹配
            collection_name=milvus_dense_collection_name,
            overwrite=True,
        )
        milvus_dense_storage_context = StorageContext.from_defaults(vector_store=milvus_dense_store)
        milvus_dense_index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model,
            storage_context=milvus_dense_storage_context,
            show_progress=True
        )
        semantic_retriever = milvus_dense_index.as_retriever(
            similarity_top_k=semantic_retriever_top_k,
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
            nodes=nodes,
            storage_context=milvus_sparse_storage_context,
            show_progress=True,
        )
        keywords_retriever = milvus_sparse_index.as_retriever(
            vector_store_query_mode="sparse",
            similarity_top_k=keywords_retriever_top_k,
        )
    else:
        # 连接不上数据库直接创建语料库, 不做本地存储
        vector_retrieve_index = VectorStoreIndex( # 创建向量索引
            nodes=nodes,
            embed_model=embed_model,
            show_progress=True,
        )
        semantic_retriever = vector_retrieve_index.as_retriever(
            similarity_top_k=semantic_retriever_top_k,
            vector_store_query_mode="default",
        )
        keywords_retriever = BM25Retriever.from_defaults( # 创建字面索引
            nodes=nodes,
            similarity_top_k=keywords_retriever_top_k,
            tokenizer=bm25_chinese_tokenizer,
        )
    return semantic_retriever, keywords_retriever

# 从数据库加载文本语料库
def load_text_corpus(
    embed_model: DashScopeEmbedding,
    milvus_dense_collection_name: str,
    milvus_sparse_collection_name: str,
    milvus_uri: str = "http://localhost:19530",
    semantic_retriever_top_k: int = 10,
    keywords_retriever_top_k: int = 10,
) -> Tuple[BaseRetriever, BaseRetriever]:
    # 检查milvus数据库能否连接上, 若未连接上, 报错
    milvus_connected = False

    try:
        connections.connect(uri=milvus_uri)
        version = utility.get_server_version() # 尝试获取Milvus的版本以确认连接
        milvus_connected = True
    except Exception as e:
        milvus_connected = False

    if True == milvus_connected:
        # 语义检索向量库
        milvus_dense_store = MilvusVectorStore(
            uri=milvus_uri,
            dim=1024,  # 向量维度需与嵌入模型匹配
            collection_name=milvus_dense_collection_name,
            overwrite=False,
        )
        milvus_dense_storage_context = StorageContext.from_defaults(vector_store=milvus_dense_store)
        milvus_dense_index = VectorStoreIndex.from_vector_store(
            embed_model=embed_model,
            vector_store=milvus_dense_store,
            storage_context=milvus_dense_storage_context,
            show_progress=True
        )
        semantic_retriever = milvus_dense_index.as_retriever(
            similarity_top_k=semantic_retriever_top_k,
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
            overwrite=False,
        )
        milvus_sparse_storage_context = StorageContext.from_defaults(vector_store=milvus_sparse_store)
        milvus_sparse_index = VectorStoreIndex.from_vector_store(
            vector_store=milvus_sparse_store,
            storage_context=milvus_sparse_storage_context,
            show_progress=True,
        )
        keywords_retriever = milvus_sparse_index.as_retriever(
            vector_store_query_mode="sparse",
            similarity_top_k=keywords_retriever_top_k,
        )
    else:
        # TODO(wangjintao): 连接不上数据库就报错
        pass

    return semantic_retriever, keywords_retriever

# 建立多模态语料库
def build_multi_modal_corpus(
    nodes: List[BaseNode],
    category: str,
    embed_model: DashScopeEmbedding,
    milvus_dense_collection_name: str,
    milvus_image_collection_name: str,
    milvus_sparse_collection_name: str,
    milvus_uri: str = "http://localhost:19530",
    use_milvus: bool = True,
    semantic_retriever_top_k: int = 10,
    image_retriever_top_k: int = 10,
    keywords_retriever_top_k: int = 10,
) -> Tuple[BaseRetriever, BaseRetriever]:
    # 检查milvus数据库能否连接上, 若未连接上, 则直接创建向量检索和字面检索库, 不再创建本地存储
    milvus_connected = False
    if True == use_milvus:
        try:
            connections.connect(uri=milvus_uri)
            version = utility.get_server_version() # 尝试获取Milvus的版本以确认连接
            milvus_connected = True
        except Exception as e:
            milvus_connected = False

    text_nodes = [node for node in nodes if type(node) is TextNode]

    if True == milvus_connected:
        # 能连接上数据库, 用milvus建立语义检索和字面检索的向量库(用稀疏量模拟)

        # 语义检索向量库
        milvus_dense_store = MilvusVectorStore(
            uri=milvus_uri,
            dim=1024,  # 向量维度需与嵌入模型匹配
            collection_name=milvus_dense_collection_name,
            overwrite=True,
        )
        milvus_image_store = MilvusVectorStore(
            uri=milvus_uri,
            dim=1024,  # 向量维度需与嵌入模型匹配
            collection_name=milvus_image_collection_name,
            overwrite=True,
        )
        milvus_dense_storage_context = StorageContext.from_defaults(
            vector_store=milvus_dense_store,
            image_store=milvus_image_store,
        )
        milvus_dense_index = MultiModalVectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model,
            image_embed_model=embed_model,
            storage_context=milvus_dense_storage_context,
            show_progress=True
        )
        semantic_retriever = milvus_dense_index.as_retriever(
            similarity_top_k=semantic_retriever_top_k,
            image_retriever_top_k=image_retriever_top_k,
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
            nodes=text_nodes,
            storage_context=milvus_sparse_storage_context,
            show_progress=True,
        )
        keywords_retriever = milvus_sparse_index.as_retriever(
            vector_store_query_mode="sparse",
            similarity_top_k=keywords_retriever_top_k,
        )

    else:
        # 连接不上数据库直接创建语料库, 不做本地存储
        vector_retrieve_index = MultiModalVectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model,
            image_embed_model=embed_model,
            show_progress=True
        )
        semantic_retriever = vector_retrieve_index.as_retriever(
            similarity_top_k=semantic_retriever_top_k,
            vector_store_query_mode="default",
        )
        keywords_retriever = BM25Retriever.from_defaults( # 创建字面索引
            nodes=text_nodes,
            similarity_top_k=keywords_retriever_top_k,
            tokenizer=bm25_chinese_tokenizer,
        )
    return semantic_retriever, keywords_retriever

# 从数据库存储加载多模态语料库
def load_multi_modal_corpus(
    embed_model: DashScopeEmbedding,
    milvus_dense_collection_name: str,
    milvus_image_collection_name: str,
    milvus_sparse_collection_name: str,
    milvus_uri: str = "http://localhost:19530",
    semantic_retriever_top_k: int = 10,
    image_retriever_top_k: int = 10,
    keywords_retriever_top_k: int = 10,
) -> Tuple[BaseRetriever, BaseRetriever]:
    # 检查milvus数据库能否连接上, 若未连接上, 则直接创建向量检索和字面检索库, 不再创建本地存储
    milvus_connected = False
    try:
        connections.connect(uri=milvus_uri)
        version = utility.get_server_version() # 尝试获取Milvus的版本以确认连接
        milvus_connected = True
    except Exception as e:
        milvus_connected = False

    #text_nodes = [node for node in nodes if type(node) is TextNode]

    if True == milvus_connected:
        # 能连接上数据库, 用milvus建立语义检索和字面检索的向量库(用稀疏量模拟)

        # 语义检索向量库
        milvus_dense_store = MilvusVectorStore(
            uri=milvus_uri,
            dim=1024,  # 向量维度需与嵌入模型匹配
            collection_name=milvus_dense_collection_name,
            overwrite=False,
        )
        milvus_image_store = MilvusVectorStore(
            uri=milvus_uri,
            dim=1024,  # 向量维度需与嵌入模型匹配
            collection_name=milvus_image_collection_name,
            overwrite=False,
        )
        milvus_dense_storage_context = StorageContext.from_defaults(
            vector_store=milvus_dense_store,
            image_store=milvus_image_store,
        )
        milvus_dense_index = MultiModalVectorStoreIndex.from_vector_store(
            embed_model=embed_model,
            image_embed_model=embed_model,
            vector_store=milvus_dense_store,
            image_vector_store=milvus_image_store,
            #storage_context=milvus_dense_storage_context,
            show_progress=True
        )
        semantic_retriever = milvus_dense_index.as_retriever(
            similarity_top_k=semantic_retriever_top_k,
            image_retriever_top_k=image_retriever_top_k,
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
            overwrite=False,
        )
        milvus_sparse_storage_context = StorageContext.from_defaults(vector_store=milvus_sparse_store)
        milvus_sparse_index = VectorStoreIndex.from_vector_store(
            vector_store=milvus_sparse_store,
            storage_context=milvus_sparse_storage_context,
            show_progress=True,
        )
        keywords_retriever = milvus_sparse_index.as_retriever(
            vector_store_query_mode="sparse",
            similarity_top_k=keywords_retriever_top_k,
        )

    else:
        # TODO(wangjintao): 连接不上数据库就报错
        pass

    return semantic_retriever, keywords_retriever