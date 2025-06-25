# 检索功能(向量+字面检索 配合问句改写实现多路召回)

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer, Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank

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

RERANK_CUT_OFF_SCORE = os.getenv("RERANK_CUT_OFF_SCORE", "0.5")
RERANK_CUT_OFF_SCORE = float(RERANK_CUT_OFF_SCORE)

MILVUS_URI = "http://" + MILVUS_HOST + ":" + MILVUS_PORT

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
# 配置重排序模型
rerank_model = DashScopeRerank(
    api_key=DASHSCOPE_API_KEY,
    model="gte-rerank-v2",
    top_n=3
)

Settings.llm = llm
Settings.embed_model = embed_model

# 加载稠密向量库(向量检索)
milvus_dense_vector_store = MilvusVectorStore(
    uri=MILVUS_URI,
    dim=1024,  # 向量维度需与嵌入模型匹配
    overwrite=False,
    collection_name=MILVUS_DENSE_COLLECTION_NAME
)
milvus_dense_vector_storage_context = StorageContext.from_defaults(
    vector_store=milvus_dense_vector_store
)
milvus_dense_vector_index = VectorStoreIndex.from_vector_store(
    vector_store=milvus_dense_vector_store,
    storage_context=milvus_dense_vector_storage_context,
    show_progress=True
)
dense_vector_retriever = milvus_dense_vector_index.as_retriever(
    similarity_top_k=5,
    vector_store_query_mode="default",
    alpha=0.5,
)

# 加载稀疏向量库(模拟字面检索)
bm25_function = BM25BuiltInFunction(
    analyzer_params={
        "type": "chinese",
        "tokenizer": "jieba",  # 使用 Jieba 中文分词器
        "filter": [
            {"type": "stop", "stop_words": ["的", "了", "是"]},  # 中文停用词
            {"type": "length", "min": 2, "max": 20},           # 过滤超短/超长词
        ],
    },
    enable_match=True,
)
Settings.embed_model = None # 显式禁用嵌入模型
milvus_sparse_vector_store = MilvusVectorStore(
    uri=MILVUS_URI,
    enable_dense=False,  # 不使用稠密向量
    enable_sparse=True,  # 启用稀疏向量
    sparse_embedding_function=bm25_function,
    collection_name=MILVUS_SPARSE_COLLECTION_NAME,
    overwrite=False,
)
milvus_sparse_vector_storage_context = StorageContext.from_defaults(
    vector_store=milvus_sparse_vector_store
)
milvus_sparse_vector_index = VectorStoreIndex.from_vector_store(
    vector_store=milvus_sparse_vector_store,
    storage_context=milvus_sparse_vector_storage_context,
    show_progress=True
)
sparse_vector_retriever = milvus_sparse_vector_index.as_retriever(
    vector_store_query_mode="sparse",
    similarity_top_k=5,
    alpha=0.5,
)

query = "吸引-结队-排斥规则是什么"
query_augment_prompt = "你是一个查询重写助手，将用户查询分解为多个角度的具体问题。\
          注意，你不需要对问题进行回答，只需要根据问题的字面意思进行子问题拆分，输出不要超过 3 条.\
          下面是一个简单的例子：\
          输入：RAG是什么？\
          输出：RAG的定义是什么？\
               RAG是什么领域内的名词？\
               RAG有什么特点？\
               \
          用户输入为："
query_augment_messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content=query_augment_prompt
    ),
    ChatMessage(
        role=MessageRole.USER, content=query
    )
]

# 查询扩展
responses = llm.chat(messages=query_augment_messages)
augmented_queries = responses.message.content.split("\n")

# TODO(wangjintao): 意图判断, 识别问题对应哪一类文档, 直接去对应类别的文档中搜索

# 查询示例
duplicated_retrieved_nodos = [] # 带重复节点的召回结果
for one_query in augmented_queries:
    # 稠密向量检索查询示例(向量检索)
    dense_retrieved_nodes = dense_vector_retriever.retrieve(one_query)
    # 稀疏向量检索查询示例(模拟字面检索)
    sparse_retrieved_nodes = sparse_vector_retriever.retrieve(one_query)
    # 保存多路召回的结果
    duplicated_retrieved_nodos.extend(dense_retrieved_nodes)
    duplicated_retrieved_nodos.extend(sparse_retrieved_nodes)

    #print("向量检索结果: ")
    #for node in dense_retrieved_nodes:
    #    print(node.get_content())
    #print("字面检索结果: ")
    #for node in sparse_retrieved_nodes:
    #    print(node.get_content())

# 去除重复的召回结果
seen_ids = set()
unique_retrieved_nodes = [] # 不带重复节点的召回结果
for node in duplicated_retrieved_nodos:
    if node.node_id not in seen_ids:
        seen_ids.add(node.node_id)
        unique_retrieved_nodes.append(node)

# 召回排序
reranked_nodes = rerank_model.postprocess_nodes(
    nodes=unique_retrieved_nodes,
    query_str=query
)
# 过滤掉低于阈值的节点
filtered_nodes = [node for node in reranked_nodes if node.score > RERANK_CUT_OFF_SCORE]

#for node in filtered_nodes:
#    # 获取当前节点内容
#    print("Text: ", node.get_content(), "Score: ", node.score)
#    # 溯源上级段落或文本（假设有parent_node或metadata字段）
#    parent_content = None
#    if hasattr(node, "parent_node") and node.parent_node is not None:
#        parent_content = node.parent_node.get_content()
#    elif hasattr(node, "metadata") and "parent_text" in node.metadata:
#        parent_content = node.metadata["parent_text"]
#    if parent_content:
#        print("Parent Text:", parent_content)
#    print("-" * 40)

if not filtered_nodes:
    print("没有找到相关内容")
else:
    response_synthesizer = get_response_synthesizer(
        llm=llm,
        verbose=True
    )
    response = response_synthesizer.synthesize(
        query=query,
        nodes=filtered_nodes
    )
    print("最终回答: ", response.response)