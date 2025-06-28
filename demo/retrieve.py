# 检索功能(向量+字面检索 配合问句改写实现多路召回)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer, Settings
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from utils.dashscope_embedding import DashScopeEmbedding
from utils.doc_handler import load_text_corpus, load_multi_modal_corpus
from utils.retrieve_pipline import expand_query

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

# 文本检索的demo
def run_text_retrievel_demo():
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
        top_n=3
    )

    semantic_retriever, keywords_retriever = load_text_corpus(
        embed_model=text_embed_model,
        milvus_dense_collection_name=milvus_dense_collection_name,
        milvus_sparse_collection_name=milvus_sparse_collection_name,
        milvus_uri=milvus_uri,
        semantic_retriever_top_k=5,
        keywords_retriever_top_k=5,
    )

    query = "吸引-结队-排斥规则是什么"
    # 查询扩写
    sub_queries, hypothesises = expand_query(
        llm=llm,
        query=query,
        sub_query_num=3,
        hypothesis_num=2
    )

    duplicated_retrieved_nodos = [] # 带重复节点的召回结果
    # 子问题查询
    for sub_query in sub_queries:
        semantic_retrieved_nodes = semantic_retriever.retrieve(sub_query)
        keywords_retrieved_nodes = keywords_retriever.retrieve(sub_query)
        duplicated_retrieved_nodos.extend(semantic_retrieved_nodes)
        duplicated_retrieved_nodos.extend(keywords_retrieved_nodes)

    # hypothesis查询
    for hypothesis in hypothesises:
        semantic_retrieved_nodes = semantic_retriever.retrieve(hypothesis)
        keywords_retrieved_nodes = keywords_retriever.retrieve(hypothesis)
        duplicated_retrieved_nodos.extend(semantic_retrieved_nodes)
        duplicated_retrieved_nodos.extend(keywords_retrieved_nodes)

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
    filtered_nodes = [node for node in reranked_nodes if node.score > 0.5]

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

def run_multi_modal_retrievel_demo():
    milvus_dense_collection_name  = "RAG_CHATBOT_" + "MM_EMBED" + "_DENSE_COLLECTION"
    milvus_image_collection_name  = "RAG_CHATBOT_" + "MM_EMBED" + "_IMAGE_COLLECTION"
    milvus_sparse_collection_name = "RAG_CHATBOT_" + "MM_EMBED" + "_SPARSE_COLLECTION"

    # 配置大语言模型
    llm = DashScope(
        api_key=DASHSCOPE_API_KEY,
        model_name=DASHSCOPE_LLM_MODEL_NAME,
    )
    ## 配置嵌入模型
    #text_embed_model = DashScopeEmbedding(
    #    api_key=DASHSCOPE_API_KEY,
    #    model_name=DASHSCOPE_TEXT_EMBED_MODEL_NAME,
    #    embed_batch_size=10
    #)
    # 配置多模态嵌入模型
    mm_embed_model = DashScopeEmbedding(
        api_key=DASHSCOPE_API_KEY,
        model_name=DASHSCOPE_MM_EMBED_MODEL_NAME,
        embed_batch_size=10
    )
    # 配置重排序模型
    rerank_model = DashScopeRerank(
        api_key=DASHSCOPE_API_KEY,
        model="gte-rerank-v2",
        top_n=3
    )

    semantic_retriever, keywords_retriever = load_multi_modal_corpus(
        embed_model=mm_embed_model,
        milvus_dense_collection_name=milvus_dense_collection_name,
        milvus_image_collection_name=milvus_image_collection_name,
        milvus_sparse_collection_name=milvus_sparse_collection_name,
        milvus_uri=milvus_uri,
        semantic_retriever_top_k=5,
        image_retriever_top_k=5,
        keywords_retriever_top_k=5,
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

    # 语义检索(向量检索)
    dense_retrieved_nodes = semantic_retriever.retrieve(query)
    # 字面检索(稀疏向量模拟)
    sparse_retrieved_nodes = keywords_retriever.retrieve(query)

    for node in dense_retrieved_nodes:
        print(node.get_content(), "Score: ", node.score)
    for node in sparse_retrieved_nodes:
        print(node.get_content(), "Score: ", node.score)

    ## 查询扩展
    #responses = llm.chat(messages=query_augment_messages)
    #augmented_queries = responses.message.content.split("\n")

    ## TODO(wangjintao): 意图判断, 识别问题对应哪一类文档, 直接去对应类别的文档中搜索

    ## 查询示例
    #duplicated_retrieved_nodos = [] # 带重复节点的召回结果
    #for one_query in augmented_queries:
    #    # 稠密向量检索查询示例(向量检索)
    #    dense_retrieved_nodes = semantic_retriever.retrieve(one_query)
    #    # 稀疏向量检索查询示例(模拟字面检索)
    #    sparse_retrieved_nodes = keywords_retriever.retrieve(one_query)
    #    # 保存多路召回的结果
    #    duplicated_retrieved_nodos.extend(dense_retrieved_nodes)
    #    duplicated_retrieved_nodos.extend(sparse_retrieved_nodes)

    #    # 去除重复的召回结果
    #    seen_ids = set()
    #    unique_retrieved_nodes = [] # 不带重复节点的召回结果
    #    for node in duplicated_retrieved_nodos:
    #        if node.node_id not in seen_ids:
    #            seen_ids.add(node.node_id)
    #            unique_retrieved_nodes.append(node)

    ## 召回排序
    #reranked_nodes = rerank_model.postprocess_nodes(
    #    nodes=unique_retrieved_nodes,
    #    query_str=query
    #)
    ## 过滤掉低于阈值的节点
    #filtered_nodes = [node for node in reranked_nodes if node.score > 0.5]

    #if not filtered_nodes:
    #    print("没有找到相关内容")
    #else:
    #    response_synthesizer = get_response_synthesizer(
    #        llm=llm,
    #        verbose=True
    #    )
    #    response = response_synthesizer.synthesize(
    #        query=query,
    #        nodes=filtered_nodes
    #    )
    #    print("最终回答: ", response.response)

if __name__ == "__main__":
    run_text_retrievel_demo()
    #run_multi_modal_retrievel_demo()