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
from llama_index.multi_modal_llms.dashscope import DashScopeMultiModal
from utils.dashscope_embedding import DashScopeEmbedding
from utils.doc_handler import load_text_corpus, load_multi_modal_corpus
from utils.retrieve_pipline import expand_query
import dashscope

# 加载.env文件
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
DASHSCOPE_LLM_MODEL_NAME = os.getenv("DASHSCOPE_LLM_MODEL_NAME")
DASHSCOPE_MLLM_MODEL_NAME = os.getenv("DASHSCOPE_MLLM_MODEL_NAME")
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
        top_n=5
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
        use_multi_modal=False,
        dashscope_api_key=DASHSCOPE_API_KEY,
        model_name=DASHSCOPE_LLM_MODEL_NAME,
        #llm=llm,
        query=query,
        sub_query_num=3,
        hypothesis_num=1
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

    # 重排序
    reranked_nodes = rerank_model.postprocess_nodes(
        nodes=unique_retrieved_nodes,
        query_str=query
    )
    # 过滤掉低于阈值的节点
    filtered_nodes = [node for node in reranked_nodes if node.score > 0.4]

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

    # 配置多模态模型
    mllm = DashScopeMultiModal(
        api_key=DASHSCOPE_API_KEY,
        model_name=DASHSCOPE_MLLM_MODEL_NAME,
    )

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

    query = "吸引-结队-排斥规则是什么"
    # 查询扩写
    sub_queries, hypothesises = expand_query(
        use_multi_modal=True,
        dashscope_api_key=DASHSCOPE_API_KEY,
        model_name=DASHSCOPE_MLLM_MODEL_NAME,
        query=query,
        sub_query_num=3,
        hypothesis_num=1
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

    # 重排序
    reranked_nodes = rerank_model.postprocess_nodes(
        nodes=unique_retrieved_nodes,
        query_str=query
    )
    # 过滤掉低于阈值的节点
    filtered_nodes = [node for node in reranked_nodes if node.score > 0.4]

    if not filtered_nodes:
        print("没有找到相关内容")
    else:
        # llamaindex自带的response synthesizer不支持dashscope多模态模型
        # 需要自己写生成回答部分
        context = "\n".join(
            f"[Source {i+1}]: {node.text}"
            for i, node in enumerate(filtered_nodes)
        )
        response_gen_str = (
            "请按照以下步骤分析问题与上下文：\n"
            "1.识别关键实体及其关联关系\n"
            "2.核查不同信息源之间的逻辑矛盾\n"
            "3.综合多维度上下文信息\n"
            "4.构建结构化应答框架\n"
            "\n"
            "上下文背景：\n"
            "{context}\n"
            "\n"
            "问题：\n"
            "\n"
            "{query}\n"
            "\n"
            "生成回答："
        )
        response_gen_messages = [
            {
                "role": "system",
                "content": [{"text": response_gen_str.format(context=context, query=query)}]
            },
            {
                "role": "user",
                "content": [{"text": query}]
            }
        ]
        response=dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model=DASHSCOPE_MLLM_MODEL_NAME,
            messages=response_gen_messages,
        )
        answer = response.output.choices[0].message.content[0]["text"]
        print("最终回答: ", answer)

if __name__ == "__main__":
    run_text_retrievel_demo()
    #run_multi_modal_retrievel_demo()