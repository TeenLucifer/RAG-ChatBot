# 检索功能(向量+字面检索 配合问句改写实现多路召回)

import os
from typing import List, Union, Sequence
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate, get_response_synthesizer, Settings
from llama_index.core.schema import ImageNode, ImageDocument
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.multi_modal_llms.dashscope import DashScopeMultiModal
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.core.base.base_retriever import BaseRetriever
import dashscope

# 查询扩展(子问题拆解/意图识别/HyDE)
def expand_query(
    use_multi_modal: bool,
    dashscope_api_key: str,
    model_name: str,
    query: str,
    sub_query_num: int = 3,
    hypothesis_num: int = 1
) -> List[str]:
    # 子问题拆解
    query_gen_str = (
        "你是一个智能助手，能将用户的查询改写成多角度的具体问题。\n"
        "请为以下查询生成 {num_queries} 个相关问题（每行一个）：\n"
        "原查询：{query}\n"
        '生成问题："""'
    )
    if True == use_multi_modal:
        query_gen_message = [
            {
                "role": "system",
                "content": [{"text": query_gen_str.format(num_queries=sub_query_num, query=query)}]
            },
            {
                "role": "user",
                "content": [{"text": query}]
            }
        ]
        response=dashscope.MultiModalConversation.call(
            api_key=dashscope_api_key,
            model=model_name,
            messages=query_gen_message,
        )
        sub_queries = response.output.choices[0].message.content[0]["text"].split("\n")
    else:
        query_gen_message = [
            {
                "role": "system",
                "content": query_gen_str.format(num_queries=sub_query_num, query=query)
            },
            {
                "role": "user",
                "content": query
            }
        ]
        response=dashscope.Generation.call(
            api_key=dashscope_api_key,
            model=model_name,
            messages=query_gen_message,
        )
        sub_queries = response.output["text"].split("\n")

    # TODO(wangjintao): 意图识别, 待捋清楚文档归类逻辑后实现
    # HyDE
    hyde_str = (
        "你是一个智能助手, 能够根据用户的查询撰写 {num_hypothesises} 段简要问题解答，\n"
        "要求：\n"
        "1. 逻辑清晰，使用专业标书\n"
        "2. 用简短的语句回答\n"
        "3. 不同段落的回答之间用制表符(\\t)分隔\n"
        "用户查询：{query}\n"
        '生成回答："""\n'
    )

    if True == use_multi_modal:
        hyde_message = [
            {
                "role": "system",
                "content": [{"text": hyde_str.format(num_hypothesises=hypothesis_num, query=query)}]
            },
            {
                "role": "user",
                "content": [{"text": query}]
            }
        ]
        response=dashscope.MultiModalConversation.call(
            api_key=dashscope_api_key,
            model=model_name,
            messages=hyde_message,
        )
        hypothesises = response.output.choices[0].message.content[0]["text"].split("\n")
    else:
        hyde_message = [
            {
                "role": "system",
                "content": hyde_str.format(num_hypothesises=hypothesis_num, query=query)
            },
            {
                "role": "user",
                "content": query
            }
        ]
        response=dashscope.Generation.call(
            api_key=dashscope_api_key,
            model=model_name,
            messages=hyde_message,
        )
        hypothesises = response.output["text"].split("\t")

    return sub_queries, hypothesises 

def retrieve_documents(
    query: str,
    llm: DashScope,
    text_embed_model: DashScopeEmbedding,
    rerank_model: DashScopeRerank,
    semantic_retriever: BaseRetriever,
    keywords_retriever: BaseRetriever,
    dashscope_api_key: str,
    dashscope_llm_model_name: str,
) -> RESPONSE_TYPE:
    # 查询扩写
    sub_queries, hypothesises = expand_query(
        use_multi_modal=False,
        dashscope_api_key=dashscope_api_key,
        model_name=dashscope_llm_model_name,
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
    filtered_nodes = [node for node in reranked_nodes if node.score > 0.2]

    if not filtered_nodes:
        return "没有找到相关内容"
    else:
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            verbose=True,
            streaming=True
        )
        response = response_synthesizer.synthesize(
            query=query,
            nodes=filtered_nodes
        )
        return response