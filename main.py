import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer, Settings
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.llms.dashscope import DashScope
from utils.dashscope_embedding import DashScopeEmbedding
from utils.doc_handler import document_segmentation, build_text_corpus, load_text_corpus, load_multi_modal_corpus
from utils.retrieve_pipline import expand_query, retrieve_documents
from pymilvus import connections, utility

from llama_index.vector_stores.milvus import MilvusVectorStore

# 加载.env文件
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
DASHSCOPE_LLM_MODEL_NAME = os.getenv("DASHSCOPE_LLM_MODEL_NAME")
DASHSCOPE_MLLM_MODEL_NAME = os.getenv("DASHSCOPE_MLLM_MODEL_NAME")
DASHSCOPE_TEXT_EMBED_MODEL_NAME = os.getenv("DASHSCOPE_TEXT_EMBED_MODEL_NAME")
DASHSCOPE_MM_EMBED_MODEL_NAME = os.getenv("DASHSCOPE_MM_EMBED_MODEL_NAME")
MILVUS_URI = os.getenv("MILVUS_URI")
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

def run_rag_chatbog():
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
        milvus_uri=MILVUS_URI,
        semantic_retriever_top_k=5,
        keywords_retriever_top_k=5,
    )

    while True:
        query = input("\n请输入相关问题（输入q退出）: ")
        if query.lower() == 'q':
            break
        response = retrieve_documents(
            query=query,
            llm=llm,
            text_embed_model=text_embed_model,
            rerank_model=rerank_model,
            semantic_retriever=semantic_retriever,
            keywords_retriever=keywords_retriever,
            dashscope_api_key=DASHSCOPE_API_KEY,
            dashscope_llm_model_name=DASHSCOPE_LLM_MODEL_NAME,
        )
        print(f"\n智能助手回答：\n{response}")


if __name__ == "__main__":
    run_rag_chatbog()