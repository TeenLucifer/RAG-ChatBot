import os
from dotenv import load_dotenv
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.llms.dashscope import DashScope
from .dashscope_embedding import DashScopeEmbedding

class RagConfig:
    def __init__(self) -> None:
        load_dotenv()
        self.dashscope_api_key               = os.getenv("DASHSCOPE_API_KEY")
        self.dashscope_base_url              = os.getenv("DASHSCOPE_BASE_URL")
        self.dashscope_llm_model_name        = os.getenv("DASHSCOPE_LLM_MODEL_NAME")
        self.dashscope_mllm_model_name       = os.getenv("DASHSCOPE_MLLM_MODEL_NAME")
        self.dashscope_text_embed_model_name = os.getenv("DASHSCOPE_TEXT_EMBED_MODEL_NAME")
        self.dashscope_mm_embed_model_name   = os.getenv("DASHSCOPE_MM_EMBED_MODEL_NAME")
        self.milvus_uri                      = os.getenv("MILVUS_URI")
        self.collection_map_file_name        = os.getenv("COLLECTION_MAP_FILE_NAME")
        #self.milvus_dense_collection_name    = "RAG_CHATBOT_" + "TEXT_EMBED" + "_DENSE_COLLECTION"
        #self.milvus_sparse_collection_name   = "RAG_CHATBOT_" + "TEXT_EMBED" + "_SPARSE_COLLECTION"
        # 配置大语言模型
        self.llm = DashScope(
            api_key=self.dashscope_api_key,
            model_name=self.dashscope_llm_model_name,
        )
        # 配置嵌入模型
        self.text_embed_model = DashScopeEmbedding(
            api_key=self.dashscope_api_key,
            model_name=self.dashscope_text_embed_model_name,
            embed_batch_size=10
        )
        # 配置重排序模型
        self.rerank_model = DashScopeRerank(
            api_key=self.dashscope_api_key,
            model="gte-rerank-v2",
            top_n=5
        )