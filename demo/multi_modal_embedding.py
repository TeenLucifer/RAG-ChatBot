from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode, ImageNode
from llama_index.embeddings.dashscope import DashScopeEmbedding
from dashscope_embedding import DashScopeEmbedding
import dashscope
import logging
from http import HTTPStatus
import base64
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SimpleNodeParser
from matplotlib import pyplot as plt
import os
from PIL import Image
from llama_index.core.response.notebook_utils import display_source_node


def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break

def image_to_base64(image_path: str) -> str:
    # 本地图片需要以base64编码的形式传入
    image_format = image_path.split('.')[-1].lower()
    if "jpg" == image_format:
        image_format = "jpeg"
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    image_data = f"data:image/{image_format};base64,{base64_image}"
    return image_data

def batch_list(lst, batch_size):
    # 按batch_size分组
    ret_list = []
    for i in range(0, len(lst), batch_size):
        rest_num = len(lst) - i
        if rest_num < batch_size:
            ret_list.append(lst[i:])
        else:
            ret_list.append(lst[i:i + batch_size])
    return ret_list

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Create the MultiModal index
    documents = SimpleDirectoryReader("./data_wiki_mini/").load_data()
    parser = SimpleNodeParser.from_defaults(chunk_size=150, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents(documents)

    # Embedding Setting
    mm_embed_model = DashScopeEmbedding(
        api_key="sk-4934b9ab077448e594033f2c95bc41c8",
        model_name="multimodal-embedding-v1",
        embed_batch_size=10
    )
    #input_ = [{'text': "hello"}, {'text': "home"}]
    #res = mm_embed_model.get_multimodal_embedding(
    #    input_
    #)

    index = MultiModalVectorStoreIndex(
        nodes=nodes,
        #storage_context=storage_context,
        embed_model=mm_embed_model,
        image_embed_model=mm_embed_model,
        show_progress=True
    )
    retriever = index.as_retriever(
        similarity_top_k=3,
        image_similarity_top_k=3,
    )
    query = "what is tesla?"
    retrieved_nodes = retriever.retrieve(str_or_query_bundle=query)
    retrieved_image = []
    for res_node in retrieved_nodes:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)

    plot_images(retrieved_image)
    print(retrieved_image)
    #response_synthesizer = get_response_synthesizer(
    #    llm=llm,
    #    verbose=True
    #)
    #response = response_synthesizer.synthesize(
    #    query=query,
    #    nodes=retrieved_nodes,
    #)
    #print("最终回答: ", response.response)

#if __name__ == "__main__":
#    # 加载文档
#    documents = SimpleDirectoryReader("./data_wiki_mini/").load_data()
#    # 使用llama_index内置的文本切分工具随便切分
#    parser = SimpleNodeParser.from_defaults(chunk_size=300, chunk_overlap=50)
#    split_documents = parser.get_nodes_from_documents(documents)
#    documents = split_documents
#    image_batch_size = 1
#    text_batch_size = 5
#
#    # 分别筛选出ImageNode和TextNode
#    image_nodes = [doc for doc in documents if type(doc) is ImageNode]
#    text_nodes = [doc for doc in documents if type(doc) is TextNode]
#    batched_image_nodes = batch_list(image_nodes, image_batch_size)
#    batched_text_nodes = batch_list(text_nodes, text_batch_size)
#
#    for batch in batched_image_nodes:
#        input = [{"image": image_doc.image_path} for image_doc in batch]
#        response = dashscope.MultiModalEmbedding.call(
#            api_key="sk-4934b9ab077448e594033f2c95bc41c8",
#            model="multimodal-embedding-v1",
#            input=input,
#        )
#        embedding_results = [None] * len(input)
#        embedding_indexs = [None] * len(input)
#        if response.status_code == HTTPStatus.OK:
#            for emb in response.output["embeddings"]:
#                embedding_results[emb["index"]] = emb["embedding"]
#                batch[emb["index"]].embedding = emb["embedding"]
#                embedding_indexs[emb["index"]] = emb["index"]
#        else:
#            logger.error("Calling MultiModalEmbedding failed, details: %s" % response)
#
#    for batch in batched_text_nodes:
#        input = [{"text": text_doc.text} for text_doc in batch]
#        response = dashscope.MultiModalEmbedding.call(
#            api_key="sk-4934b9ab077448e594033f2c95bc41c8",
#            model="multimodal-embedding-v1",
#            input=input,
#        )
#        embedding_results = [None] * len(input)
#        embedding_indexs = [None] * len(input)
#        if response.status_code == HTTPStatus.OK:
#            for emb in response.output["embeddings"]:
#                embedding_results[emb["index"]] = emb["embedding"]
#                batch[emb["index"]].embedding = emb["embedding"]
#                embedding_indexs[emb["index"]] = emb["index"]
#        else:
#            logger.error("Calling MultiModalEmbedding failed, details: %s" % response)
#
#        embeded_nodes = []
#        for batch in batched_image_nodes:
#            embeded_nodes.extend(batch)
#        for batch in batched_text_nodes:
#            embeded_nodes.extend(batch)
#
#        VectorStoreIndex(
#        )


    # index = MultiModalVectorStoreIndex.from_documents(
    #     documents,
    #     storage_context=storage_context,
    # )

#if __name__ == "__main__":
#    # nltk.download('stopwords')
#    # 模拟上传的文件列表
#    uploaded_files = ["example.pdf"]  # 这里可以替换为实际的文件路径或文件对象
#    uploaded_files = "./rag_docs/uav_swarm_page23-25.pdf"
#
#    # 配置嵌入模型和大语言模型
#    embed_model = None  # 替换为实际的嵌入模型实例
#    mllm = None  # 替换为实际的大语言模型实例
#    milvus_uri = "tcp://localhost:19530"  # 替换为实际的Milvus URI
#
#    # 处理文档并建立向量库
#    process_documents(uploaded_files, embed_model, mllm, milvus_uri)