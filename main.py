import os
import bs4
import re
from langchain import hub
from langchain_community.llms import Tongyi
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank

DASHSCOPE_API_KEY = 'sk-4934b9ab077448e594033f2c95bc41c8'

# 指定模型
llm=Tongyi(
    model_name="qwen-plus",
    dashscope_api_key=DASHSCOPE_API_KEY,
    temperature=0.1
) # 用通义plus模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 指定模型版本
    dashscope_api_key=DASHSCOPE_API_KEY
) # 用通义text-embedding-v3模型嵌入
# 嵌入模型测试, 单文本向量化
query_embedding = embeddings.embed_query("自然语言处理是什么？")
print(f"向量维度：{len(query_embedding)}")  # 输出维度（如1024）
# 重排序模型用阿里云的gte-rerank-v2
compressor = DashScopeRerank(
    dashscope_api_key=DASHSCOPE_API_KEY,
    model="gte-rerank-v2",
    top_n=2
)

# 1. 解析文档
loader = PyPDFLoader("./rag_docs/china_labor_laws.pdf")
documents = loader.load()
full_text = "\n".join(doc.page_content for doc in documents)


# 2. 分割
# 定义正则表达式匹配"第X条"格式
law_separators = re.compile(r'第[一二三四五六七八九十百零]+条')     # 匹配行内的条款头
text_splitter = RecursiveCharacterTextSplitter(
    #separators=[s.pattern for s in law_separators] + ["\n\n", "\n", "。", "，", " "],
    separators=[law_separators.pattern],
    chunk_size=20,          # 每块最大字符数
    chunk_overlap=0,        # 块间重叠字符数
    length_function=len,
    is_separator_regex=True   # 启用正则模式
)
split_docs = text_splitter.create_documents([full_text]) # 分割文档

# 3. 嵌入
persist_dir = "./chroma_db"
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    # 本地已存在向量数据库，直接加载
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
else:
    # 本地不存在，重新生成并保存
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

# 4. 召回
retriever = vectorstore.as_retriever()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
) # 重排序召回器

# 辅助函数, 把多份检索结果的内容整齐地合并成一段文本，方便后续作为上下文输入给大语言模型。
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = hub.pull("rlm/rag-prompt") # langchain内部的预设模板
rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    query = input("请输入问题：")
    if query.lower() == "exit":
        break
    compressed_docs = compression_retriever.get_relevant_documents(query)
    print(compressed_docs)
    result = rag_chain.invoke(query)
    print(result)