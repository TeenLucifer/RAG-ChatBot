# 用于切分转换后的Markdown文件
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import SentenceSplitter
from pathlib import Path

markdown_file_path = "./converted_docs/uav_swarm_page23-25/uav_swarm_page23-25.md"

# 切分markdown文件
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

# TODO(wangjintao): 需要进一步处理公式/图标/代码块

# 打印切分后的节点
for node in refined_nodes:
    #print(repr(node.get_content()))
    print(node.get_content())
    print("-" * 80)  # 分隔符