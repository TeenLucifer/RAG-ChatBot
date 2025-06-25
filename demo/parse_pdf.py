# 用于解析pdf, 转换为Markdown格式

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
import os

pdf_file_path = "./rag_docs/uav_swarm_page23-25.pdf"
pdf_file_path = "./rag_docs/uav_swarm_page34-36.pdf"
# 获取PDF文件名（不带扩展名）
pdf_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
markdown_file_path = "./converted_docs/" + pdf_name

# 配置Qwen-VL参数
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
rendered = converter(pdf_file_path)
text, _, images = text_from_rendered(rendered)

# 输出目录
os.makedirs(markdown_file_path, exist_ok=True)

# 保存图片
for img_path, img_data in images.items():
    img_data.save(f"{markdown_file_path}/{img_path}")

# 构建Markdown文件路径
output_md_path = os.path.join(markdown_file_path, pdf_name + ".md")
# 保存为Markdown文件
with open(output_md_path, "w", encoding="utf-8") as f:
    f.write(text)