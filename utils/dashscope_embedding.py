"""DashScope embeddings file."""

import logging
from enum import Enum
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import ImageType

logger = logging.getLogger(__name__)


class DashScopeTextEmbeddingType(str, Enum):
    """DashScope TextEmbedding text_type."""

    TEXT_TYPE_QUERY = "query"
    TEXT_TYPE_DOCUMENT = "document"


class DashScopeTextEmbeddingModels(str, Enum):
    """DashScope TextEmbedding models."""

    TEXT_EMBEDDING_V1 = "text-embedding-v1"
    TEXT_EMBEDDING_V2 = "text-embedding-v2"
    TEXT_EMBEDDING_V3 = "text-embedding-v3"


class DashScopeBatchTextEmbeddingModels(str, Enum):
    """DashScope TextEmbedding models."""

    TEXT_EMBEDDING_ASYNC_V1 = "text-embedding-async-v1"
    TEXT_EMBEDDING_ASYNC_V2 = "text-embedding-async-v2"
    TEXT_EMBEDDING_ASYNC_V3 = "text-embedding-async-v3"


EMBED_MAX_INPUT_LENGTH = 2048
EMBED_MAX_BATCH_SIZE = 25


class DashScopeMultiModalEmbeddingModels(str, Enum):
    """DashScope MultiModalEmbedding models."""

    MULTIMODAL_EMBEDDING_ONE_PEACE_V1 = "multimodal-embedding-one-peace-v1"
    MULTIMODAL_EMBEDDING_V1 = "multimodal-embedding-v1"

def convert_input(input_data: Union[List[str], List[Dict]]) -> List[Dict]:
    """
    将输入转换为 List[Dict] 格式，其中每个 dict 包含 'text' 键。
    
    参数:
        input_data (Union[List[str], List[Dict]]): 待转换的输入
    
    返回:
        List[Dict]: 转换后的列表，每个元素为 {'text': ...}
    """
    if not isinstance(input_data, list):
        raise TypeError("输入必须是一个列表")

    # 检查是否已经是 List[Dict] 并且每个 dict 都包含 'text' 键
    is_valid_dict_list = all(
        isinstance(item, dict) and 'text' in item 
        for item in input_data
    )
    if is_valid_dict_list:
        return input_data

    # 检查是否是 List[str]
    is_str_list = all(isinstance(item, str) for item in input_data)
    if is_str_list:
        return [{'text': item} for item in input_data]

    # 否则报错
    raise ValueError("输入必须是 List[str] 或 List[Dict]，且每个 Dict 必须包含 'text' 键")

def get_text_embedding(
    model: str,
    text: Union[str, List[str]],
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> List[List[float]]:
    """Call DashScope text embedding.
       ref: https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-api-details.

    Args:
        model (str): The `DashScopeTextEmbeddingModels`
        text (Union[str, List[str]]): text or list text to embedding.

    Raises:
        ImportError: need import dashscope

    Returns:
        List[List[float]]: The list of embedding result, if failed return empty list.
            if some of test no output, the correspond index of output is None.
    """
    try:
        import dashscope
    except ImportError:
        raise ImportError("DashScope requires `pip install dashscope")
    # 用多模态嵌入的时候请求要调用多模态的api
    dashscope_multi_modal_embedding_models = [member.value for member in DashScopeMultiModalEmbeddingModels]
    if model in dashscope_multi_modal_embedding_models:
        if isinstance(text, str):
            input = [{'text': text}]
        else:
            input = text
        input = convert_input(input_data=input)
        embedding_results = get_multimodal_text_embedding(
            model=model, input=input, api_key=api_key, kwargs=kwargs
        )
    else:
        if isinstance(text, str):
            text = [text]
        response = dashscope.TextEmbedding.call(
            model=model, input=text, api_key=api_key, kwargs=kwargs
        )
        embedding_results = [None] * len(text)
        if response.status_code == HTTPStatus.OK:
            for emb in response.output["embeddings"]:
                embedding_results[emb["text_index"]] = emb["embedding"]
        else:
            logger.error("Calling TextEmbedding failed, details: %s" % response)

    return embedding_results


def get_batch_text_embedding(
    model: str, url: str, api_key: Optional[str] = None, **kwargs: Any
) -> Optional[str]:
    """Call DashScope batch text embedding.

    Args:
        model (str): The `DashScopeMultiModalEmbeddingModels`
        url (str): The url of the file to embedding which with lines of text to embedding.

    Raises:
        ImportError: Need install dashscope package.

    Returns:
        str: The url of the embedding result, format ref:
        https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-async-api-details
    """
    try:
        import dashscope
    except ImportError:
        raise ImportError("DashScope requires `pip install dashscope")
    response = dashscope.BatchTextEmbedding.call(
        model=model, url=url, api_key=api_key, kwargs=kwargs
    )
    if response.status_code == HTTPStatus.OK:
        return response.output["url"]
    else:
        logger.error("Calling BatchTextEmbedding failed, details: %s" % response)
        return None

def get_multimodal_text_embedding(
    model: str, input: list[Dict], api_key: Optional[str] = None, **kwargs: Any
) -> List[List[float]]:
    """Call DashScope multimodal embedding.
       ref: https://help.aliyun.com/zh/dashscope/developer-reference/one-peace-multimodal-embedding-api-details.

    Args:
        model (str): The `DashScopeBatchTextEmbeddingModels`
        input (str): The input of the embedding, eg:
             [{'factor': 1, 'text': '你好'},
             {'factor': 2, 'audio': 'https://dashscope.oss-cn-beijing.aliyuncs.com/audios/cow.flac'},
             {'factor': 3, 'image': 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png'}]

    Raises:
        ImportError: Need install dashscope package.

    Returns:
        List[float]: Embedding result, if failed return empty list.
    """
    try:
        import dashscope
    except ImportError:
        raise ImportError("DashScope requires `pip install dashscope")
    response = dashscope.MultiModalEmbedding.call(
        model=model, input=input, api_key=api_key, kwargs=kwargs
    )
    embedding_results = [None] * len(input)
    if response.status_code == HTTPStatus.OK:
        for emb in response.output["embeddings"]:
            embedding_results[emb["index"]] = emb["embedding"]
            #embedding_results = emb["embedding"]
    else:
        logger.error("Calling MultiModalEmbedding failed, details: %s" % response)

    return embedding_results

def get_multimodal_embedding(
    model: str, input: list[Dict], api_key: Optional[str] = None, **kwargs: Any
) -> List[float]:
    """Call DashScope multimodal embedding.
       ref: https://help.aliyun.com/zh/dashscope/developer-reference/one-peace-multimodal-embedding-api-details.

    Args:
        model (str): The `DashScopeBatchTextEmbeddingModels`
        input (str): The input of the embedding, eg:
             [{'factor': 1, 'text': '你好'},
             {'factor': 2, 'audio': 'https://dashscope.oss-cn-beijing.aliyuncs.com/audios/cow.flac'},
             {'factor': 3, 'image': 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png'}]

    Raises:
        ImportError: Need install dashscope package.

    Returns:
        List[float]: Embedding result, if failed return empty list.
    """
    try:
        import dashscope
    except ImportError:
        raise ImportError("DashScope requires `pip install dashscope")
    response = dashscope.MultiModalEmbedding.call(
        model=model, input=input, api_key=api_key, kwargs=kwargs
    )
    embedding_results = [None] * len(input)
    if response.status_code == HTTPStatus.OK:
        for emb in response.output["embeddings"]:
            #embedding_results[emb["index"]] = emb["embedding"]
            embedding_results = emb["embedding"]
    else:
        logger.error("Calling MultiModalEmbedding failed, details: %s" % response)

    return embedding_results


class DashScopeEmbedding(MultiModalEmbedding):
    """DashScope class for text embedding.

    Args:
        model_name (str): Model name for embedding.
            Defaults to DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2.
                Options are:

                - DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1
                - DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2
        text_type (str): The input type, ['query', 'document'],
            For asymmetric tasks such as retrieval, in order to achieve better
            retrieval results, it is recommended to distinguish between query
            text (query) and base text (document) types, clustering Symmetric
            tasks such as classification and classification do not need to
            be specially specified, and the system default
            value "document" can be used.
        api_key (str): The DashScope api key.
    """

    _api_key: Optional[str] = PrivateAttr()
    _text_type: Optional[str] = PrivateAttr()

    def __init__(
        self,
        model_name: str = DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        text_type: str = "document",
        api_key: Optional[str] = None,
        embed_batch_size: int = EMBED_MAX_BATCH_SIZE,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        self._api_key = api_key
        self._text_type = text_type

    @classmethod
    def class_name(cls) -> str:
        return "DashScopeEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        emb = get_text_embedding(
            self.model_name,
            query,
            api_key=self._api_key,
            text_type="query",
        )
        if len(emb) > 0 and emb[0] is not None:
            return emb[0]
        else:
            return []

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        emb = get_text_embedding(
            self.model_name,
            text,
            api_key=self._api_key,
            text_type=self._text_type,
        )
        if len(emb) > 0 and emb[0] is not None:
            return emb[0]
        else:
            return []

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return get_text_embedding(
            self.model_name,
            texts,
            api_key=self._api_key,
            text_type=self._text_type,
        )

    # TODO: use proper async methods
    async def _aget_text_embedding(self, query: str) -> List[float]:
        """Get text embedding."""
        return self._get_text_embedding(query)

    # TODO: user proper async methods
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._get_query_embedding(query)

    def get_batch_query_embedding(self, embedding_file_url: str) -> Optional[str]:
        """Get batch query embeddings.

        Args:
            embedding_file_url (str): The url of the file to embedding which with lines of text to embedding.

        Returns:
            str: The url of the embedding result, format ref:
                 https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-async-api-details.
        """
        return get_batch_text_embedding(
            self.model_name,
            embedding_file_url,
            api_key=self._api_key,
            text_type=self._text_type,
        )

    def get_batch_text_embedding(self, embedding_file_url: str) -> Optional[str]:
        """Get batch text embeddings.

        Args:
            embedding_file_url (str): The url of the file to embedding which with lines of text to embedding.

        Returns:
            str: The url of the embedding result, format ref:
                 https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-async-api-details.
        """
        return get_batch_text_embedding(
            self.model_name,
            embedding_file_url,
            api_key=self._api_key,
            text_type=self._text_type,
        )

    def _get_image_embedding(self, img_file_path: ImageType) -> List[float]:
        """
        Embed the input image synchronously.
        """
        input = [{"image": img_file_path}]
        return get_multimodal_embedding(
            self.model_name, input=input, api_key=self._api_key
        )

    async def _aget_image_embedding(self, img_file_path: ImageType) -> List[float]:
        """
        Embed the input image asynchronously.

        """
        return self._get_image_embedding(img_file_path=img_file_path)

    def get_multimodal_embedding(
        self, input: List[Dict], auto_truncation: bool = False
    ) -> List[float]:
        """Call DashScope multimodal embedding.
        ref: https://help.aliyun.com/zh/dashscope/developer-reference/one-peace-multimodal-embedding-api-details.

        Args:
            input (str): The input of the multimodal embedding, eg:
                [{'factor': 1, 'text': '你好'},
                {'factor': 2, 'audio': 'https://dashscope.oss-cn-beijing.aliyuncs.com/audios/cow.flac'},
                {'factor': 3, 'image': 'https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png'}]

        Raises:
            ImportError: Need install dashscope package.

        Returns:
            List[float]: The embedding result
        """
        return get_multimodal_embedding(
            self.model_name,
            input=input,
            api_key=self._api_key,
            auto_truncation=auto_truncation,
        )