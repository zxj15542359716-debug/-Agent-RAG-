#模型工厂
import os
from abc import ABC, abstractmethod
from typing import Optional

# 解决：Embeddings、BaseChatModel 未解析引用
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

# 千问Embedding
from langchain_community.embeddings import DashScopeEmbeddings
#from langchain_dashscope import DashScopeEmbeddings
# DeepSeek 兼容openai接口，复用ChatOpenAI
from langchain_openai import ChatOpenAI

from utils.config_handler import rag_config

class BaseModelFactory(ABC):
    @abstractmethod
    def get_model(self)->Optional[Embeddings | BaseChatModel]:
        pass

#聊天工厂
class ChatModelFactory(BaseModelFactory):
    def get_model(self)->Optional[Embeddings | BaseChatModel]:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        return ChatOpenAI(
            model=rag_config["chat_model_name"],
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )

#嵌入工厂
class DashScopeModelFactory(BaseModelFactory):
    def get_model(self)->Optional[Embeddings | BaseChatModel]:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return None
        return DashScopeEmbeddings(
            model=rag_config["embedding_model_name"],
            dashscope_api_key=api_key
        )

chat_model = ChatModelFactory().get_model()
embed_model = DashScopeModelFactory().get_model()