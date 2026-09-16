#模型工厂
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

from utils.config_handler import rag_config, require_env

class BaseModelFactory(ABC):
    @abstractmethod
    def get_model(self)->Optional[Embeddings | BaseChatModel]:
        pass

#聊天工厂
class ChatModelFactory(BaseModelFactory):
    def get_model(self)->Optional[Embeddings | BaseChatModel]:
        #【修改】原来缺失 Key 时静默返回 None（服务照常启动、用户一提问才报错）；
        #改为缺失即抛错（fail-fast），配置问题在启动阶段暴露
        api_key = require_env("DEEPSEEK_API_KEY")
        return ChatOpenAI(
            model=rag_config["chat_model_name"],
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )

#嵌入工厂
class DashScopeModelFactory(BaseModelFactory):
    def get_model(self)->Optional[Embeddings | BaseChatModel]:
        #【修改】同 ChatModelFactory：缺失 Key 即抛错（fail-fast）
        api_key = require_env("DASHSCOPE_API_KEY")
        return DashScopeEmbeddings(
            model=rag_config["embedding_model_name"],
            dashscope_api_key=api_key
        )

chat_model = ChatModelFactory().get_model()
embed_model = DashScopeModelFactory().get_model()