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

from utils.config_handler import orchestration_config, rag_config, require_env
#【第2步·2.4】用量记账入口（向量化消耗由 UsageRecordingEmbeddings 显式入账）
from utils.usage_ledger import record_embedding, usage_tokens

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
            base_url="https://api.deepseek.com/v1",
            #【第2步·2.4】流式返回 usage：本项目的 base_url 是自定义域名，
            #langchain-openai 只在"默认 base_url"下自动开启 stream_usage，这里必须显式打开，
            #否则流式回答拿不到 token 用量（记账会退化成"记 0"）
            stream_usage=bool(orchestration_config["usage"].get("stream_usage", True)),
        )

#嵌入工厂
class DashScopeModelFactory(BaseModelFactory):
    def get_model(self)->Optional[Embeddings | BaseChatModel]:
        #【修改】同 ChatModelFactory：缺失 Key 即抛错（fail-fast）
        api_key = require_env("DASHSCOPE_API_KEY")
        #【第2步·2.4】改用带记账的子类（见 UsageRecordingEmbeddings 说明）
        return UsageRecordingEmbeddings(
            model=rag_config["embedding_model_name"],
            dashscope_api_key=api_key
        )

#DashScope 向量模型的分批上限（官方文档：v4 单批最多 10 条）
_EMBED_BATCH_SIZE = 10

class UsageRecordingEmbeddings(DashScopeEmbeddings):
    """在 DashScopeEmbeddings 上补一层用量记账（第2步·2.4）。

    为什么需要：langchain_community 的实现取到了 resp 却丢掉了 resp.usage（源码里
    只用了 resp.output["embeddings"]），于是"嵌入花了多少 token"在账上永远是 0。
    为什么用子类重写而不是打补丁改第三方源码：改 site-packages 会在重装依赖后丢失，
    也不利于课程交付时说明"我们改了什么"。
    兼容性：分批大小、失败重试、非 200 抛错的行为与父类实现保持一致（父类同样是
    "按 BATCH_SIZE 分批 + tenacity 重试"），只是顺带把每批的 usage 记进账本。
    """

    def _embed_batches(self, inputs: list[str], text_type: str) -> list[dict]:
        import time
        result: list[dict] = []
        for start in range(0, len(inputs), _EMBED_BATCH_SIZE):
            batch = inputs[start:start + _EMBED_BATCH_SIZE]
            last_err: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    resp = self.client.call(input=batch, text_type=text_type, model=self.model)
                    if resp.status_code == 200:
                        result += resp.output["embeddings"]
                        #记账：拿得到 usage 就记，拿不到就跳过（绝不影响向量化本身）
                        record_embedding(usage_tokens(getattr(resp, "usage", None)))
                        last_err = None
                        break
                    last_err = RuntimeError(
                        f"DashScope 向量化失败：status_code={resp.status_code} "
                        f"code={getattr(resp, 'code', '')} message={getattr(resp, 'message', '')}")
                except Exception as e:      #网络类错误：退避重试
                    last_err = e
                time.sleep(min(2 ** attempt, 4))
            if last_err is not None:
                raise last_err
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [item["embedding"] for item in self._embed_batches(list(texts), "document")]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batches([text], "query")[0]["embedding"]

chat_model = ChatModelFactory().get_model()
embed_model = DashScopeModelFactory().get_model()