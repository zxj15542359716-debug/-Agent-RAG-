#总结服务
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.fectory import chat_model
from rag.hybrid_retriever import HybridRetriever, RetrievedChunk
from utils.Prompt_loader import load_rag_prompts


class RagSummarizeService(object):
    def __init__(self):
        # 混合检索器（内部持有共享的 FAISS 向量库实例与 chunk 账本，BM25 索引惰性构建）
        self.retriever = HybridRetriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[RetrievedChunk]:
        """检索：混合流水线（两路召回 → 加权融合 → 精排），参数见 config/retrieval.yml"""
        return self.retriever.search(query)

    def rag_summarize(self, query: str) -> tuple[str, list[dict]]:
        """检索 + 总结，返回 (答案, 来源列表)。"""
        hits = self.retriever_docs(query)

        # 拼上下文：编号从 1 起，与返回的来源列表严格对齐（溯源可解引用的前提）
        context_parts: list[str] = []
        sources: list[dict] = []
        for i, hit in enumerate(hits, 1):
            context_parts.append(
                f"[参考资料{i}]（来源：{hit.source}·{hit.title}）：{hit.text}")
            sources.append(hit.to_source_item(i))
        context = "\n".join(context_parts)

        answer = self.chain.invoke(
            {
                "input": query,     #用户查询
                "context": context, #参考资料（条目级完整文本，替代原先的 200 字碎片）
            }
        )
        return answer, sources


if __name__ == "__main__":
    #运行方式：cd 项目根 && .venv/Scripts/python.exe -m rag.rag_service
    #演示：一次完整问答 + 打印结构化来源（检索自身不依赖生成模型；此处会调用一次总结模型）
    rag = RagSummarizeService()
    answer, sources = rag.rag_summarize("哪种键盘适合打射击游戏")
    print(answer)
    print("\n参考来源：")
    for s in sources:
        print(f"  [{s['n']}] {s['source']}#{s['entry']} {s['title']}  score={s['score']}")