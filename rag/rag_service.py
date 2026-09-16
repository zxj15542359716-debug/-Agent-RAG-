#总结服务（第1步·检索纵深改造版）：用户提问 -> 混合检索 -> 提交模型 -> 总结回复 + 溯源来源
#【第1步·1.3 改造】检索由"纯向量 k=3"改为 HybridRetriever（向量+BM25 → 融合 → 重排），
#模式与条数由 config/retrieval.yml 控制；
#【第1步·1.4 改造】rag_summarize 返回 (答案, 来源列表)：来源与提示词中"参考资料N"编号一一对应，
#经工具层写入运行时上下文，最终以 SSE sources 事件下发到前端（引用溯源）。
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
        """检索 + 总结，返回 (答案, 来源列表)。

        来源列表与提示词中的"参考资料N"编号一致（n 从 1 起），每项含：
        id（来源#条目号）/ source（文件）/ entry（条目号）/ title（标题）/ snippet（片段）/ score。
        注意：同一轮对话内多次调用本方法时，编号按"每次调用"各自从 1 计——
        前端展示按返回顺序重新罗列即可，不依赖跨调用编号的全局唯一性。
        """
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


# ============================================================================================
# 【第 1 步 · 1.3/1.4 说明】总结服务改造（本文件的改动与设计）
# --------------------------------------------------------------------------------------------
# 改动 1（1.3）：检索接入混合流水线
#   retriever_docs 由 `FAISS as_retriever(k=3)` 换成 HybridRetriever.search()：
#   向量 + BM25 两路召回 → 0.4/0.6 加权融合 → gte-rerank 精排取 5 条；
#   52 题评测实测：recall@5 0.962 → 1.000、MRR@10 0.865 → 0.950（见 eval/results/comparison.md）。
# 改动 2（1.4）：rag_summarize 返回 (答案, 来源列表)  —— 引用溯源的数据源
#   - 提示词里的 [参考资料N] 编号与 sources 列表的 n 严格对齐，保证答案中
#     "（参考资料N）"能对应到具体来源条目；
#   - 来源携带 来源文件/条目号/标题/片段/分数，前端以"参考来源"折叠列表展示；
#   - 条目级切分（1.1）让来源可以精确到"某文件的第几条"，而不是"某个 200 字碎片"。
# 已知取舍：
#   本服务内部仍会调用一次总结模型（工具返回答案文本），外层 Agent 再生成最终回答——
#   每条 RAG 问题有两次 LLM 调用。砍掉其中一次的改法（工具只回片段、由外层统一生成）
#   需要同时迁移 rag_summarize.txt 的回答规则到主提示词，属答案质量改动，
#   应先用 eval/run_answer.py 做 A/B 再动手（列入后续计划，不盲目改）。
# 验证方式：
#   python -m rag.rag_service           → 看答案与结构化来源；
#   python -m eval.run_retrieval        → 看检索指标；
#   网页侧：一次提问后浏览器控制台可见 sources 事件、气泡下方出现"参考来源"。
# ============================================================================================
