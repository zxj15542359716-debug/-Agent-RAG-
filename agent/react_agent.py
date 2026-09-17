from langchain.agents import create_agent
from model.fectory import chat_model
from utils.Prompt_loader import load_system_prompts
#【重构】只注册生产工具；mock 工具（get_weather 等）已迁至 agent/tools/mock_tools.py，生产 Agent 不使用
from agent.tools.agent_tools import (rag_summarize,get_user_id,get_current_month,fetch_external_data,fill_context_for_report,query_warranty)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize,get_user_id,get_current_month,fetch_external_data,fill_context_for_report,query_warranty],
            middleware=[monitor_tool,log_before_model,report_prompt_switch],
            #【第2步·2.1】关掉子图自带的 checkpointer：会话持久化统一由父图
            # （agent/orchestration/graph.py + checkpoint.py）承担——内层再存一份会
            # 产生两套历史（父图只留 user+assistant、子图留全量含工具消息），
            # 既浪费空间又让"到底以谁为准"变得含糊
            checkpointer=False,
        )

    def execute_stream(self,query:str):
        input_dict = {
            "messages":[
                {"role":"user","content":query},
            ]
        }
        #第三个参数context就是上下文runtime中的信息，是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict,stream_mode="values",context={"report":False}):
            latest_message=chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip()+"\n"

if __name__=="__main__":
    agent=ReactAgent()

    for chunk in agent.execute_stream("在我所在的地区怎么选键盘"):
        print(chunk,end="",flush=True)


# ============================================================================================
# 【第 2 步 · 2.1 说明】本文件在第 2 步的改动（子 Agent 让出持久化职责）
# --------------------------------------------------------------------------------------------
# 改动点：create_agent 增加一个参数 checkpointer=False，其余（模型/工具/中间件/系统提示词）
# 一字未动。
# 为什么：第 2 步起，会话历史统一由父图（agent/orchestration/graph.py + checkpoint.py）
# 的 SqliteSaver 持久化。内层 Agent 若也开 checkpointer，会同时存在两套历史——
# 父图只存 user+assistant（口径同改造前），子图存全量（含工具调用中间消息），
# 空间翻倍且"以谁为准"变得含糊。
# 为什么保留这个类：它仍是日常问答分支的实际执行体（agent/orchestration/nodes.py 的
# normal 节点命令式调用它），也仍是唯一被第 1 步评测覆盖过的问答链路。
# 验证：.venv/Scripts/python.exe -m agent.react_agent（自检，不联网），
# 或 .venv/Scripts/python.exe -m pytest tests/test_orchestrator_graph.py
# ============================================================================================
