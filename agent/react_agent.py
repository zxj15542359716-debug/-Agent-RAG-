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
