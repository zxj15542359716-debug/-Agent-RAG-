#教学演示用 Mock 工具（隔离区）
#【重构】原 agent_tools.py 中扫地机器人售后示例遗留的随机数工具全部迁入本文件，
#仅用于教学演示，生产 Agent（agent/react_agent.py）不注册这些工具。
#注意：mock 的返回值与真实数据格式故意不匹配（get_user_id 返回 "00"~"09" 而非四位数字ID、
#get_current_month 返回 "01"~"12" 而非 YYYY-MM），可用于对比教学：靠随机数拿不到真实记录。
import random

from langchain_core.tools import tool

user_ids = ["00","01","02","03","04","05","06","07","08","09"]
month_arr = ["01","02","03","04","05","06","07","08","09","10","11","12"]

@tool(description="获取城市天气，返回字符串")
def get_weather(city:str)->str:
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级"

@tool(description="获取城市名，返回字符串")
def get_user_city(city:str)->str:
    return random.choice(["深圳","大连","芜湖"])

@tool(description="获取用户ID")
def get_user_id(city:str)->str:
    #注意：参数名 city 是从 get_user_city 复制遗留的 bug，mock 保留原貌
    return random.choice(user_ids)

@tool(description="获取月份,返回字符串")
def get_current_month()->str:
    return random.choice(month_arr)


if __name__ == "__main__":
    #运行方式：cd 项目根 && .venv/Scripts/python.exe -m agent.tools.mock_tools
    print("get_weather:", get_weather.invoke({"city": "深圳"}))
    print("get_user_city:", get_user_city.invoke({"city": "深圳"}))
    print("get_user_id:", get_user_id.invoke({"city": "深圳"}))
    print("get_current_month:", get_current_month.invoke({}))
