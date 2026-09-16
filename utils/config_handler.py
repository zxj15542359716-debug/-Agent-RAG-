"""通过K读取V，K就是文件，V就是配置文件中的对象"""
import os

import yaml
from dotenv import load_dotenv

from utils.path_tool import get_abs_path

#【新增】加载项目根目录的 .env（不存在则静默跳过；已存在于系统环境变量中的值优先，
#不会被 .env 覆盖）。JWT_SECRET 等本地敏感配置放 .env，避免硬编码进源码。
load_dotenv(get_abs_path(".env"))


def require_env(name: str) -> str:
    """读取必需的环境变量；缺失时抛错（fail-fast：配置问题在启动阶段暴露，
    而不是服务照常启动、用户请求时才失败）"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"缺少必需的环境变量 {name}。请在项目根目录的 .env 文件"
            f"（可从 .env.example 复制）或系统环境变量中配置后重启。")
    return value

#rag文件
def load_rag_config(configer_path:str = get_abs_path("config/rag.yml"),encoding:str="utf-8"):
    #打开config文件
    with open(configer_path, "r", encoding=encoding) as f:
        #全量加载
        return yaml.load(f, Loader=yaml.FullLoader)

#向量数据库
# 【修改】向量库由 Chroma 更换为 FAISS：函数改名 load_faiss_config，默认读取 config/faiss.yml
def load_faiss_config(configer_path:str = get_abs_path("config/faiss.yml"),encoding:str="utf-8"):
    #打开config文件
    with open(configer_path, "r", encoding=encoding) as f:
        #全量加载
        return yaml.load(f, Loader=yaml.FullLoader)

#提示词文件
def load_prompts_config(configer_path:str = get_abs_path("config/prompts.yml"),encoding:str="utf-8"):
    #打开config文件
    with open(configer_path, "r", encoding=encoding) as f:
        #全量加载
        return yaml.load(f, Loader=yaml.FullLoader)


#agent文件
def load_agent_config(configer_path:str = get_abs_path("config/agent.yml"),encoding:str="utf-8"):
    #打开config文件
    with open(configer_path, "r", encoding=encoding) as f:
        #全量加载
        return yaml.load(f, Loader=yaml.FullLoader)

rag_config = load_rag_config()
# 【修改】向量库由 Chroma 更换为 FAISS：模块级配置对象改名 faiss_config
faiss_config = load_faiss_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()

if __name__ == '__main__':
    print(rag_config["chat_model_name"])