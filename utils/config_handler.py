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

#检索配置（第1步·检索纵深：混合召回/融合权重/重排档位参数）
def load_retrieval_config(configer_path:str = get_abs_path("config/retrieval.yml"),encoding:str="utf-8"):
    #打开config文件
    with open(configer_path, "r", encoding=encoding) as f:
        #全量加载
        return yaml.load(f, Loader=yaml.FullLoader)

#编排配置（第2步·编排架构：路由词表/checkpoint/报告生成/token记账）
def load_orchestration_config(configer_path:str = get_abs_path("config/orchestration.yml"),encoding:str="utf-8"):
    #打开config文件
    with open(configer_path, "r", encoding=encoding) as f:
        #全量加载
        return yaml.load(f, Loader=yaml.FullLoader)

rag_config = load_rag_config()
# 【修改】向量库由 Chroma 更换为 FAISS：模块级配置对象改名 faiss_config
faiss_config = load_faiss_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()
# 【第1步·检索纵深】检索参数（mode/召回条数/融合权重/重排），见 config/retrieval.yml
retrieval_config = load_retrieval_config()
# 【第2步·编排架构】编排参数（路由词表/checkpoint/报告生成/token记账），见 config/orchestration.yml
orchestration_config = load_orchestration_config()

if __name__ == '__main__':
    print(rag_config["chat_model_name"])


# ============================================================================================
# 【第 1 步 · 1.3 说明】本文件在第 1 步的改动（检索配置加载）
# --------------------------------------------------------------------------------------------
# 新增 load_retrieval_config() 与模块级 retrieval_config：读取 config/retrieval.yml，
# 供 rag/hybrid_retriever.py 使用（mode / 召回条数 / 融合权重 / 重排档位）。
# 为什么配置放 yaml 而不是代码常量：这三个旋钮是评测调参的主要对象
# （实测 keyword 权重 0.7→0.4 使 recall@5 从 0.904 提到 1.000），
# 配置与代码分离后，调参不动代码、评测脚本可复跑同一套参数。
# 与前序改动的关系：本文件顶部仍负责 .env 加载（0.4 步）与 require_env 的 fail-fast 校验。
# ============================================================================================

# ============================================================================================
# 【第 2 步 · 2.1 说明】本文件在第 2 步的改动（编排配置加载）
# --------------------------------------------------------------------------------------------
# 新增 load_orchestration_config() 与模块级 orchestration_config：读取 config/orchestration.yml，
# 供 agent/orchestration/（checkpoint/router/graph）与 utils/usage_ledger.py 使用：
#   ① router   意图路由词表（2.2）  ② checkpoint 会话持久化（2.1）
#   ③ report   并行子研究者与报告预算（2.3）  ④ usage token 记账开关（2.4）
# 写法完全照抄 load_retrieval_config（同一个函数体、同一套默认路径约定），
# 保持"配置加载"在本项目里只有一种读法。
# 为什么配置放 yaml：与第 1 步同理——保留轮数、报告预算、路由词表都是要调参与对比的量，
# 放进配置后调参不动代码，回归脚本可复跑同一套参数。
# ============================================================================================