"""通过K读取V，K就是文件，V就是配置文件中的对象"""
import yaml
from utils.path_tool import get_abs_path

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