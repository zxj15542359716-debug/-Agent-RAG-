#提示词加载工具
from utils.config_handler import prompts_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def load_system_prompts():
    try:
        system_prompts_path = get_abs_path(prompts_config["main_prompts_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置中没有main_prompt_path配置项")
        raise e

    try:
        return open(system_prompts_path, "r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e

def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_config["rag_summarize_prompt"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置中没有rag_summarize_prompt配置项")
        raise e

    try:
        return open(rag_prompt_path, "r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析系统提示词出错，{str(e)}")
        raise e

def load_report_prompts():
    try:
        report_prompts_path = get_abs_path(prompts_config["report_prompts_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompts_path, "r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析系统提示词出错，{str(e)}")
        raise e

def load_researcher_prompts(kind: str) -> str:
    """读取某个子研究者的系统提示词（kind: fault / warranty / history）"""
    key = f"researcher_{kind}_prompt"
    try:
        path = get_abs_path(prompts_config[key])
    except KeyError as e:
        logger.error(f"[load_researcher_prompts]在yaml配置中没有{key}配置项")
        raise e

    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_researcher_prompts]解析提示词出错，{str(e)}")
        raise e

def load_report_synthesize_prompt() -> str:
    """读取报告合成提示词（第2步·2.3：三路结论合并为结构化报告）"""
    try:
        path = get_abs_path(prompts_config["report_synthesize_prompt"])
    except KeyError as e:
        logger.error(f"[load_report_synthesize_prompt]在yaml配置中没有report_synthesize_prompt配置项")
        raise e

    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_synthesize_prompt]解析提示词出错，{str(e)}")
        raise e

if __name__ == "__main__":
    print(load_system_prompts())


# ============================================================================================
# 【第 2 步 · 2.3 说明】本文件在第 2 步的改动（子研究者与合成提示词加载）
# --------------------------------------------------------------------------------------------
# 新增 load_researcher_prompts(kind) 与 load_report_synthesize_prompt()，路径键在
# config/prompts.yml（与既有三个提示词同一套约定：路径进配置、文件进 prompts/）。
# 为什么每个研究者一个提示词文件：三个研究者的工具面与产出结构都不同（故障诊断查知识库、
# 保修政策查真实保修记录、历史工单查使用/维修记录），混在一个提示词里模型容易串工具；
# 分开后"角色-工具-输出结构"三者一一对应，也便于单独调某一个研究者的表现。
# 与本文件既有函数的关系：异常处理、日志口径完全一致（KeyError 与读文件失败分别记日志再抛）。
# ============================================================================================