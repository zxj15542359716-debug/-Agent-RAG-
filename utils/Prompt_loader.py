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

if __name__ == "__main__":
    print(load_system_prompts())