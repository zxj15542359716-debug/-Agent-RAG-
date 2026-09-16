#日志工具
import os
import logging
from datetime import datetime
from utils.path_tool import get_abs_path
from logging.handlers import TimedRotatingFileHandler

#日志保存的根目录
LOG_FORMAT_DIR = get_abs_path("logs")

# 保留最近7天日志，旧日志自动删除
LOG_BACKUP_COUNT = 7

#确保目录存在
os.makedirs(LOG_FORMAT_DIR, exist_ok=True)

#配置日志格式
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d- %(message)s',
)

def get_logger(
        name:str = "agent",
        console_level:int = logging.INFO,
        file_level:int = logging.DEBUG,
        log_file = None
)->logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    #避免重复添加handler
    if logger.handlers:
        return logger

    #控制台handler
    console_handler=logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    #配置 文件handler
    if log_file is None:
        # 基础日志文件名，轮转组件会自动追加 .20260908 后缀
        base_log_name = os.path.join(LOG_FORMAT_DIR, f"{name}.log")
        # when='D' 按天切割；interval=1 间隔1天；backupCount保留N份旧日志
        file_handler = TimedRotatingFileHandler(
            filename=base_log_name,
            when="D",
            interval=1,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
            delay=True
        )
    else:
        # 如果外部传入log_file，则使用普通FileHandler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")

    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger

#快捷获取
logger = get_logger()

if __name__ == "__main__":
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")