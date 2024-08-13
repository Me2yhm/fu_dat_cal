# logger.py
from pathlib import Path
import logging

parent_path = Path(__file__).parent / "log"


def setup_logger(name, log_file="run.log", level=logging.INFO):
    """设置日志器"""
    # 创建一个日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 创建一个文件处理器
    handler = logging.FileHandler(Path.joinpath(parent_path, log_file))
    handler.setLevel(level)

    # 创建一个控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 创建一个日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志器
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger


Logger = setup_logger("index", "index.log")
