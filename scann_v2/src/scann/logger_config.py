"""统一日志配置模块

提供全局日志配置，确保整个应用使用统一的日志系统。
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[Path] = None,
    log_level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """配置全局日志系统

    Args:
        log_file: 日志文件路径，如果为None则使用默认路径
        log_level: 日志级别（默认为INFO）
        console_output: 是否输出到控制台（默认为True）

    Returns:
        配置好的根logger

    Example:
        >>> logger = setup_logging()
        >>> logger.info("这是一条日志消息")
    """
    # 确定日志文件路径
    if log_file is None:
        # 默认路径：项目根目录下的logs/scann.log
        current_file = Path(__file__).resolve()
        # __file__ = src/scann/logger_config.py
        # current_file.parent = src/scann
        # current_file.parent.parent = src
        # current_file.parent.parent.parent = 项目根目录 (scann_v2)
        logs_dir = current_file.parent.parent.parent / 'logs'
        logs_dir.mkdir(exist_ok=True)  # 确保logs目录存在
        log_file = logs_dir / 'scann.log'
    else:
        log_file = Path(log_file).resolve()
        # 如果指定了文件路径，确保其目录存在
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # 配置根logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 清除已有的handlers（避免重复添加）
    logger.handlers.clear()

    # 创建formatter
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件handler
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except (IOError, OSError) as e:
        # 如果无法创建日志文件，至少保证程序能运行
        print(f"警告：无法创建日志文件 {log_file}: {e}", file=sys.stderr)

    # 控制台handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 记录初始化信息
    logger.info(f"日志系统已初始化，日志文件: {log_file}")
    logger.info(f"日志级别: {logging.getLevelName(log_level)}")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取logger实例

    Args:
        name: logger名称，通常为__name__

    Returns:
        Logger实例

    Example:
        >>> from scann.logger_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("这是一条日志消息")
    """
    return logging.getLogger(name)


def close_logging():
    """关闭日志系统（清理所有handlers）

    通常在程序退出时调用。这个函数会遍历并关闭所有logger的所有handlers，
    包括root logger和子logger的handlers，以确保在Windows等系统上释放文件锁。
    """
    # 获取root logger
    root_logger = logging.getLogger()
    
    # 关闭root logger的所有handlers
    for handler in root_logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        root_logger.removeHandler(handler)
    
    # 清理所有已存在的logger
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass
            logger.removeHandler(handler)
