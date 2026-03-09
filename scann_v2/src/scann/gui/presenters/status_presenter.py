"""状态栏与日志消息展示器。"""

from __future__ import annotations

import logging

from PyQt5.QtWidgets import QStatusBar


class StatusPresenter:
    """统一封装状态栏消息与日志输出。"""

    def __init__(self, status_bar: QStatusBar, logger: logging.Logger) -> None:
        self._status_bar = status_bar
        self._logger = logger

    def show_message(self, message: str, timeout: int = 3000, level: str = "INFO") -> None:
        """显示消息并按级别写入日志。"""
        self._status_bar.showMessage(message, timeout)
        log_level = getattr(logging, level.upper(), logging.INFO)
        self._logger.log(log_level, message)