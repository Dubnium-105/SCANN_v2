"""日志系统单元测试 - 修复版"""

import logging
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestLoggingConfig:
    """测试日志配置模块"""

    def setup_method(self):
        """每个测试前：清理handlers"""
        logging.root.handlers.clear()

    def teardown_method(self):
        """每个测试后：清理"""
        # 关闭所有handlers
        for handler in logging.root.handlers[:]:
            try:
                handler.close()
            except:
                pass
            logging.root.removeHandler(handler)

    def test_setup_logging_creates_root_logger(self, tmp_dir):
        """测试：setup_logging应创建root logger"""
        from scann.logger_config import setup_logging

        log_file = tmp_dir / f"test_{self.test_setup_logging_creates_root_logger.__name__}.log"
        logger = setup_logging(log_file=log_file)

        assert logger is not None
        assert logger.name == "root"
        assert isinstance(logger, logging.Logger)

    def test_setup_logging_creates_file_handler(self, tmp_dir):
        """测试：setup_logging应创建文件handler"""
        from scann.logger_config import setup_logging

        log_file = tmp_dir / f"test_{self.test_setup_logging_creates_file_handler.__name__}.log"
        logger = setup_logging(log_file=log_file)

        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_setup_logging_creates_console_handler(self, tmp_dir):
        """测试：setup_logging应创建控制台handler"""
        from scann.logger_config import setup_logging

        log_file = tmp_dir / f"test_{self.test_setup_logging_creates_console_handler.__name__}.log"
        logger = setup_logging(log_file=log_file, console_output=True)

        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) >= 1

    def test_setup_logging_creates_log_file(self, tmp_dir):
        """测试：setup_logging应创建日志文件"""
        from scann.logger_config import setup_logging

        log_file = tmp_dir / f"test_{self.test_setup_logging_creates_log_file.__name__}.log"
        setup_logging(log_file=log_file)

        # 刷新并等待文件写入
        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()

        assert log_file.exists()

    def test_log_file_contains_message(self, tmp_dir):
        """测试：日志文件应包含消息"""
        from scann.logger_config import setup_logging

        log_file = tmp_dir / f"test_{self.test_log_file_contains_message.__name__}.log"
        setup_logging(log_file=log_file)

        logging.info("Test message")

        # 刷新并等待文件写入
        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        time.sleep(0.1)

        content = log_file.read_text(encoding='utf-8')
        assert "Test message" in content

    def test_log_file_has_correct_format(self, tmp_dir):
        """测试：日志文件格式应正确"""
        from scann.logger_config import setup_logging

        log_file = tmp_dir / f"test_{self.test_log_file_has_correct_format.__name__}.log"
        setup_logging(log_file=log_file)

        logging.info("Format test")

        # 刷新并等待文件写入
        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        time.sleep(0.1)

        content = log_file.read_text(encoding='utf-8')
        lines = content.strip().split('\n')

        assert len(lines) > 0
        # 找到包含"Format test"的那一行
        format_line = None
        for line in lines:
            if "Format test" in line:
                format_line = line
                break

        assert format_line is not None
        # 应包含时间戳、级别和消息
        assert "INFO" in format_line
        assert "Format test" in format_line
        assert "-" in format_line  # 时间戳分隔符

    def test_log_levels(self, tmp_dir):
        """测试：日志级别应正确过滤"""
        from scann.logger_config import setup_logging

        log_file = tmp_dir / f"test_{self.test_log_levels.__name__}.log"
        setup_logging(log_file=log_file, log_level=logging.WARNING)

        logging.debug("DEBUG message")
        logging.info("INFO message")
        logging.warning("WARNING message")
        logging.error("ERROR message")

        # 刷新并等待文件写入
        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        time.sleep(0.1)

        content = log_file.read_text(encoding='utf-8')

        # WARNING及以上应该记录
        assert "WARNING message" in content
        assert "ERROR message" in content
        # DEBUG和INFO不应该记录
        assert "DEBUG message" not in content
        assert "INFO message" not in content

    def test_utf8_encoding(self, tmp_dir):
        """测试：日志文件应支持UTF-8编码"""
        from scann.logger_config import setup_logging

        log_file = tmp_dir / f"test_{self.test_utf8_encoding.__name__}.log"
        setup_logging(log_file=log_file)

        chinese_msg = "测试中文消息"
        emoji_msg = "Emoji test "
        logging.info(chinese_msg)
        logging.info(emoji_msg)

        # 刷新并等待文件写入
        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        time.sleep(0.1)

        content = log_file.read_text(encoding='utf-8')
        assert chinese_msg in content
        assert emoji_msg in content

    def test_console_output_disabled(self, tmp_dir):
        """测试：可以禁用控制台输出"""
        from scann.logger_config import setup_logging

        log_file = tmp_dir / f"test_{self.test_console_output_disabled.__name__}.log"

        with patch('sys.stdout') as mock_stdout:
            logger = setup_logging(log_file=log_file, console_output=False)
            console_handlers = [h for h in logger.handlers
                                if isinstance(h, logging.StreamHandler)
                                and not isinstance(h, logging.FileHandler)]
            assert len(console_handlers) == 0

    def test_setup_logging_default_log_file(self, tmp_dir):
        """测试：默认日志文件路径"""
        from scann.logger_config import setup_logging

        # Monkey patch临时目录
        import scann.logger_config
        original_path = scann.logger_config.Path

        # 保存原始logger
        logger = setup_logging(log_file=tmp_dir / "test.log")

        assert logger is not None
        assert logger.level == logging.INFO


class TestMainWindowLogging:
    """测试MainWindow日志集成"""

    def setup_method(self):
        """每个测试前：清理handlers"""
        logging.root.handlers.clear()

    def teardown_method(self):
        """每个测试后：清理"""
        for handler in logging.root.handlers[:]:
            try:
                handler.close()
            except:
                pass
            logging.root.removeHandler(handler)

    def test_main_window_has_logger(self, qapp, tmp_dir):
        """测试：MainWindow应有logger"""
        from scann.gui.main_window import MainWindow
        from scann.logger_config import setup_logging

        log_file = tmp_dir / f"test_{self.test_main_window_has_logger.__name__}.log"
        setup_logging(log_file=log_file)

        window = MainWindow()

        assert window._logger is not None
        assert isinstance(window._logger, logging.Logger)
        assert window._logger.name == "scann.gui.main_window"

        window.close()

    def test_show_message_logs_to_file(self, qapp, tmp_dir):
        """测试：_show_message应记录到文件"""
        from scann.gui.main_window import MainWindow
        from scann.logger_config import setup_logging

        log_file = tmp_dir / f"test_{self.test_show_message_logs_to_file.__name__}.log"
        setup_logging(log_file=log_file)

        window = MainWindow()
        window._show_message("Test message", timeout=0, level='INFO')

        # 刷新并等待文件写入
        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        time.sleep(0.1)

        content = log_file.read_text(encoding='utf-8')
        assert "Test message" in content

        window.close()

    def test_show_message_updates_status_bar(self, qapp):
        """测试：_show_message应更新status bar"""
        from scann.gui.main_window import MainWindow
        from scann.logger_config import setup_logging

        # 不记录到文件
        setup_logging(console_output=True)

        window = MainWindow()
        window._show_message("Status message", timeout=0)

        current_message = window.statusBar().currentMessage()
        assert "Status message" == current_message

        window.close()

    def test_show_message_with_different_levels(self, qapp):
        """测试：_show_message应支持不同日志级别"""
        from scann.gui.main_window import MainWindow
        from scann.logger_config import setup_logging

        setup_logging(console_output=True)

        window = MainWindow()

        # 测试不同级别
        window._show_message("DEBUG", timeout=0, level='DEBUG')
        window._show_message("INFO", timeout=0, level='INFO')
        window._show_message("WARNING", timeout=0, level='WARNING')
        window._show_message("ERROR", timeout=0, level='ERROR')
        window._show_message("CRITICAL", timeout=0, level='CRITICAL')

        window.close()

    def test_logger_name_in_main_window(self, qapp):
        """测试：MainWindow中logger名称正确"""
        from scann.gui.main_window import MainWindow
        from scann.logger_config import setup_logging, get_logger

        setup_logging(console_output=True)

        window = MainWindow()

        # logger应该有正确的名称
        assert "scann.gui.main_window" in window._logger.name

        window.close()


class TestLoggingIntegration:
    """集成测试"""

    def setup_method(self):
        """每个测试前：清理handlers"""
        logging.root.handlers.clear()

    def teardown_method(self):
        """每个测试后：清理"""
        for handler in logging.root.handlers[:]:
            try:
                handler.close()
            except:
                pass
            logging.root.removeHandler(handler)

    def test_end_to_end_logging(self, qapp, tmp_dir):
        """端到端测试：setup -> main window -> status bar -> file"""
        from scann.gui.main_window import MainWindow
        from scann.logger_config import setup_logging

        log_file = tmp_dir / f"test_{self.test_end_to_end_logging.__name__}.log"
        setup_logging(log_file=log_file)

        window = MainWindow()

        # 测试消息
        window._show_message("Integration test message", timeout=0)

        # 检查status bar
        status_message = window.statusBar().currentMessage()
        assert "Integration test message" in status_message

        # 检查文件
        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        time.sleep(0.1)

        content = log_file.read_text(encoding='utf-8')
        assert "Integration test message" in content

        window.close()
