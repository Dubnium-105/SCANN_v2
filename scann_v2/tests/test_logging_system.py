"""Test Logging System - TDD Test Suite"""

import logging
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import pytest


class TestLoggingSystem:
    """Logging System Test Suite"""

    def setup_method(self):
        """Execute before each test method"""
        # Clear existing handlers
        logging.root.handlers.clear()
        # Test log file path
        self.test_log_file = Path("test_scann.log")

    def teardown_method(self):
        """Execute after each test method"""
        # Clear handlers
        logging.root.handlers.clear()
        # Delete test log file
        if self.test_log_file.exists():
            self.test_log_file.unlink()

    def test_global_logger_initialization(self):
        """Test 1: Global logger initialization should succeed"""
        from scann.logger_config import setup_logging

        logger = setup_logging(log_file=self.test_log_file)

        # Verify logger is created
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "root"

    def test_logger_has_file_handler(self):
        """Test 2: Logger should have file handler"""
        from scann.logger_config import setup_logging

        logger = setup_logging(log_file=self.test_log_file)

        # Verify file handler exists
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) > 0, "Should have file handler"

    def test_logger_has_console_handler(self):
        """Test 3: Logger should have console handler"""
        from scann.logger_config import setup_logging

        logger = setup_logging(log_file=self.test_log_file)

        # Verify console handler exists
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) > 0, "Should have console handler"

    def test_log_file_is_created(self):
        """Test 4: Log file should be created"""
        from scann.logger_config import setup_logging

        setup_logging(log_file=self.test_log_file)

        # Write log
        logging.info("Test message")

        # Verify log file exists
        assert self.test_log_file.exists(), "Log file should exist"

    def test_log_file_contains_message(self):
        """Test 5: Log file should contain message"""
        from scann.logger_config import setup_logging

        setup_logging(log_file=self.test_log_file)
        test_message = "This is a test log message"

        logging.info(test_message)

        # Read log file
        log_content = self.test_log_file.read_text(encoding='utf-8')
        assert test_message in log_content, "Log file should contain test message"

    def test_log_file_has_correct_format(self):
        """Test 6: Log file should have correct format"""
        from scann.logger_config import setup_logging

        setup_logging(log_file=self.test_log_file)
        test_message = "Format test message"

        logging.info(test_message)

        # Read log file
        log_content = self.test_log_file.read_text(encoding='utf-8')
        lines = log_content.strip().split('\n')

        # Verify format: should contain timestamp, level and message
        assert len(lines) > 0
        log_line = lines[0]
        assert "INFO" in log_line, "Should contain log level"
        assert test_message in log_line, "Should contain message"
        # Should have timestamp format
        assert any(c in log_line for c in ['-', ':']), "Should contain timestamp format"

    def test_logger_log_levels(self):
        """Test 7: Logger should support different log levels"""
        from scann.logger_config import setup_logging

        setup_logging(log_file=self.test_log_file)

        # Test different levels
        logging.debug("DEBUG message")
        logging.info("INFO message")
        logging.warning("WARNING message")
        logging.error("ERROR message")

        # Read log file
        log_content = self.test_log_file.read_text(encoding='utf-8')

        # INFO and above should be recorded (since logger is set to INFO)
        assert "INFO message" in log_content
        assert "WARNING message" in log_content
        assert "ERROR message" in log_content
        # DEBUG should not be recorded
        assert "DEBUG message" not in log_content

    def test_logger_utf8_encoding(self):
        """Test 8: Log file should support UTF-8 encoding"""
        from scann.logger_config import setup_logging

        setup_logging(log_file=self.test_log_file)
        test_message = "Test Chinese message"

        logging.info(test_message)

        # Read log file
        log_content = self.test_log_file.read_text(encoding='utf-8')
        assert test_message in log_content, "Should handle UTF-8 encoding correctly"

    def test_show_message_outputs_to_statusbar(self):
        """Test 9: _show_message should output to status bar"""
        from PyQt5.QtWidgets import QApplication
        from scann.gui.main_window import MainWindow

        if not QApplication.instance():
            app = QApplication([])
        else:
            app = QApplication.instance()

        # Create main window (but don't show)
        window = MainWindow()

        # Mock status bar
        mock_statusbar = Mock()
        window.setStatusBar(mock_statusbar)

        # Call _show_message
        test_message = "Test status bar message"
        window._show_message(test_message)

        # Verify status bar is called
        mock_statusbar.showMessage.assert_called_once()
        args = mock_statusbar.showMessage.call_args[0]
        assert test_message in args[0]

        app.quit()

    def test_show_message_outputs_to_logger(self):
        """Test 10: _show_message should output to logger"""
        from PyQt5.QtWidgets import QApplication
        from scann.gui.main_window import MainWindow
        import logging

        # Initialize logger for testing
        logging.basicConfig(level=logging.INFO, format='%(message)s')

        if not QApplication.instance():
            app = QApplication([])
        else:
            app = QApplication.instance()

        # Create main window (but don't show)
        window = MainWindow()

        # Mock status bar (to avoid actual display)
        mock_statusbar = Mock()
        window.setStatusBar(mock_statusbar)

        # Mock logger output
        with patch.object(window._logger, 'log') as mock_log:
            test_message = "Test logger message"
            window._show_message(test_message)

            # Verify logger is called
            assert mock_log.called, "Logger should be called"
            call_args = mock_log.call_args[0]
            assert test_message in call_args[1], "Message should be passed to logger"

        app.quit()

    def test_show_message_with_different_levels(self):
        """Test 11: _show_message should support different log levels"""
        from PyQt5.QtWidgets import QApplication
        from scann.gui.main_window import MainWindow
        import logging

        # Initialize logger for testing
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')

        if not QApplication.instance():
            app = QApplication([])
        else:
            app = QApplication.instance()

        # Create main window
        window = MainWindow()

        # Mock
        mock_statusbar = Mock()
        window.setStatusBar(mock_statusbar)

        # Test different levels
        with patch.object(window._logger, 'log') as mock_log:
            window._show_message("INFO message", level='INFO')
            assert mock_log.called
            call_args = mock_log.call_args[0]
            assert call_args[0] == logging.INFO

            mock_log.reset_mock()
            window._show_message("ERROR message", level='ERROR')
            assert mock_log.called
            call_args = mock_log.call_args[0]
            assert call_args[0] == logging.ERROR

        app.quit()

    def test_main_window_uses_global_logger(self):
        """Test 12: MainWindow should use global logger"""
        from PyQt5.QtWidgets import QApplication
        from scann.logger_config import setup_logging
        from scann.gui.main_window import MainWindow

        # Initialize global logging
        logger = setup_logging(log_file=self.test_log_file)

        if not QApplication.instance():
            app = QApplication([])
        else:
            app = QApplication.instance()

        # Create main window
        window = MainWindow()

        # Verify window's logger is a child of global logger
        assert window._logger is not None
        assert isinstance(window._logger, logging.Logger)

        # Mock status bar
        mock_statusbar = Mock()
        window.setStatusBar(mock_statusbar)

        # Call _show_message
        test_message = "Global logger test"
        window._show_message(test_message)

        # Verify log file contains message
        log_content = self.test_log_file.read_text(encoding='utf-8')
        assert test_message in log_content

        app.quit()


def test_integration_logging_with_app():
    """Integration test: Verify complete app logging flow"""
    from scann.logger_config import setup_logging
    from PyQt5.QtWidgets import QApplication
    from scann.gui.main_window import MainWindow
    import logging

    # Setup logging
    test_log_file = Path("test_integration.log")
    logger = setup_logging(log_file=test_log_file)

    try:
        if not QApplication.instance():
            app = QApplication([])
        else:
            app = QApplication.instance()

        # Create main window
        window = MainWindow()

        # Mock status bar
        mock_statusbar = Mock()
        window.setStatusBar(mock_statusbar)

        # Send multiple messages
        messages = [
            ("Startup message", "INFO"),
            ("Warning message", "WARNING"),
            ("Error message", "ERROR"),
        ]

        for msg, level in messages:
            window._show_message(msg, level=level)

        # Verify status bar calls
        assert mock_statusbar.showMessage.call_count == len(messages)

        # Verify log file
        log_content = test_log_file.read_text(encoding='utf-8')
        for msg, level in messages:
            assert msg in log_content, f"Log should contain: {msg}"

        app.quit()

    finally:
        # Cleanup
        if test_log_file.exists():
            test_log_file.unlink()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
