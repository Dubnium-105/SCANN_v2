"""SCANN v2 Application Entry Point"""

import sys


def main():
    """启动 SCANN v2 应用程序"""
    from PyQt5.QtWidgets import QApplication
    from scann.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("SCANN v2")
    app.setApplicationVersion("2.0.0")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
