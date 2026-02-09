"""测试按钮信号是否正常连接"""
import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget


def test_button_click():
    """测试按钮点击事件"""
    app = QApplication(sys.argv)

    widget = QWidget()
    layout = QVBoxLayout(widget)

    button = QPushButton("测试按钮")
    layout.addWidget(button)

    # 连接信号
    def on_clicked():
        print("按钮被点击了!")

    button.clicked.connect(on_clicked)

    widget.show()

    print("按钮点击测试已启动，请点击按钮...")
    sys.exit(app.exec_())


if __name__ == "__main__":
    test_button_click()
