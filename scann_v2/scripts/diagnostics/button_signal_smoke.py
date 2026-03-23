"""Manual Qt smoke test for button signal wiring."""

from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)

    widget = QWidget()
    widget.setWindowTitle("SCANN Button Signal Smoke Test")

    layout = QVBoxLayout(widget)
    button = QPushButton("Click to verify signal")
    layout.addWidget(button)

    def on_clicked() -> None:
        print("Button click signal received.")

    button.clicked.connect(on_clicked)

    widget.show()
    print("Window opened. Click the button to verify signal delivery.")
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
