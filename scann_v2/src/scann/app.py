"""SCANN v2 Application Entry Point"""

import os
import sys
from pathlib import Path

# 添加src目录到sys.path以确保模块可以被导入
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 设置PyTorch模型下载路径到项目内（必须在导入torch之前设置）
# app.py 位于 scann_v2/src/scann/app.py，需要向上3级到 scann_v2/
project_root = Path(__file__).parent.parent.parent
model_cache_dir = project_root / "models" / "torch_cache"
model_cache_dir.mkdir(parents=True, exist_ok=True)

# 设置环境变量
os.environ['TORCH_HOME'] = str(model_cache_dir)
os.environ['TORCH_HUB_DIR'] = str(model_cache_dir)

import torch  # 导入后再次确认
torch.hub.set_dir(str(model_cache_dir))


def main():
    """启动 SCANN v2 应用程序"""
    from scann.logger_config import setup_logging, get_logger
    from PyQt5.QtWidgets import QApplication
    from scann.gui.main_window import MainWindow

    # 初始化日志系统
    setup_logging()
    logger = get_logger(__name__)
    logger.info("SCANN v2 启动中...")

    try:
        app = QApplication(sys.argv)
        app.setApplicationName("SCANN v2")
        app.setApplicationVersion("2.0.0")

        window = MainWindow()
        window.show()
        logger.info("主窗口已显示")

        exit_code = app.exec_()
        logger.info(f"程序退出，退出码: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"程序启动失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
