"""SCANN v2 Application Entry Point"""

import os
import sys
from pathlib import Path


def _configure_frozen_dll_env() -> None:
    """在 PyInstaller 冻结模式下补充 DLL 搜索路径，避免 torch 依赖初始化失败。"""
    if not getattr(sys, "frozen", False):
        return

    exe_dir = Path(sys.executable).resolve().parent
    meipass = Path(getattr(sys, "_MEIPASS", exe_dir))
    dll_dirs = [
        meipass,
        exe_dir,
        meipass / "torch" / "lib",
        meipass / "numpy.libs",
        meipass / "scipy.libs",
        meipass / "sklearn.libs",
        meipass / "cv2",
        meipass / "PyQt5" / "Qt5" / "bin",
    ]

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for dll_dir in dll_dirs:
        if not dll_dir.exists():
            continue

        dll_dir_str = str(dll_dir)
        if dll_dir_str not in path_parts:
            path_parts.insert(0, dll_dir_str)

        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            try:
                add_dll_directory(dll_dir_str)
            except OSError:
                # 某些目录可能不被允许加入，忽略后继续。
                pass

    os.environ["PATH"] = os.pathsep.join(path_parts)

    # 规避部分 OpenMP 运行时冲突导致的 DLL 初始化失败。
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


_configure_frozen_dll_env()

# 添加src目录到sys.path以确保模块可以被导入
if not getattr(sys, "frozen", False):
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

# 设置PyTorch模型下载路径到项目内（必须在导入torch之前设置）
# 非冻结模式：app.py 位于 scann_v2/src/scann/app.py，需要向上3级到 scann_v2/
# 冻结模式：以可执行文件所在目录为根目录
if getattr(sys, "frozen", False):
    project_root = Path(sys.executable).resolve().parent
else:
    project_root = Path(__file__).parent.parent.parent
model_cache_dir = project_root / "models" / "torch_cache"
model_cache_dir.mkdir(parents=True, exist_ok=True)

# 设置环境变量
os.environ['TORCH_HOME'] = str(model_cache_dir)
os.environ['TORCH_HUB_DIR'] = str(model_cache_dir)

try:
    import torch  # 导入后再次确认
    torch.hub.set_dir(str(model_cache_dir))
except OSError as e:
    err_msg = (
        "SCANN v2 启动失败：PyTorch GPU 运行库初始化失败。\n\n"
        f"详细错误: {e}\n\n"
        "请检查：\n"
        "1) 已安装 Microsoft Visual C++ 2015-2022 x64 运行库；\n"
        "2) NVIDIA 显卡驱动版本满足当前 CUDA 版本要求；\n"
        "3) 使用最新打包产物（已禁用 UPX 压缩）。"
    )
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, err_msg, "SCANN v2 启动错误", 0x10)
        except Exception:
            pass
    raise SystemExit(err_msg)


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
        window.showMaximized()
        logger.info("主窗口已显示")

        exit_code = app.exec_()
        logger.info(f"程序退出，退出码: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"程序启动失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
