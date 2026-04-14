"""PyInstaller runtime hook: stabilize torch GPU DLL loading on Windows."""

import ctypes
import os
import sys
from pathlib import Path


def _add_dir(path: Path) -> None:
    if not path.exists():
        return

    path_str = str(path)
    os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        try:
            add_dll_directory(path_str)
        except OSError:
            pass


def _configure() -> None:
    if not getattr(sys, "frozen", False):
        return

    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    torch_lib = base / "torch" / "lib"

    # Prefer packaged runtime first to avoid picking incompatible system DLLs.
    for d in (
        torch_lib,
        base / "numpy.libs",
        base / "scipy.libs",
        base / "sklearn.libs",
        base / "cv2",
        base / "PyQt5" / "Qt5" / "bin",
        base,
    ):
        _add_dir(d)

    # OpenMP conflict mitigations.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Preload common torch runtime DLLs in deterministic order.
    for dll in (
        "libiomp5md.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "c10.dll",
    ):
        dll_path = torch_lib / dll if dll.endswith(".dll") else torch_lib / f"{dll}.dll"
        if dll_path.exists():
            try:
                ctypes.WinDLL(str(dll_path))
            except OSError:
                # Let torch raise its own error with full context later.
                pass


_configure()
