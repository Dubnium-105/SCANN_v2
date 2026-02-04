# -*- coding: utf-8 -*-
"""
SCANN - Supernova Candidate Analysis via Neural Network
========================================================

模块化结构:
    - config: 配置管理 (ProcessingConfig, ConfigManager)
    - database: 数据库操作 (AsyncDatabaseWriter, DatabaseManager)
    - processing: 图像处理 (process_stage_a, _prepare_patch_tensor_80_static)
    - downloader: 下载引擎 (LinkedDownloader, DBDownloadWindow)
    - widgets: 自定义UI组件 (ImageViewer, SuspectListWidget, SuspectGlobalKeyFilter)
    - workers: 后台线程 (BatchWorker)
    - main_window: 主窗口 (SCANN)
"""

__version__ = '2.0.0'
__author__ = 'SCANN Team'

from .config import ProcessingConfig, ConfigManager
from .database import AsyncDatabaseWriter, DatabaseManager
from .processing import process_stage_a, _prepare_patch_tensor_80_static
from .downloader import LinkedDownloader, DBDownloadWindow
from .widgets import ImageViewer, SuspectListWidget, SuspectGlobalKeyFilter
from .workers import BatchWorker
from .main_window import SCANN, main

__all__ = [
    # 配置
    'ProcessingConfig',
    'ConfigManager',
    # 数据库
    'AsyncDatabaseWriter',
    'DatabaseManager',
    # 图像处理
    'process_stage_a',
    '_prepare_patch_tensor_80_static',
    # 下载
    'LinkedDownloader',
    'DBDownloadWindow',
    # 组件
    'ImageViewer',
    'SuspectListWidget',
    'SuspectGlobalKeyFilter',
    # 后台线程
    'BatchWorker',
    # 主窗口
    'SCANN',
    'main',
]
