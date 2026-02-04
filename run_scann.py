# -*- coding: utf-8 -*-
"""
SCANN - Supernova Candidate Analysis via Neural Network
========================================================

启动脚本 - 运行此文件以启动应用程序

用法:
    python run_scann.py
"""

import sys
import os

# 确保包路径在系统路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scann import main

if __name__ == '__main__':
    main()
