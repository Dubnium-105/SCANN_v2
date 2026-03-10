"""配置服务，封装配置的读取与持久化。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from scann.core.config import load_config as core_load_config
from scann.core.config import save_config as core_save_config
from scann.core.models import AppConfig


class ConfigService:
    """集中管理配置的加载与保存入口。"""

    def load_config(self, path: Optional[Union[str, Path]] = None) -> AppConfig:
        """加载配置对象。"""
        return core_load_config(path)

    def save_config(
        self,
        config: AppConfig,
        path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """保存配置对象。"""
        return core_save_config(config, path)