"""配置模块单元测试"""

import json
from pathlib import Path

import pytest


class TestConfig:
    """测试配置加载与保存"""

    def test_load_config(self, config_file):
        from scann.core.config import load_config

        cfg = load_config(str(config_file))
        assert cfg is not None

    def test_load_config_telescope(self, config_file):
        from scann.core.config import load_config

        cfg = load_config(str(config_file))
        assert cfg.telescope.focal_length_mm == 2000.0
        assert cfg.telescope.pixel_size_um == 9.0

    def test_load_config_observatory(self, config_file):
        from scann.core.config import load_config

        cfg = load_config(str(config_file))
        assert cfg.observatory.code == "C42"

    def test_save_config_roundtrip(self, config_file, tmp_dir):
        from scann.core.config import load_config, save_config

        cfg = load_config(str(config_file))
        out = tmp_dir / "saved_config.json"
        save_config(cfg, str(out))

        assert out.exists()
        cfg2 = load_config(str(out))
        assert cfg2.telescope.focal_length_mm == cfg.telescope.focal_length_mm
        assert cfg2.observatory.code == cfg.observatory.code

    def test_load_missing_file_returns_default(self, tmp_dir):
        from scann.core.config import load_config

        cfg = load_config(str(tmp_dir / "nonexistent.json"))
        assert cfg is not None  # 返回默认配置
        # 默认值应为 AppConfig 的字段默认值
        assert cfg.thresh == 80
        assert cfg.blink_speed_ms == 500

    def test_save_config_json_valid(self, config_file, tmp_dir):
        from scann.core.config import load_config, save_config

        cfg = load_config(str(config_file))
        out = tmp_dir / "valid.json"
        save_config(cfg, str(out))

        # 验证是有效 JSON
        content = out.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_detection_mode_default_patch_when_missing(self, config_file):
        from scann.core.config import load_config

        cfg = load_config(str(config_file))
        assert cfg.detection_mode == "patch"

    def test_detection_mode_roundtrip(self, config_file, tmp_dir):
        from scann.core.config import load_config, save_config

        cfg = load_config(str(config_file))
        cfg.detection_mode = "hybrid"
        out = tmp_dir / "detection_mode_roundtrip.json"
        save_config(cfg, str(out))

        cfg2 = load_config(str(out))
        assert cfg2.detection_mode == "hybrid"

    def test_detection_mode_invalid_value_fallback_to_patch(self, tmp_dir):
        from scann.core.config import load_config

        cfg_path = tmp_dir / "invalid_detection_mode.json"
        cfg_path.write_text(
            json.dumps({"detection_mode": "not_a_mode"}, ensure_ascii=False),
            encoding="utf-8",
        )
        cfg = load_config(str(cfg_path))
        assert cfg.detection_mode == "patch"

    def test_hybrid_defaults_when_missing(self, config_file):
        from scann.core.config import load_config

        cfg = load_config(str(config_file))
        assert cfg.hybrid_primary_mode == "full_image"
        assert cfg.hybrid_low_confidence == pytest.approx(0.5)

    def test_hybrid_config_roundtrip(self, config_file, tmp_dir):
        from scann.core.config import load_config, save_config

        cfg = load_config(str(config_file))
        cfg.hybrid_primary_mode = "patch"
        cfg.hybrid_low_confidence = 0.77
        out = tmp_dir / "hybrid_config_roundtrip.json"
        save_config(cfg, str(out))

        cfg2 = load_config(str(out))
        assert cfg2.hybrid_primary_mode == "patch"
        assert cfg2.hybrid_low_confidence == pytest.approx(0.77)

    def test_hybrid_config_invalid_values_fallback(self, tmp_dir):
        from scann.core.config import load_config

        cfg_path = tmp_dir / "invalid_hybrid_config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "hybrid_primary_mode": "invalid_mode",
                    "hybrid_low_confidence": "bad",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        cfg = load_config(str(cfg_path))
        assert cfg.hybrid_primary_mode == "full_image"
        assert cfg.hybrid_low_confidence == pytest.approx(0.5)
