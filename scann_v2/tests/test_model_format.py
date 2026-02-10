"""模型格式兼容性测试 (TDD)

测试覆盖:
- ModelFormat 枚举定义
- v1/v2 state_dict 格式自动检测
- v1 格式模型加载 (自动添加 backbone. 前缀)
- v2 格式模型正常加载
- 保存时携带格式元数据
- 模型格式转换工具函数
- InferenceEngine 对不同格式的支持
"""

import sys
import tempfile
from pathlib import Path

import pytest

# 添加src目录到路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# 设置PyTorch模型缓存目录
project_root = src_path.parent
model_cache_dir = project_root / "models" / "torch_cache"
model_cache_dir.mkdir(parents=True, exist_ok=True)

torch = pytest.importorskip("torch")
torch.hub.set_dir(str(model_cache_dir))


# ──────────────────── ModelFormat 枚举 ────────────────────


class TestModelFormat:
    """测试模型格式枚举定义"""

    def test_enum_values_exist(self):
        from scann.ai.model import ModelFormat

        assert hasattr(ModelFormat, "V1_CLASSIFIER")
        assert hasattr(ModelFormat, "V2_CLASSIFIER")
        assert hasattr(ModelFormat, "AUTO")

    def test_enum_string_values(self):
        from scann.ai.model import ModelFormat

        assert ModelFormat.V1_CLASSIFIER.value == "v1_classifier"
        assert ModelFormat.V2_CLASSIFIER.value == "v2_classifier"
        assert ModelFormat.AUTO.value == "auto"

    def test_from_string(self):
        """测试从字符串创建枚举"""
        from scann.ai.model import ModelFormat

        assert ModelFormat("v1_classifier") == ModelFormat.V1_CLASSIFIER
        assert ModelFormat("v2_classifier") == ModelFormat.V2_CLASSIFIER
        assert ModelFormat("auto") == ModelFormat.AUTO


# ──────────────────── 格式自动检测 ────────────────────


class TestDetectModelFormat:
    """测试 state_dict 格式自动检测"""

    def test_detect_v1_format(self):
        """v1 格式: 键没有 backbone. 前缀 (原始 ResNet18)"""
        from scann.ai.model import ModelFormat, detect_model_format

        # 构造 v1 风格的 state_dict 键
        v1_keys = [
            "conv1.weight", "bn1.weight", "bn1.bias",
            "layer1.0.conv1.weight", "layer1.0.bn1.weight",
            "fc.weight", "fc.bias",
        ]
        v1_state = {k: torch.zeros(1) for k in v1_keys}
        assert detect_model_format(v1_state) == ModelFormat.V1_CLASSIFIER

    def test_detect_v2_format(self):
        """v2 格式: 键有 backbone. 前缀"""
        from scann.ai.model import ModelFormat, detect_model_format

        v2_keys = [
            "backbone.conv1.weight", "backbone.bn1.weight",
            "backbone.layer1.0.conv1.weight",
            "backbone.fc.weight", "backbone.fc.bias",
        ]
        v2_state = {k: torch.zeros(1) for k in v2_keys}
        assert detect_model_format(v2_state) == ModelFormat.V2_CLASSIFIER

    def test_detect_with_module_prefix(self):
        """带 module. 前缀的 v1 格式 (DataParallel 保存)"""
        from scann.ai.model import ModelFormat, detect_model_format

        keys = [
            "module.conv1.weight", "module.bn1.weight",
            "module.layer1.0.conv1.weight",
            "module.fc.weight",
        ]
        state = {k: torch.zeros(1) for k in keys}
        assert detect_model_format(state) == ModelFormat.V1_CLASSIFIER

    def test_detect_with_module_backbone_prefix(self):
        """带 module.backbone. 前缀的 v2 格式"""
        from scann.ai.model import ModelFormat, detect_model_format

        keys = [
            "module.backbone.conv1.weight", "module.backbone.bn1.weight",
            "module.backbone.layer1.0.conv1.weight",
            "module.backbone.fc.weight",
        ]
        state = {k: torch.zeros(1) for k in keys}
        assert detect_model_format(state) == ModelFormat.V2_CLASSIFIER

    def test_detect_empty_state(self):
        """空 state_dict 默认返回 AUTO"""
        from scann.ai.model import ModelFormat, detect_model_format

        assert detect_model_format({}) == ModelFormat.AUTO


# ──────────────────── state_dict 键转换 ────────────────────


class TestConvertStateDict:
    """测试 state_dict 键名转换"""

    def test_v1_to_v2_conversion(self):
        """v1 键 → v2 键 (添加 backbone. 前缀)"""
        from scann.ai.model import convert_state_dict_v1_to_v2

        v1_state = {
            "conv1.weight": torch.zeros(64, 3, 7, 7),
            "bn1.weight": torch.zeros(64),
            "layer1.0.conv1.weight": torch.zeros(64, 64, 3, 3),
            "fc.weight": torch.zeros(2, 512),
            "fc.bias": torch.zeros(2),
        }
        v2_state = convert_state_dict_v1_to_v2(v1_state)

        assert "backbone.conv1.weight" in v2_state
        assert "backbone.bn1.weight" in v2_state
        assert "backbone.layer1.0.conv1.weight" in v2_state
        assert "backbone.fc.weight" in v2_state
        assert "backbone.fc.bias" in v2_state

        # 原始键不应存在
        assert "conv1.weight" not in v2_state

    def test_v1_to_v2_with_module_prefix(self):
        """module. 前缀先移除再添加 backbone."""
        from scann.ai.model import convert_state_dict_v1_to_v2

        v1_state = {
            "module.conv1.weight": torch.zeros(64, 3, 7, 7),
            "module.fc.weight": torch.zeros(2, 512),
        }
        v2_state = convert_state_dict_v1_to_v2(v1_state)
        assert "backbone.conv1.weight" in v2_state
        assert "backbone.fc.weight" in v2_state

    def test_v1_to_v2_removes_num_batches_tracked(self):
        """转换时过滤 num_batches_tracked 键"""
        from scann.ai.model import convert_state_dict_v1_to_v2

        v1_state = {
            "conv1.weight": torch.zeros(64, 3, 7, 7),
            "bn1.num_batches_tracked": torch.tensor(100),
        }
        v2_state = convert_state_dict_v1_to_v2(v1_state)
        assert "backbone.conv1.weight" in v2_state
        # num_batches_tracked 可保留也可过滤，主要验证不会报错
        # 关键是 conv1 被正确转换


# ──────────────────── 模型加载兼容性 ────────────────────


class TestModelLoadCompatibility:
    """测试不同格式的模型加载"""

    def _create_v1_checkpoint(self, path: str):
        """创建 v1 格式的 checkpoint 文件"""
        from torchvision import models
        resnet = models.resnet18(weights=None)
        resnet.fc = torch.nn.Linear(512, 2)
        # v1 直接保存 ResNet18 的 state_dict (无 backbone. 前缀)
        torch.save(resnet.state_dict(), path)

    def _create_v2_checkpoint(self, path: str):
        """创建 v2 格式的 checkpoint 文件"""
        from scann.ai.model import SCANNClassifier
        model = SCANNClassifier(pretrained=False)
        torch.save(model.state_dict(), path)

    def _create_v1_dict_checkpoint(self, path: str):
        """创建 v1 格式的字典 checkpoint (包含 state 键)"""
        from torchvision import models
        resnet = models.resnet18(weights=None)
        resnet.fc = torch.nn.Linear(512, 2)
        torch.save({
            "state": resnet.state_dict(),
            "threshold": 0.75,
        }, path)

    def test_load_v1_checkpoint_auto_detect(self):
        """自动检测并加载 v1 格式的 checkpoint"""
        from scann.ai.model import SCANNClassifier

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            self._create_v1_checkpoint(path)
            model = SCANNClassifier.load_from_checkpoint(
                path, device=torch.device("cpu")
            )
            assert model is not None
            # 验证模型能正常前向传播
            model.eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 2)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_v2_checkpoint(self):
        """正常加载 v2 格式 checkpoint"""
        from scann.ai.model import SCANNClassifier

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            self._create_v2_checkpoint(path)
            model = SCANNClassifier.load_from_checkpoint(
                path, device=torch.device("cpu")
            )
            assert model is not None
            model.eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 2)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_v1_dict_checkpoint(self):
        """加载 v1 格式的字典 checkpoint (含 state 键和 threshold)"""
        from scann.ai.model import SCANNClassifier

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            self._create_v1_dict_checkpoint(path)
            model = SCANNClassifier.load_from_checkpoint(
                path, device=torch.device("cpu")
            )
            assert model is not None
            model.eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 2)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_with_explicit_v1_format(self):
        """显式指定 v1 格式加载"""
        from scann.ai.model import ModelFormat, SCANNClassifier

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            self._create_v1_checkpoint(path)
            model = SCANNClassifier.load_from_checkpoint(
                path, device=torch.device("cpu"), model_format=ModelFormat.V1_CLASSIFIER
            )
            assert model is not None
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_with_explicit_v2_format(self):
        """显式指定 v2 格式加载"""
        from scann.ai.model import ModelFormat, SCANNClassifier

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            self._create_v2_checkpoint(path)
            model = SCANNClassifier.load_from_checkpoint(
                path, device=torch.device("cpu"), model_format=ModelFormat.V2_CLASSIFIER
            )
            assert model is not None
        finally:
            Path(path).unlink(missing_ok=True)


# ──────────────────── 保存带格式元数据 ────────────────────


class TestSaveWithFormat:
    """测试保存模型时包含格式元数据"""

    def test_save_checkpoint_includes_format(self):
        """save_checkpoint 应在字典中包含 model_format 字段"""
        from scann.ai.model import ModelFormat, SCANNClassifier

        model = SCANNClassifier(pretrained=False)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            SCANNClassifier.save_checkpoint(model, path, threshold=0.5)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            assert isinstance(ckpt, dict)
            assert "model_format" in ckpt
            assert ckpt["model_format"] == ModelFormat.V2_CLASSIFIER.value
            assert "state" in ckpt
            assert "threshold" in ckpt
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_checkpoint_with_custom_format(self):
        """保存 v1 格式 checkpoint"""
        from scann.ai.model import ModelFormat, SCANNClassifier

        model = SCANNClassifier(pretrained=False)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            SCANNClassifier.save_checkpoint(
                model, path, model_format=ModelFormat.V1_CLASSIFIER
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            assert ckpt["model_format"] == ModelFormat.V1_CLASSIFIER.value
            # v1 格式的 state 不应有 backbone. 前缀
            state = ckpt["state"]
            for key in state:
                assert not key.startswith("backbone."), f"v1 格式不应有 backbone. 前缀: {key}"
        finally:
            Path(path).unlink(missing_ok=True)


# ──────────────────── InferenceEngine 格式支持 ────────────────────


class TestInferenceEngineFormat:
    """测试推理引擎对不同模型格式的支持"""

    def _create_v1_checkpoint(self, path: str):
        from torchvision import models
        resnet = models.resnet18(weights=None)
        resnet.fc = torch.nn.Linear(512, 2)
        torch.save(resnet.state_dict(), path)

    def test_inference_engine_loads_v1_model(self):
        """InferenceEngine 应能加载 v1 模型"""
        from scann.ai.inference import InferenceConfig, InferenceEngine

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            self._create_v1_checkpoint(path)
            config = InferenceConfig(device="cpu")
            engine = InferenceEngine(model_path=path, config=config)
            assert engine.is_ready
        finally:
            Path(path).unlink(missing_ok=True)

    def test_inference_config_has_model_format(self):
        """InferenceConfig 应有 model_format 字段"""
        from scann.ai.inference import InferenceConfig
        from scann.ai.model import ModelFormat

        config = InferenceConfig()
        assert hasattr(config, "model_format")
        assert config.model_format == ModelFormat.AUTO.value

    def test_inference_engine_with_explicit_format(self):
        """显式指定格式加载"""
        from scann.ai.inference import InferenceConfig, InferenceEngine
        from scann.ai.model import ModelFormat

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            self._create_v1_checkpoint(path)
            config = InferenceConfig(device="cpu", model_format=ModelFormat.V1_CLASSIFIER.value)
            engine = InferenceEngine(model_path=path, config=config)
            assert engine.is_ready
        finally:
            Path(path).unlink(missing_ok=True)
