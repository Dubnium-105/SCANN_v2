"""简单测试CUDA是否可用"""
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA不可用，可能的原因:")
    print("  1. 安装的是CPU版本的PyTorch")
    print("  2. 没有安装NVIDIA显卡或驱动")
    print("  3. NVIDIA驱动版本太旧")

    # 检查nvidia-smi
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi"], 
            capture_output=True, 
            text=True,
            timeout=2,
            shell=True
        )
        print(f"\nnvidia-smi测试:")
        if result.returncode == 0:
            print("  ✅ nvidia-smi可用，NVIDIA驱动已安装")
            print("  原因: 可能是CPU版PyTorch")
            print("  解决: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        else:
            print(f"  ❌ nvidia-smi失败，错误码: {result.returncode}")
    except Exception as e:
        print(f"\n无法运行nvidia-smi: {e}")
        print("  请确认安装了NVIDIA驱动")
