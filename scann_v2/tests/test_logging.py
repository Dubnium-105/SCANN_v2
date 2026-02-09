"""测试日志功能"""
import logging
import os

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 文件handler
file_handler = logging.FileHandler('scann.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 控制台handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 测试日志输出
logger.info("这是一条INFO级别的日志")
logger.warning("这是一条WARNING级别的日志")
logger.error("这是一条ERROR级别的日志")

print("\n日志文件路径:", os.path.abspath('scann.log'))
print("\n日志文件内容:")
with open('scann.log', 'r', encoding='utf-8') as f:
    print(f.read())
