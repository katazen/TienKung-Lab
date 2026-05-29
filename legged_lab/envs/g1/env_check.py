import os
import sys

print("-" * 50)
print(f"Python 解释器路径: {sys.executable}")
print("-" * 50)

# 获取所有环境变量并按字母顺序排序
env_vars = dict(os.environ)
for key in sorted(env_vars.keys()):
    print(f"{key}: {env_vars[key]}")

print("-" * 50)
print(f"总计环境变量数量: {len(env_vars)}")