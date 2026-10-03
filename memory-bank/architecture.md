# ShopCare 代码职责记录

## Step 1.1：环境基线

| 文件 | 职责 |
| --- | --- |
| `requirements.txt` | 锁定 ShopCare 的 13 个核心 Python 依赖版本；不承载 CPU Torch 安装，也不引入 CUDA/GPU 依赖。 |
| `.env.example` | 提供 LLM、Redis、Chroma、Embedding、Skills、监控、Composer 与评测变量的安全占位模板；不包含真实密钥或生产密码。 |
| `tests/test_environment_contract.py` | 校验依赖名称/版本精确匹配、GPU/CUDA 依赖排除、环境变量覆盖和敏感凭据占位符安全性。 |

Step 1.1 未创建 API 模型、内部 dataclass、业务路由或外部业务工具；Step 1.2 保持未开始。
