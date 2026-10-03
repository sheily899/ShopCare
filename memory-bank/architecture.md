# ShopCare 代码职责记录

## Step 1.1：环境基线

| 文件 | 职责 |
| --- | --- |
| `requirements.txt` | 锁定 ShopCare 的 13 个核心 Python 依赖版本；不承载 CPU Torch 安装，也不引入 CUDA/GPU 依赖。 |
| `.env.example` | 提供 LLM、Redis、Chroma、Embedding、Skills、监控、Composer 与评测变量的安全占位模板；不包含真实密钥或生产密码。 |
| `tests/test_environment_contract.py` | 校验依赖名称/版本精确匹配、GPU/CUDA 依赖排除、环境变量覆盖和敏感凭据占位符安全性。 |

Step 1.1 未创建 API 模型、内部 dataclass、业务路由或外部业务工具；Step 1.2 保持未开始。

## Step 1.2：API 边界契约

| 文件 | 职责 |
| --- | --- |
| `api/main.py` | 提供空的 FastAPI 应用实例，以及 API 边界专用的 `ChatRequest` / `ChatResponse` Pydantic V2 模型；不包含 lifespan、业务路由或外部系统调用。 |
| `tests/test_api_contracts.py` | 验证请求默认值、空白消息拒绝、响应字段和默认集合、置信度/延迟边界、JSON 与 dict 序列化。 |

`ChatRequest` 仅在边界校验 `message` 非空白；`ChatResponse` 的 `routing_confidence` / `intent_confidence` 限制为 0..1，`latency_ms` 限制为非负数。内部 dataclass 尚未开始。

## Step 1.3：内部核心 Dataclass 契约

| 文件 | 职责 |
| --- | --- |
| `core/intent_recognizer.py` | 定义 `IntentCategory`、`UrgencyLevel` 与 `IntentResult`；只承载枚举和识别结果字段，不包含识别算法。 |
| `agents/agent_orchestrator.py` | 定义 `AgentType`、`AgentProfile`、`AgentResponse`、`Request`、`RoutingDecision`、`OrchestratorResult`；只承载内部总线结构，不包含编排逻辑。 |
| `tests/test_internal_contracts.py` | 校验枚举值/别名、UrgencyLevel 数值、dataclass 默认值、UUID 请求号、可变集合隔离和非 Pydantic 边界。 |

内部模块仅依赖 Python 标准库和彼此的类型枚举；Pydantic 仍只存在于 API 边界模块。
