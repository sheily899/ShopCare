# ShopCare 施工进度

## Step 1.1：环境基线

- 状态：已完成，等待用户确认后再进入 Step 1.2。
- 已创建：`requirements.txt`、`.env.example`、`tests/test_environment_contract.py`。
- 依赖基线：13 个核心包版本与 `tech-stack.md` 锁定值完全匹配；未加入 CUDA/GPU 依赖。
- 环境模板：覆盖 LLM、Redis、Chroma、知识库模型、Skills、监控、Composer 与评测配置；仅使用无害占位符。
- 验证：`python -m pytest -q tests/test_environment_contract.py` → `4 passed`。
- 备注：`py -3.12` 启动器不可用，但当前 `python --version` 为 `Python 3.12.7`；未开始 Step 1.2。

## Step 1.2：API 边界契约

- 状态：已完成，等待用户确认后再进入 Step 1.3。
- 已创建：`api/main.py`、`tests/test_api_contracts.py`。
- `ChatRequest`：`message` 必填且拒绝空白文本；`user_id` 默认 `anonymous`；`conv_id` 默认 `None`。
- `ChatResponse`：严格保留 system-spec 2.1 的字段、默认值、独立可变集合和范围约束；未添加业务路由或 lifespan。
- 验证：`.venv\Scripts\python.exe -m pytest -q tests/test_environment_contract.py tests/test_api_contracts.py` → `16 passed`。
- 环境：项目 `.venv` 已验证 FastAPI 0.115.5、Pydantic 2.10.3、pytest 8.3.4。
- 边界：`core/intent_recognizer.py`、`agents/agent_orchestrator.py`、`tests/test_internal_contracts.py` 尚未创建，Step 1.3 未开始。
