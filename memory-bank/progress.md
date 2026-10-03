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

## Step 1.3：内部核心 Dataclass 契约

- 状态：已完成，等待人工确认后再进入 Step 2。
- 已创建：`core/intent_recognizer.py`、`agents/agent_orchestrator.py`、`tests/test_internal_contracts.py`。
- 枚举：18 个 `IntentCategory` value，`ESCALATION` 为 `HUMAN_HANDOFF` 别名；`UrgencyLevel` 为 1–4 的 `IntEnum`；`AgentType` 为四类字符串枚举。
- Dataclass：`IntentResult`、`AgentProfile`、`AgentResponse`、`Request`、`RoutingDecision`、`OrchestratorResult`，默认可变集合均使用 `default_factory`。
- 验证：`.venv\Scripts\python.exe -m pytest -q tests/test_environment_contract.py tests/test_api_contracts.py tests/test_internal_contracts.py` → `23 passed`。
- 边界：未实现意图识别算法、路由评分、Agent 执行、LLM、Redis、Chroma 或 Step 2 逻辑。

## Step 2.1：Pattern 规则、实体抽取与紧急度判定

- 状态：已完成，等待人工确认后再进入 Step 2.2。
- 修改：`core/intent_recognizer.py` 增加 18 类 Specific/Generic Pattern 表、`match_patterns`、`extract_entities`、`calculate_urgency`、`apply_500_override`。
- Pattern：单规则命中 0.5，双规则 0.75，三次及以上封顶 1.0；返回候选、命中数、分数、最高候选和 `specific_score`。
- 实体：固定返回 `order_id`、`product`、`date`、`amount`、`error_code` 五个 key；`product` 保持空数组。
- 紧急度：按 CRITICAL > HIGH > MEDIUM > 意图兜底 > LOW；500 非登录意图覆盖为 `technical_crash` 且置信度至少 0.8。
- 验证：`.venv\Scripts\python.exe -m pytest -q tests/test_environment_contract.py tests/test_api_contracts.py tests/test_internal_contracts.py tests/test_intent_patterns.py` → `40 passed`。
- 边界：未实现 LLM Tool Use、Embedding、三路融合、复合意图或调度逻辑。
