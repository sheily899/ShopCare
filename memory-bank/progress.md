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

## Step 2.2：LLM Tool Use 意图识别适配器

- 状态：已完成，等待人工确认后再进入 Step 2.3。
- 修改：`core/intent_recognizer.py` 增加严格的 `classify_intent` Tool Use schema、`LLMIntentResult` dataclass、异步 `_recognize_with_llm` 及 Anthropic 响应解析与失败降级。
- 契约：工具名固定为 `classify_intent`；18 个意图值由 `IntentCategory` 生成；`intent`、`confidence`、`reasoning` 为必填字段；`additionalProperties=False`；强制 `tool_choice`。
- 解析与容错：合法 payload 返回 `success=True`；未知意图、缺失/额外字段、无 tool_use、非法 reasoning、非有限置信度和客户端异常统一返回 `OTHER`、`0.0`、`success=False`；数值置信度越界安全截断到 0..1。
- 测试：新增 `tests/test_intent_llm_tool.py`，使用 `AsyncMock`，不发真实公网请求；覆盖成功解析、调用参数、schema、未知意图、越界置信度、缺失字段、无 tool_use 与超时异常。
- 验证：`.venv\Scripts\python.exe -m pytest -q` → `46 passed`；`.venv\Scripts\python.exe -m compileall -q core tests` 通过。
- 边界：未实现 Embedding、三路加权融合、复合意图和 Step 2.3 逻辑；默认 Anthropic client 仅在调用方未传 client 时惰性创建。

## Step 2.3：Embedding 与三路加权融合

- 状态：已完成，等待人工确认后再进入 Step 2.4。
- 修改：`core/intent_recognizer.py` 增加 18 类模板文本、固定 256 维字符 1/2/3-gram MD5 哈希向量、余弦相似度、可选远程 Embedding 尝试与本地降级，以及异步 `recognize_intent` 三路汇聚。
- 融合：LLM、Embedding、Pattern 使用 `asyncio.create_task` 与 `gather(return_exceptions=True)`；Embedding 成功时权重为 0.7/0.2/0.1，来源失败时切换为 0.85/0.15。
- 阈值：融合分低于 0.5 强制 `OTHER`；LLM 失败时仅接受 Embedding 分数不低于 0.45；特异 Pattern 分数不低于 0.5 且融合低于 0.8 时细化泛化意图；500 非登录请求最终覆盖为 `technical_crash` 且置信度至少 0.8。
- 诊断：`IntentResult` 增加 `intent_diagnostics`，记录来源意图/分数、权重、失败状态、fallback、细化、最终意图与置信度；同时装配 urgency、entities 与 latency。
- 测试：新增 `tests/test_intent_fusion.py`、`tests/test_intent_diagnostics.py`，覆盖本地 256 维向量、三路权重、0.5、0.45、0.8、500 覆盖和诊断字段。
- 验证：`.venv\Scripts\python.exe -m pytest -q` → `54 passed`；`.venv\Scripts\python.exe -m compileall -q core tests` 与 `git diff --check` 通过。
- 边界：未实现 Step 2.4 复合意图 clause 切分、secondary intents 或 scoped retrieval query；远程 Embedding 失败只回退本地哈希向量，不引入训练模型依赖。

## Step 2.4：复合意图解析与跨域次要意图切分

- 状态：已完成，Checkpoint B 已闭环；未开始 Step 3。
- 修改：`core/intent_recognizer.py` 增加连接词/标点白名单切分、最长 Pattern 命中候选、三大业务域映射、同域与泛化候选过滤、跨域次意图去重及最多 3 个的硬上限。
- 终止与诊断：主意图为 `HUMAN_HANDOFF` 时无条件清空 secondary；`compound_trigger` 记录切分触发词；诊断字段记录 clauses、clause matches、secondary intents 与 scoped retrieval queries。
- 检索范围：存在合法 secondary 时按领域合并对应子句；没有合法 secondary 时保留完整原始消息作为默认查询。
- 测试：新增 `tests/test_compound_intent.py`，覆盖单意图守门、技术+账单跨域、同域过滤、3 个上限、人工终止和主链路字段装配。
- 验证：`.venv\Scripts\python.exe -m pytest -q` → `62 passed`；`.venv\Scripts\python.exe -m compileall -q core tests` 与 `git diff --check` 通过。
- 边界：未实现 Agent 并发执行与真正的 RAG 调用；scoped query 目前作为识别诊断输出，Step 6 再由编排器消费。

## Checkpoint B：意图与路由

- [x] 18 类意图和 UrgencyLevel 通过。
- [x] Tool Use、embedding、pattern 融合通过。
- [x] 0.5、0.45、0.8 阈值和 500 覆盖通过。
- [x] 复合意图最多 3 个且不会因单纯关键词共现触发跨域并发。
- [x] 更新 progress.md 与 architecture.md。
