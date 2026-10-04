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

## Step 3.1：Message 与 Redis 工作记忆

- 状态：已完成，等待确认后再进入 Step 3.2。
- 创建：`memory/conversation_memory.py` 实现 `MsgRole`、`Message`、`MemoryContext` 和 `MemoryManager` 的 Redis 工作记忆读写。
- 数据结构：`Message` 支持 timestamp/metadata 的 JSON 字典往返；`MemoryContext.to_prompt_text()` 保留 summary、最多 3 条 relevant history 和最后 8 条 recent messages。
- Redis 规则：key 为 `wm:{user_id}:{conv_id}`；消息使用 `LPUSH` 写入、随后 `EXPIRE 86400`；读取最多 20 条并反转恢复时间正序。
- 压缩边界：达到 15 条只设置压缩入口信号并记录日志，保留 5 条常量；摘要生成、旧消息压缩与 Chroma episodic 写入留给 Step 3.2。
- 异常：Redis 读写失败记录 error 日志并返回安全结果，不伪造上下文。
- 测试：新增 `tests/test_memory_working.py`，覆盖序列化、key、LPUSH/EXPIRE、顺序恢复、15 条阈值、Prompt 截断和 Redis 异常。
- 验证：`.venv\Scripts\python.exe -m pytest -q` → `68 passed`；`compileall` 与 `git diff --check` 通过。
- 边界：尚未实现摘要 LLM、Chroma 情景记忆、用户画像和后台异步画像更新；Step 3.2 未开始。

## Step 3.2：会话摘要与 Chroma 情景记忆

- 状态：已完成，等待人工确认后再进入 Step 3.3。
- 修改：`memory/conversation_memory.py` 增加工作记忆压缩、摘要生成、`episodic` Chroma 写入和两级情景记忆检索。
- 压缩：达到 `LLEN >= 15` 后读取最多 20 条，按 `messages[:-5]` / `messages[-5:]` 切分；摘要调用固定 `temperature=0`、`max_tokens=256`，并限制为 800 字符。
- 降级：摘要 LLM 异常统一使用 `对话包含 {N} 条消息（摘要生成失败）`；Redis 写回失败和 Chroma 连接/检索失败均记录日志并隔离，不伪造情景上下文。
- Redis：摘要写入 `summary:{user_id}:{conv_id}` 并使用 86400 秒 TTL；工作记忆列表删除后按时间顺序重建最新 5 条。
- Chroma：固定 `episodic` 集合；写入 `user_id`、`conv_id`、`ts`、最多 500 字符的 `full_text`；同步 Chroma 操作统一通过 `asyncio.to_thread`。
- 检索：先按用户+会话召回，再按用户补足，内存去重后最多返回 5 条。
- 测试：新增 `tests/test_memory_episodic.py`，覆盖 15/5 滑窗、摘要截断与失败兜底、Chroma 元数据/截断、两级优先级检索、`to_thread` 和异常隔离。
- 验证：`.venv\Scripts\python.exe -m pytest -q tests/test_memory_episodic.py` → `5 passed`；全量 `.venv\Scripts\python.exe -m pytest -q` → `73 passed`；`compileall` 与 `git diff --check` 通过。
- 边界：未实现用户画像异步写回、`asyncio.create_task(update_profile)` 和 Step 3.3 逻辑。

## Step 3.3：用户画像异步更新与检索

- 状态：已完成，Checkpoint C-1 已闭环；等待人工确认后再进入 Step 4。
- 修改：`memory/conversation_memory.py` 增加 `update_profile`、`get_user_profile`、`user_profile` Chroma 集合解析与异步持久化逻辑；新增 `tests/test_memory_profile_async.py`。
- 提取：读取当前会话最新 10 条消息，使用 Anthropic 结构化 JSON 提取 `preferences` 与 `entities`，固定 `temperature=0`、`max_tokens=512`。
- 校验：严格拒绝非 JSON、缺少关键字段、非字符串数组或畸变实体结构；校验失败不调用 Chroma 更新，不覆盖已有画像。
- 持久化：稳定主键 `profile:{user_id}`，写入合法 JSON document 与 `user_id`、`conv_id`、`updated_at` 元数据；同步 Chroma 调用统一通过 `asyncio.to_thread`。
- 异步隔离：`update_profile` 捕获所有未知异常，适配 API 层后续使用 `asyncio.create_task` 的非阻塞后台写回；失败只记录 warning，不影响对话主链路。
- 读取：`get_user_profile` 能反序列化已有画像；画像不存在、损坏或 Chroma 异常时返回 `{"preferences": [], "entities": {}}`。
- 验证：`.venv\Scripts\python.exe -m pytest -q tests/test_memory_profile_async.py` → `4 passed`；记忆相关回归 → `15 passed`；全量 `.venv\Scripts\python.exe -m pytest -q` → `77 passed`；compileall 与 `git diff --check` 通过。
- 边界：未在 API 路由中接入画像后台任务，未实现 Step 4 知识库 RAG 与工具断路器。

## Checkpoint C-1：三层记忆基础能力

- [x] Redis 工作记忆通过。
- [x] episodic 摘要压缩与两级召回通过。
- [x] user_profile 异步写回、严格校验与读取兜底通过。
- [x] `progress.md` 与 `architecture.md` 已同步。

## Step 4.1：KnowledgeBase 初始化与文档切片

- 状态：已完成，等待人工确认后再进入 Step 4.2。
- 创建：`mcp/knowledge_base.py`、`data/knowledge/*.md` 六篇产品源文档、`scripts/init_knowledge_base.py`、`tests/test_knowledge_base_embedding.py`、`tests/test_knowledge_base_metadata.py`、`tests/test_knowledge_base_ingestion.py`。
- Chroma：HTTP client 优先读取 `CHROMA_HOST`/`CHROMA_PORT`；连接失败时显式记录 warning 并 fallback 到 `PersistentClient`，默认路径为 `.data/chroma`；新 `knowledge_base_zh` 为空时支持迁移旧 `knowledge_base`。
- Embedding：默认 `BAAI/bge-small-zh-v1.5`，固定 `device="cpu"`、`normalize_embeddings=True`；collection metadata 固定 `hnsw:space=l2`。
- 切片与元数据：升级为 Markdown Header-Aware 切片，优先按 `##`/`###` 标题边界拆分并注入 `[一级标题 > 二级标题 > 三级标题]` 面包屑；超长小节按段落、句号和硬截断拆分，单片（含前缀）最多 500 字符。每片写入 `title`、`source_document_id`、`source_chunk_id`、`content_hash`、`chunk_index`、`total_chunks`、`corpus_version` 七项字段，source_chunk_id 格式为 `{source_document_id}_chunk_{chunk_index}`。
- 检索：返回 `title`、`content`、`score`、`id` 与稳定 chunk 身份；`score=round(1-distance,4)`，不设置 score threshold，也不将 score 解释为概率。
- 源文本：仅落位 6 篇产品客服 Markdown；未复制老工程 `data/chroma` 二进制，也未混入 `knowledge_base_benchmark_v2.1_main100.json`。
- 入库：`python scripts/init_knowledge_base.py` 扫描 `data/knowledge/*.md`，默认写入项目 `.data/chroma` 的产品 collection，并输出文档数、chunk 数和耗时。
- 启动兼容：新增 `mcp/__init__.py` 并固定脚本将项目根目录置于 `sys.path` 首位，避免系统同名第三方 `mcp` 包导致 `ModuleNotFoundError`。
- 验证：知识库专项测试 → `11 passed`；Header-Aware 专项断言六篇文档生成 18 个 chunk、最大 244 字符并聚焦 401 片段；全量 pytest → `88 passed`；脚本 `--help`、compileall 与 `git diff --check` 通过。
- 回归修复：HTTP Chroma 失败后的 PersistentClient fallback 改为使用独立的 `SegmentAPI` Settings，避免复用被 `HttpClient` 原地改写的 FastAPI 配置；真实 0.5.23 客户端已验证可从 `localhost:8000` 的 404 回退到本地目录。
- 本地模型复用：从 Lenovo RAG 模块复制可加载的 BGE Small 中文模型到 `models/BAAI/bge-small-zh-v1.5`（约 192 MB）；当前运行时通过 `CUSTOMEROPS_EMBEDDING_MODEL` 指向该目录，CPU 生成 512 维向量，已完成 6 篇文档、18 个 chunk 的实际入库与 `401 认证失败` 检索验证。模型二进制仅作为本机运行时资产，不提交 Git。
- 边界：Step 4.3 的 Query Rewrite、并行召回与去重已完成；LLM Rerank 在 Step 4.4 独立实现。

## Step 4.2：Tool、缓存与 Circuit Breaker

- 状态：已完成；Step 4.3 与 Step 4.4 已在后续完成。
- 创建：`mcp/tool_manager.py` 与 `tests/test_tool_manager_circuit.py`。
- 契约：`Tool`、`ToolResult`、`ToolStats` 使用 dataclass；兼容 `name`/`tool_name`、`schema`/`input_schema` 等旧调用别名；默认工具 TTL 为 0、超时为 30 秒，知识工具注册 TTL 固定为 300 秒。
- 熔断：`CLOSED -> OPEN -> HALF_OPEN -> CLOSED`；连续失败 5 次打开，OPEN 冷却 60 秒后仅放行一次探测，探测失败重置冷却。
- 缓存：标准库 `OrderedDict` 内存缓存，仅缓存成功结果；TTL 到期删除；容量超过 5000 条时批量淘汰最早 1250 条。
- 调用顺序：Cache 查询、Breaker 判定、required/type/additionalProperties 校验、`asyncio.wait_for`、统计与熔断更新、fallback 或成功缓存；fallback 成功仍保留主 handler 的 `error`。
- 验证：`.venv\Scripts\python.exe -m pytest -q tests/test_tool_manager_circuit.py` → `7 passed`；全量 pytest → `95 passed`。
- 边界：Query Rewrite、并行召回、去重编排和 LLM Rerank 已完成；未接入真实外部业务写操作。

## Step 4.3：Query Rewrite、并行召回与稳定去重

- 状态：已完成；Checkpoint C/D 的 RAG 与工具韧性能力已闭环。
- 修改：扩展 `mcp/tool_manager.py`，增加可注入 Anthropic client、Query Rewrite、并行召回和多级稳定 identity 去重；新增 `tests/test_tool_manager_reasoning.py`。
- Query Rewrite：保留原始 query；LLM 最多生成 3 条改写；固定 `temperature=0.3`、`max_tokens=256`，不传 history、MemoryContext 或用户画像；返回非法或异常时回退为原始 query。
- 并行召回：按 `recall_k=max(top_k,5)` 为每个 query 调用现有 Tool，使用 `asyncio.gather(..., return_exceptions=True)`，单路异常不阻断其他成功召回。
- 去重：按 `source_chunk_id -> content_hash -> runtime_id -> content SHA256` 生成稳定 identity；同一 identity 保留较高 score，缺失 identity 的候选独立保留；不添加 score threshold，不进行最终排序或 Top-K 截断。
- 边界：Step 4.3 本身保持 `ToolResult.reranked=False`、`rerank_degraded=False`；Rerank 由 Step 4.4 独立处理，不引入 Cross-Encoder 或真实外部业务写操作。
- 验证：Step 4.3 专项 `.venv\Scripts\python.exe -m pytest -q tests/test_tool_manager_reasoning.py` → `9 passed`；Step 4.2 回归 → `7 passed`；全量 pytest → `104 passed`；compileall 与 `git diff --check` 通过。

## Step 4.4：LLM Rerank 重排序与降级状态机

- 状态：已完成，Checkpoint C/D 的 RAG 与工具治理验收通过；等待人工确认后再进入 Step 5。
- 修改：扩展 `mcp/tool_manager.py`，新增 `rerank_results(...)`、`rerank_results` Tool Use schema、完整 index permutation 校验和确定性降级；新增 `tests/test_tool_manager_rerank.py`。
- 触发门槛：只有去重候选数量大于 `top_k` 时才调用 LLM；候选数量不超过 `top_k` 时直接返回原候选集，不产生额外 LLM 调用。
- 正常路径：Anthropic Tool Use 固定 `temperature=0`、`max_tokens=256`，要求返回完整 `indices` 置换；校验无遗漏、无重复、无越界后按置换重排并返回前 `top_k`，标记 `reranked=True`、`rerank_degraded=False`。
- 降级路径：LLM 不可用、调用异常、缺失 tool_use 或置换非法时保留原物理召回顺序前 `top_k`，保留错误信息并标记 `reranked=False`、`rerank_degraded=True`。
- 安全边界：未引入 Cross-Encoder、数值型 rerank score 或 score threshold；不执行任何真实订单、支付、退款或人工坐席写操作。
- 验证：Step 4.4 专项、Step 4.3 与 Step 4.2 回归共 `21 passed`；全量 pytest `109 passed`；compileall 与 `git diff --check` 通过。

## Checkpoint C/D：RAG 与工具韧性

- [x] Chroma 基础检索、切片和 metadata 通过。
- [x] Circuit Breaker 5/60、TTL 300、timeout 30 通过。
- [x] Query Rewrite、并行召回、多级去重和高分保留通过。
- [x] LLM Rerank 完整置换校验、正常重排和降级状态通过。
- [x] `progress.md` 与 `architecture.md` 已同步。

## Step 5.1：AgentToolSpec 与通用工具

- 状态：已完成，等待人工确认后再进入 Step 5.2。
- 创建：`agents/tools.py` 定义 `AgentToolSpec`、`make_tool`、通用工具 schema 与注册表；新增 `tests/test_agent_tools_general.py`。
- 契约：工具 schema 强制 `type=object` 与 `additionalProperties=false`，`to_anthropic_tool()` 输出独立 schema 副本，handler 支持同步或异步执行。
- 通用工具：`inspect_request_context` 只返回 request_id、intent、脱敏 user_id 和时间戳；`suggest_required_fields` 按意图返回下一轮补字段建议。
- 安全边界：两个工具均为内存只读计算，不调用订单、支付、退款、改地址或人工坐席系统；非法入参统一返回结构化错误，不向主链路冒泡异常。
- 验证：`.venv\Scripts\python.exe -m pytest -q tests/test_agent_tools_general.py` → `5 passed`；全量 pytest → `114 passed`；`compileall agents tests` 与 `git diff --check` 通过。
- 边界：未实现 Technical/Billing/Escalation 专属工具和四类 Agent profile；Step 5.2 未开始。

## Step 5.2：Technical 与 Billing 专属工具

- 状态：已完成，等待人工确认后再进入 Step 5.3。
- 修改：扩展 `agents/tools.py`，新增静态错误码查询、低风险诊断计划、账单字段完整度检查和金额比较工具；新增 `tests/test_agent_tools_domain.py`。
- Technical：`lookup_error_code` 仅读取 400/401/403/404/429/500 静态目录，始终返回 `server_log_checked=False`；`build_diagnostic_plan` 只生成记录错误、检查网络、清缓存和重启应用等 low-risk 步骤。
- Billing：`check_billing_fields` 仅检查 `order_id`、`amount`、`date`、`payment_channel`，无论字段是否齐全都固定返回 `can_confirm_refund=False`；`compare_amounts` 只计算实际金额减预期金额的差值和大小关系。
- 安全边界：所有工具均为内存只读计算，不读取后台日志、不调用订单/支付接口、不确认退款，也不推断重复扣款等法律结论；所有工具 schema 均显式 `type=object` 与 `additionalProperties=false`。
- 验证：Step 5.2 专项 `.venv\Scripts\python.exe -m pytest -q tests/test_agent_tools_domain.py` → `9 passed`；Step 5.1/5.2 回归 → `14 passed`；全量 pytest → `123 passed`；`compileall agents tests` 与 `git diff --check` 通过。
- 边界：未实现 Escalation 工具、Agent profile 和真实业务写操作；Step 5.3 未开始。

## Step 5.3：Escalation 人工交接工具与安全红线

- 状态：已完成，等待人工确认后再进入 Step 5.4。
- 修改：扩展 `agents/tools.py`，新增 `create_handoff_summary`、严格 Anthropic schema 和 `escalation_tools()` 注册表；新增 `tests/test_agent_tools_escalation.py`。
- 交接契约：生成 8 位 `request_id`，返回 `reason`、`urgency`、脱敏后的 `entities`、`context_summary`、`handoff_status` 与升级建议文本；`urgency` 严格限制为 1–4 的整数。
- 安全边界：固定返回 `sensitive_data_required=False` 与 `external_ticket_created=False`；递归过滤 `password`、`verification_code`、`full_card_number` 等禁止字段及敏感提示词；不调用 JIRA、工单、订单、支付或人工坐席 API，不声称人工已接手。
- 异常：非法入参统一返回结构化安全错误，仍保留敏感数据禁止和无外部工单标志。
- 验证：Step 5.3 专项 `4 passed`；Step 5.1–5.3 回归 `18 passed`；全量 pytest `127 passed`；`compileall agents tests` 与 `git diff --check` 通过。
- 边界：未实现 Agent profile、编排器和真实人工系统接入；Step 5.4 未开始。

## Step 5.4：四大领域 Agent 实体与工具白名单隔离

- 状态：已完成，Checkpoint E-Agents 已闭环；等待人工确认后再进入 Step 6。
- 修改：扩展 `agents/agent_orchestrator.py`，新增 `AgentStats`、`BaseAgent`、`GeneralAgent`、`TechnicalAgent`、`BillingAgent`、`EscalationAgent` 及四套固定 Profile；新增共享只读 `search_knowledge_base` 工具占位；创建 `tests/test_agent_orchestrator.py`。
- Profile：General `0.3/900`，Technical `0.1/1200`，Billing `0.0/1100`，Escalation `0.0/500`（temperature/max_tokens）。
- 执行：BaseAgent 仅向 Anthropic 暴露自身 `tools` 白名单，最多执行 3 轮 Tool Use；未知工具直接拒绝；异常边界返回 `success=False` 并记录 total、success、total_ms。
- Escalation：重写 `handle()` 为零 Token 确定性摘要，只生成会话内升级建议，不创建工单、不设置真实接入状态。
- 验证：Step 5.4 专项 `5 passed`；Step 5 契约与工具回归 `30 passed`；全量 pytest `132 passed`；`compileall agents tests` 与 `git diff --check` 通过。
- 边界：尚未实现路由、Fan-out、ResponseComposer、真实 KnowledgeBase handler 注入和 Step 6 编排逻辑。

## Checkpoint E-Agents

- [x] 四类 Agent Profile、temperature/max_tokens 与白名单隔离通过。
- [x] BaseAgent 三轮 Tool Use 上限、异常隔离与统计通过。
- [x] EscalationAgent 确定性摘要、不调用 LLM、不声明人工已接入通过。
- [x] `progress.md` 与 `architecture.md` 已同步。
