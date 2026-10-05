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
- 边界：`intent/intent_recognizer.py`、`agents/agent_orchestrator.py`、`tests/test_internal_contracts.py` 尚未创建，Step 1.3 未开始。

## Step 1.3：内部核心 Dataclass 契约

- 状态：已完成，等待人工确认后再进入 Step 2。
- 已创建：`intent/intent_recognizer.py`、`agents/agent_orchestrator.py`、`tests/test_internal_contracts.py`。
- 枚举：18 个 `IntentCategory` value，`ESCALATION` 为 `HUMAN_HANDOFF` 别名；`UrgencyLevel` 为 1–4 的 `IntEnum`；`AgentType` 为四类字符串枚举。
- Dataclass：`IntentResult`、`AgentProfile`、`AgentResponse`、`Request`、`RoutingDecision`、`OrchestratorResult`，默认可变集合均使用 `default_factory`。
- 验证：`.venv\Scripts\python.exe -m pytest -q tests/test_environment_contract.py tests/test_api_contracts.py tests/test_internal_contracts.py` → `23 passed`。
- 边界：未实现意图识别算法、路由评分、Agent 执行、LLM、Redis、Chroma 或 Step 2 逻辑。

## Step 2.1：Pattern 规则、实体抽取与紧急度判定

- 状态：已完成，等待人工确认后再进入 Step 2.2。
- 修改：`intent/intent_recognizer.py` 增加 18 类 Specific/Generic Pattern 表、`match_patterns`、`extract_entities`、`calculate_urgency`、`apply_500_override`。
- Pattern：单规则命中 0.5，双规则 0.75，三次及以上封顶 1.0；返回候选、命中数、分数、最高候选和 `specific_score`。
- 实体：固定返回 `order_id`、`product`、`date`、`amount`、`error_code` 五个 key；`product` 保持空数组。
- 紧急度：按 CRITICAL > HIGH > MEDIUM > 意图兜底 > LOW；500 非登录意图覆盖为 `technical_crash` 且置信度至少 0.8。
- 验证：`.venv\Scripts\python.exe -m pytest -q tests/test_environment_contract.py tests/test_api_contracts.py tests/test_internal_contracts.py tests/test_intent_patterns.py` → `40 passed`。
- 边界：未实现 LLM Tool Use、Embedding、三路融合、复合意图或调度逻辑。

## Step 2.2：LLM Tool Use 意图识别适配器

- 状态：已完成，等待人工确认后再进入 Step 2.3。
- 修改：`intent/intent_recognizer.py` 增加严格的 `classify_intent` Tool Use schema、`LLMIntentResult` dataclass、异步 `_recognize_with_llm` 及 Anthropic 响应解析与失败降级。
- 契约：工具名固定为 `classify_intent`；18 个意图值由 `IntentCategory` 生成；`intent`、`confidence`、`reasoning` 为必填字段；`additionalProperties=False`；强制 `tool_choice`。
- 解析与容错：合法 payload 返回 `success=True`；未知意图、缺失/额外字段、无 tool_use、非法 reasoning、非有限置信度和客户端异常统一返回 `OTHER`、`0.0`、`success=False`；数值置信度越界安全截断到 0..1。
- 测试：新增 `tests/test_intent_llm_tool.py`，使用 `AsyncMock`，不发真实公网请求；覆盖成功解析、调用参数、schema、未知意图、越界置信度、缺失字段、无 tool_use 与超时异常。
- 验证：`.venv\Scripts\python.exe -m pytest -q` → `46 passed`；`.venv\Scripts\python.exe -m compileall -q core tests` 通过。
- 边界：未实现 Embedding、三路加权融合、复合意图和 Step 2.3 逻辑；默认 Anthropic client 仅在调用方未传 client 时惰性创建。

## Step 2.3：Embedding 与三路加权融合

- 状态：已完成，等待人工确认后再进入 Step 2.4。
- 修改：`intent/intent_recognizer.py` 增加 18 类模板文本、固定 256 维字符 1/2/3-gram MD5 哈希向量、余弦相似度、可选远程 Embedding 尝试与本地降级，以及异步 `recognize_intent` 三路汇聚。
- 融合：LLM、Embedding、Pattern 使用 `asyncio.create_task` 与 `gather(return_exceptions=True)`；Embedding 成功时权重为 0.7/0.2/0.1，来源失败时切换为 0.85/0.15。
- 阈值：融合分低于 0.5 强制 `OTHER`；LLM 失败时仅接受 Embedding 分数不低于 0.45；特异 Pattern 分数不低于 0.5 且融合低于 0.8 时细化泛化意图；500 非登录请求最终覆盖为 `technical_crash` 且置信度至少 0.8。
- 诊断：`IntentResult` 增加 `intent_diagnostics`，记录来源意图/分数、权重、失败状态、fallback、细化、最终意图与置信度；同时装配 urgency、entities 与 latency。
- 测试：新增 `tests/test_intent_fusion.py`、`tests/test_intent_diagnostics.py`，覆盖本地 256 维向量、三路权重、0.5、0.45、0.8、500 覆盖和诊断字段。
- 验证：`.venv\Scripts\python.exe -m pytest -q` → `54 passed`；`.venv\Scripts\python.exe -m compileall -q core tests` 与 `git diff --check` 通过。
- 边界：未实现 Step 2.4 复合意图 clause 切分、secondary intents 或 scoped retrieval query；远程 Embedding 失败只回退本地哈希向量，不引入训练模型依赖。

## Step 2.4：复合意图解析与跨域次要意图切分

- 状态：已完成，Checkpoint B 已闭环；未开始 Step 3。
- 修改：`intent/intent_recognizer.py` 增加连接词/标点白名单切分、最长 Pattern 命中候选、三大业务域映射、同域与泛化候选过滤、跨域次意图去重及最多 3 个的硬上限。
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
- 产品知识源文本：仅落位 6 篇产品客服 Markdown；未复制老工程 `data/chroma` 二进制，也未将评测语料混入产品 collection。评测基线语料已独立迁移为 `backend/evaluation/datasets/benchmark_corpus.json`。
- 评测适配：在保留原始基线文件的前提下，已生成 `shopcare_evaluation_cases.json` 与 `shopcare_evaluation_corpus.json`，统一当前 ShopCare 的 message、意图、Agent、知识门控、工具与安全约束字段。
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

## Step 6.1：路由评分算法、澄清逻辑与执行降级

- 状态：已完成，等待人工确认后再进入 Step 6.2。
- 修改：扩展 `agents/agent_orchestrator.py`，新增 `_domain_scores`、`_route_decision`、`_execute`、领域关键词/意图矩阵、`AGENT_POOL` 和固定澄清文案；新增 `tests/test_routing_decision.py`。
- 评分：General 基础 `0.1`；主意图加分 General `0.55`、Technical/Billing `0.75`；Technical/Billing 关键词每个 `0.18`、封顶 `0.45`；General 关键词每个 `0.12`、封顶 `0.35`；实体加分 error_code `0.2`、amount `0.15`、order_id `0.1`。
- 路由：CRITICAL 以置信度 `1.0` 直接进入 Escalation；HUMAN_HANDOFF 以 `max(intent_confidence, 0.8)` 进入 Escalation；OTHER 且文本长度大于 `2`、置信度低于 `0.5` 时返回固定澄清问题并使用 `primary_agent=None`，不派发 Agent。
- 辅助与降级：supporting 只从识别器给出的 `secondary_intents` 映射，过滤同域、重复和 Escalation；Technical/Billing 失败自动调用 General，General/Escalation 失败只返回友好失败提示。
- 验证：Step 6.1 专项 `5 passed`；全量 pytest `137 passed`；`compileall agents tests` 与 `git diff --check` 通过。
- 边界：尚未实现 Fan-out、ResponseComposer、强制知识检索和 Step 6.2。

## Step 6.2：并发 Fan-out、强制知识检索与 Trace 收集

- 状态：已完成，等待人工确认后再进入 Step 6.3。
- 修改：扩展 `agents/agent_orchestrator.py`，新增轻量 `AgentOrchestrator`、`_fan_out`、单 Agent 知识前置检索、Prompt 上下文注入、search 工具动态剔除和 Trace 聚合；新增 `tests/test_knowledge_required_retrieval.py`、`tests/test_retrieval_stage_trace.py`。
- 强制检索：`knowledge_required=True` 时，每个实际 Agent 只调用一次 `search_knowledge_base`，固定 `top_k=5`；支持 `scoped_retrieval_queries` 按领域传递查询；同步 `KnowledgeBase.search` 使用 `asyncio.to_thread`。
- 工具隔离：检索完成后通过 `tools_override` 从该 Agent 的 Anthropic 工具声明与工具映射中移除 `search_knowledge_base`，避免模型重复召回；检索失败仍注入不可用说明，不伪造知识结果。
- Fan-out：使用 `asyncio.gather(..., return_exceptions=True)` 并发执行主/辅助 Agent；辅助异常只记录日志并跳过，成功主结果继续返回；本阶段不实现 ResponseComposer。
- Trace：聚合去重后的 `tools_used`，保留 `tool_name`、`input`、`latency_ms`、`status`、`retrieval_stage`、召回切片和摘要预览；在 `intent_diagnostics["retrieval_stage"]` 与内存 Trace 队列中保存。
- 验证：Step 6.2 专项 `4 passed`；相关 Agent/路由回归 `14 passed`；全量 pytest `141 passed`；`compileall agents tests` 与 `git diff --check` 通过。
- 边界：未实现 ResponseComposer 语义合并、最终 Fan-in 冲突消解和 Step 6.3。

## Step 6.3：ResponseComposer 语义合并与 OrchestratorResult 装配

- 状态：已完成，Checkpoint D 已闭环；等待人工确认后再进入 Step 7。
- 修改：扩展 `agents/agent_orchestrator.py`，新增 `ResponseComposer`、多结果合成/确定性降级和 `AgentOrchestrator` 的完整 `OrchestratorResult` 装配；新增 `tests/test_response_composer.py`。
- 单结果：仅一个成功分支时直接返回其 content，不额外调用 Composer LLM，降低 Token 消耗与延迟。
- 多结果：以路由主 Agent 结论为主体调用 Anthropic 合成，默认 `temperature=0.1`、`max_tokens=1000`，支持 `CUSTOMEROPS_COMPOSER_TEMPERATURE` 与 `CUSTOMEROPS_COMPOSER_MAX_TOKENS` 覆盖；Prompt 要求去重、平滑冲突、只输出中文客户段落且不编造外部业务结果。
- 降级：Composer 未配置、超时、异常或空文本时，按主结果加 `\n\n补充说明：\n` 和辅助结果固定拼接；全盘失败返回 `抱歉，所有 Agent 均处理失败。`。
- 结果装配：`OrchestratorResult` 使用 `@dataclass(frozen=True)` 汇聚合成文本、主/辅助 Agent、路由原因与置信度、耗时、去重 `tools_used`、`tool_traces`、检索阶段诊断、secondary intents 和 handoff 标志。
- 安全边界：只处理已成功的内部 Agent 文本，不执行订单、支付、退款、后台查询或真实人工接入；不引入新第三方依赖。
- 验证：`tests/test_response_composer.py` → `5 passed`；Agent/Fan-out 回归 → `9 passed`；全量 pytest → `146 passed`；compileall 与 `git diff --check` 通过。

## Checkpoint D：编排与 Fan-in

- [x] 单结果直出、多结果语义合成和 Composer 异常确定性拼接通过。
- [x] 全盘失败固定安全文案和局部异常隔离通过。
- [x] `OrchestratorResult` 完整装配合成文本、路由、Agent、工具、Trace 与检索诊断元数据。
- [x] `progress.md` 与 `architecture.md` 已同步。

## Step 7.1：FastAPI Lifespan 生命周期与全局组件装配

- 状态：已完成，等待人工确认后再进入 Step 7.2。
- 修改：扩展 `api/main.py`，新增现代 FastAPI `lifespan` 上下文管理器、环境配置快照、Redis/Anthropic 客户端工厂、KnowledgeBase 与 MemoryManager 注入、IntentRecognizer 适配器、四类 Agent 池、AgentOrchestrator、SkillManager、PerformanceMonitor 与 EndToEndEvaluator 状态装配；新增 `tests/test_api_lifespan.py`。
- 知识工具：注册 `knowledge_search` 及兼容的 `search_knowledge_base` 只读工具，`cache_ttl=300`、`supports_rerank=True`，异常 fallback 明确返回知识库不可用，不伪造业务结果。
- 生命周期：启动阶段读取 `.env`/环境变量并保存到 `app.state.settings`；缺少 `ANTHROPIC_API_KEY` 时记录警告并保留安全降级；退出时关闭可用异步资源并隔离关闭异常。
- 安全边界：启动阶段不访问订单、支付、退款、工单或人工坐席系统；当前 PerformanceMonitor 与 EndToEndEvaluator 仅完成生命周期装配，详细监控与评测流水线分别留待 Step 8、Step 9。
- 验证：`.venv\Scripts\python.exe -m pytest tests/test_api_lifespan.py -q` → `3 passed`；全量 pytest → `149 passed`；`compileall api tests` 与 `git diff --check` 通过。
- 边界：未实现 `/health`、`/chat`、记忆回写与支持性 API 路由；Step 7.2 未开始。

## Step 7.2：/health、/chat 与记忆时序闭环

- 状态：已完成，等待人工确认后再进入 Step 7.3。
- 修改：扩展 `api/main.py`，新增 `GET /health`、`POST /chat`、上下文读取兼容层、意图识别调用、内部 `Request` 装配、ChatResponse 全字段映射、消息安全写回和画像后台任务异常消费；扩展 `memory/conversation_memory.py`，新增 `MemoryManager.get_context` 统一组装摘要、近期消息、情景历史和画像；新增 `tests/test_chat_integration.py`。
- `/health`：未就绪返回 503；就绪时返回 status、Agent stats 以及 collection_name、embedding_model、doc_count。
- `/chat` 时序：分配/继承 conv_id → `get_context` → 最近 5 条历史意图识别 → 内部 Request → `orchestrator.run` → USER/ASSISTANT 工作记忆写回 → `asyncio.create_task(update_profile)` → 完整 ChatResponse。
- 异常边界：记忆读写失败只记录 warning 并保留响应链路；画像任务通过 done callback 消费异常，禁止后台异常冒泡改变 HTTP 响应。
- 验证：`.venv\Scripts\python.exe -m pytest tests/test_chat_integration.py -q` → `4 passed`；记忆与集成回归 → `19 passed`；全量 pytest → `153 passed`；compileall 与 `git diff --check` 通过。
- 边界：未实现 `/search`、知识上传、trace/supporting routes、监控路由和评测路由；Step 7.3 未开始。

## Step 7.3：知识检索、文档上传、Trace 追溯与技能重载路由

- 状态：已完成，Step 7 主链路支持性 API 已闭环；等待人工确认后再进入 Step 8。
- 修改：扩展 `api/main.py`，新增 `/search`、`/knowledge/add`、`/knowledge/upload`、`/knowledge/stats`、`/trace/tool/{request_id}`、`/trace/tools`、`/skills` 与 `/skills/reload`；新增 `tests/test_api_supporting_routes.py`。
- 知识管理：`/search` 复用 `KnowledgeBase.search`；文本和上传文件复用 `add_documents` 切片入库；上传仅允许 `.txt`、`.md`、`.json`，单文件上限 10MB，非 JSON 使用 UTF-8 `errors="ignore"` 解码；同步知识库操作经 `asyncio.to_thread` 隔离。
- Trace 与技能：Trace 查询最多读取编排器内存 deque 的 200 条记录，按 request_id 返回单条详情；响应明确标注 `storage=memory`、`restart_loses_data=true`。SkillManager 增加本地 Markdown 技能目录 `reload_skills()` 热扫描。
- 安全边界：保留开发期开放 CORS；管理与上传路由当前无鉴权，已在 API 文档中声明生产前需接入鉴权网关；不调用订单、支付、退款、工单或人工坐席系统。
- 验证：支持性路由专项 `4 passed`；指定 API 回归 `20 passed`；全量 pytest `157 passed`；`compileall api tests` 与 `git diff --check` 通过。

## Checkpoint 主链路

- [x] `/health` 在 fake 依赖下通过并返回产品知识库配置与 Agent 统计。
- [x] `/chat` 完成上下文读取、意图识别、编排、记忆回写和画像后台任务隔离。
- [x] `/search`、知识入库/上传、知识库统计、Trace 查询、Skill 列表/热加载路由通过离线测试。
- [x] Trace 明确为最多 200 条内存运行期数据，服务重启后丢失；更新 `progress.md` 与 `architecture.md`。

## Step 7.4：Vue 3 + Vite 前端工作台与质感层级重构

- 状态：已完成，Step 7 前后端主链路与管理界面已闭环；等待人工确认后再进入 Step 8。
- 修改：保留 `frontend/` Vue 3 + Vite 工程与后端 API 边界，重构 `App.vue`、`ChatView.vue`、`AdminView.vue` 及全局样式，建立浅灰画布、白色表面、深森林绿主色和环境光阴影令牌。
- 会话视图：顶部导航提供会话/管理切换、历史会话数量、新对话；历史记录改为 320px 抽屉式滑出面板，继续使用 `shopcare_sessions` localStorage 持久化；会话主体采用 960px 悬浮白色工作卡片，保留 ChatResponse 元数据、转人工提示与安全 Markdown 链接渲染。
- 管理视图：知识库、切片数量、嵌入模型与断路器状态拆分为四个统计浮岛；左侧仅保留文档拖拽上传与状态反馈，右侧保留检索探针，底部保留 Trace 表格，继续接入既有 `/health`、`/search`、`/knowledge/upload`、`/trace/tools`；技能列表与技能重载不再由前端管理台展示或调用。
- 视觉边界：画布 `#F8FAFC`、表面 `#FFFFFF`、嵌套灰槽 `#F1F5F9`、主色 `#2D6A4F`、边框 `#E2E8F0`；字号限制为标题 18/20px、正文 14px、元数据 12px；无 Emoji、机器人头像、状态点和发光特效。
- 验证：`npm install` 成功安装 35 个依赖；`npm run build` 通过；`npm run dev -- --host localhost` 在 `http://localhost:5173/` 正常启动；`git diff --check` 通过。

## Step 7.4 UI 可读性优化：耗时单位与 Markdown 渲染

- 修改：会话消息元数据和管理台 Trace 表中的 `latency_ms` 改为自适应人类可读单位：低于 60 秒显示两位小数秒，达到分钟后显示分钟与秒；缺失、非法或负值仍显示 `—`。
- 修改：客服回复增加安全 Markdown 渲染，支持标题、加粗、斜体、删除线、行内代码、代码块、有序/无序列表和表格；先转义 HTML，仅允许 `http`、`https`、`mailto` 链接。
- 修改：后端 `/chat` 的 `ChatResponse.latency_ms` 现在覆盖上下文读取、意图识别、Agent 编排和同步记忆写回；内部编排耗时单独保存在 `intent_diagnostics.orchestrator_latency_ms`。
- 修改：会话页从浏览器发起请求开始计时，使用客户端请求墙钟时间作为用户可见的“端到端耗时”，同时保留 `server_latency_ms` 供网络与服务端耗时差异排查。
- 边界：不改变 ChatResponse 字段契约；仅补充诊断字段和前端展示计时语义。

## Checkpoint 主链路

- [x] FastAPI `/health`、`/chat` 与支持性管理 API 已有前端调用入口。
- [x] 会话历史可本地持久化，新建/切换会话与 `conv_id` 对齐。
- [x] 管理界面接入知识检索、文档上传、技能热加载和内存 Trace 查询，并保留后端安全边界说明。
- [x] Step 7.4 前端构建与开发服务器启动验证通过；Step 8 未开始。

## 目录重构：前后端分离

- 状态：已完成目录移动，保留历史施工记录和根目录 `memory-bank/`。
- `backend/`：集中 Python 后端源代码、脚本、测试、六篇知识源、依赖文件、Chroma 运行数据和本地 BGE 模型目录。
- `frontend/`：继续作为独立 Vue 3 + Vite 工程；根目录 `pytest.ini` 将后端测试路径映射到 `backend/tests/`。
- 根目录保留：`AGENTS.md`、`.gitignore`、`pytest.ini`、`memory-bank/`、Git 元数据；`.venv`、缓存、模型和运行时数据不提交。

## Docker Compose 一键部署

- 新增：`backend/Dockerfile` 使用 `python:3.12-slim`，先安装 `torch==2.4.1+cpu`，再安装后端锁定依赖；以 UID 1000 非 root 用户运行，暴露 8000，并通过 `/health` 做容器健康检查。
- 新增：`frontend/Dockerfile` 采用 Node 20 Alpine 构建 Vue 静态产物，再由 Nginx Alpine 提供服务；`frontend/nginx.conf` 支持 Vue Router 回退和 `/api/` 到 `backend:8000` 的反向代理。
- 新增：根目录 `docker-compose.yml` 编排 Redis 7、ChromaDB 0.5.23、FastAPI 后端和 Nginx 前端，统一使用 `shopcare-net`，并持久化 Redis AOF 与 Chroma 数据卷。
- 启动顺序：Redis 与 Chroma 健康后启动后端，后端 `/health` 健康后启动前端；容器内通过服务名互联，不依赖宿主机 localhost。
- 边界：Compose 仅透传根目录 `.env` 的 Anthropic 配置，不提交密钥；BGE 模型默认在后端首次启动时按模型标识下载，个人本地模型目录不会打入镜像。
- 兼容：Windows 宿主机的 HTTP.sys 可能占用 80 端口，前端宿主端口改为 `${SHOPCARE_FRONTEND_PORT:-8080}`；容器内仍监听 80，Linux 或端口空闲环境可设置 `SHOPCARE_FRONTEND_PORT=80`。
- 验证：`docker compose config --quiet` 通过；前端生产镜像与后端生产镜像均构建成功；Docker Desktop 首次构建曾受本机 buildx 权限拦截，授权后已完成验证。
- 运行入口：`python -m uvicorn api.main:app --app-dir backend --reload`；知识库脚本：`python backend/scripts/init_knowledge_base.py`。

## Skill 能力落地与 Agent Prompt 注入

- 状态：已完成，三个领域 Skill 已从既有 EchoMind 工程复制到 ShopCare 后端，保持 Markdown 原文和 frontmatter，不复制运行缓存或模型数据。
- 新增：`backend/skills/technical/SKILL.md`、`backend/skills/billing/SKILL.md`、`backend/skills/general/SKILL.md`。
- 新增：`backend/skills/skill_manager.py` 与 `backend/skills/__init__.py`；SkillManager 递归扫描根目录 Markdown 和嵌套 `SKILL.md`，解析 `name`、`description`、`keywords`、`agents`、`enabled`，单文件异常只隔离该文件。
- Agent 集成：四类 Agent 通过注入同一个 SkillManager，按照 Skill frontmatter 的 `agents` 范围动态拼接系统 Prompt；单 Agent Prompt 的 Skill 文本上限为 `CUSTOMEROPS_SKILLS_MAX_PROMPT_CHARS`（默认 5000），避免上下文无限增长。
- 路径边界：`CUSTOMEROPS_SKILLS_DIR` 默认 `skills`；从仓库根目录使用 `--app-dir backend` 启动时自动回退解析到 `backend/skills`，容器内仍使用 `/app/skills`。
- 安全边界：Skill 只提供本地只读规则文本，不执行工具、不调用外部订单/支付/退款/工单系统；`/skills` 与 `/skills/reload` 仅返回本地扫描快照。
- 验证：`test_skill_manager.py`、`test_agent_orchestrator.py`、`test_api_lifespan.py` 聚焦回归通过；完整结果以最终全量测试为准。

## Skill 品牌名称统一

- 三个领域 `SKILL.md` 中的旧品牌 `CustomerOps` 已统一替换为 `ShopCare`；规则正文、Agent 范围、安全约束和加载逻辑未改变。
- 验证：Skill/Agent 回归 `10 passed`；`git diff --check` 通过。
