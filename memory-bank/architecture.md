# ShopCare 代码职责记录

## 当前仓库目录布局（2026-10-04）

当前可执行工程已按前后端分离整理：

| 目录 | 职责 |
| --- | --- |
| `frontend/` | Vue 3 + Vite 会话与管理界面。 |
| `backend/` | Python/FastAPI API、Agent、意图、记忆、MCP、脚本、测试、知识源和后端依赖配置。 |
| `memory-bank/` | 规格、设计、实施计划、进度与架构记录，保留在仓库根目录。 |
| `AGENTS.md`、`.gitignore`、`pytest.ini` | 项目级规则、忽略边界和根目录测试入口。 |

本节之后早期 Step 记录中的 `api/`、`agents/`、`tests/` 等路径是重构前的历史路径；当前对应位置统一为 `backend/<原路径>/`。历史记录保留用于追踪，不代表仍存在同名根目录。

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

## Step 2.1：Pattern 规则、实体抽取与紧急度判定

| 文件 | 职责 |
| --- | --- |
| `core/intent_recognizer.py` | 在既有枚举/结果契约之上增加 18 类 Specific/Generic 正则规则、命中计分、五键实体抽取、固定紧急度优先级和 500 状态码覆盖辅助函数。所有逻辑均为标准库纯函数。 |
| `tests/test_intent_patterns.py` | 覆盖 Pattern 表完整性、0.5/0.75/1.0 计分、退款/物流/投诉/人工、实体字段、紧急度优先级、意图兜底和 500 特例。 |

Pattern 结果同时保留旧兼容字段 `intent`/`confidence` 与后续融合所需的 `candidates`、`scores`、`hits`、`specific_score`；本阶段不发起网络、LLM、Embedding 或存储调用。

## Step 2.2：LLM Tool Use 意图识别适配器

| 文件 | 职责 |
| --- | --- |
| `core/intent_recognizer.py` | 增加 `CLASSIFY_INTENT_TOOL` 严格 JSON Schema、内部 `LLMIntentResult` dataclass、历史上下文提示构造、Anthropic `messages.create` 异步 Tool Use 调用、`tool_use` payload 校验和统一失败降级。客户端采用依赖注入；未传 client 时才惰性创建 SDK 客户端。 |
| `tests/test_intent_llm_tool.py` | 使用 `AsyncMock` 验证 Tool Use 成功解析、工具名与 `tool_choice`、18 类枚举、`additionalProperties=False`、非法意图、缺失/额外字段、越界置信度、无 tool_use 和超时异常。 |

`LLMIntentResult` 保持内部 dataclass 边界，并提供属性访问及兼容的 `result["field"]` / `result.get(...)` 读取方式；本阶段不执行 Embedding、Pattern 加权融合或任何真实公网请求。

## Step 2.3：Embedding 与三路加权融合

| 文件 | 职责 |
| --- | --- |
| `core/intent_recognizer.py` | 增加 `EmbeddingIntentResult`、18 类意图模板、固定 256 维 MD5 1/2/3-gram 向量、余弦相似度和可选远程 Embedding 适配；实现 `recognize_intent` 的 LLM/Embedding/Pattern 并发、0.7/0.2/0.1 与 0.85/0.15 权重、0.5/0.45/0.8 阈值、500 覆盖、实体/紧急度/延迟装配及诊断记录。 |
| `tests/test_intent_fusion.py` | 验证三路正常融合、Embedding 失败权重切换、0.5 硬拦截、LLM 失败时 0.45 fallback、Pattern 细化和 500 终极覆盖。 |
| `tests/test_intent_diagnostics.py` | 验证本地向量为固定 256 维、相似度范围、local fallback 标志以及 source/final diagnostics 完整性。 |

本阶段没有实现复合意图拆分；`secondary_intents` 与 `compound_trigger` 保持契约默认值，等待 Step 2.4。

## Step 2.4：复合意图解析与跨域次要意图切分

| 文件 | 职责 |
| --- | --- |
| `core/intent_recognizer.py` | 定义连接词/标点白名单与 clause 切分；按最长规则命中选择子句候选；通过 technical/billing/general/escalation 域映射过滤泛化、同域和人工辅助候选；保留最多 3 个跨域 `secondary_intents`，生成 `compound_trigger`、clause diagnostics 与按领域的 scoped retrieval queries，并将结果装配到 `IntentResult`。 |
| `tests/test_compound_intent.py` | 验证无分割信号时的单意图守门、技术+账单跨域识别、同域过滤、次意图 3 个上限、人工终止规则及 `recognize_intent` 字段装配。 |

本阶段仍不执行 Agent fan-out 或知识库检索；scoped query 只是识别层输出，后续编排器必须只消费已确认的 `secondary_intents`，不能重新通过关键词猜测辅助 Agent。

## Step 3.1：Message 与 Redis 工作记忆

| 文件 | 职责 |
| --- | --- |
| `memory/conversation_memory.py` | 定义 `MsgRole`、`Message`、`MemoryContext`；实现消息 JSON 往返、Prompt 上下文截断，以及 `MemoryManager` 的 Redis `LPUSH`/`EXPIRE`/`LRANGE`/`LLEN` 工作记忆读写、时间顺序恢复和 15 条压缩入口信号。 |
| `tests/test_memory_working.py` | 使用内存 Fake Redis 验证消息序列化、key 命名、TTL、逆序恢复、压缩阈值、Prompt 截断和 Redis 异常安全路径。 |

Step 3.1 的短期 Redis 工作记忆已完成；用户画像异步更新仍由 Step 3.3 单独实现。

## Step 3.2：会话摘要与 Chroma 情景记忆

| 文件 | 职责 |
| --- | --- |
| `memory/conversation_memory.py` | 在 Redis 工作记忆之上实现 15 条阈值触发的 `messages[:-5]` 摘要压缩、固定参数 Anthropic 摘要调用、800 字符摘要上限、摘要 Redis TTL 写回和最新 5 条热记忆重建；通过 `asyncio.to_thread` 写入固定 `episodic` Chroma 集合，并按当前会话优先、同用户补足的顺序检索最多 5 条历史。 |
| `tests/test_memory_episodic.py` | 使用内存 Fake Redis、LLM 和 Chroma collection 验证摘要输入切分、固定降级文案、Redis 状态重置、Chroma 元数据与 500 字符 `full_text` 限制、两级召回优先级、`to_thread` 和异常隔离。 |

Step 3.2 仍不包含用户画像集合、画像异步后台写回或 Step 3.3 的 profile 更新任务。

## Step 3.3：用户画像异步更新与检索

| 文件 | 职责 |
| --- | --- |
| `memory/conversation_memory.py` | 从当前会话最新 10 条工作记忆生成结构化用户画像；校验 `preferences` 与 `entities` 后，以稳定 `profile:{user_id}` 主键写入 `user_profile` Chroma 集合，并通过 `asyncio.to_thread` 隔离同步操作；提供画像读取、JSON 反序列化和空画像/异常兜底。画像更新协程捕获全部异常，适合 API 层后续用 `asyncio.create_task` 调度。 |
| `tests/test_memory_profile_async.py` | 使用 Fake Redis、LLM 和 Chroma collection 验证画像正常提取、固定 LLM 参数、稳定主键与元数据、畸变 JSON 防覆盖、后台异常隔离、画像读取和默认空结构。 |

Checkpoint C-1 已完成；Step 3.3 不在 API 路由中提前接入后台任务，也不包含 Step 4 的知识库检索和工具断路器。

## Step 4.1：KnowledgeBase 初始化与文档切片

| 文件 | 职责 |
| --- | --- |
| `mcp/knowledge_base.py` | 提供产品知识库的 Chroma HTTP 优先连接、本地 PersistentClient fallback、`knowledge_base_zh` collection 初始化、旧 `knowledge_base` 迁移、BGE CPU/归一化 embedding 配置、Markdown Header-Aware 面包屑切片、500 字符硬上限、七项 chunk metadata 和基础向量检索；不包含 ToolManager、缓存、熔断、Query Rewrite 或 Rerank。 |
| `tests/test_knowledge_base_embedding.py` | 使用 fake Chroma client 验证 HTTP 失败后的显式日志与 PersistentClient fallback、默认 collection、BGE CPU/normalize 配置、HNSW l2 和旧 collection 迁移。 |
| `tests/test_knowledge_base_metadata.py` | 验证切片长度上限、七项 metadata 完整性、稳定 `source_chunk_id` 与 `score=round(1-distance,4)` 检索结果；低分结果不因 threshold 被过滤。 |

Step 4.1 已完成；Step 4.2 的 Tool、缓存与 CircuitBreaker、Step 4.3 的召回前处理和 Step 4.4 的 LLM Rerank 均已完成。

### Step 4.1 知识源文本与初始化脚本

| 文件 | 职责 |
| --- | --- |
| `data/knowledge/refund_policy.md` | 退款申请、审核、原路退回时效与只读边界。 |
| `data/knowledge/order_logistics.md` | 订单状态、物流时效、配送和改地址边界。 |
| `data/knowledge/account_security.md` | 密码重置、两步验证和敏感信息保护。 |
| `data/knowledge/technical_troubleshooting.md` | 401/403/404/500 错误码、网络和应用排查。 |
| `data/knowledge/membership_points.md` | 会员等级、积分累计、抵扣和有效期规则。 |
| `data/knowledge/invoice_and_payment.md` | 发票字段、支付失败和重复扣款核验边界。 |
| `scripts/init_knowledge_base.py` | 扫描六篇 Markdown 源文档，复用 KnowledgeBase 切片和 metadata 规则，校验后写入产品 collection 并输出统计；不处理 benchmark 语料。 |
| `tests/test_knowledge_base_ingestion.py` | 验证六篇源文档读取、500 字符切片上限、七项 metadata 和关键词召回 score。 |

源文本是可审计的重建输入；老工程 `data/chroma` 的二进制索引不纳入 ShopCare，评测库保持独立 collection。

### Step 4.1：Header-Aware 语义切片升级

| 文件 | 职责 |
| --- | --- |
| `mcp/knowledge_base.py` | 按 Markdown `##`/`###` 标题优先拆分，在每片正文前注入标题面包屑；对超长小节按段落、句号和硬截断拆分，并以 `{source_document_id}_chunk_{chunk_index}` 生成稳定片段身份。 |
| `mcp/__init__.py` | 将本地 MCP/RAG 目录声明为明确 Python 包，避免直接执行初始化脚本时被环境中的同名第三方 `mcp` 包覆盖。 |
| `scripts/init_knowledge_base.py` | 复用带标题上下文的切片器，校验七项 metadata、统计最大 chunk 长度，并输出文档数、chunk 数和耗时。 |
| `tests/test_knowledge_base_chunking.py` | 验证六篇文档生成 16–22 个 chunk、面包屑、500 字符上限、metadata 完整性及 401 专属片段召回。 |

本次升级实测六篇源文档生成 18 个 chunk，最大长度 244 字符；Step 4.2、Step 4.3 和 Step 4.4 已在后续完成。

### Step 4.1 回归修复：Chroma 本地 fallback 设置隔离

| 文件 | 职责变化 |
| --- | --- |
| `mcp/knowledge_base.py` | HTTP `HttpClient` 失败后重新创建独立的本地 Settings，并明确固定 `chromadb.api.segment.SegmentAPI`；不再复用会被 HTTP 客户端原地改写为 FastAPI 的 Settings，确保 `PersistentClient` 真正访问本地持久化目录。 |
| `tests/test_knowledge_base_embedding.py` | 模拟真实 `HttpClient` 对 Settings 的原地污染，并断言 fallback 使用 SegmentAPI 配置，防止 404 后再次走 HTTP。 |

验证：全量 pytest `88 passed`；真实 Chroma 0.5.23 在 `localhost:8000` 返回 404 时可成功创建本地 `PersistentClient`。

### Step 4.1：本机 BGE 模型复用

运行时发现并复用了 Lenovo RAG 模块中的本地模型目录，复制到 `backend/models/BAAI/bge-small-zh-v1.5`。当前目录包含完整 tokenizer、Transformer 配置、Pooling 配置和 `model.safetensors`/`pytorch_model.bin`，使用 ShopCare `.venv` 在 CPU 上加载成功，输出维度为 512；由于源目录的内部 `_name_or_path` 记录为 `BAAI_bge-small-zh`，文档将其标记为已验证可运行的本机复用候选，而非凭文件名声称远程仓库版本绝对一致。

正式运行时通过环境变量指定本地目录：

    $env:CUSTOMEROPS_EMBEDDING_MODEL="D:\\EchoMind\\ShopCare\\models\\BAAI\\bge-small-zh-v1.5"

该模型目录和 Hugging Face 运行缓存已加入 `.gitignore`，不进入版本库；产品 collection 仍固定为 `knowledge_base_zh`，未改变模型职责或 Step 4.2 范围。

## Step 4.2：Tool、缓存与 Circuit Breaker

| 文件 | 职责 |
| --- | --- |
| `mcp/tool_manager.py` | 提供 `Tool`/`ToolResult`/`ToolStats` dataclass 契约；按 Cache → Breaker → 参数校验 → `asyncio.wait_for` → 统计/状态更新 → fallback 或成功缓存的顺序执行工具。使用标准库 `OrderedDict` 实现 5000 条容量、1250 条批量淘汰和 TTL；实现 5 次失败、60 秒冷却、半开探测的三态熔断；后续增加 Query Rewrite、并行召回、稳定去重和 LLM Rerank。 |
| `tests/test_tool_manager_circuit.py` | 覆盖五次失败、60 秒恢复、半开探测失败、超时、参数校验不污染熔断、TTL 命中、5000/1250 淘汰和 fallback 保留原 error。异步测试使用标准库 `asyncio.run`，未新增 pytest 插件依赖。 |

验证：Step 4.2 专项测试 `7 passed`；全量 pytest 在 Step 4.4 完成后为 `109 passed`。

## Step 4.3：Query Rewrite、并行召回与稳定去重

| 文件 | 职责变化 |
| --- | --- |
| `mcp/tool_manager.py` | 在既有 Tool/缓存/熔断边界上增加可注入 Anthropic client、`rewrite_query(query, n=3)` 与 `search_with_rewrite(...)`；改写固定使用 `temperature=0.3`、`max_tokens=256`，不携带 history/context；每路以 `recall_k=max(top_k,5)` 并发调用基础工具，并通过 `return_exceptions=True` 隔离局部召回异常。 |
| `tests/test_tool_manager_reasoning.py` | 使用 fake Anthropic client 和 fake Tool handler 覆盖改写参数、JSON/fenced JSON 解析、异常回退、并发启动、recall_k、局部失败、全失败和多级稳定 identity 去重。 |

去重优先级固定为 `source_chunk_id`、`content_hash`、`runtime_id`、content SHA256；同一 identity 保留较高 score，不同 identity 保留首次出现顺序，无 identity 的结果不强行合并。Step 4.3 返回完整去重候选集，不执行最终 Top-K、不调用 Rerank，也不设置 score threshold。

验证：Step 4.3 专项测试 `9 passed`；Step 4.2 回归 `7 passed`；在 Step 4.4 完成前全量 pytest 为 `104 passed`；compileall 与 `git diff --check` 通过。

## Step 4.4：LLM Rerank 与降级状态机

| 文件 | 职责变化 |
| --- | --- |
| `mcp/tool_manager.py` | 新增 `RERANK_TOOL` Tool Use schema 与 `rerank_results(...)` 异步方法。只有候选数量超过 `top_k` 才触发 LLM；固定 `temperature=0`、`max_tokens=256`，要求返回完整 `indices` index permutation，并校验长度、类型、遗漏、重复和越界。合法置换按模型顺序重排并截取前 `top_k`；LLM 不可用、调用异常或协议非法时保留原物理召回顺序前 `top_k`，设置 `rerank_degraded=True` 并保留错误信息。 |
| `tests/test_tool_manager_rerank.py` | 使用 fake Anthropic client 验证候选未超限时跳过 LLM、合法置换重排、网络异常降级、非法 index 降级、tool_use 缺失和无 score threshold 行为。 |

Rerank 只消费 Step 4.3 已完成去重的候选列表，不引入 Cross-Encoder、不产生数值型 rerank score，也不设置 Chroma score threshold。Step 4.3 的 `search_with_rewrite` 仍保持完整候选集语义，具体 Rerank 由调用方显式使用 `rerank_results(...)`。

验证：Step 4.4 专项、Step 4.3 和 Step 4.2 回归共 `21 passed`；全量 pytest `109 passed`；compileall 与 `git diff --check` 通过。

## Step 5.1：AgentToolSpec 与通用工具

| 文件 | 职责 |
| --- | --- |
| `agents/tools.py` | 定义 `AgentToolSpec` 工具元数据与 handler 契约；构造时强制校验 JSON Schema 为 object 且 `additionalProperties=false`，通过 `to_anthropic_tool()` 导出 Anthropic Tool Use 声明。实现 `inspect_request_context` 的只读脱敏请求快照、`suggest_required_fields` 的意图字段建议和 `general_tools()` 注册表；不连接任何外部订单/支付系统。 |
| `tests/test_agent_tools_general.py` | 覆盖 schema 硬约束、Anthropic 声明导出、通用工具注册、user_id 脱敏、退款/技术字段建议和非法入参安全错误。 |

本阶段只建立通用只读工具层，未实现 Technical/Billing/Escalation 工具、Agent profile 或真实业务写操作；Step 5.2 继续保持未开始状态。

验证：Step 5.1 专项测试 `5 passed`；全量 pytest `114 passed`；`compileall agents tests` 与 `git diff --check` 通过。

## Step 5.2：Technical 与 Billing 专属工具

| 文件 | 职责变化 |
| --- | --- |
| `agents/tools.py` | 增加 `lookup_error_code` 静态错误码目录与 `server_log_checked=False` 声明；增加 `build_diagnostic_plan` 低风险步骤生成；增加 `check_billing_fields` 字段完整度检查与固定 `can_confirm_refund=False`；增加 `compare_amounts` 纯数学差值；为 Technical/Billing 工具分别提供 JSON Schema、`AgentToolSpec` 和注册表工厂。 |
| `tests/test_agent_tools_domain.py` | 验证 401/403/404/500 静态解释、未知码与非法入参、低风险诊断步骤、账单字段缺失/完整两条路径、退款安全标志、金额算术和工具 schema。 |

本阶段不读取真实服务端日志、不调用订单或支付系统、不确认退款，也不对重复扣款作法律定性；Escalation 工具和 Agent profile 留待后续步骤。

验证：Step 5.2 专项测试 `9 passed`；Step 5.1/5.2 回归 `14 passed`；全量 pytest `123 passed`；`compileall agents tests` 与 `git diff --check` 通过。

## Step 5.3：Escalation 人工交接工具与安全边界

| 文件 | 职责变化 |
| --- | --- |
| `agents/tools.py` | 新增 `create_handoff_summary` 及其严格 object schema，生成 8 位 request_id、reason、urgency、脱敏实体和上下文摘要；通过 `sensitive_data_required=False`、`external_ticket_created=False` 与 `handoff_status=recommended` 明确这是会话内升级建议，不是工单创建或人工已接入。递归过滤禁止的 password、verification_code、full_card_number 字段和敏感提示词，非法入参走结构化错误路径。 |
| `tests/test_agent_tools_escalation.py` | 验证交接摘要字段、8 位 request_id、敏感数据标志、禁止字段过滤、非法入参安全返回、无外部网络副作用及 Escalation schema 注册。 |

Step 5.3 只提供结构化上下文快照，不执行任何真实工单、订单、支付、退款或人工坐席写操作；Step 5.4 的 Agent profile 与编排逻辑尚未开始。

验证：专项测试 `4 passed`；Step 5.1–5.3 回归 `18 passed`；全量 pytest `127 passed`；`compileall agents tests` 与 `git diff --check` 通过。

## Step 5.4：四大领域 Agent 实体与工具白名单隔离

| 文件 | 职责变化 |
| --- | --- |
| `agents/agent_orchestrator.py` | 新增 `AgentStats` 统计契约、`BaseAgent` Tool Use 执行基类和四类领域 Agent。BaseAgent 每次仅把自身 `tools` 转为 Anthropic schema，最多执行 3 轮 Tool Use，并在未知工具、LLM 异常和空响应时返回局部失败结果；统计 total/success/total_ms。General、Technical、Billing 分别绑定共享 `search_knowledge_base` 占位与各自领域工具；Escalation 保留 profile 声明但覆盖 `handle()`，直接生成会话内升级建议，不调用 LLM。 |
| `tests/test_agent_orchestrator.py` | 验证四类 Profile 的 temperature/max_tokens、Technical/Billing 白名单互斥、Escalation 零 LLM 确定性、BaseAgent 异常统计和三轮 Tool Use 上限。 |

共享 `search_knowledge_base` 当前仅是只读注入占位，返回不可用提示，不连接外部系统；真正的 KnowledgeBase handler、路由、Fan-out 和 ResponseComposer 留给 Step 6。Escalation 的 `escalate=True` 只表示升级建议，不等价于 API 的 `escalated=True`。

验证：Step 5.4 专项 `5 passed`；Step 5 契约与工具回归 `30 passed`；全量 pytest `132 passed`；`compileall agents tests` 与 `git diff --check` 通过。

## Step 6.1：路由评分、澄清与执行降级

| 文件 | 职责变化 |
| --- | --- |
| `agents/agent_orchestrator.py` | 新增 `_domain_scores(request, intent_result)`，按基础分、主意图、关键词封顶和实体权重计算三域相对分；新增 `_route_decision` 的 CRITICAL/HUMAN_HANDOFF 安全优先路由、低置信度 OTHER 澄清、主 Agent 选择和显式 secondary 映射；新增 `_execute` 的 Technical/Billing → General fallback 与 General/Escalation 友好失败隔离。 |
| `tests/test_routing_decision.py` | 验证 CRITICAL/HUMAN_HANDOFF 终止路由、OTHER 低置信度澄清不派发、Technical 领域加分、secondary 仅显式跨域映射和 Technical 异常降级 General。 |

评分值只用于相对路由，不代表概率；澄清分支用 `primary_agent=None` 表示无执行目标。Step 6.1 不创建并发任务，不实现 Fan-out、ResponseComposer 或真实业务接口。

验证：专项 `5 passed`；全量 pytest `137 passed`；`compileall agents tests` 与 `git diff --check` 通过。

## Step 6.2：并发 Fan-out、强制知识检索与 Trace 收集

| 文件 | 职责变化 |
| --- | --- |
| `agents/agent_orchestrator.py` | 为 `BaseAgent` 增加可选 `tools_override` 和上下文注入；新增单 Agent 知识前置检索、同步 KnowledgeBase 线程隔离、`_fan_out` 异常隔离和 `AgentOrchestrator.run_parallel/run` 入口。业务请求每个实际 Agent 只执行一次 `search_knowledge_base(query, top_k=5)`，随后从 Anthropic 工具白名单中移除 search；工具调用统一记录 `tool_name`、`input`、`latency_ms`、`status`，检索 Trace 记录召回切片摘要。 |
| `tests/test_knowledge_required_retrieval.py` | 验证每个 Agent 单次知识检索、固定 `top_k=5`、Prompt 上下文注入和 search 工具剔除。 |
| `tests/test_retrieval_stage_trace.py` | 验证 `asyncio.gather(return_exceptions=True)` 的局部异常隔离、工具去重聚合和 `retrieval_stage` Trace 元数据。 |

本阶段仅完成 Fan-out 与证据/Trace 汇聚，不实现 ResponseComposer 的语义合并；多路成功时暂保留主 Agent 文本，专业分支完整结果通过结构化 Trace 返回。知识检索失败只标记 failed 状态并注入谨慎提示，不伪造召回内容。

验证：Step 6.2 专项 `4 passed`；相关 Agent/路由回归 `14 passed`；全量 pytest `141 passed`；`compileall agents tests` 与 `git diff --check` 通过。

## Step 6.3：ResponseComposer 语义合并与结果装配

| 文件 | 职责变化 |
| --- | --- |
| `agents/agent_orchestrator.py` | 新增 `ResponseComposer`，按成功分支数量执行单结果直出、多结果 Anthropic 语义合并、异常确定性拼接和全盘失败固定兜底；`AgentOrchestrator.run_parallel` 将合成文本、Agent 类型、路由信息、耗时、工具清单、工具 Trace、检索阶段诊断和 handoff 标志完整装配进 `@dataclass(frozen=True) OrchestratorResult`。 |
| `tests/test_response_composer.py` | 覆盖单结果不调用合成模型、多结果参数与安全 Prompt、合成异常固定拼接、编排结果元数据和全盘失败文案。 |

合成默认使用 `temperature=0.1`、`max_tokens=1000`，可由 `CUSTOMEROPS_COMPOSER_TEMPERATURE` 和 `CUSTOMEROPS_COMPOSER_MAX_TOKENS` 覆盖。合成 Prompt 只允许输出中文客户段落，要求以主结论为主体、去除重复寒暄、平滑冲突，并禁止编造订单、退款或后台查询结果。合成器不执行任何外部业务写操作，也不引入 Cross-Encoder 或其他新依赖。

验证：Step 6.3 专项 `5 passed`；Agent/Fan-out 回归 `9 passed`；全量 pytest `146 passed`；`compileall agents tests` 与 `git diff --check` 通过。
## Step 7.1：FastAPI Lifespan 生命周期与全局组件装配

| 文件 | 职责变化 |
| --- | --- |
| `api/main.py` | 以现代 FastAPI `lifespan` 管理应用级依赖。启动时读取 `.env`/环境变量，构造 Redis 与可选 Anthropic 客户端，初始化 `KnowledgeBase`、`MCPToolManager`、`MemoryManager`、`IntentRecognizer`、四类领域 Agent、`AgentOrchestrator`、`SkillManager`、`PerformanceMonitor` 和 `EndToEndEvaluator`，并统一挂载至 `app.state`。 |
| `tests/test_api_lifespan.py` | 使用离线 fake Redis/KnowledgeBase 验证 TestClient 生命周期、核心状态装配、`knowledge_search` 工具注册、缺少 API key 的明确告警和资源关闭。 |

### 生命周期边界

- `KnowledgeBase` 继续负责 HTTP Chroma 优先与本地持久化 fallback；API 层只传递环境配置，不复制连接策略。
- `knowledge_search` 与兼容的 `search_knowledge_base` 均为只读工具，`cache_ttl=300`、`supports_rerank=True`，fallback 仅返回不可用说明，不伪造业务结果。
- `ANTHROPIC_API_KEY` 缺失时只记录 warning 并让应用进入安全降级状态；有 key 时才构造 Anthropic 客户端，启动阶段不主动调用模型。
- Redis 客户端仅创建连接对象，不在 lifespan 启动时执行 ping；退出阶段统一调用 `aclose`/`close` 并隔离关闭异常。
- 当前 `PerformanceMonitor`、`EndToEndEvaluator` 和 `SkillManager` 是为生命周期装配提供的轻量组件；详细监控、评测流水线与 `/skills` 路由仍属于后续 Step 7.3/Step 8/Step 9。
- API 层不向订单、支付、退款、工单或人工坐席系统发起网络调用；Step 7.2 的 `/health`、`/chat` 和记忆回写尚未实现。

验证：Step 7.1 专项 `3 passed`；全量 pytest `149 passed`；`compileall api tests` 与 `git diff --check` 通过。

## Step 7.2：/health、/chat 与记忆时序闭环

| 文件 | 职责变化 |
| --- | --- |
| `api/main.py` | 新增 `GET /health` 与 `POST /chat`。`/health` 检查 lifespan ready 状态并返回 Agent 统计、产品 collection/model/doc_count；`/chat` 严格执行上下文读取、最近 5 条历史意图识别、内部 Request 装配、AgentOrchestrator 执行、USER/ASSISTANT 消息写回、画像后台任务和完整 ChatResponse 映射。 |
| `memory/conversation_memory.py` | 新增 `MemoryManager.get_context`，统一读取 Redis 摘要、最多 20 条工作记忆、最多 5 条 episodic 历史和 user_profile，返回 `MemoryContext`；各存储层故障只返回安全空值。 |
| `tests/test_chat_integration.py` | 使用 fake Redis/Chroma/Memory/Recognizer/Orchestrator 覆盖 `/health` 正常与 503、/chat 完整响应、最近 5 条历史、消息写回顺序和画像任务失败隔离。 |

### 主链路与安全边界

```text
ChatRequest
  -> conv_id/request_id
  -> MemoryManager.get_context
  -> recent_messages[-5:] + IntentRecognizer.recognize_intent
  -> agents.Request
  -> AgentOrchestrator.run
  -> add_message(USER) -> add_message(ASSISTANT)
  -> create_task(update_profile) + done callback
  -> ChatResponse
```

- `knowledge_required` 由意图门控决定并进入内部 Request；知识检索仍由 Step 6.2 编排层负责，API 不复制 RAG 算法。
- ChatResponse 的 intent、Agent、routing、tools、handoff、latency、knowledge、entities、source scores 和 diagnostics 均从内部结果映射，不伪造外部订单、支付、退款或人工接入状态。
- `/health` 的 collection count 通过 `asyncio.to_thread` 读取，避免阻塞事件循环；未就绪时不返回 healthy。
- 画像更新是非阻塞增强任务；任务失败只写日志，不影响已完成的 HTTP 响应。

验证：Step 7.2 专项 `4 passed`；记忆与集成回归 `19 passed`；全量 pytest `153 passed`；`compileall api tests` 与 `git diff --check` 通过。

## Step 7.3：支持性 API 与运行期运维边界

| 文件 | 职责变化 |
| --- | --- |
| `api/main.py` | 新增知识库 `/search`、`/knowledge/add`、`/knowledge/upload`、`/knowledge/stats` 路由；新增内存 Trace `/trace/tool/{request_id}`、`/trace/tools` 查询；新增 `/skills` 与 `/skills/reload` 本地技能目录接口；`SkillManager` 增加 `reload_skills()`；保留开发期 CORS 开放基线。 |
| `tests/test_api_supporting_routes.py` | 使用离线 KnowledgeBase、Redis、SkillManager 和编排器替身，覆盖检索/入库、10MB 与扩展名校验、统计、Trace 内存边界和技能热加载。 |

### 支持性路由数据流与安全边界

```text
SearchRequest / UploadFile / KnowledgeAddRequest
        -> API 校验（扩展名、10MB、JSON 格式）
        -> asyncio.to_thread(KnowledgeBase.search/add_documents)
        -> 结构化检索或入库结果

AgentOrchestrator._recent_traces (deque maxlen=200)
        -> /trace/tools 或 /trace/tool/{request_id}
        -> storage=memory, restart_loses_data=true

SkillManager.skills_dir/*.md
        -> /skills/reload -> 本地重新扫描 -> /skills 返回新快照
```

- `/knowledge/upload` 只处理产品知识源文本，不导入评测库；非 JSON 文件使用 UTF-8 `errors="ignore"`，JSON 先做解析校验。
- `/search` 输出 `title`、`chunk_id`、`content`、`score`，不增加 score threshold，也不伪造检索概率含义。
- `/trace/*` 读取运行期内存数据，不是审计级持久化日志；应用重启后历史 Trace 丢失。
- 管理类 API 与 CORS 当前为开发期开放状态，代码文档已明确生产前需要鉴权网关；所有路由都不连接外部订单、支付、退款、工单或人工系统。

验证：支持性路由专项 `4 passed`；指定 API 回归 `20 passed`；全量 pytest `157 passed`；`compileall api tests` 与 `git diff --check` 通过。

## Step 7.4：Vue 3 + Vite 前端工作台与质感层级重构

| 文件 | 职责变化 |
| --- | --- |
| `frontend/package.json` / `frontend/vite.config.js` | 锁定 Vue `^3.5.25`、Vite `^7.2.7` 与 Vue 插件 `^6.0.2`，提供 dev/build/preview 脚本；开发服务默认 localhost:5173。 |
| `frontend/src/App.vue` | 提供会话/管理切换、统一的新对话操作与历史会话数量；通过抽屉式面板展示 `ChatView` 发出的 localStorage 会话摘要，不复制后端业务逻辑。 |
| `frontend/src/views/ChatView.vue` | 实现悬浮白色工作卡片、空状态引导、3 个快捷问题、localStorage 会话持久化、ChatRequest 发送、消息元数据、两位小数耗时、转人工文案和安全 HTML 转义 Markdown 链接；对外暴露新建/切换会话方法。 |
| `frontend/src/views/AdminView.vue` | 实现四个统计浮岛、文档上传与检索双卡片、10MB 文件上传状态反馈和内存 Trace 表格；移除技能列表与技能重载入口，耗时统一显示两位小数，嵌入模型展示为 CPU 简称。 |
| `frontend/src/api/index.js` | 统一封装对 `http://localhost:8000` 的 fetch 请求：`/chat`、`/health`、`/search`、`/knowledge/upload`、`/skills`、`/skills/reload`、`/trace/tools`。 |
| `frontend/src/styles.css` | 固化 `#F8FAFC` 画布、`#FFFFFF` 表面、`#F1F5F9` 灰槽、`#2D6A4F` 主色、环境光阴影和 12/14/18/20px 字号层级；实现导航胶囊、抽屉过渡、悬浮卡片、响应式表格，不引入 Emoji、图标库、霓虹特效或渐变。 |

### 前端数据流

```text
ChatView localStorage(shopcare_sessions)
        -> POST /chat {message, user_id, conv_id}
        -> ChatResponse
        -> assistant message + metadata
        -> localStorage/session switch

AdminView
  -> GET /health          -> collection/model/doc_count
  -> POST /search         -> result table
  -> POST /knowledge/upload -> chunk ingestion notice
  -> POST /skills/reload  -> skill snapshot
  -> GET /trace/tools     -> memory trace table
```

- 前端只负责展示与请求编排，不执行订单查询、退款、支付、改地址、工单创建或真实人工接入。
- Markdown 在浏览器端先做 HTML 字符转义，仅允许 `http`、`https`、`mailto` 链接；后端仍是事实安全边界。
- 管理 API 的开放鉴权缺口与 Trace 重启丢失仍由后端事实边界负责；前端只展示接口返回的运行数据，断路器统计卡属于当前开发期工作台展示项，不新增后端能力。

验证：`npm install` 成功安装 35 个依赖；`npm run build` 通过；`npm run dev -- --host localhost` 正常启动；`git diff --check` 通过。
