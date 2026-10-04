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

Step 4.1 已完成；Step 4.2 的 Tool、缓存与 CircuitBreaker 尚未开始。

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

本次升级实测六篇源文档生成 18 个 chunk，最大长度 244 字符；Step 4.2 的 ToolManager/CircuitBreaker 仍未开始。

### Step 4.1 回归修复：Chroma 本地 fallback 设置隔离

| 文件 | 职责变化 |
| --- | --- |
| `mcp/knowledge_base.py` | HTTP `HttpClient` 失败后重新创建独立的本地 Settings，并明确固定 `chromadb.api.segment.SegmentAPI`；不再复用会被 HTTP 客户端原地改写为 FastAPI 的 Settings，确保 `PersistentClient` 真正访问本地持久化目录。 |
| `tests/test_knowledge_base_embedding.py` | 模拟真实 `HttpClient` 对 Settings 的原地污染，并断言 fallback 使用 SegmentAPI 配置，防止 404 后再次走 HTTP。 |

验证：全量 pytest `88 passed`；真实 Chroma 0.5.23 在 `localhost:8000` 返回 404 时可成功创建本地 `PersistentClient`。

### Step 4.1：本机 BGE 模型复用

运行时发现并复用了 Lenovo RAG 模块中的本地模型目录，复制到 `models/BAAI/bge-small-zh-v1.5`。当前目录包含完整 tokenizer、Transformer 配置、Pooling 配置和 `model.safetensors`/`pytorch_model.bin`，使用 ShopCare `.venv` 在 CPU 上加载成功，输出维度为 512；由于源目录的内部 `_name_or_path` 记录为 `BAAI_bge-small-zh`，文档将其标记为已验证可运行的本机复用候选，而非凭文件名声称远程仓库版本绝对一致。

正式运行时通过环境变量指定本地目录：

    $env:CUSTOMEROPS_EMBEDDING_MODEL="D:\\EchoMind\\ShopCare\\models\\BAAI\\bge-small-zh-v1.5"

该模型目录和 Hugging Face 运行缓存已加入 `.gitignore`，不进入版本库；产品 collection 仍固定为 `knowledge_base_zh`，未改变模型职责或 Step 4.2 范围。
