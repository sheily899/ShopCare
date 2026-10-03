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
