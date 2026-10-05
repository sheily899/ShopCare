# 系统事实基线与架构规格书

审计对象：D:\EchoMind\EchoMind（Python 服务端）与 D:\EchoMind\EchoMindFrontend（Vue 前端）。
审计日期：2026-10-03。本文仅记录源码已验证事实；未实现、Mock 或预留能力均明确标识。源代码项目名 CustomerOps 保持为事实名称。

## 0. 已验证主链路与边界

主链路：Vue 前端或 CLI -> POST /chat -> MemoryManager.get_context -> IntentRecognizer（LLM Tool Use + embedding + pattern 融合） -> AgentOrchestrator（单 Agent 或 fan-out） -> 业务请求强制 RAG -> ResponseComposer（多成功结果时） -> Redis 工作记忆同步写入、Chroma 用户画像异步更新。

- 已实现：客服问答、意图路由、向量检索、字段检查、金额算术、交接摘要、监控和评测。
- 未实现：订单查询、退款执行、支付核验、改地址、真实后台日志读取、真实工单创建、人工坐席系统。
- EscalationAgent 与 create_handoff_summary 只生成会话内文本或结构化摘要，不证明人工已接手。
- 前端预置 java 后端 URL（默认 /api/java，端口 8080），但两个审计源码目录中未发现 Java 服务端实现。

## 1. 业务域与意图分类体系

证据：intent/intent_recognizer.py、agents/agent_orchestrator.py。

### 1.1 硬编码分类

| 枚举 | value | 默认路由 |
| --- | --- | --- |
| QUERY | query | general |
| COMPLAINT | complaint | general |
| REQUEST | request | general |
| GREETING | greeting | general |
| TECHNICAL | technical | technical |
| BILLING | billing | billing |
| ACCOUNT | account | billing |
| FEEDBACK | feedback | general |
| ORDER_STATUS | order_status | general |
| LOGISTICS | logistics | general |
| REFUND | refund | billing |
| INVOICE | invoice | billing |
| PAYMENT_ISSUE | payment_issue | billing |
| ACCOUNT_SECURITY | account_security | billing |
| TECHNICAL_LOGIN | technical_login | technical |
| TECHNICAL_CRASH | technical_crash | technical |
| HUMAN_HANDOFF | human_handoff | escalation |
| OTHER | other | general |

ESCALATION = human_handoff 是 HUMAN_HANDOFF 别名；旧值 escalation 会归一为 human_handoff。UrgencyLevel 是 LOW=1、MEDIUM=2、HIGH=3、CRITICAL=4。

### 1.2 意图输出契约

IntentResult 字段：intent: IntentCategory、confidence: float、urgency: UrgencyLevel、intent_group: str、entities: Dict[str, List[str]]、reasoning: str、latency_ms: float、source_scores: dict={}、source_intents: dict={}、secondary_intents: list=[]、compound_trigger: str=""。

实体 key 固定为 order_id、product、date、amount、error_code；product 当前只初始化为空数组，未见商品实体提取。分组：order_status/logistics -> query；refund/invoice/payment_issue -> billing；account_security -> account；technical_login/technical_crash -> technical。

### 1.3 主通路、降级与阈值

1. 主通路使用 Anthropic Tool Use：强制 classify_intent 工具，JSON Schema 禁止额外属性；不是自由文本 JSON Mode。
2. LLM 与 embedding 通过 asyncio.create_task 并发，pattern 同步计算，再 asyncio.gather。
3. embedding 可用时权重为 LLM 0.7、embedding 0.2、pattern 0.1；不可用时 LLM 0.85、pattern 0.15。
4. 最高融合分 < 0.5 归 OTHER。LLM 失败时，embedding 仅达到默认 0.45 才可降级使用。
5. 特异 pattern 分数 >= 0.5 且融合结果 < 0.8 时可细化泛化意图。消息含 500 且最终不是 technical_login 时，强制 technical_crash，置信度至少 0.8。
6. embedding 优先尝试 voyage-3-lite；无接口时使用本地固定 256 维字符 1/2/3-gram MD5 哈希向量，不是训练好的备用语义模型。
7. pattern 一次命中得 0.5，多命中为 min(1.0, 0.5 + 0.25 * (hits - 1))。

### 1.4 单意图、多意图与紧急度

- 必须用 同时、另外、还想、还要、而且、并且、此外、分别、两个问题、两个诉求、除此之外、但是、不过、以及 等连接词或标点切出至少两个 clause，才可能形成复合请求。
- 仅保留跨域 secondary_intents，最多 3 个；human_handoff 是终止意图并清空次意图。不是两个关键词就并发。
- 紧急度：紧急/emergency/urgent/asap/立刻 -> CRITICAL；今天/马上/尽快/hurry/now -> HIGH；这周/soon/快点 -> MEDIUM。无命中时 human_handoff 为 HIGH，complaint 为 MEDIUM，其他 LOW。

## 2. 状态总线与核心数据契约

证据：api/main.py、agents/agent_orchestrator.py、memory/conversation_memory.py。

### 2.1 API 契约

POST /chat 输入 ChatRequest：message: str、user_id: str = anonymous、conv_id: Optional[str] = None；未传 conv_id 时生成 UUID。

ChatResponse 的字段完整为：conv_id、request_id=""、response、intent、secondary_intents=[]、intent_group="other"、agent_type、agent_types=[]、primary_agent=""、supporting_agents=[]、tools_used=[]、routing_reason=""、routing_confidence=0.0、escalated、agent_recommended_handoff=False、latency_ms、knowledge_used=False、entities={}、intent_confidence=0.0、intent_source_scores={}、intent_diagnostics={}。

knowledge_used 只是 search_knowledge_base 是否在 tools_used 中，并不验证回答证据正确性。

### 2.2 Agent 输入、输出与最终合成

Request：message、user_id、conv_id、context=""、history=None、entities={}、intent=None、intent_group=None、urgency=None、intent_confidence=1.0、request_id=UUID 前 8 位、intent_diagnostics={}、secondary_intents=[]、knowledge_required=False。

不存在独立名为 SubTaskOutput 的 Pydantic 模型。领域 Agent 的统一输出为 AgentResponse(agent_type, content, success, confidence=1.0, latency_ms=0.0, escalate=False, tools_used=[], tool_traces=[])。

| Agent | Profile 输入关注点 | 专属工具的真实输出 |
| --- | --- | --- |
| general | 历史、画像、意图/紧急度、知识上下文 | 请求快照、建议待补字段 |
| technical | 错误码、时间、环境、影响范围、变更、知识上下文 | 401/403/404/500 静态排查、低风险计划 |
| billing | 订单、金额币种、时间/渠道、预期、知识上下文 | 缺失字段、两数差值 |
| escalation | 消息、意图、紧急度、实体、背景 | 交接摘要，不创建工单 |

OrchestratorResult：request_id、response、agent_type、intent、escalated=False、agent_recommended_handoff=False、latency_ms=0.0、agent_types=[]、primary_agent=None、supporting_agents=[]、tools_used=[]、tool_traces=[]、routing_reason=""、routing_confidence=0.0、intent_diagnostics={}、secondary_intents=[]、knowledge_required=False。

### 2.3 Profile 与统计

AgentProfile：role、mission、workflow、input_contract、output_contract、handoff_conditions=()、tool_scope=()、model=None、temperature=0.2、max_tokens=1024。

| Agent | temperature / max_tokens | 工具 |
| --- | --- | --- |
| general | 0.3 / 900 | search、请求上下文、补字段 |
| technical | 0.1 / 1200 | search、错误码、诊断计划 |
| billing | 0.0 / 1100 | search、账单字段、金额比较 |
| escalation | 0.0 / 500 | profile 声明 search/交接；实际覆盖 handle，直接确定性返回，不调用工具 |

AgentStats(total=0, success=0, total_ms=0.0, monitor_penalty=0.0) 的评分公式为：(success_rate * 0.7 + (1 / (1 + avg_ms / 1000)) * 0.3) * max(0, 1 - monitor_penalty)。

## 3. 会话记忆引擎

证据：memory/conversation_memory.py、api/main.py。

| 层 | 存储 | 名称 | 内容 |
| --- | --- | --- | --- |
| 工作记忆 | Redis list/string | wm:{user}:{conv} 与 summary key | Message(role, content, timestamp, metadata) 与摘要 |
| 情景记忆 | ChromaDB | episodic | 压缩会话摘要和 user/conv/ts/full_text |
| 用户画像 | ChromaDB | user_profile | preferences: list[str]、entities: dict[str, list[str]] |

- 常量：WORKING_MAX=20、COMPRESS_AT=15、HISTORY_TOP_K=5、SUMMARY_MAX_CHARS=800。
- 每条消息 LPUSH 后 TTL 是 86400 秒。达到 15 条时，读取最多 20 条、压缩旧消息、保留最新 5 条。摘要 LLM 为 max_tokens=256、temperature=0。
- 摘要失败写入“对话包含 N 条消息（摘要生成失败）”；合并失败回退最后 800 字符；episodic 的 full_text 最多 500 字符。
- 读取先取同 conversation，再同 user 补足最多 5 条；Prompt 最多展示 3 条 relevant history 和最后 8 条 recent message。
- /chat 同步读上下文、同步写入用户/助手消息；画像使用 asyncio.create_task(update_profile) 非阻塞更新。
- 文件注释称 embeddings 都由 Anthropic 提供，但 episodic/profile collection 没有显式 embedding function；注释与实现不一致。

## 4. RAG 知识检索流水线

证据：mcp/knowledge_base.py、mcp/tool_manager.py、agents/tools.py、api/main.py。

### 4.1 知识库与切片

- 默认 collection 是 knowledge_base_zh；旧 collection 是 knowledge_base。空新库时尝试迁移旧库，否则装载 6 篇默认文档。
- embedding model 是 BAAI/bge-small-zh-v1.5，SentenceTransformerEmbeddingFunction(device=cpu, normalize_embeddings=True)，HNSW space=l2。
- 当前产品知识库采用 Markdown Header-Aware 切片：优先按 `##`/`###` 标题边界拆分，并为每片注入 `[一级标题 > 二级标题 > 三级标题]` 面包屑；单片（含前缀）硬上限 500 字符，超长小节再按段落、句号和硬截断拆分。元数据含 title、chunk_index、total_chunks、source_document_id、source_chunk_id、content_hash、corpus_version；新建片段的 source_chunk_id 格式为 `{source_document_id}_chunk_{chunk_index}`。
- 检索 score=round(1-distance,4)，未定义为概率，也没有最低 score 阈值。

### 4.2 门控、rewrite、召回与 rerank

- _should_use_knowledge 对 greeting、feedback、escalation/human_handoff、other 跳过；对业务意图强制启用。
- knowledge_required=True 时，每个实际执行 Agent 在 LLM 回复前强制且只检索一次 search_knowledge_base(query, top_k=5)，将结果注入上下文并移除该轮 LLM 的 search 工具。多 Agent 时每个 Agent 独立检索。
- api/main.py 有 _build_knowledge_context(message, intent, top_k=3)，但主 /chat 没有调用；主路径是 Agent 内强制检索。
- rewrite：原 query 加 LLM 改写，目标 n=3，temperature=0.3、max_tokens=256；不传历史/记忆；失败只保留原 query。
- 并发召回：每个改写用 asyncio.gather(return_exceptions=True)，recall_k=max(top_k,5)。
- 去重优先 source_chunk_id，再为 content_hash/runtime_id/content hash，重复保留较高 score。
- 候选多于 top_k 时，LLM Tool Use rerank_results 返回完整 index 排列，temperature=0、max_tokens=256。没有交叉编码 reranker、数值 rerank 分数或 score threshold。失败则取原顺序前 top_k，rerank_degraded=True。

## 5. 多 Agent 编排与容错

证据：agents/agent_orchestrator.py。

- 初始 pool 每种 AgentType 仅一个实例。代码支持同类多实例按 stats 选优，但当前构造路径没有副本。
- CRITICAL 直接 escalation（confidence 1.0）；human_handoff 直接 escalation（confidence 至少 max(intent_confidence,0.8)）。
- 领域评分：general 基础 0.1；general 主意图 +0.55，technical/billing 主意图 +0.75；technical/billing keyword 每个 +0.18、最多 0.45，general 每个 +0.12、最多 0.35；error_code +0.2，amount +0.15，order_id +0.1。
- OTHER 且消息长度大于 2 且置信度 <0.5 时，固定澄清，不运行 Agent。
- 多 Agent 用 asyncio.gather(*tasks, return_exceptions=True) fan-out。多个成功结果由 ResponseComposer LLM 语义合并；Composer 失败则主结果加“补充说明”拼接。
- _collaboration_targets 仍存在关键词协作函数，但 _route_decision 不调用它，不是实际主路径。
- Agent 内部失败被捕获为 success=False；technical/billing 失败降级 General；并行局部异常不阻断其他任务；全部失败返回“抱歉，所有 Agent 均处理失败。”

## 6. 安全、熔断与人工协同

证据：agents/tools.py、agents/agent_orchestrator.py、mcp/tool_manager.py。

- 真实 escalated=True 的条件只有 CRITICAL 或 human_handoff。响应出现 转人工、人工客服、escalate、specialist、无法处理 只设 agent_recommended_handoff=True，不转接。
- technical 工具不会读服务端日志；billing 工具固定 can_confirm_refund=False，金额工具只做算术；人工摘要工具不建工单。
- Tool 默认 cache_ttl=0、timeout=30。熔断状态 closed/open/half_open，默认失败阈值 5、恢复时间 60 秒。
- 缓存最多 5000 条，超限删除最早 1250；知识库工具 TTL 300 秒。
- 工具失败后若 fallback 成功，ToolResult.success=True 但携带原 error；不能将其等同外部服务真实成功。

## 7. 监控与日志基线

证据：monitor/performance_monitor.py、api/main.py。

- 基础 logging，默认 LOG_LEVEL=INFO；提供 /health、/monitor、/metrics、工具 trace API。
- tool trace 是内存 deque，默认最大 200（CUSTOMEROPS_TOOL_TRACE_MAX），进程重启丢失。
- Monitor 默认每 10 秒采集。Z-score window=60、sensitivity=2.5，不足半窗不检测。
- 阈值：agent success <0.90、tool success <0.95、agent avg >3000ms、tool avg >5000ms。
- 可选 Prometheus 指标：agent success、agent latency histogram、tool success、requests_total；源码未见 requests_total 的递增。
- 未发现 Langfuse、OpenTelemetry 或跨 HTTP/LLM/Redis/Chroma 的持久分布式追踪。现有是基础日志 + 内存 trace + 可选 Prometheus/Webhook，不是完整 APM。rewrite 仍保留 print DEBUG 输出。

## 8. 评测与测试基线

证据：evaluation/evaluator.py、evaluation/deterministic_metrics.py、evaluation/datasets/golden_benchmark.json、evaluation/datasets/benchmark_corpus.json、evaluation/run_phase2_cases.ps1、tests；其中 benchmark corpus 仅用于离线评测，不进入产品知识库 collection。

### 8.1 API 评测

- EndToEndEvaluator 计算意图 accuracy、per-class Precision/Recall/F1、macro-F1。
- 对话先经 orchestrator，再由 LLM-as-Judge 对 relevance、accuracy、completeness、helpfulness 四项 0..1 打分；Judge JSON 输出，temperature=0、max_tokens=256。
- PASS_THRESHOLD=0.75。Judge 失败返回 0.5 占位、标记失败且不计入质量均分。
- 当前/上次或 EVAL_BASELINE_PATH 的基线相比下降超过 5% 记 regression。内置 11 条意图样本、5 条对话样本。

### 8.2 100 条基准与确定性指标

golden_benchmark.json 顶层是对象：dataset_id=customerops-golden-v2.1.1-100、schema_version=2.1、case_count=100、8 条 regression baseline + 92 条新增 case、knowledge_base_version=kb-v2.1-main-100。

真实样本在 cases，包含 case_id、task_types、domain、scenario、difficulty、risk_level、user_question、expected_intent、conversation、gold_intent、gold_route、gold_chunk、gold_retrieval、gold_answer、gold_tools、gold_safety、evaluation。它提供 ground truth intent/route/chunk、required facts/actions、forbidden claims、工具顺序和安全约束。

deterministic_metrics.py 不调用 Judge，消费 case、API 结果和 Trace 计算意图、路由、知识门控、Recall/MRR/nDCG、答案约束、工具顺序、升级和稳定性。旧中文自然语言约束使用有限词面兜底，不是语义等价评测。

run_phase2_cases.ps1 会先查 /health 校验 collection/model，再请求 /chat 与 trace。默认期望 knowledge_base_benchmark_v2 和 BAAI/bge-small-zh-v1.5，区别于产品默认 knowledge_base_zh；运行 benchmark 前必须切换数据环境。

tests 有 19 个 test_*.py，覆盖意图、路由、工具、记忆、RAG、监控和确定性评测。前端有 src/lib/markdown.test.js，但 package.json 没有 test script，不能表述为已配置可运行的前端测试命令。

## 9. 真实技术栈与部署

### 9.1 后端

| 类别 | 锁定依赖 |
| --- | --- |
| LLM/API | anthropic==0.40.0、fastapi==0.115.5、uvicorn[standard]==0.32.1、httpx==0.28.1 |
| 契约 | pydantic==2.10.3 |
| 存储/RAG | redis==5.2.1、chromadb==0.5.23、sentence-transformers==3.3.1、transformers==4.46.3 |
| 支撑 | python-dotenv==1.0.1、prometheus-client==0.21.1、pytest==8.3.4、python-multipart==0.0.12 |

Docker 额外固定 CPU torch==2.4.1+cpu，基础 python:3.12-slim，生产镜像 UID 1000 非 root 运行。运行范式是原生 Python asyncio + dataclass/Pydantic + Anthropic SDK；未使用 LangGraph、Celery、Kafka、Temporal 等重型编排框架。

### 9.2 前端与部署

前端是 Vue SPA：vue ^3.5.25、vite ^7.2.7、@vitejs/plugin-vue ^6.0.2。实际 UI 入口只有聊天、知识库；请求层虽有 eval/monitor/trace，但顶部导航未暴露 evaluation。设置与会话写入 localStorage 的 customerops.frontend.settings。

Markdown 先 escape HTML，只允许 http:、https:、mailto: 链接。上传后端上限 10MB，非 JSON 使用 UTF-8 errors=ignore 解码。

Compose 使用 Redis 7-alpine、Chroma 0.5.23、Prometheus latest、应用和 Nginx；端口 Redis 6379、Chroma 8001:8000、应用 8000、Prometheus 9090、Nginx 80；网络 172.28.0.0/16；Prometheus retention 200h。Redis AOF 开启且默认密码回退 customerops123，部署必须替换。

API CORS 是 allow_origins=["*"] 且允许全部 method/header。未见认证/鉴权中间件；知识库写入、上传、Skill reload、评测和 trace 均无访问控制。这是当前安全缺口，不是已实现治理能力。

## 10. 复刻锚点与禁止误读

应保持的兼容行为：意图 value、0.5/0.45/0.8 阈值和 500 特例；Request/Result/ChatResponse 默认字段；15 条压缩/保留 5 条/86400 秒 TTL；knowledge_base_zh、BGE、500 字符切片；每 Agent 一次 top_k=5 强制检索；并发 gather/局部隔离；Tool 熔断 5/60 秒；rewrite 三路、LLM 排序且无分数阈值。

不得误当成熟能力复制：真实业务写操作、人工接手回执、持久 trace/APM、Java 服务端、未接入主路径的 _build_knowledge_context 和 _collaboration_targets。

