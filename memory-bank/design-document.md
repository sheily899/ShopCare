# ShopCare 系统详细设计文档

版本：0.1
输入规格：PRD.md、system-spec.md
设计原则：保留原工程已验证的数据流、阈值、依赖和降级边界；不把未实现的外部业务能力设计成现有能力。

## 1. 系统全景架构与模块拓扑

### 1.1 端到端数据流

    Web Vue / CLI
        |
        v
    FastAPI POST /chat
        |
        +-- ChatRequest(message, user_id, conv_id)
        |
        v
    MemoryManager.get_context
        |-- Redis working memory
        |-- Chroma episodic retrieval
        |-- Chroma user_profile
        v
    AgentOrchestrator.recognize_intent
        |
        +-- async LLM classify_intent
        +-- async embedding candidate
        +-- synchronous pattern candidate
        +-- weighted fusion / compound split / urgency
        v
    _should_use_knowledge + _route_decision
        |
        +-- OTHER + confidence < 0.5 -> clarification
        +-- CRITICAL or human_handoff -> EscalationAgent
        +-- one domain -> one Agent
        +-- confirmed cross-domain -> primary + supporting Agents
        v
    Agent execution
        |
        +-- knowledge_required
        |     -> search_knowledge_base once, top_k=5
        |     -> rewrite n=3 -> parallel recall -> deduplicate -> optional LLM rerank
        |
        +-- domain tools / Anthropic Tool Use, max 3 tool rounds
        v
    asyncio.gather(return_exceptions=True) when multi-agent
        |
        +-- one successful result -> return it
        +-- multiple results -> ResponseComposer LLM
        +-- Composer failure -> primary + deterministic supplement
        v
    OrchestratorResult -> ChatResponse
        |
        +-- synchronous Redis user/assistant messages
        +-- asyncio.create_task(MemoryManager.update_profile)
        v
    Vue renders escaped Markdown and status metadata

### 1.2 目录结构

    ShopCare/
    ├── api/
    │   └── main.py                 FastAPI app、lifespan、ChatRequest/Response、路由
    ├── core/
    │   ├── intent_recognizer.py    IntentCategory、UrgencyLevel、IntentResult、识别融合
    │   ├── llm_utils.py            文本抽取和 JSON 解析
    │   ├── skill_loader.py         Skill/SkillManager、热加载与 prompt 注入
    │   └── config.py               可选；仅承载环境变量读取，不改变默认值
    ├── agents/
    │   ├── agent_orchestrator.py   AgentType、Request、AgentResponse、路由、并发、Composer
    │   └── tools.py                AgentToolSpec、领域工具、共享 RAG 工具
    ├── memory/
    │   └── conversation_memory.py MemoryManager、Message、Redis/Chroma 读写
    ├── mcp/
    │   ├── knowledge_base.py      Chroma collection、切片、搜索、迁移
    │   └── tool_manager.py        Tool、ToolResult、缓存、熔断、rewrite、rerank
    ├── monitor/
    │   └── performance_monitor.py Agent/tool 统计、Z-score、告警、Prometheus
    ├── evaluation/
    │   ├── evaluator.py            EndToEndEvaluator、LLMJudge、回归
    │   ├── deterministic_metrics.py Golden trace 确定性指标
    │   ├── datasets/               100 case 与冻结片段
    │   └── run_phase2_cases.ps1    /health、/chat、trace 运行器
    ├── tests/                      Python 单元/集成测试
    └── frontend/                   Vue 源码、backends.js、Markdown 安全渲染

模块依赖方向必须保持：api -> agents/memory/mcp/monitor/evaluation；agents -> core/mcp；mcp 不依赖 api；monitor 只读取 orchestrator/tool_manager 公开统计；evaluation 通过 API 或 orchestrator 运行，不反向改变业务逻辑。

## 2. 状态机与核心数据契约

### 2.1 API Pydantic 模型

ChatRequest：

| 字段 | 类型 | 默认 / 校验 |
| --- | --- | --- |
| message | str | 必填；调用前 trim，空文本应拒绝 |
| user_id | str | anonymous；不替换为认证身份 |
| conv_id | Optional[str] | None；None 时 api 生成 UUID |

ChatResponse：

| 字段 | 类型 | 默认 |
| --- | --- | --- |
| conv_id | str | 必填 |
| request_id | str | "" |
| response | str | 必填 |
| intent | str | 必填 |
| secondary_intents | List[str] | [] |
| intent_group | str | other |
| agent_type | str | 必填 |
| agent_types | List[str] | [] |
| primary_agent | str | "" |
| supporting_agents | List[str] | [] |
| tools_used | List[str] | [] |
| routing_reason | str | "" |
| routing_confidence | float | 0.0；应限制在 0..1 |
| escalated | bool | 必填 |
| agent_recommended_handoff | bool | false |
| latency_ms | float | 必填；不得负数 |
| knowledge_used | bool | false |
| entities | Dict[str, List[str]] | {} |
| intent_confidence | float | 0.0；应限制在 0..1 |
| intent_source_scores | Dict[str, float] | {} |
| intent_diagnostics | Dict[str, Any] | {} |

### 2.2 内部流转模型

原工程中这些对象是 dataclass；设计保持 dataclass，不强行改成 Pydantic。

IntentResult：

    intent: IntentCategory
    confidence: float
    urgency: UrgencyLevel
    intent_group: str
    entities: Dict[str, List[str]]
    reasoning: str
    latency_ms: float
    source_scores: Dict[str, float] = {}
    source_intents: Dict[str, str] = {}
    secondary_intents: List[IntentCategory] = []
    compound_trigger: str = ""

校验：confidence 归一到 0..1；entities 保留 order_id、product、date、amount、error_code 五个 key；secondary_intents 最多 3 个；human_handoff 为主意图时必须清空辅助意图。

AgentProfile：

    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    handoff_conditions: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024

AgentResponse：

    agent_type: AgentType
    content: str
    success: bool
    confidence: float = 1.0
    latency_ms: float = 0.0
    escalate: bool = False
    tools_used: List[str] = []
    tool_traces: List[Dict[str, Any]] = []

校验：content 允许失败时的确定性文案；latency_ms 不得负数；tools_used 只能来自该 Agent 的白名单；escalate 只表示 Agent 建议，不直接等于 escalated。

OrchestratorResult：

    request_id: str
    response: str
    agent_type: AgentType
    intent: Optional[IntentCategory]
    escalated: bool = False
    agent_recommended_handoff: bool = False
    latency_ms: float = 0.0
    agent_types: List[AgentType] = []
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = []
    tools_used: List[str] = []
    tool_traces: List[Dict[str, Any]] = []
    routing_reason: str = ""
    routing_confidence: float = 0.0
    intent_diagnostics: Dict[str, Any] = {}
    secondary_intents: List[IntentCategory] = []
    knowledge_required: bool = False

### 2.3 记忆模型与存储记录

源码有 Message 和 MemoryContext；没有名为 WorkingMemory、UserProfile、EpisodicMemory 的现成 Pydantic 类。设计使用以下记录形状，不新增独立业务语义：

Message：

    role: MsgRole                  # user / assistant / system
    content: str
    timestamp: datetime = now()
    metadata: Dict[str, Any] = {}

WorkingMemory 是 Redis list 的逻辑记录，不是新表：

    key = "wm:{user_id}:{conv_id}"
    value = JSON(Message)
    max_read = 20
    compress_at = 15
    keep_after_compress = 5
    ttl_seconds = 86400

UserProfile 是 Chroma user_profile collection 的 document：

    document_id = stable user profile id
    document = JSON({preferences: List[str], entities: Dict[str, List[str]]})
    metadata = {user_id, conv_id, updated_at}

EpisodicMemory 是 Chroma episodic collection 的 document：

    document = compressed summary, max stored text 500 chars
    metadata = {user_id, conv_id, ts, full_text}
    retrieval = same conversation first, same user fallback, max 5

## 3. 核心子系统详细设计

### 3.1 IntentRecognizer

#### 3.1.1 三通路并发

1. 创建 llm_task = asyncio.create_task(_recognize_with_llm(message, history))。
2. 创建 embedding_task = asyncio.create_task(_recognize_with_embedding(message))。
3. 同步执行 _recognize_with_pattern(message)。
4. await asyncio.gather(llm_task, embedding_task, return_exceptions=True)。
5. 记录 source_intents/source_scores/diagnostics；失败来源不阻断其他来源。
6. embedding 可用时使用权重 0.7/0.2/0.1；不可用时使用 0.85/0.15。

伪代码：

    llm, embedding = await gather(llm_task, embedding_task, return_exceptions=True)
    pattern = pattern_result
    if llm.failed and pattern.specific_score > 0:
        candidate = pattern.specific
    elif llm.failed and embedding.score >= 0.45:
        candidate = embedding.intent
    else:
        candidate = weighted_vote(llm, embedding, pattern)
    if candidate.score < 0.5:
        candidate = OTHER
    if pattern.specific_score >= 0.5 and candidate.score < 0.8:
        candidate = pattern.specific
    if "500" in message and candidate.intent != TECHNICAL_LOGIN:
        candidate = TECHNICAL_CRASH
        candidate.confidence = max(candidate.confidence, 0.8)

#### 3.1.2 阈值状态分支

    START
      |
      +-- LLM success? -- no --> embedding >= 0.45?
      |                              | yes -> embedding fallback
      |                              | no  -> specific pattern or OTHER
      |
      +-- yes --> weighted fusion
                       |
                       +-- max score < 0.5 -> OTHER
                       +-- specific pattern >= 0.5 and fused < 0.8
                       |       -> specific pattern refinement
                       +-- message contains 500 and not technical_login
                               -> TECHNICAL_CRASH, confidence >= 0.8

#### 3.1.3 复合意图算法

1. 使用连接词及标点切分原文，保留非空 clause。
2. clause 数少于 2，返回单意图。
3. 每个 clause 计算 pattern 候选，按命中 pattern 长度取最长候选。
4. 去除 generic 标签；去除与 primary 同路由域的候选。
5. 去除 escalation 候选，除非它本身是 primary。
6. 按原顺序去重，最多保留 3 个 secondary_intents。
7. 生成 compound_trigger；没有跨域结果则保持空数组。
8. retrieval query 只在确有 secondary_intents 时按 Agent 领域关键词抽取 clause，否则使用完整原文。

#### 3.1.4 紧急度

    if contains(CRITICAL_WORDS):
        urgency = CRITICAL
    elif contains(HIGH_WORDS):
        urgency = HIGH
    elif contains(MEDIUM_WORDS):
        urgency = MEDIUM
    elif intent == HUMAN_HANDOFF:
        urgency = HIGH
    elif intent == COMPLAINT:
        urgency = MEDIUM
    else:
        urgency = LOW

CRITICAL 或主意图 HUMAN_HANDOFF 由编排器直接选择 escalation。Agent 输出关键词只设置 agent_recommended_handoff。

### 3.2 AgentOrchestrator 与 Composer

#### 3.2.1 节点配置

| 节点 | 职责 | 专属工具 | 温度 / tokens |
| --- | --- | --- | --- |
| GeneralAgent | 首轮接待、澄清、通用知识回答 | inspect_request_context、suggest_required_fields，加共享 search | 0.3 / 900 |
| TechnicalAgent | 错误码和环境排障 | lookup_error_code、build_diagnostic_plan，加共享 search | 0.1 / 1200 |
| BillingAgent | 账单、退款、发票字段核验 | check_billing_fields、compare_amounts，加共享 search | 0.0 / 1100 |
| EscalationAgent | 记录升级原因和上下文 | handle 当前不调用工具；profile 仍保留声明 | 0.0 / 500 |

所有工具通过 AgentToolSpec 白名单暴露，input schema additionalProperties=false。不得增加订单写操作工具。

#### 3.2.2 路由

1. request.intent 已存在时跳过重复识别。
2. CRITICAL -> escalation。
3. HUMAN_HANDOFF -> escalation。
4. OTHER 且文本长度 >2 且 confidence <0.5 -> 固定澄清。
5. 计算 general/technical/billing domain_scores。
6. primary 为可用 Agent 中最高分。
7. supporting 只由 secondary_intents 映射，跳过 primary 与 escalation。
8. 返回 RoutingDecision(primary_agent, supporting_agents, reason, confidence)。

#### 3.2.3 Fan-out、Fan-in 与隔离

    tasks = [_execute(req, primary)]
    tasks += [_execute(req, support) for support in supporting]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    valid = [x for x in responses if isinstance(x, AgentResponse)]

    if len([x for x in valid if x.success]) == 1:
        output = successful[0].content
    elif len(successful) > 1:
        output = await composer.compose(req, successful)
    else:
        output = "抱歉，所有 Agent 均处理失败。"

每个 Agent 的异常由 BaseAgent.handle 转成 success=false；gather 的 return_exceptions=true 保证其他任务继续。technical/billing 失败时 _execute 调 General；Escalation 不降级。

#### 3.2.4 ResponseComposer

Prompt 必须包含：主 Agent 类型、用户原文、每个成功 Agent 的标签与 content；要求：

- 以 primary 结论为主；
- 删除重复与冲突表述；
- 不补造订单、退款、后台查询结果；
- 保留排查步骤、核验字段和升级边界；
- 只输出给用户看的中文，不提 Agent。

配置：原模型，max_tokens 默认 1000，可由 CUSTOMEROPS_COMPOSER_MAX_TOKENS 覆盖；temperature 默认 0.1，可由 CUSTOMEROPS_COMPOSER_TEMPERATURE 覆盖。

失败降级：

    return primary.content
           + for each remaining:
               "

补充说明：
" + supporting.content

### 3.3 KnowledgeBasePipeline

#### 3.3.1 基础配置

collection=knowledge_base_zh；旧 collection=knowledge_base；embedding=BAAI/bge-small-zh-v1.5；CPU；normalize_embeddings=true；HNSW space=l2。文档采用 Markdown Header-Aware 切片：优先以 `##`/`###` 标题作为边界，为正文注入 `[一级标题 > 二级标题 > 三级标题]` 面包屑；每片（含前缀）最多 500 字符，超长小节按段落、句号和硬截断继续拆分。每片保存 source_document_id、source_chunk_id、content_hash、chunk_index、total_chunks、title、corpus_version，其中 source_chunk_id 格式为 `{source_document_id}_chunk_{chunk_index}`。

#### 3.3.2 业务门控

_should_use_knowledge(message, intent)：

- 空文本 -> false。
- greeting、feedback、escalation、human_handoff、other -> false。
- query、request、technical、billing、account、complaint、order_status、logistics、refund、invoice、payment_issue、account_security、technical_login、technical_crash -> true。
- 未知意图：精确 greeting 集合 -> false；否则 len(message) >= 4 或业务关键词命中 -> true。

主链路 knowledge_required 时，每个实际执行 Agent 只强制调用一次 search_knowledge_base，top_k=5；调用成功后从 LLM 工具列表删除 search，避免重复调用。

#### 3.3.3 四步检索

    queries = rewrite_query(original, n=3)
    rows = await gather(
        recall(query, top_k=max(requested_top_k, 5))
        for query in queries
    )
    merged = deduplicate(rows)
    if len(merged) > requested_top_k:
        ordered = await rerank_results(original, merged)
    else:
        ordered = merged
    return ordered[:requested_top_k]

rewrite 的 LLM 不接收 history 或 MemoryContext，temperature=0.3、max_tokens=256。去重键依次为 source_chunk_id、content_hash、runtime_id、内容 hash；同键保留 score 高者。Chroma score 是 round(1-distance,4)，不是概率。

rerank 使用 LLM Tool Use 返回候选 index 的完整排列，temperature=0、max_tokens=256。没有交叉编码器和 score threshold。调用失败、index 不完整或越界时返回原顺序前 top_k，并设置 rerank_degraded=true。

### 3.4 MemoryManager

#### 3.4.1 主链路时序

    /chat
      -> get_context(user, conv, query)
      -> recognize/routing/agent response
      -> add_message(USER)
      -> add_message(ASSISTANT)
      -> asyncio.create_task(update_profile(user, conv))
      -> return ChatResponse

#### 3.4.2 工作记忆

key=wm:{user_id}:{conv_id}。add_message 将 Message JSON LPUSH，随后 EXPIRE 86400。LLEN >= 15 时：

1. LRANGE 读取最多 20 条。
2. 按时间恢复顺序，old = messages[:-5]，keep = messages[-5:]。
3. old 交给摘要 LLM，max_tokens=256、temperature=0。
4. 摘要最大 800 字符；失败使用“对话包含 N 条消息（摘要生成失败）”。
5. 将摘要写入 Redis summary key 和 episodic collection。
6. 删除工作列表，重新 LPUSH keep，并重新 EXPIRE 86400。

#### 3.4.3 情景记忆与画像

episodic：Chroma query 先过滤 user_id + conv_id，再按 user_id 补足，去重后最多 5 条；同步 Chroma 操作通过 asyncio.to_thread。

user_profile：update_profile 读取最新 10 条消息，要求 LLM 返回 preferences list 与 entities object，校验结构后替换固定 profile document。API 使用 asyncio.create_task，不阻塞 ChatResponse。任务异常只记录日志，不回滚已经写入的两条消息。

### 3.5 Resilience、Safety 与 Handoff

#### 3.5.1 Circuit Breaker

    CLOSED:
      handler success -> failure_count = 0
      handler failure -> failure_count += 1
      failure_count >= 5 -> OPEN

    OPEN:
      now - opened_at < 60s -> reject/fallback
      now - opened_at >= 60s -> HALF_OPEN

    HALF_OPEN:
      probe success -> CLOSED
      probe failure -> OPEN

Tool 层顺序：cache -> breaker -> required/type 校验 -> asyncio.wait_for(timeout=30) -> stats/breaker -> optional rerank -> cache。knowledge_search cache_ttl=300；总缓存上限 5000，超限删最早 1250。

#### 3.5.2 异常隔离矩阵

| 故障 | 隔离范围 | 确定性路径 | 对外状态 |
| --- | --- | --- | --- |
| classify_intent 失败 | 意图来源 | pattern；embedding >=0.45 才采用，否则 OTHER | diagnostics 标记来源失败 |
| embedding 接口不可用 | embedding 来源 | LLM + pattern，权重 0.85/0.15；本地 256 维实现可作为实现内 fallback | 不阻断请求 |
| RAG handler 超时 | 当前工具调用 | breaker/fallback；返回不可用说明、score=0 | 不声称检索成功 |
| rewrite 失败 | 改写阶段 | 使用原 query 单路召回 | 正常继续 |
| rerank 失败 | 排序阶段 | 原召回顺序前 top_k，rerank_degraded=true | 保留降级标志 |
| technical/billing Agent 异常 | 单一 Agent | 降级 General | 其他 Agent 不受影响 |
| 多 Agent 单任务异常 | 单协程 | gather 继续，保留成功结果 | Composer 处理局部结果 |
| Composer 失败 | Fan-in | primary + 补充说明字符串 | 不丢失专业结果 |
| update_profile 失败 | 后台画像任务 | 记录日志，消息已保留 | 不影响已返回答复 |

#### 3.5.3 Human Handoff

状态由两个字段分离表达：

- escalated=true：仅 CRITICAL 或 intent=human_handoff。
- agent_recommended_handoff=true：Agent 输出含转人工、人工客服、escalate、specialist、无法处理。

EscalationAgent 生成 request_id、原因、意图、紧急度、实体；固定声明不索取密码、短信验证码、完整支付凭证。没有真实工单 ID、队列 ID、人工接手时间或处理 SLA，因此不得向用户承诺这些状态。

## 4. 评测与可观测性设计

### 4.1 100 条基准流水线

1. 读取 golden_benchmark_v2.1_100.json，顶层对象取 cases，校验 case_count=100。
2. 启动前 GET /health，验证 collection 与 embedding model；产品库 knowledge_base_zh 与 benchmark 库 knowledge_base_benchmark_v2 必须明确区分。
3. 每个 case 为同一 conversation 复用 user_id/conv_id；多轮只提交 role=user 的 turns。
4. POST /chat，保存 request_id、intent、route、tools、knowledge_used、latency、response。
5. GET /trace/tool/{request_id}，保存 retrieval_stage、tool_calls、executed_agents、retrieved_chunks。
6. 保存 case 的 gold_intent、gold_route、gold_retrieval、gold_answer、gold_tools、gold_safety 与运行结果。
7. deterministic_metrics.py 计算 intent、route、knowledge gate、RAG Recall/MRR/nDCG、required facts、forbidden claims、tool order、handoff 和 stability。
8. EndToEndEvaluator 额外对对话调用 LLMJudge，四维 0..1，单例综合 >=0.75 才 passed；Judge 失败标记 judge_failed，不混入均值。
9. 与历史或 baseline.json 比较，任何指标相对下降超过 5% 写入 regressions。

### 4.2 监控与指标

PerformanceMonitor 每 10 秒执行 collect：

1. 读取 orchestrator.get_stats 与 tool_manager.get_stats。
2. 对 agent success_rate、agent avg_ms 写入 Z-score detector。
3. 检查 agent success <0.90、tool success <0.95、agent avg >3000、tool avg >5000。
4. 创建 Alert，记录 severity、metric、value、threshold、ts；可异步发送 webhook。
5. 计算 routing penalty，范围 0..0.9，回写 orchestrator.update_routing_penalties。
6. 生成连续失败 >=3 的工具建议以及 Agent 成功率 <0.85 且 total>10 的路由建议。

Z-score：

    buffer = deque(maxlen=60)
    append(value)
    if len(buffer) < 30: return None
    mean = statistics.mean(buffer)
    stdev = statistics.stdev(buffer)
    if stdev == 0: return None
    z = abs(value - mean) / stdev
    if z > 2.5: emit anomaly

Prometheus 可选指标：agent_success_rate{agent}、agent_latency_ms{agent}、tool_success_rate{tool}、requests_total。当前源码未见 requests_total 递增，施工时不得把该指标宣传为准确请求计数。

### 4.3 Trace 结构

每个 request_id 的内存 trace 至少保留：

    request_id
    intent_diagnostics
    primary_agent
    executed_agents
    tool_calls
    tools_used
    retrieval_stage
    retrieval_call_stages
    retrieved_chunks
    reranked
    rerank_degraded
    latency_ms
    final response metadata

deque maxlen 默认 200。Trace 是运行期排障数据，不是审计级持久化日志；重启丢失。

## 5. 编码验收检查点

### 5.1 ASCII 类关系图

    ChatRequest / ChatResponse (Pydantic)
                 |
                 v
    AgentOrchestrator ----> IntentRecognizer
          |                       |
          |                       +--> IntentResult
          |
          +--> Request ---------> RoutingDecision
          |                         |
          |                         +--> AgentType
          |
          +--> BaseAgent
          |       +--> GeneralAgent
          |       +--> TechnicalAgent
          |       +--> BillingAgent
          |       +--> EscalationAgent
          |                    |
          |                    +--> AgentResponse
          |
          +--> ResponseComposer --> OrchestratorResult
          |
          +--> AgentToolSpec --> MCPToolManager --> ToolResult
                                      |
                                      +--> KnowledgeBase --> ChromaDB

    MemoryManager --> Redis working list / summary
                  --> Chroma episodic / user_profile
    PerformanceMonitor -reads-> AgentStats / ToolStats
    EndToEndEvaluator -reads-> API result / ToolTrace / Golden case

### Checkpoint A：契约

- ChatRequest、ChatResponse 与内部 dataclass 字段/默认值对齐。
- Pydantic 仅用于 API 边界；Message 等内部记录与既有 dataclass 兼容。
- confidence、routing_confidence、intent_confidence 范围为 0..1。

### Checkpoint B：意图与路由

- 0.5、0.45、0.8、500 特例测试齐全。
- 复合意图最多 3 个 secondary；关键词偶合不触发并发。
- CRITICAL/HUMAN_HANDOFF 到 escalation，低置信度 OTHER 到 clarification。

### Checkpoint C：RAG 与记忆

- collection、模型、切片、元数据和 top_k=5 对齐。
- rewrite n=3、无 history、去重和 rerank_degraded 可观测。
- Redis 15/5/20/86400 参数与异步 profile 更新正确。

### Checkpoint D：故障

- 工具 5 次失败 -> Open，60 秒后 Half-Open。
- 专属 Agent 失败 -> General；并发局部失败不阻塞整体。
- Composer、rewrite、rerank、画像失败均有确定性 fallback。

### Checkpoint E：评测与观测

- 100 cases 可复现运行并保存 trace。
- Judge failed 不污染均值；回归超过 5% 被记录。
- /health、/monitor、/metrics、trace API 可读取，且明确内存数据的丢失边界。


