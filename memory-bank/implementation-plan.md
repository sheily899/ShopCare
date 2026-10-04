# ShopCare 实施计划

版本：0.1  
计划依据：memory-bank/system-spec.md、PRD.md、design-document.md、tech-stack.md  
执行规则：每个微任务只处理 1–2 个紧密关联文件；先写测试或验证夹具，再实现；验证通过后才勾选进度并更新架构记录。

## 0. 全局施工约束

### 0.1 不可变边界

- Python 3.12；依赖严格遵循 tech-stack.md。
- API 边界使用 Pydantic；内部 Request、IntentResult、AgentProfile、AgentResponse、OrchestratorResult 和记忆记录保持 dataclass。
- 不实现真实订单查询、支付扣款、退款执行、改地址、工单创建或人工坐席接入。
- 不加入 LangGraph、交叉编码器 Rerank、Java 后端、分布式 APM 或未声明第三方库。
- 每个微任务完成后必须运行该任务的独立验证命令；命令失败时不得更新为完成。

### 0.2 每步的固定收尾动作

1. 运行本步骤的 focused pytest 或脚本。
2. 检查 git diff 与实际影响文件。
3. 在 memory-bank/progress.md 勾选已验证任务，记录命令和结果。
4. 在 memory-bank/architecture.md 追加新建/修改文件的职责、依赖和边界。
5. 若接口、阈值、依赖或范围发生变化，先停止施工，更新规格文档后再继续。

### 0.3 依赖图

Step 1 契约与环境
  -> Step 2 意图
  -> Step 3 记忆
  -> Step 4 RAG 与工具基础设施
  -> Step 5 领域 Agent
  -> Step 6 编排与合成
  -> Step 7 FastAPI 集成
  -> Step 8 监控
  -> Step 9 评测

## Step 1：环境依赖文件与数据总线契约（Checkpoint A）

### Step 1.1 环境基线

【目标与修改文件清单】

- 创建 requirements.txt。
- 创建 .env.example。

【具体的代码实现逻辑要点】

- requirements.txt 严格写入 anthropic==0.40.0、fastapi==0.115.5、uvicorn[standard]==0.32.1、pydantic==2.10.3、python-multipart==0.0.12、redis==5.2.1、chromadb==0.5.23、sentence-transformers==3.3.1、transformers==4.46.3、prometheus-client==0.21.1、httpx==0.28.1、python-dotenv==1.0.1、pytest==8.3.4。
- Docker 安装 torch==2.4.1+cpu；不要把 CUDA 依赖加入 requirements。
- .env.example 只放变量名和无敏感占位值：ANTHROPIC_API_KEY、ANTHROPIC_MODEL、ANTHROPIC_BASE_URL、REDIS_URL、REDIS_PASSWORD、CHROMA_HOST、CHROMA_PORT、CHROMA_PERSIST_DIRECTORY、CUSTOMEROPS_KB_COLLECTION、CUSTOMEROPS_EMBEDDING_MODEL、LOG_LEVEL、PROMETHEUS_PORT、MONITOR_INTERVAL、CUSTOMEROPS_TOOL_TRACE_MAX、EVAL_BASELINE_PATH。
- 不写真实 key、密码或运行期数据。

【前置依赖条件】

- 已完整阅读 tech-stack.md。
- Python 3.12 可用；项目目录已初始化。

【独立验证测试命令】

    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --dry-run -r requirements.txt
    pytest -q tests/test_environment_contract.py

测试断言：所有锁定包存在且版本完全匹配；.env.example 不包含疑似真实 secret。

### Step 1.2 API 边界契约

【目标与修改文件清单】

- 创建 api/main.py 的 Pydantic 输入/输出模型部分。
- 创建 tests/test_api_contracts.py。

【具体的代码实现逻辑要点】

- ChatRequest：message 必填 str，user_id 默认 anonymous，conv_id 默认 None；空 message 进入 API 校验错误。
- ChatResponse 完整保留 system-spec 中的字段和默认值。
- routing_confidence、intent_confidence 限制为 0..1；latency_ms 不得为负。
- 暂不实现 lifespan 和业务路由；本子任务只建立边界模型。

【前置依赖条件】

- Step 1.1 完成。
- Pydantic 版本通过环境验证。

【独立验证测试命令】

    pytest -q tests/test_api_contracts.py

测试断言：默认值、字段序列化、非法置信度、空 message 和未知字段策略符合设计。

### Step 1.3 内部 dataclass 契约

【目标与修改文件清单】

- 创建 core/intent_recognizer.py 的 Enum、IntentResult、UrgencyLevel。
- 创建 agents/agent_orchestrator.py 的 AgentType、AgentProfile、AgentResponse、Request、OrchestratorResult、RoutingDecision。
- 创建 tests/test_internal_contracts.py。

【具体的代码实现逻辑要点】

- IntentCategory 必须包含 18 个 value；ESCALATION 与 HUMAN_HANDOFF 为同值别名。
- AgentProfile 默认 handoff_conditions=()、tool_scope=()、model=None、temperature=0.2、max_tokens=1024。
- AgentResponse 默认 confidence=1.0、latency_ms=0.0、escalate=false、tools_used=[]、tool_traces=[]。
- 内部模型保持 dataclass，不新增 Pydantic 版本。
- 默认可变列表使用 field(default_factory=...)。

【前置依赖条件】

- Step 1.2 完成。
- 不依赖 Redis、Chroma 或 Anthropic 实例。

【独立验证测试命令】

    pytest -q tests/test_internal_contracts.py

测试断言：字段、默认值、Enum value、dataclass 类型与设计文档一致。

### Checkpoint A

- [ ] requirements.txt 与 .env.example 完成。
- [ ] API 边界模型通过测试。
- [ ] 内部 dataclass 契约通过测试。
- [ ] 未引入未声明第三方库。
- [ ] 更新 progress.md 和 architecture.md。

## Step 2：意图枚举与多通路融合识别器（Checkpoint B）

### Step 2.1 Pattern 与实体规则

【目标与修改文件清单】

- 完善 core/intent_recognizer.py 的 pattern、entity、urgency 纯函数。
- 创建 tests/test_intent_patterns.py。

【具体的代码实现逻辑要点】

- 建立 specific/generic pattern 表；一次命中 0.5，多命中按 0.25 增量，封顶 1.0。
- 提取 order_id、product 空列表、date、amount、error_code。
- urgency 优先级：CRITICAL > HIGH > MEDIUM > human_handoff 默认 HIGH > complaint 默认 MEDIUM > LOW。
- 500 特殊规则只在非 technical_login 时覆盖为 technical_crash。

【前置依赖条件】

- Step 1.3 完成。
- 只使用标准库正则、枚举和 dataclass。

【独立验证测试命令】

    pytest -q tests/test_intent_patterns.py

测试用例必须覆盖：退款、物流、401/403/404/500、投诉、人工、订单号/金额/日期/错误码实体、紧急度优先级。

### Step 2.2 LLM Tool Use 适配器

【目标与修改文件清单】

- 修改 core/intent_recognizer.py。
- 创建 tests/test_intent_llm_tool.py。

【具体的代码实现逻辑要点】

- 使用 Anthropic messages.create 的 classify_intent Tool Use。
- 校验 intent 是否属于 IntentCategory；confidence 是否为 0..1；reasoning 是否可读。
- 工具返回非法字段或调用异常时返回 failed 标记、OTHER 和“LLM 失败”说明。
- 不使用自由文本 JSON 作为主协议。

【前置依赖条件】

- Step 1.1、Step 1.3、Step 2.1 完成。
- 测试使用 fake Anthropic client，不发真实网络请求。

【独立验证测试命令】

    pytest -q tests/test_intent_llm_tool.py

测试断言：tool_choice、工具名、additionalProperties=false、非法 payload 和异常 fallback。

### Step 2.3 Embedding 与加权融合

【目标与修改文件清单】

- 修改 core/intent_recognizer.py。
- 创建 tests/test_intent_diagnostics.py。

【具体的代码实现逻辑要点】

- LLM 与 embedding 使用 asyncio.create_task；pattern 同步；用 gather(return_exceptions=true) 汇聚。
- embedding 可用时权重 0.7/0.2/0.1；不可用时 0.85/0.15。
- max score <0.5 -> OTHER。
- LLM 失败时 embedding <0.45 不得成为 fallback。
- specific pattern >=0.5 且 fused <0.8 时细化；500 规则最后覆盖。
- diagnostics 必须保留来源意图、来源分数、final intent/confidence 和 fallback 状态。

【前置依赖条件】

- Step 2.1、Step 2.2 完成。
- 本地 embedding fallback 仅使用原工程约定的 256 维哈希向量；不新增模型依赖。

【独立验证测试命令】

    pytest -q tests/test_intent_diagnostics.py tests/test_intent_fusion.py

测试断言：0.5、0.45、0.8、0.7/0.2/0.1、0.85/0.15、500 特例全部覆盖。

### Step 2.4 复合意图与路由映射

【目标与修改文件清单】

- 修改 core/intent_recognizer.py。
- 创建 tests/test_compound_intent.py。

【具体的代码实现逻辑要点】

- 用指定连接词和标点切分 clause。
- clause 少于 2 不进入复合分支；按最长匹配确定候选。
- 过滤同域、generic、非主流程 escalation；保留跨域 secondary_intents，最多 3 个。
- primary=human_handoff 时清空 secondary。
- 保存 compound_trigger；为 Agent 领域生成 scoped retrieval query。

【前置依赖条件】

- Step 2.3 完成。
- AgentType 映射表已在 Step 1.3 建立。

【独立验证测试命令】

    pytest -q tests/test_compound_intent.py

测试断言：单意图不并发；技术+账单只保留跨域辅助；超过 3 个截断；人工主意图终止。

### Checkpoint B

- [ ] 18 类意图和 UrgencyLevel 通过。
- [ ] Tool Use、embedding、pattern 融合通过。
- [ ] 阈值和 500 覆盖。
- [ ] 复合意图最多 3 个且不因偶然关键词并发。
- [ ] 更新 progress.md 和 architecture.md。

## Step 3：Redis 工作记忆与 Chroma 基础记忆（Checkpoint C）

### Step 3.1 Message 与 Redis 工作记忆

【目标与修改文件清单】

- 创建 memory/conversation_memory.py 的 Message、MsgRole、MemoryContext、Redis 读写。
- 创建 tests/test_memory_working.py。

【具体的代码实现逻辑要点】

- key 使用 wm:{user_id}:{conv_id}。
- Message JSON LPUSH，写后 EXPIRE 86400。
- 读取最多 20 条；LLEN >=15 触发压缩入口；保留最新 5 条。
- Redis 连接失败必须产生可识别日志/异常路径，不静默伪造记忆。
- MemoryContext.to_prompt_text 按原工程显示 summary、最多 3 条 relevant、最后 8 条 recent。

【前置依赖条件】

- Step 1.1、Step 1.3 完成。
- 测试使用 fake async Redis。

【独立验证测试命令】

    pytest -q tests/test_memory_working.py

测试断言：LPUSH、EXPIRE、key 格式、15/20/5 边界、角色序列化和 Prompt 截断。

### Step 3.2 摘要与 episodic Chroma

【目标与修改文件清单】

- 修改 memory/conversation_memory.py。
- 创建 tests/test_memory_episodic.py。

【具体的代码实现逻辑要点】

- old=messages[:-5] 交给摘要 LLM，max_tokens=256、temperature=0。
- 摘要最大 800 字符；失败使用固定失败摘要。
- 写 episodic 时 full_text 最多 500 字符，保存 user_id、conv_id、ts。
- 同 conversation 优先召回，同 user 补足到 5 条；用 asyncio.to_thread 包装同步 Chroma。
- 不把 Chroma 内置 embedding 误写为 Anthropic embedding。

【前置依赖条件】

- Step 3.1 完成。
- Chroma fake client 和 fake LLM 可用。

【独立验证测试命令】

    pytest -q tests/test_memory_episodic.py

测试断言：摘要成功/失败、800/500 限制、相同会话优先、同用户补足和异常隔离。

### Step 3.3 用户画像异步更新

【目标与修改文件清单】

- 修改 memory/conversation_memory.py。
- 创建 tests/test_memory_profile_async.py。

【具体的代码实现逻辑要点】

- 读取最新 10 条消息。
- 要求 LLM JSON 返回 preferences list 与 entities object，并严格校验。
- 替换固定 profile document；保存 user_id、conv_id、updated_at。
- 由 API 层以 asyncio.create_task 调用；画像异常不影响已返回答复。

【前置依赖条件】

- Step 3.1、Step 3.2 完成。
- API 层尚未集成时，用独立 coroutine 测试后台任务。

【独立验证测试命令】

    pytest -q tests/test_memory_profile_async.py

测试断言：任务创建、任务异常不冒泡、非法 JSON 不覆盖旧画像。

### Checkpoint C-1

- [ ] Redis 工作记忆通过。
- [ ] episodic 摘要和召回通过。
- [ ] 画像异步写回通过。
- [ ] 更新 progress.md 和 architecture.md。

## Step 4：知识库 RAG 与工具断路器（Checkpoint C/D）

### Step 4.1 KnowledgeBase 初始化与文档切片

【目标与修改文件清单】

- 创建 mcp/knowledge_base.py。
- 创建 tests/test_knowledge_base_embedding.py。

【具体的代码实现逻辑要点】

- HTTP Chroma 优先，PersistentClient fallback。
- collection=knowledge_base_zh，旧库 knowledge_base；使用 BAAI/bge-small-zh-v1.5、CPU、normalize_embeddings=true、HNSW l2。
- 文档优先按 Markdown `##`/`###` 标题层级切片，正文前注入 `[一级标题 > 二级标题 > 三级标题]` 面包屑；每片（含前缀）最多 500 字符，超长小节再按段落、句号和硬截断拆分。
- 写入 title、source_document_id、source_chunk_id、content_hash、chunk_index、total_chunks、corpus_version。
- 新建片段的 source_chunk_id 格式固定为 `{source_document_id}_chunk_{chunk_index}`。
- 检索返回 title/content/score/id/稳定身份；score=round(1-distance,4)，无 score threshold。

【前置依赖条件】

- Step 1.1 完成。
- 允许使用已声明的 Chroma 与 sentence-transformers。

【独立验证测试命令】

    pytest -q tests/test_knowledge_base_embedding.py tests/test_knowledge_base_metadata.py

测试断言：collection 名、模型名、迁移、切片上限、metadata 和 score 计算。

### Step 4.2 Tool、缓存与 Circuit Breaker

【目标与修改文件清单】

- 创建 mcp/tool_manager.py。
- 创建 tests/test_tool_manager_circuit.py。

【具体的代码实现逻辑要点】

- Tool/ToolResult 字段按设计实现。
- 调用顺序：cache -> breaker -> required/type 校验 -> wait_for(timeout=30) -> stats -> fallback/cache。
- breaker：closed/open/half_open；连续失败 5 次打开；60 秒后探测。
- 缓存最多 5000，超限删除最早 1250；fallback 成功仍保留原 error。
- 知识工具 TTL 在注册处设 300 秒。

【前置依赖条件】

- Step 4.1 完成。
- 使用 asyncio 和标准库时间，不新增缓存库。

【独立验证测试命令】

    pytest -q tests/test_tool_manager_circuit.py

测试断言：5 次失败、60 秒恢复、超时、缓存命中、淘汰和 fallback 状态。

### Step 4.3 Rewrite、并行召回与去重

【目标与修改文件清单】

- 修改 mcp/tool_manager.py。
- 创建 tests/test_tool_manager_reasoning.py。

【具体的代码实现逻辑要点】

- rewrite 目标 n=3、temperature=0.3、max_tokens=256；不传 history/context；失败保留原 query。
- 每个 query 并发 recall，recall_k=max(top_k,5)。
- source_chunk_id -> content_hash -> runtime_id -> content hash 多级去重；保留高分。
- 召回异常用 return_exceptions=true，不让单 query 阻断整体。

【前置依赖条件】

- Step 4.1、Step 4.2 完成。

【独立验证测试命令】

    pytest -q tests/test_tool_manager_reasoning.py -k "rewrite or deduplicat or recall"

测试断言：改写数量、无历史、并发调用、稳定 identity 和高分保留。

### Step 4.4 LLM Rerank 与降级

【目标与修改文件清单】

- 修改 mcp/tool_manager.py。
- 创建 tests/test_tool_manager_rerank.py。

【具体的代码实现逻辑要点】

- 候选多于 top_k 才调用 rerank_results。
- 使用完整 index permutation；校验缺失、重复、越界。
- temperature=0、max_tokens=256。
- 失败取原序前 top_k，设置 reranked=false、rerank_degraded=true。
- 严禁引入交叉编码器或 score threshold。

【前置依赖条件】

- Step 4.3 完成。
- fake Anthropic client 可返回合法及非法 index 排列。

【独立验证测试命令】

    pytest -q tests/test_tool_manager_rerank.py

### Checkpoint C/D

- [ ] Chroma 基础检索和 metadata 通过。
- [ ] Circuit Breaker 5/60、TTL 300、timeout 30 通过。
- [ ] Rewrite、并行召回、去重、Rerank fallback 通过。
- [ ] 更新 progress.md 和 architecture.md。

## Step 5：领域专属 Agents 与工具白名单

### Step 5.1 AgentToolSpec 与通用工具

【目标与修改文件清单】

- 创建 agents/tools.py 的 AgentToolSpec、make_tool、general_tools。
- 创建 tests/test_agent_tools_general.py。

【具体的代码实现逻辑要点】

- JSON Schema type=object、additionalProperties=false。
- inspect_request_context 只返回脱敏请求快照。
- suggest_required_fields 按 intent 返回下一轮必要字段。
- 不能调用外部订单/支付系统。

【前置依赖条件】

- Step 1.3、Step 2.1 完成。

【独立验证测试命令】

    pytest -q tests/test_agent_tools_general.py

### Step 5.2 Technical/Billing 工具

【目标与修改文件清单】

- 修改 agents/tools.py。
- 创建 tests/test_agent_tools_domain.py。

【具体的代码实现逻辑要点】

- lookup_error_code 只覆盖静态 401/403/404/500，server_log_checked=false。
- build_diagnostic_plan 只生成低风险步骤。
- check_billing_fields 只检查 order_id、amount、date、payment_channel，can_confirm_refund=false。
- compare_amounts 只计算两数差，不判断重复扣款。

【前置依赖条件】

- Step 5.1 完成。

【独立验证测试命令】

    pytest -q tests/test_agent_tools_domain.py

### Step 5.3 Escalation 工具与安全边界

【目标与修改文件清单】

- 修改 agents/tools.py。
- 创建 tests/test_agent_tools_escalation.py。

【具体的代码实现逻辑要点】

- create_handoff_summary 生成 request_id、reason、intent、urgency、entities、sensitive_data_required=false。
- 不创建真实工单。
- 不索取 password、verification_code、full_card_number。

【前置依赖条件】

- Step 5.1、Step 1.3 完成。

【独立验证测试命令】

    pytest -q tests/test_agent_tools_escalation.py

### Step 5.4 四个 Agent Profile

【目标与修改文件清单】

- 创建 agents/agent_profiles.py 或在 agents/agent_orchestrator.py 中实现四个 Agent 类。
- 创建 tests/test_agent_orchestrator.py 的 profile/白名单部分。

【具体的代码实现逻辑要点】

- General：0.3/900；Technical：0.1/1200；Billing：0.0/1100；Escalation：0.0/500。
- 每类只看到 shared search 加自己的工具集。
- EscalationAgent.handle 使用确定性文本，不调用 LLM，不设置真实转接状态。
- BaseAgent.handle 捕获异常，记录 stats，返回 AgentResponse。

【前置依赖条件】

- Step 5.1 至 Step 5.3 完成。
- Step 4.2 的 ToolManager 可被 shared RAG tool 包装。

【独立验证测试命令】

    pytest -q tests/test_agent_orchestrator.py -k "profile or scope or escalation"

### Checkpoint E-Agents

- [ ] 四类 Agent profile、温度、max_tokens、工具白名单通过。
- [ ] 工具均为只读建议/算术/交接摘要。
- [ ] 更新 progress.md 和 architecture.md。

## Step 6：Agent 编排器与结果合成（Checkpoint D）

### Step 6.1 路由评分与执行降级

【目标与修改文件清单】

- 修改 agents/agent_orchestrator.py 的 _domain_scores、_route_decision、_execute。
- 创建 tests/test_routing_decision.py。

【具体的代码实现逻辑要点】

- CRITICAL -> escalation；human_handoff -> escalation。
- OTHER + 长度 >2 + confidence <0.5 -> clarification。
- 按意图、关键词、entities 计算 general/technical/billing 分数。
- supporting 只从 secondary_intents 映射，不重新猜测。
- technical/billing 失败降级 General；escalation 不降级。

【前置依赖条件】

- Step 2.4、Step 5.4 完成。

【独立验证测试命令】

    pytest -q tests/test_routing_decision.py

### Step 6.2 Fan-out 与工具 trace

【目标与修改文件清单】

- 修改 agents/agent_orchestrator.py。
- 创建 tests/test_knowledge_required_retrieval.py。

【具体的代码实现逻辑要点】

- 业务请求每个实际 Agent 强制 search 一次，top_k=5，检索后从 LLM tools 移除 search。
- 多 Agent 使用 asyncio.gather(tasks, return_exceptions=true)。
- 收集成功 Agent、tools_used、tool_traces、retrieval_stage。
- 局部异常保留成功响应，不把一个异常升级为全局异常。

【前置依赖条件】

- Step 4.3、Step 5.4、Step 6.1 完成。

【独立验证测试命令】

    pytest -q tests/test_knowledge_required_retrieval.py tests/test_retrieval_stage_trace.py

### Step 6.3 ResponseComposer

【目标与修改文件清单】

- 修改 agents/agent_orchestrator.py。
- 创建 tests/test_response_composer.py。

【具体的代码实现逻辑要点】

- 一个成功响应直接返回。
- 多个成功响应调用 Composer；Prompt 明确主 Agent、用户问题、候选结果、去重/不编造/冲突说明要求。
- Composer 失败时输出 primary.content，再按顺序追加“补充说明”。
- 所有失败返回固定失败文案。

【前置依赖条件】

- Step 6.2 完成。
- Anthropic fake client 可模拟 Composer 成功与异常。

【独立验证测试命令】

    pytest -q tests/test_response_composer.py tests/test_agent_orchestrator.py -k "composer or parallel"

### Checkpoint D

- [ ] 路由评分、澄清、降级、fan-out/fan-in 通过。
- [ ] 强制检索只执行一次且 trace 保留。
- [ ] Composer 成功/失败路径通过。
- [ ] 更新 progress.md 和 architecture.md。

## Step 7：FastAPI 路由集成与主链路联调

### Step 7.1 lifespan 与组件初始化

【目标与修改文件清单】

- 修改 api/main.py 的 lifespan。
- 创建 tests/test_api_lifespan.py。

【具体的代码实现逻辑要点】

- 按环境变量初始化 IntentRecognizer、SkillManager、AgentOrchestrator、MemoryManager、MCPToolManager、KnowledgeBase、PerformanceMonitor、EndToEndEvaluator。
- 注册 knowledge_search，cache_ttl=300、supports_rerank=true、fallback 明确不可用。
- 未设置 ANTHROPIC_API_KEY 时启动失败信息必须清晰。
- 不在启动阶段连接任何未批准业务系统。

【前置依赖条件】

- Step 1 至 Step 6 完成。
- Redis/Chroma 可由 fake clients 替代进行单测。

【独立验证测试命令】

    pytest -q tests/test_api_lifespan.py

### Step 7.2 /health、/chat 与记忆时序

【目标与修改文件清单】

- 修改 api/main.py。
- 创建 tests/test_chat_integration.py。

【具体的代码实现逻辑要点】

- /chat 顺序必须是 get_context -> history last 5 -> recognize -> Request -> run -> 写入 user/assistant -> create_task(update_profile) -> ChatResponse。
- /health 返回 status、Agent stats、knowledge_base collection/model/doc_count；未就绪返回 503。
- ChatResponse 的 intent、agent、tools、routing、knowledge_used、entities、diagnostics 映射准确。
- 用户画像任务异常不得改变已返回响应。

【前置依赖条件】

- Step 7.1 完成。
- Step 3、Step 6 通过独立测试。

【独立验证测试命令】

    pytest -q tests/test_chat_integration.py
    pytest -q tests/test_api_contracts.py tests/test_chat_integration.py

### Step 7.3 搜索、知识与 trace 路由

【目标与修改文件清单】

- 修改 api/main.py。
- 创建 tests/test_api_supporting_routes.py。

【具体的代码实现逻辑要点】

- 实现 /search、/knowledge/add、/knowledge/upload、/knowledge/stats、/trace/tool/{request_id}、/trace/tools。
- 上传限制 10MB；支持 txt/md/json；错误码符合 system-spec。
- /skills、/skills/reload 保持 SkillManager 语义。
- 不新增认证假设；在文档中标记当前无鉴权。

【前置依赖条件】

- Step 4、Step 7.1 完成。

【独立验证测试命令】

    pytest -q tests/test_api_supporting_routes.py

### Checkpoint 主链路

- [ ] /health 在 fake 依赖下通过。
- [ ] /chat 完成一轮端到端数据流。
- [ ] 记忆回写、trace、知识门控可观察。
- [ ] 更新 progress.md 和 architecture.md。

## Step 8：可观测性与监控告警（Checkpoint E）

### Step 8.1 Stats、Trace deque 与指标

【目标与修改文件清单】

- 修改 agents/agent_orchestrator.py 与 mcp/tool_manager.py 的统计/trace 接口。
- 创建 tests/test_observability_contract.py。

【具体的代码实现逻辑要点】

- AgentStats 维护 total、success、total_ms、monitor_penalty、success_rate、avg_ms、routing_score。
- Tool stats 维护 success rate、avg latency、consecutive_fails、circuit_state。
- request trace deque 默认 maxlen=200；保存 request_id、intent diagnostics、agents、tool calls、retrieval、latency。
- Prometheus 指标名称与 labels 对齐；不宣称 requests_total 已准确递增，除非实现并测试递增点。

【前置依赖条件】

- Step 4.2、Step 6、Step 7.2 完成。

【独立验证测试命令】

    pytest -q tests/test_observability_contract.py

### Step 8.2 PerformanceMonitor 与 Z-score

【目标与修改文件清单】

- 创建 monitor/performance_monitor.py。
- 创建 tests/test_performance_monitor.py。

【具体的代码实现逻辑要点】

- interval 默认 10 秒。
- AnomalyDetector window=60、sensitivity=2.5；len(buffer)<30 或 stdev=0 时不告警。
- 阈值：Agent success <0.90、tool success <0.95、Agent avg >3000、tool avg >5000。
- routing penalty 范围 0..0.9，回写 orchestrator。
- 连续工具失败 >=3 生成建议；Agent success <0.85 且 total>10 生成路由建议。
- Webhook 使用 httpx timeout=5，不阻塞采集循环。

【前置依赖条件】

- Step 8.1 完成。
- Prometheus server 只在 PROMETHEUS_PORT 非零时启动。

【独立验证测试命令】

    pytest -q tests/test_performance_monitor.py

### Step 8.3 监控 API

【目标与修改文件清单】

- 修改 api/main.py。
- 创建 tests/test_monitor_routes.py。

【具体的代码实现逻辑要点】

- /monitor 返回 agent_stats、tool_stats、active_alerts 最近 10 条、suggestions 前 5 条。
- /metrics 返回 prometheus_client.generate_latest。
- start/stop 使用 asyncio task；停止时取消并吸收 CancelledError。
- trace API 只读取内存，不写持久化审计库。

【前置依赖条件】

- Step 8.1、Step 8.2、Step 7.1 完成。

【独立验证测试命令】

    pytest -q tests/test_monitor_routes.py

### Checkpoint E-Observability

- [ ] /monitor、/metrics、trace 可用。
- [ ] Z-score、阈值、penalty 和 Webhook 通过。
- [ ] trace 上限 200 且重启丢失边界已记录。
- [ ] 更新 progress.md 和 architecture.md。

## Step 9：100 条基准评测流水线集成（Checkpoint E）

### Step 9.1 Golden Dataset 加载与运行器

【目标与修改文件清单】

- 创建 evaluation/datasets/golden_benchmark_v2.1_100.json 的路径校验与加载器。
- 修改 evaluation/run_phase2_cases.ps1。
- 创建 tests/test_phase2_runner_contract.py。

【具体的代码实现逻辑要点】

- 校验顶层对象、cases 数组、case_count=100。
- 启动前 GET /health，校验 collection 与 embedding model。
- 每个 case 复用 user_id/conv_id；多轮只执行 user turns。
- POST /chat 后按 request_id 获取 trace。
- 保存实际结果与 gold_intent、gold_route、gold_retrieval、gold_answer、gold_tools、gold_safety。

【前置依赖条件】

- Step 7.2、Step 8.3 完成。
- benchmark collection 与产品 collection 已明确隔离。

【独立验证测试命令】

    pytest -q tests/test_phase2_runner_contract.py
    pwsh -File evaluation/run_phase2_cases.ps1 -CaseIds <single-case-id> -BaseUrl http://localhost:8000

### Step 9.2 deterministic_metrics

【目标与修改文件清单】

- 创建/完善 evaluation/deterministic_metrics.py。
- 创建 tests/test_deterministic_metrics.py 与 tests/test_deterministic_retrieval_scope.py。

【具体的代码实现逻辑要点】

- 不调用 LLM Judge。
- 计算 intent accuracy/F1、route exact match、secondary agent 约束、knowledge gate、retrieval Recall/MRR/nDCG、required facts/actions、forbidden claims、tool order、handoff 与 stability。
- source_chunk_id/content_hash 作为稳定 retrieval identity。
- 旧中文说明句只允许有限词面兜底，并在结果中保留 match mode，不冒充语义等价。
- 多 Agent retrieval 按 agent-local top-k union 计算，不做全局字符串拼接。

【前置依赖条件】

- Step 9.1 完成。
- Trace schema 已在 Step 8.1 固定。

【独立验证测试命令】

    pytest -q tests/test_deterministic_metrics.py tests/test_deterministic_retrieval_scope.py

### Step 9.3 EndToEndEvaluator 与 /eval/run

【目标与修改文件清单】

- 修改 evaluation/evaluator.py。
- 修改 api/main.py 的 /eval/run。
- 创建 tests/test_evaluator.py。

【具体的代码实现逻辑要点】

- 意图评测计算 accuracy、per-class precision/recall/F1、macro-F1。
- 对话 Judge 四维 relevance/accuracy/completeness/helpfulness，值域 0..1，PASS_THRESHOLD=0.75。
- Judge 失败只标记 judge_failed 并排除质量均值，不把 0.5 当成真实质量。
- baseline 对比下降超过 5% 记 regressions。
- /eval/run 支持请求体覆盖 intent_cases/dialog_cases，否则使用默认 11 条意图与 5 条对话样本。

【前置依赖条件】

- Step 6、Step 7、Step 9.2 完成。
- Anthropic fake client 支持合法、缺失字段和越界评分。

【独立验证测试命令】

    pytest -q tests/test_evaluator.py
    pytest -q tests/test_evaluator.py tests/test_deterministic_metrics.py

### Checkpoint E

- [ ] 100 case 可读取、执行、保存 trace 与 gold 对照。
- [ ] deterministic metrics 通过。
- [ ] LLM Judge 0.75、失败排除均值、5% 回归通过。
- [ ] /eval/run 返回完整报告。
- [ ] 更新 progress.md 和 architecture.md。

## 10. 全局 Definition of Done

每个微任务完成必须满足：

- [ ] 只修改计划允许的 1–2 个紧密文件，测试文件算作同一微任务的验证配套。
- [ ] 没有新增未声明第三方依赖。
- [ ] 没有改变未批准的 API、阈值、collection、Redis key 或外部业务边界。
- [ ] focused pytest 或脚本通过，并记录命令与结果。
- [ ] progress.md 已勾选并含证据。
- [ ] architecture.md 已记录新建文件职责。
- [ ] 失败路径、fallback、trace 和安全边界有测试。
- [ ] git diff、git status 和实际影响范围已检查。

## 11. 阶段性交付顺序

1. 完成 Step 1 并停在 Checkpoint A，确认契约后再进入 Step 2。
2. 完成 Step 2 与 Step 3，确认意图和记忆后再进入 RAG。
3. 完成 Step 4 与 Step 5，确认工具与 Agent 白名单后再进入编排。
4. 完成 Step 6 与 Step 7，形成可运行 /chat 主链路。
5. 完成 Step 8，补齐 trace 与监控。
6. 完成 Step 9，最后运行 100 case 与完整回归。
7. 每个 Checkpoint 由人工确认后再继续下一阶段。

