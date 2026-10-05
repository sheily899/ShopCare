# ShopCare 产品需求文档（PRD）

版本：0.1
状态：立项规格
事实基线：system-spec.md
产品形态：轻量级、非侵入式多 Agent 智能客服与分流中枢

## 1. 产品背景与商业目标（Business Value）

### 1.1 业务背景与痛点

电商售后咨询常在一轮会话中同时出现技术故障、订单物流、退款账单、发票和账户安全等问题。若系统只做关键词回复，容易发生两类问题：

- 答非所问：将登录失败当成退款问题，或把订单问题交由账单话术回答。
- 资损与信任风险：在没有订单、支付或退款后台事实的情况下，错误确认退款成功、到账时间或账单状态。
- 运营效率损失：人工需要反复追问订单号、金额、错误码、发生时间和环境，难以获得结构化交接信息。
- 知识不一致：政策说明、技术排查和资金类边界依赖人工记忆，规则更新后难以稳定执行。

### 1.2 产品定位

ShopCare 是一个只读知识大脑加安全转人工兜底的客服分流中枢。它以 CustomerOps 源码事实为复刻目标：

- 通过 LLM Tool Use、embedding 与 pattern 融合识别意图。
- 将技术、账单和通用诉求分派给专属 Agent；确认跨域复合诉求时并发协作。
- 对业务类诉求强制读取知识库后再生成答复。
- 对低置信度、紧急和人工诉求优先澄清或生成交接信息。
- 不直接操作订单、支付、退款、地址或工单系统。

产品不定位为交易执行系统、全渠道 CRM、人工坐席系统或完整的分布式可观测平台。

### 1.3 商业价值与交付口径

| 价值方向 | 可交付能力 | 业务口径 |
| --- | --- | --- |
| 首轮分流 | 18 类意图识别、主/次意图、紧急度与领域路由 | 降低简单咨询被错误转派的概率，形成可解释路由记录 |
| 安全回答 | 强制 RAG、字段补充、明确未知边界 | 不把缺少后台证据的推测包装为订单或资金结果 |
| 运营协同 | 结构化实体、工具 trace、人工交接摘要 | 减少人工客服重新收集上下文的工作量 |
| 韧性 | 专属 Agent 降级、工具熔断、知识库 fallback | 依赖故障时保留保守答复或升级建议，不中断整条会话 |
| 持续评估 | 100 条 Golden Benchmark、确定性指标、LLM Judge | 用可重复测试校验路由、检索、安全和答复约束 |

### 1.4 核心验收与业务度量

下表区分已存在的测量机制、可作为本期验收的阈值，以及尚未具备的实测结论。

| 指标 | 当前事实或验收口径 | 不可误读的边界 |
| --- | --- | --- |
| 意图识别质量 | 以 Golden Benchmark 的 expected_intent 与确定性指标计算 accuracy、per-class F1、macro-F1 | 源码没有已验证的 91% 实测结果；91% 不能作为当前已达成成绩 |
| 意图改进提示线 | Evaluator 在 intent_accuracy < 90% 时给出补样本/优化建议 | 这是建议阈值，不是系统已承诺的 SLA |
| 跨域分流确定性 | 只接受识别器确认的跨域 secondary_intents，最多 3 个；禁止仅凭关键词并发 | 需通过 Gold Route、supporting agent 与 trace 验证 |
| 100 条业务基准 | golden_benchmark.json：100 case，8 条回归样本加 92 条新增样本 | 基准通过率必须由实际运行产生，不能在 PRD 中预填成绩 |
| 对话质量 | EndToEndEvaluator 使用四维 0..1 分；单案例 PASS_THRESHOLD=0.75 | Judge 失败的 0.5 占位分不计入正常质量均分 |
| 响应时延 | Monitor 的告警边界为 Agent 平均耗时 >3000ms、工具平均耗时 >5000ms | 源码未定义端到端 P95/P99 SLA；本期只能以这些平均耗时告警线做运行验收 |
| 资损零越权 | 没有真实订单/支付写操作；billing 工具不能确认退款；转人工不承诺业务结果 | 这是权限与话术边界，不是“已完成退款”的业务指标 |
| 可靠性 | 工具连续失败 5 次打开熔断，60 秒后恢复尝试；知识搜索 TTL 300 秒 | 不是高可用集群、容灾多活或外部 SLA 承诺 |

### 1.5 本期成功标准

1. 100 条基准可通过标准运行脚本执行并输出结果与 trace。
2. 每个业务请求在 Agent 回答前有一次 top_k=5 的强制知识检索；转人工和纯寒暄场景按门控跳过。
3. 低置信度 OTHER 不误路由，返回澄清问题。
4. 任何路径不声称已执行退款、改地址、支付核验、工单创建或人工已接手。
5. 运行监控可暴露健康状态、基础 Prometheus 指标、最近工具 trace 和告警摘要。

## 2. 用户画像与交互规范（User Journey & Conversational UI）

### 2.1 用户群体

| 用户 | 核心任务 | 关键体验要求 |
| --- | --- | --- |
| 电商消费者（C 端） | 咨询物流、退款、发票、账户、登录和故障问题 | 快速得到可执行的下一步；信息不足时只追问必要字段；高风险时不被虚假承诺误导 |
| 人工客服/售后运营（B 端） | 审核复杂或高风险请求，维护知识，定位异常 | 获得用户原始诉求、已抽取实体、路由理由、工具与检索 trace；可补充知识并热加载 Skill |

### 2.2 C 端会话流程

1. 用户在 Web 聊天框发送文本，客户端附带 user_id 与可选 conv_id。
2. 服务读取 Redis/Chroma 上下文并识别主意图、次意图、紧急度和实体。
3. 若为低置信度 OTHER，则系统直接提示用户说明订单物流、退款账单、账户资料或技术故障。
4. 业务类请求经知识门控后由对应 Agent 回答；跨域复合请求并发处理并合并。
5. UI 展示客服 Markdown 答复；请求完成后保留会话 ID，后续消息沿用上下文。
6. 若 escalated=true 或 agent_recommended_handoff=true，UI 应展示“建议人工核验/已进入人工升级路由”的状态，但不得显示“人工已接手”“退款已处理”等未证实文案。

### 2.3 会话 UI 与安全渲染

- 前端提供聊天、新对话、知识库搜索、知识新增/上传与 Skill reload。
- 聊天支持普通段落、标题、列表、引用、代码块及有限 Markdown 行内格式。
- 模型和用户文本在生成标签前必须 HTML escape。
- 链接只允许 http:、https:、mailto: 协议；其他协议不可渲染为可点击链接。
- 三个可见起始问题是退款申请、技术排查、发票咨询；它们是交互入口，不代表真实订单操作。
- 上传仅允许 txt、md、json，后端大小上限 10MB；非 JSON 文本按 UTF-8 errors=ignore 解码。

### 2.4 结构化诊断与补字段交互

| 场景 | 面向用户的输出规范 | 可用事实 |
| --- | --- | --- |
| 技术故障 | 复述现象、可能原因、编号排查步骤、验证方式、待补环境信息 | 401/403/404/500 静态含义与排障方向；不读取服务端日志 |
| 账单/退款 | 说明可核验字段、当前不能确认的边界、下一步路径 | 检查 order_id、amount、date、payment_channel 是否齐全；不确认退款 |
| 订单/物流/通用 | 知识库说明或补充订单号/下单时间 | 不连接订单系统，不返回真实物流状态 |
| 人工升级 | 展示原始诉求、意图、紧急度、已有实体与保守后续说明 | 仅生成交接摘要；不得索取密码、短信验证码、完整支付凭证 |

### 2.5 B 端运营交互

- 健康查看：GET /health 返回 Agent stats、知识库 collection、embedding model 和片段数量。
- 监控查看：GET /monitor 返回 Agent/tool 统计、告警和建议。
- 证据追溯：GET /trace/tool/{request_id} 或 GET /trace/tools 查看内存中的工具调用明细。
- 知识维护：POST /knowledge/add 与 POST /knowledge/upload 写入 Chroma；POST /skills/reload 重扫业务 Skill。
- 边界：现状没有认证/授权，B 端接口在正式外部部署前不能被视为已安全多租户运营后台。

## 3. 业务场景与意图流转规格（Functional Scope）

### 3.1 三大核心业务域映射

18 个意图归纳为三类业务处理域，加一个安全升级节点：

| 域 | 意图 | 处理 Agent | 主要动作 |
| --- | --- | --- | --- |
| 通用与订单域 | query、complaint、request、greeting、feedback、order_status、logistics、other | general | 回答、澄清、补充字段、检索知识 |
| 技术支持域 | technical、technical_login、technical_crash | technical | 错误码解释、低风险诊断、知识检索 |
| 账单与售后域 | billing、account、refund、invoice、payment_issue、account_security | billing | 核验字段、金额比较、政策说明、升级建议 |
| 安全升级节点 | human_handoff；ESCALATION 为同值别名 | escalation | 确认升级原因、整理上下文、保守交接 |

### 3.2 意图融合规则

| 来源 | 默认权重 | 作用 |
| --- | --- | --- |
| LLM Tool Use | 0.7 | 由强制 classify_intent 工具输出结构化意图与置信度 |
| Embedding | 0.2 | 模板相似度候选；优先 voyage-3-lite，无法调用时为本地 256 维哈希向量 |
| Pattern | 0.1 | 特异及通用关键词/正则规则 |

embedding 不可用时，权重变更为 LLM 0.85、Pattern 0.15。

严格规则：

- 融合最高分 <0.5：intent=other。
- LLM 失败时：embedding 分数必须 >=0.45 才可作为降级结论。
- 特异 pattern 分数 >=0.5 且融合分 <0.8：可将泛化意图细化。
- 消息含 500 且不是 technical_login：强制 technical_crash，置信度至少 0.8。
- 单次 pattern 命中 0.5；多次为 min(1.0,0.5+0.25*(hits-1))。

### 3.3 复合意图规则

1. 仅当文本经 同时、另外、还想、还要、而且、并且、此外、分别、两个问题、两个诉求、除此之外、但是、不过、以及 或标点切分后，至少有两个有效 clause，才进入复合判断。
2. 每个 clause 以最长 pattern 命中确定候选，泛化标签在同域存在特异标签时移除。
3. 只保留跨域辅助意图，最多 3 个。
4. human_handoff 为终止意图：清空 secondary_intents，只走 escalation。
5. supporting Agent 只能由已确认的 secondary_intents 映射，禁止根据另一次关键词扫描自行增加 Agent。

### 3.4 紧急度与升级仲裁

| 紧急度 | 识别条件 | 路由效果 |
| --- | --- | --- |
| CRITICAL | 紧急、emergency、urgent、asap、立刻 | 直接 escalation，routing confidence=1.0 |
| HIGH | 今天、马上、尽快、hurry、now；或 human_handoff 无更高词时 | 正常路由或人工升级意图处理 |
| MEDIUM | 这周、soon、快点；或 complaint 无更高词时 | 正常领域路由 |
| LOW | 无匹配的普通情形 | 正常领域路由 |

真正 escalated=true 仅在 CRITICAL 或主意图 human_handoff。Agent 文本中出现转人工、人工客服、escalate、specialist、无法处理，仅产生 agent_recommended_handoff=true。

## 4. 核心系统机制与技术规格（Technical Specifications）

### 4.1 数据总线契约

ChatRequest：

| 字段 | 类型 | 默认 |
| --- | --- | --- |
| message | str | 必填 |
| user_id | str | anonymous |
| conv_id | Optional[str] | None，服务端生成 UUID |

ChatResponse 必须至少返回：conv_id、request_id、response、intent、secondary_intents、intent_group、agent_type、agent_types、primary_agent、supporting_agents、tools_used、routing_reason、routing_confidence、escalated、agent_recommended_handoff、latency_ms、knowledge_used、entities、intent_confidence、intent_source_scores、intent_diagnostics。

内部 Request 必须承载 message、user_id、conv_id、context、history、entities、intent、intent_group、urgency、intent_confidence、request_id、intent_diagnostics、secondary_intents、knowledge_required。内部 AgentResponse 必须承载 agent_type、content、success、confidence、latency_ms、escalate、tools_used、tool_traces。

### 4.2 多 Agent 编排

- 每种 AgentType 当前创建一个实例；虽然评分器支持同类多实例选择，MVP 不创建多副本。
- 一般请求以主意图、关键词和实体加分选择 primary Agent。
- 有 supporting Agent 时，使用 asyncio.gather(*tasks, return_exceptions=True) 并发执行。
- 一个成功结果直接返回；多个成功结果由 ResponseComposer LLM 合并，要求以主结论为主、去重冲突、不编造订单/退款/后台结果。
- Composer 不可用时，返回主结果与“补充说明”的确定性拼接。
- technical 或 billing 专属 Agent 失败时降级 General；General 和 escalation 不再降级；全部失败返回保守失败文案。

### 4.3 RAG 流水线

| 项目 | 规格 |
| --- | --- |
| 产品 collection | knowledge_base_zh |
| 旧库兼容 | knowledge_base；新库为空时尝试迁移 |
| Embedding | BAAI/bge-small-zh-v1.5，CPU，normalize_embeddings=true |
| 索引空间 | HNSW l2 |
| 切片 | 句子/换行边界，最多 500 字符 |
| 片段身份 | source_chunk_id 优先，含 content_hash、source_document_id、chunk_index 等元数据 |
| 业务门控 | 业务类请求开启；greeting、feedback、escalation/human_handoff、other 跳过 |
| 强制检索 | 每个执行 Agent 在回答前一次 search_knowledge_base，top_k=5 |
| Rewrite | 原 query 加 LLM 改写，n=3，temperature=0.3，max_tokens=256；不携带会话历史 |
| 召回 | 多 query asyncio.gather；每路 recall_k=max(top_k,5) |
| 去重 | source_chunk_id -> content_hash -> runtime_id -> content hash；保留高 score |
| Rerank | LLM Tool Use rerank_results，temperature=0，max_tokens=256，返回完整 index 排列 |
| 降级 | rerank 失败取原顺序前 top_k 并标记 rerank_degraded=true |

不得替换为交叉编码器 reranker，不得添加不存在的 rerank score threshold。Chroma 分数 round(1-distance,4) 不等于概率，也不作为最低拦截线。

### 4.4 三层记忆时序

| 层 | 存储与策略 |
| --- | --- |
| 工作记忆 | Redis wm:{user}:{conv}。最大窗口 20 条，达到 15 条压缩，保留最新 5 条，TTL=86400 秒 |
| 情景记忆 | Chroma episodic。保存压缩摘要，按同 conversation 优先、同 user 补足，最多 5 条 |
| 用户画像 | Chroma user_profile。保存 preferences 与 entities；读取最近 10 条对话，以 asyncio.create_task 异步更新 |

摘要 LLM max_tokens=256、temperature=0；摘要最大 800 字符，episodic full_text 最大 500 字符。若摘要失败使用受控 fallback，而非丢失整个会话。

### 4.5 安全与容灾

- Agent 工具白名单及 JSON Schema 禁止 Agent 任意调用外部动作。
- 账单工具只能检查字段或算差值，固定不能确认退款。
- 转人工提醒不索取密码、短信验证码、完整支付凭证。
- Tool Circuit Breaker：closed/open/half_open，连续失败 5 次打开，60 秒后进入恢复尝试。
- Tool 默认 timeout=30 秒；缓存最多 5000 条，超限删除最早 1250 条。
- knowledge_search 缓存 TTL=300 秒，失败 fallback 返回明确不可用说明和 score=0.0。
- fallback 成功仍可能是保守文本，不能以 ToolResult.success=true 认定外部检索/业务查询真实完成。

## 5. 质量评估与监控基线（Evaluation & Observability）

### 5.1 评测机制

| 层级 | 机制 | 验收意义 |
| --- | --- | --- |
| 离线确定性评测 | golden_benchmark.json 与 deterministic_metrics.py | 校验意图、路由、门控、RAG Recall/MRR/nDCG、答案约束、工具顺序、升级和系统稳定性 |
| 端到端评测 | EndToEndEvaluator | 计算意图 accuracy/F1，并通过 LLM Judge 对对话质量评分 |
| 质量维度 | relevance、accuracy、completeness、helpfulness | 每项 0..1；综合单例阈值 0.75 |
| 回归 | 当前结果相对历史内存或 EVAL_BASELINE_PATH | 指标下降超过 5% 标记 regression |

100 条数据集字段包括 expected_intent、gold_intent、gold_route、gold_retrieval、gold_answer、gold_tools、gold_safety 及多轮 conversation，因此不是只有问答文本的样例集。

运行基准前，run_phase2_cases.ps1 会校验 /health 的 collection 和 embedding model。该脚本默认期望 benchmark collection knowledge_base_benchmark_v2；与产品 collection knowledge_base_zh 不同，执行前必须显式选择正确数据环境。

### 5.2 可观测性基线

- GET /health：服务状态、Agent stats、知识库 collection/model/片段数。
- GET /monitor：Agent/tool 统计、近 10 条未解决告警、前 5 条建议。
- GET /metrics：Prometheus 输出；可选独立 Prometheus HTTP server。
- GET /trace/tool/{request_id} 与 GET /trace/tools：最近工具 trace。
- 内存 trace deque 默认最大 200 条，服务重启丢失。
- Monitor 每 10 秒采集；Z-score window=60、sensitivity=2.5，样本数少于半窗不检测。
- 告警线：Agent success rate <0.90、tool success rate <0.95、Agent avg latency >3000ms、tool avg latency >5000ms。

本期观测定位为基础日志、内存 trace、Prometheus 指标与可选 Webhook。没有 Langfuse、OpenTelemetry、持久化 trace 或全链路分布式 APM。

## 6. MVP 分阶段实施边界（P0 / P1 / P2）

### P0：核心最小闭环

范围：

- 统一 Pydantic/API 契约与内部 Request、AgentResponse、OrchestratorResult。
- 18 类意图、LLM Tool Use + embedding + pattern 融合与 0.5/0.45/0.8 阈值。
- general、technical、billing、escalation 四个节点的基础响应与明确边界。
- Redis 工作记忆：20 条上限、15 条触发压缩、保留 5 条、86400 秒 TTL。
- Chroma 基础知识检索：knowledge_base_zh、BGE-small-zh、500 字符切片、业务请求强制单次检索。
- 只读知识答复、补字段、诊断建议、交接摘要。

验收：单领域请求可得到对应 Agent、知识检索 trace 和不越权答复；人工意图仅产生交接状态，不产生工单或交易动作。

不做：真实订单/支付/退款/地址 API，Java 后端，跨系统认证，完整 APM。

### P1：生产级韧性补齐

范围：

- Query Rewrite n=3、并发召回、稳定身份去重、LLM Rerank 与 rerank_degraded 状态。
- 已确认跨域 secondary_intents 的 asyncio 并发与 ResponseComposer 合并/fallback。
- Tool Circuit Breaker、timeout、缓存与知识库 fallback。
- Chroma 情景记忆、用户画像、异步画像更新。
- 结构化工具 trace、路由理由、工具白名单和 schema 校验。

验收：任一专属 Agent、Composer、rerank 或知识工具失败时，能以规定的降级路径给出可解释结果，不让局部异常中断所有并发任务。

不做：把 LLM rerank 替换成交叉编码器；将 fallback 伪装为真实业务查询；创建真实人工工单。

### P2：评测与监控固化

范围：

- 接入 100 条 Golden Benchmark 的自动化执行与结果归档。
- 接入 EndToEndEvaluator、四维 Judge、0.75 单例门槛与超过 5% 的回归标记。
- 启动 Z-score Monitor、基础 Prometheus 指标、/health、/monitor、/metrics 和内存 trace 查询。
- 建立基于实际 benchmark 结果的意图质量、路由质量、RAG 和安全质量报告。

验收：可重复运行数据集、追溯 request/trace、观察告警，并明确区分运行实测成绩与未验证目标。

不做：把内存 trace 说成审计级长期留存；把基础 Prometheus 说成分布式 APM；未经另行立项接入认证、告警平台、人工队列或业务执行系统。

## 附：全局红线

1. 所有对外话术必须把“建议、核验、交接”与“已执行、已确认、已到账”分开。
2. 没有真实后台证据时，不得确认订单状态、退款结果、支付结果、地址变更或人工接手。
3. 不得引入 system-spec.md 未证明的 Java 服务、交叉编码 reranker、分布式 APM 或外部写操作。
4. 新能力若改变稳定接口、验收标准或产品范围，必须先更新 PRD，再进入设计和实现阶段。

