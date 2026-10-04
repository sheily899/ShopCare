# ShopCare 技术栈与环境配置

版本：0.1  
事实来源：system-spec.md、design-document.md、原工程 requirements.txt 与 Docker 配置。

## 1. 技术栈总览

ShopCare 保持原工程的原生 Python 异步架构，不引入新的 Agent 编排框架。

| 层次 | 锁定方案 | 职责 |
| --- | --- | --- |
| 运行时 | Python 3.12 | 本地 venv；容器 python:3.12-slim |
| Web/API | FastAPI 0.115.5 + Uvicorn 0.32.1 | 异步 REST API 与 lifespan |
| 契约 | Pydantic 2.10.3 | API 边界模型；内部编排保留 dataclass |
| LLM | Anthropic SDK 0.40.0 | Tool Use、回答、摘要、rewrite、rerank、Judge |
| 异步 | Python asyncio | create_task、gather、wait_for、to_thread |
| 向量库 | ChromaDB 0.5.23 | knowledge_base_zh、episodic、user_profile |
| Embedding | sentence-transformers 3.3.1 + BGE | BAAI/bge-small-zh-v1.5，CPU，归一化 |
| 模型加载 | transformers 4.46.3 + torch 2.4.1+cpu | 固定 BGE CPU 链路 |
| 工作记忆 | Redis 7-alpine + redis 5.2.1 | Redis list/string、AOF、TTL |
| HTTP/观测 | httpx 0.28.1 + prometheus-client 0.21.1 | Webhook、Prometheus 基础指标 |
| 测试/配置 | pytest 8.3.4 + python-dotenv 1.0.1 | 后端测试与 .env |
| 上传 | python-multipart 0.0.12 | FastAPI multipart 文件上传 |

不采用：LangGraph、Celery、Kafka、Temporal、交叉编码器 Reranker、Java 服务、Langfuse、OpenTelemetry，以及真实订单/支付/退款/工单 SDK。

## 2. 锁定依赖与职责

### 2.1 标准 requirements.txt

    anthropic==0.40.0
    fastapi==0.115.5
    uvicorn[standard]==0.32.1
    pydantic==2.10.3
    python-multipart==0.0.12
    redis==5.2.1
    chromadb==0.5.23
    sentence-transformers==3.3.1
    transformers==4.46.3
    prometheus-client==0.21.1
    httpx==0.28.1
    python-dotenv==1.0.1
    pytest==8.3.4

Docker 构建额外执行：

    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu

### 2.2 职责边界

| 依赖 | 具体用途 |
| --- | --- |
| anthropic | classify_intent、Agent Tool Use、摘要、Query Rewrite、LLM Rerank、LLM Judge |
| fastapi | /chat、/health、/search、知识库、评测、trace 路由 |
| uvicorn | ASGI 运行 |
| pydantic | ChatRequest、ChatResponse、上传与评测输入校验 |
| python-multipart | /knowledge/upload |
| redis | async 工作记忆、summary key、TTL |
| chromadb | 产品知识库、情景记忆、用户画像 |
| sentence-transformers | BGE EmbeddingFunction 与 CPU 编码 |
| transformers | BGE 模型加载兼容链 |
| torch | 2.4.1+cpu CPU 推理基础 |
| prometheus-client | Agent/tool Gauge、Histogram、Counter、/metrics |
| httpx | Monitor Webhook HTTP 客户端 |
| python-dotenv | 环境变量加载 |
| pytest | Python 测试与确定性评测测试 |

不得用新增依赖替换上述职责，也不得把工具 fallback 当成真实业务查询。

## 3. Python 环境

本地安装：

    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu
    pip install -r requirements.txt

校验：

    python --version
    pip show anthropic fastapi uvicorn pydantic redis chromadb sentence-transformers transformers torch
    python -m pytest

容器基础镜像固定为 python:3.12-slim。生产镜像以 UID 1000 的非 root 用户运行，暴露 8000，并以 /health 作为健康检查。

## 4. LLM 配置

必需：

    ANTHROPIC_API_KEY=<secret>

可选：

    ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
    ANTHROPIC_BASE_URL=<compatible-endpoint-or-empty>

关键行为参数：

| 用途 | temperature | max_tokens |
| --- | ---: | ---: |
| Memory 摘要 | 0 | 256 |
| 用户画像 | 0 | 256 |
| Query Rewrite，目标 n=3 | 0.3 | 256 |
| Rerank | 0 | 256 |
| ResponseComposer | 0.1 | 1000 |
| LLM Judge | 0 | 256 |
| GeneralAgent | 0.3 | 900 |
| TechnicalAgent | 0.1 | 1200 |
| BillingAgent | 0.0 | 1100 |
| EscalationAgent | 0.0 | 500 |

意图、Rerank 和 Judge 必须继续使用结构化 Tool Use/JSON 校验。LLM 失败进入既定 fallback，不得转换为退款、订单或人工已接手等业务事实。

## 5. Redis 中间件

### 5.1 服务配置

| 配置 | 值 |
| --- | --- |
| 镜像 | redis:7-alpine |
| 默认端口 | 6379 |
| 命令 | redis-server --appendonly yes --requirepass <password> |
| 健康检查 | redis-cli -a <password> ping |
| 容器连接 | redis://:<password>@redis:6379/0 |
| 本地连接 | redis://:<password>@localhost:6379/0 |

环境变量：

    REDIS_PASSWORD=<strong-random-secret>
    REDIS_URL=redis://:<password>@redis:6379/0

原工程默认回退密码为 customerops123，仅允许本地开发；部署必须覆盖，不得提交密码。

### 5.2 数据规则

工作记忆 key：

    wm:{user_id}:{conv_id}

- Message JSON 使用 LPUSH 写入。
- 每次写入 EXPIRE 86400。
- LLEN >= 15 触发摘要压缩。
- 读取最多 20 条，压缩后保留最新 5 条。
- summary 写 Redis，并写入 episodic Chroma。
- Redis 只负责短期工作记忆和摘要，不负责向量检索或订单状态。

## 6. ChromaDB 中间件

### 6.1 独立服务

| 配置 | 产品值 |
| --- | --- |
| 镜像 | chromadb/chroma:0.5.23 |
| 容器端口 | 8000 |
| 宿主映射 | 8001:8000 |
| 持久化 | IS_PERSISTENT=TRUE |
| 数据目录 | /chroma/chroma |
| telemetry | ANONYMIZED_TELEMETRY=FALSE |
| Compose 应用连接 | CHROMA_HOST=chromadb，CHROMA_PORT=8000 |
| 宿主应用连接 | CHROMA_HOST=localhost，CHROMA_PORT=8001 |

### 6.2 本地持久化 fallback

连接 HTTP Chroma 失败时允许 PersistentClient：

    CHROMA_PERSIST_DIRECTORY=/app/data/chroma

本地可使用：

    CHROMA_PERSIST_DIRECTORY=.data/chroma

HTTP 优先、PersistentClient fallback 的顺序必须保留；fallback 发生时应记录日志，不能隐藏独立服务故障。

### 6.3 Collection 与模型

产品知识库：

    CUSTOMEROPS_KB_COLLECTION=knowledge_base_zh
    legacy_collection=knowledge_base
    CUSTOMEROPS_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
    device=cpu
    normalize_embeddings=true
    hnsw:space=l2

本机离线运行可将 `CUSTOMEROPS_EMBEDDING_MODEL` 覆盖为相对目录
`models/BAAI/bge-small-zh-v1.5`（该目录为已验证可加载的本机复用模型，二进制不纳入 Git）；默认值仍保持远程模型标识，避免把个人机器路径写入通用配置。

长期记忆：

    episodic
    user_profile

知识片段最多 500 字符，保存 title、source_document_id、source_chunk_id、content_hash、chunk_index、total_chunks、corpus_version。产品默认 collection 与 benchmark collection 必须分开：

    产品：knowledge_base_zh
    benchmark runner 默认：knowledge_base_benchmark_v2

## 7. 应用环境变量

### 7.1 核心变量

    ANTHROPIC_API_KEY
    ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
    ANTHROPIC_BASE_URL=
    REDIS_URL
    REDIS_PASSWORD
    CHROMA_HOST=chromadb
    CHROMA_PORT=8000
    CHROMA_PERSIST_DIRECTORY=/app/data/chroma
    CUSTOMEROPS_KB_COLLECTION=knowledge_base_zh
    CUSTOMEROPS_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
    CUSTOMEROPS_SKILLS_DIR=/app/skills
    CUSTOMEROPS_SKILLS_MAX_PROMPT_CHARS=5000

### 7.2 监控/评测变量

    LOG_LEVEL=INFO
    PROMETHEUS_PORT=0
    MONITOR_INTERVAL=10
    ALERT_WEBHOOK_URL=
    CUSTOMEROPS_TOOL_TRACE_MAX=200
    CUSTOMEROPS_COMPOSER_MAX_TOKENS=1000
    CUSTOMEROPS_COMPOSER_TEMPERATURE=0.1
    EVAL_BASELINE_PATH=/app/data/eval/baseline.json

PROMETHEUS_PORT=0 表示不启动独立指标 server，但保留 /metrics。Monitor 默认每 10 秒采集；Trace deque 默认最多 200 条，重启丢失。

## 8. Docker Compose 拓扑

| 服务 | 镜像/端口 | 职责 |
| --- | --- | --- |
| redis | redis:7-alpine / 6379 | 工作记忆 |
| chromadb | chromadb/chroma:0.5.23 / 8001:8000 | 知识库与长期记忆 |
| prometheus | prom/prometheus:latest / 9090 | 指标，retention=200h |
| customerops | ShopCare Python / 8000 | FastAPI |
| nginx | nginx:alpine / 80 | 静态前端与 API 代理 |

Compose 网络为 172.28.0.0/16。持久卷包括 Redis、Chroma、Prometheus、应用 eval/chroma、skills 和 Nginx logs。

后端启动命令：

    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

## 9. 安全与升级纪律

- API key、Redis 密码和 .env 不提交 Git。
- 原工程 CORS 为 allow_origins=*；这只能作为开发兼容基线，不能宣称生产鉴权。
- 当前没有认证/授权；知识写入、上传、Skill reload、评测和 trace 不得直接暴露到不可信公网。
- 不增加真实订单、支付、退款、地址或工单凭据。
- 未经兼容性评估不得升级锁定版本。
- 变更 collection、embedding model、Redis key、ChatResponse 字段或阈值，必须先更新 system-spec.md、PRD.md、design-document.md。
- 增加依赖必须记录用途、成本、许可证和替代方案，不得以“更先进”为理由引入重型框架。

## 10. 环境验收

- Python 3.12，torch 2.4.1+cpu。
- requirements.txt 无版本漂移。
- Redis 7-alpine 健康、AOF 开启、部署密码已替换。
- Chroma 0.5.23 独立服务可访问，或明确记录 PersistentClient fallback。
- /health 返回 knowledge_base_zh 与 BAAI/bge-small-zh-v1.5。
- /chat 可读写 Redis 工作记忆。
- 业务请求可产生 Chroma 检索或明确 fallback。
- /metrics、/monitor、trace API 可访问。
- pytest 与 100 case runner 使用目标 collection/model。

