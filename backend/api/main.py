"""ShopCare 公共 API 边界。

本模块只负责 FastAPI 应用壳与 Pydantic 请求/响应契约；业务编排、记忆
访问和外部工具调用均留在该边界之外。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional
import uuid

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from handoff.feature_flag import handoff_enabled
from handoff.config import callback_due_at, load_handoff_copy_config, load_handoff_urgency_config
from handoff.generation import validate_handoff_configuration
from handoff.processor import enrich_handoff, refresh_handoff_after_messages
from handoff.urgency import is_explicit_handoff_requested, score_temporary_urgency

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentType,
    BillingAgent,
    build_business_field_state,
    EscalationAgent,
    GeneralAgent,
    ResponseComposer,
    Request,
    TechnicalAgent,
)
from agents.case_facts import CaseFacts
from agents.handoff import build_handoff_payload
from agents.response_guard import (
    asked_fields_in_text,
    build_return_refund_request_reply,
    filter_external_refund_instructions,
    guard_response,
)
from intent.intent_recognizer import IntentCategory, recognize_intent
from intent.emotion_llm import analyze_emotion
from intent.beauty_intent_ticket_registry import (
    BEAUTY_INTENT_TICKET_BY_ID,
    policy_for_intents,
    should_prefetch_knowledge,
)
from intent.slot_registry import (
    apply_pending_text_answer,
    get_slot,
    normalize_value,
    pending_text_answer_field_id,
)
from mcp.knowledge_base import BUSINESS_KB_DOMAINS, KnowledgeBase
from mcp.tool_manager import MCPToolManager, Tool
from memory.conversation_memory import MemoryContext, MemoryManager, MsgRole
from monitor.token_usage import TokenUsageLedger
from skills.skill_manager import SkillManager
from storage.sqlite_case_store import SQLiteCaseStore


logger = logging.getLogger(__name__)

# Why：10MiB 是 Step7.3 的单文件安全上限，先读上限+1 字节即可拒绝超大上传，
# 避免管理接口把任意大文件完整载入内存。
MAX_KNOWLEDGE_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_KNOWLEDGE_SUFFIXES = {".txt", ".md", ".json"}
TRACE_MEMORY_LIMIT = 200


class IntentRecognizer:
    """为 API 生命周期提供意图识别函数的轻量适配器。

    Args:
        model: 可选的 Anthropic 模型名；实际识别函数仍遵循核心模块的注入边界。

    Returns:
        无；实例通过 ``recognize`` 暴露异步识别入口。
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model

    async def recognize(
        self,
        message: str,
        history: Any = None,
    ) -> Any:
        """调用核心三路意图识别，不在 API 层复制识别算法。

        Args:
            message: 待识别的用户消息。
            history: 可选历史，仅由核心识别器决定是否使用。

        Returns:
            核心模块返回的 ``IntentResult`` dataclass。
        """

        return await recognize_intent(message, history=history)

    async def recognize_intent(
        self,
        message: str,
        history: Any = None,
    ) -> Any:
        """兼容 API 主链路约定的 ``recognize_intent`` 方法名。

        Args:
            message: 待识别的用户消息。
            history: 最近会话消息列表。

        Returns:
            核心模块返回的 ``IntentResult`` dataclass。
        """

        return await self.recognize(message, history)


class PerformanceMonitor:
    """保存生命周期级监控配置的轻量组件。

    详细 Prometheus 与 Z-score 逻辑留给 Step 8，启动阶段不产生外部调用。
    """

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        """标记监控组件关闭，保持关闭操作可重复执行。"""

        self.closed = True


class EndToEndEvaluator:
    """保存评测运行配置的轻量组件。

    100 条基准评测流水线留待 Step 9；此处只完成全局依赖装配。
    """

    def __init__(self, dataset_path: Optional[str] = None) -> None:
        self.dataset_path = dataset_path

    async def close(self) -> None:
        """评测组件当前无外部资源，保留统一生命周期接口。"""


def _read_settings() -> Dict[str, str]:
    """读取生命周期所需环境变量并返回不含派生状态的配置快照。

    Returns:
        配置键到字符串值的映射；敏感值只保存在内存，不写入日志。
    """

    defaults = {
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_MODEL": "claude-3-5-sonnet-20241022",
        "ANTHROPIC_BASE_URL": "",
        "REDIS_URL": "redis://localhost:6379/0",
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": "8000",
        "CHROMA_PERSIST_DIRECTORY": ".data/chroma",
        "CUSTOMEROPS_KB_COLLECTION": "knowledge_base_zh",
        "CUSTOMEROPS_EMBEDDING_MODEL": "BAAI/bge-small-zh-v1.5",
        "CUSTOMEROPS_SKILLS_DIR": "skills",
        "CUSTOMEROPS_SKILLS_MAX_PROMPT_CHARS": "5000",
        "EVALUATION_DATASET": "evaluation/datasets/golden_benchmark.json",
    }
    return {key: os.getenv(key, default) for key, default in defaults.items()}


def _create_redis_client(redis_url: str) -> Any:
    """创建异步 Redis 客户端；不在启动阶段主动 ping 外部服务。

    Args:
        redis_url: Redis 连接 URL。

    Returns:
        配置好的异步 Redis 客户端。

    Raises:
        RuntimeError: Redis SDK 不可用时抛出明确错误。
    """

    try:
        import redis.asyncio as redis
    except Exception as exc:  # pragma: no cover - 依赖环境决定
        raise RuntimeError(f"Redis SDK 不可用: {exc}") from exc
    return redis.from_url(redis_url, decode_responses=True)


def _create_anthropic_client(settings: Dict[str, str]) -> Any:
    """仅在配置 API key 时构造 Anthropic 异步客户端。

    Args:
        settings: 生命周期配置快照。

    Returns:
        ``AsyncAnthropic`` 客户端；未配置 key 时返回 ``None``。
    """

    api_key = settings.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        # Why：允许离线启动并使用确定性降级，避免把缺失凭证变成白屏启动故障。
        logger.warning("未设置 ANTHROPIC_API_KEY，LLM 能力将使用安全降级路径")
        return None
    try:
        import anthropic

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if settings.get("ANTHROPIC_BASE_URL"):
            kwargs["base_url"] = settings["ANTHROPIC_BASE_URL"]
        return anthropic.AsyncAnthropic(**kwargs)
    except Exception as exc:  # pragma: no cover - SDK 安装环境决定
        logger.warning("Anthropic 客户端初始化失败，将使用安全降级：%s", exc)
        return None


async def _knowledge_handler(params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
    """为已装配的知识库工具执行一次只读检索。

    Args:
        params: 包含 query 与可选 top_k 的工具参数。
        context: 工具调用上下文；当前只读 handler 不修改该上下文。

    Returns:
        知识库切片列表；知识库不可用时由 ToolManager fallback 处理。
    """

    knowledge_base = getattr(app.state, "knowledge_base", None)
    if knowledge_base is None:
        raise RuntimeError("KnowledgeBase 尚未装配")
    query = str(params.get("query", ""))
    top_k = int(params.get("top_k", 5))
    return await asyncio.to_thread(
        knowledge_base.search,
        query,
        top_k,
        domain=params.get("domain"),
    )


async def _knowledge_fallback(
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]],
    error: str,
) -> Dict[str, Any]:
    """返回明确的知识库不可用说明，不伪造检索结果。"""

    return {"available": False, "message": "知识库暂不可用，请稍后重试"}


def _register_knowledge_tools(manager: MCPToolManager) -> None:
    """注册 Step7.1 的知识检索工具及兼容别名。

    Args:
        manager: 已初始化的工具治理管理器。

    Returns:
        无；注册后的工具由 ``app.state.tool_manager`` 持有。
    """

    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1},
            "domain": {"type": "string", "enum": sorted(BUSINESS_KB_DOMAINS)},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    for name in ("knowledge_search", "search_knowledge_base"):
        tool = Tool(
            name=name,
            description="只读检索 ShopCare 产品知识库，不执行任何外部业务写操作。",
            schema=schema,
            handler=_knowledge_handler,
            fallback=_knowledge_fallback,
            cache_ttl=300,
            timeout=30.0,
        )
        # Why：rerank 能力由 Step4.4 的工具治理协议标记，TTL 300 秒减少重复召回压力。
        tool.supports_rerank = True
        manager.register(tool)


async def _close_resource(resource: Any) -> None:
    """尽力关闭异步或同步资源，隔离关闭阶段异常。"""

    if resource is None:
        return
    try:
        close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result
    except Exception as exc:  # pragma: no cover - 仅在依赖关闭异常时触发
        logger.warning("生命周期资源关闭失败：%s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化并释放 ShopCare 全局组件。

    Args:
        app: 当前 FastAPI 应用实例。

    Yields:
        ``None``；yield 期间应用可安全访问 ``app.state`` 中的组件。

    说明：启动阶段只创建本地客户端与只读知识组件，不访问订单、支付、工单或人工坐席系统。
    """

    try:
        from dotenv import load_dotenv

        # Why：以 api/main.py 所在 backend 目录为基准，避免从仓库根目录或
        # 其他工作目录启动 Uvicorn 时漏读 backend/.env，造成 LLM client 为空。
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
    except Exception as exc:  # pragma: no cover - python-dotenv 可选安装差异
        logger.warning(".env 加载失败，继续使用系统环境变量：%s", exc)

    settings = _read_settings()
    if handoff_enabled():
        try:
            config_versions = validate_handoff_configuration()
        except Exception as exc:
            logger.exception("HANDOFF 配置预检失败，后端不会进入就绪状态")
            raise RuntimeError(f"HANDOFF 配置预检失败：{exc}") from exc
        logger.info("HANDOFF 配置预检通过 versions=%s", config_versions)

    sqlite_store = SQLiteCaseStore(
        os.getenv("SHOPCARE_SQLITE_PATH", str(Path(".data") / "shopcare.sqlite3"))
    )
    redis_client = _create_redis_client(settings["REDIS_URL"])
    anthropic_client = _create_anthropic_client(settings)
    knowledge_base = KnowledgeBase(
        chroma_host=settings["CHROMA_HOST"],
        chroma_port=int(settings["CHROMA_PORT"]),
        chroma_path=settings["CHROMA_PERSIST_DIRECTORY"],
        embedding_model=settings["CUSTOMEROPS_EMBEDDING_MODEL"],
        collection_name=settings["CUSTOMEROPS_KB_COLLECTION"],
    )
    tool_manager = MCPToolManager(
        api_key=settings["ANTHROPIC_API_KEY"] or None,
        model=settings["ANTHROPIC_MODEL"],
        client=anthropic_client,
    )
    app.state.knowledge_base = knowledge_base
    _register_knowledge_tools(tool_manager)
    memory_manager = MemoryManager(
        redis_client=redis_client,
        redis_url=settings["REDIS_URL"],
        llm_client=anthropic_client,
        chroma_client=getattr(knowledge_base, "_client", None),
        model=settings["ANTHROPIC_MODEL"],
    )
    intent_recognizer = IntentRecognizer(model=settings["ANTHROPIC_MODEL"])
    skill_manager = SkillManager(
        settings["CUSTOMEROPS_SKILLS_DIR"],
        max_prompt_chars=int(settings["CUSTOMEROPS_SKILLS_MAX_PROMPT_CHARS"]),
    )
    agents = {
        "consult": GeneralAgent(client=anthropic_client, skill_manager=skill_manager),
        "order": TechnicalAgent(client=anthropic_client, skill_manager=skill_manager),
        "after_sales": BillingAgent(client=anthropic_client, skill_manager=skill_manager),
        "escalation": EscalationAgent(client=anthropic_client, skill_manager=skill_manager),
    }
    typed_agents = {
        AgentType.INQUIRY: agents["consult"],
        AgentType.FULFILLMENT: agents["order"],
        AgentType.REFUND_RISK: agents["after_sales"],
        AgentType.ESCALATION: agents["escalation"],
    }
    orchestrator = AgentOrchestrator(
        agent_pool=typed_agents,
        # Why：生产检索必须经过工具治理层，才能连接 query rewrite、域内并发召回和 rerank。
        knowledge_search=tool_manager,
        # Why：生命周期已创建的同一个 LLM client 必须注入合成器，避免多 Agent
        # 成功时错误地创建空 Composer 并触发确定性降级。
        composer=ResponseComposer(
            client=anthropic_client,
            model=settings["ANTHROPIC_MODEL"],
        ),
    )
    performance_monitor = PerformanceMonitor()
    evaluator = EndToEndEvaluator(settings["EVALUATION_DATASET"])
    token_usage = TokenUsageLedger(max_items=TRACE_MEMORY_LIMIT)

    app.state.settings = settings
    app.state.redis_client = redis_client
    app.state.anthropic_client = anthropic_client
    app.state.tool_manager = tool_manager
    app.state.memory_manager = memory_manager
    app.state.intent_recognizer = intent_recognizer
    app.state.agents = agents
    app.state.agent_pool = typed_agents
    app.state.agent_orchestrator = orchestrator
    app.state.skill_manager = skill_manager
    app.state.performance_monitor = performance_monitor
    app.state.evaluator = evaluator
    app.state.token_usage = token_usage
    app.state.sqlite_store = sqlite_store
    app.state.handoff_tasks = {}
    app.state.ready = True
    if handoff_enabled():
        try:
            pending_handoffs = await asyncio.to_thread(sqlite_store.list_pending_handoff_enrichments)
            for pending in pending_handoffs:
                _schedule_handoff_enrichment(
                    sqlite_store, str(pending["conv_id"]), pending
                )
        except Exception as exc:
            logger.warning("启动时恢复转人工后台任务失败：%s", exc)
    try:
        yield
    finally:
        app.state.ready = False
        handoff_tasks = list(getattr(app.state, "handoff_tasks", {}).values())
        for task in handoff_tasks:
            task.cancel()
        if handoff_tasks:
            await asyncio.gather(*handoff_tasks, return_exceptions=True)
        await _close_resource(performance_monitor)
        await _close_resource(skill_manager)
        await _close_resource(anthropic_client)
        await _close_resource(getattr(knowledge_base, "_client", None))
        await _close_resource(redis_client)


app = FastAPI(title="ShopCare", version="0.1.0", lifespan=lifespan)
# Why：默认仅允许本机开发前端跨域；部署方需显式配置可信来源，不能默认接受任意网页来源。
_DEFAULT_CORS_ORIGINS = {
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}
_CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("SHOPCARE_CORS_ORIGINS", "").split(",")
    if origin.strip()
] or sorted(_DEFAULT_CORS_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """单次 ShopCare 会话请求的边界校验契约。

    本模型只做 API 边界校验，不负责意图识别或 Agent 执行。

    Attributes:
        message: 必须包含非空白字符的用户消息。
        user_id: 稳定的用户标识，默认值为 ``anonymous``。
        conv_id: 调用方可选传入的会话标识。
    """

    message: str
    user_id: str = "anonymous"
    conv_id: Optional[str] = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        """在消息进入核心链路前拒绝纯空白文本。

        Args:
            value: API 载荷中的原始消息。

        Returns:
            包含非空白内容时返回原始消息。

        Raises:
            ValueError: 去除首尾空白后消息为空时抛出。
        """
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class RegisterConsumerRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    password: str = Field(min_length=6, max_length=200)
    display_name: str = Field(min_length=1, max_length=40)


class ChatResponse(BaseModel):
    """公共 API 边界对外序列化的响应契约。

    字段映射内部路由结果，但不暴露 dataclass 实例或客户端句柄等实现细节。

    Attributes:
        conv_id: 响应所属的会话标识。
        request_id: 请求追踪标识，缺省为空字符串。
        response: 面向用户的响应文本。
        intent: 路由层选定的主意图值。
        secondary_intents: 跨域辅助意图值列表。
        intent_group: 主意图所属的粗粒度业务分组。
        agent_type: 主响应 Agent 类型。
        agent_types: 参与响应的 Agent 类型列表。
        primary_agent: 主 Agent 名称。
        supporting_agents: 辅助 Agent 名称列表。
        tools_used: 本次请求使用过的工具名。
        routing_reason: 路由决策说明。
        routing_confidence: 路由置信度，范围为 0 到 1。
        escalated: 安全策略是否将请求标记为升级建议。
        agent_recommended_handoff: Agent 是否建议转人工。
        latency_ms: 端到端耗时，单位为毫秒。
        knowledge_used: 是否使用知识库内容。
        entities: 从用户消息中抽取的实体。
        intent_confidence: 主意图置信度，范围为 0 到 1。
        intent_source_scores: 各识别来源的置信度。
        intent_diagnostics: 意图识别与路由诊断信息。
        emotion: 识别到的消费者情绪标签。
        emotion_intensity: 情绪强度，范围为 0 到 1。
        emotion_evidence: 支撑情绪判断的原文关键词。
        usage: 本次请求已由模型提供的 token usage；供应商未返回时为空对象。
        usage_by_stage: 按意图识别、Agent 和合成器拆分的 token usage；未返回时为空对象。
    """

    conv_id: str
    request_id: str = ""
    response: str
    intent: str
    secondary_intents: List[str] = Field(default_factory=list)
    intent_group: str = "other"
    agent_type: str
    agent_types: List[str] = Field(default_factory=list)
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    escalated: bool
    agent_recommended_handoff: bool = False
    latency_ms: float = Field(..., ge=0.0)
    knowledge_used: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)
    intent_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    emotion: str = "neutral"
    emotion_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    emotion_evidence: List[str] = Field(default_factory=list)
    handoff_summary: Optional[Dict[str, Any]] = None
    ticket_draft: Optional[Dict[str, Any]] = None
    suggested_replies: List[str] = Field(default_factory=list)
    follow_up_options: List[str] = Field(default_factory=list)
    pending_questions: List[Dict[str, Any]] = Field(default_factory=list)
    usage: Dict[str, int] = Field(default_factory=dict)
    usage_by_stage: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class ServiceMessageRequest(BaseModel):
    """客服端发送公开文字消息的契约；当前版本不接受附件。"""

    agent_id: str = "agent-demo"
    content: str

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class SlotAnswerRequest(BaseModel):
    """消费者追问按钮的结构化回答；不重新走意图识别。"""

    question_id: str = Field(min_length=1, max_length=100)
    slot_id: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=200)


class TicketUpdateRequest(BaseModel):
    fields: Dict[str, Any] = Field(default_factory=dict)
    expected_version: Optional[int] = None


class SearchRequest(BaseModel):
    """知识库只读检索的 API 输入契约。

    Attributes:
        query: 待检索的自然语言问题。
        top_k: 请求的最大召回条数，必须为正整数。
    """

    query: str
    top_k: int = Field(default=5, ge=1)
    domain: Optional[str] = None

    @field_validator("domain")
    @classmethod
    def domain_must_be_registered(cls, value: Optional[str]) -> Optional[str]:
        """只接受三个美妆业务域；管理检索可省略域过滤。"""

        if value is not None and value not in BUSINESS_KB_DOMAINS:
            raise ValueError("domain must be consult, order, or after_sales")
        return value

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        """拒绝空白检索词，避免把无意义请求传给向量库。"""

        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class KnowledgeAddRequest(BaseModel):
    """知识库文本入库的 API 输入契约。"""

    title: str
    content: str
    domain: Optional[str] = None

    @field_validator("title", "content")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """拒绝没有标题或正文的入库请求。"""

        if not value.strip():
            raise ValueError("title/content must not be blank")
        return value

    @field_validator("domain")
    @classmethod
    def domain_must_be_registered(cls, value: Optional[str]) -> Optional[str]:
        """新写入知识必须归入唯一的美妆业务域。"""

        if value is not None and value not in BUSINESS_KB_DOMAINS:
            raise ValueError("domain must be consult, order, or after_sales")
        return value


async def _maybe_await(value: Any) -> Any:
    """统一处理真实组件与同步测试替身的返回值。"""

    if hasattr(value, "__await__"):
        return await value
    return value


async def _load_memory_context(
    memory_manager: Any,
    user_id: str,
    conv_id: str,
    query: str,
) -> MemoryContext:
    """读取会话上下文，并在记忆组件不可用时返回空上下文。

    Args:
        memory_manager: 生命周期装配的 MemoryManager 或测试替身。
        user_id: 用户标识。
        conv_id: 会话标识。
        query: 当前查询，用于情景记忆召回。

    Returns:
        ``MemoryContext``；存储异常只影响上下文增强，不伪造历史消息。
    """

    get_context = getattr(memory_manager, "get_context", None)
    if callable(get_context):
        try:
            context = await _maybe_await(get_context(user_id, conv_id, query))
            if isinstance(context, MemoryContext):
                return context
            if isinstance(context, Mapping):
                return MemoryContext(
                    summary=str(context.get("summary", "")),
                    recent_messages=list(context.get("recent_messages", [])),
                    relevant_history=list(context.get("relevant_history", [])),
                    user_profile=dict(context.get("user_profile", {})),
                )
        except Exception as exc:
            logger.warning("读取会话上下文失败，使用空上下文：%s", exc)
    get_recent = getattr(memory_manager, "get_recent_messages", None)
    if callable(get_recent):
        try:
            recent = await _maybe_await(get_recent(user_id, conv_id, limit=20))
            return MemoryContext(recent_messages=list(recent or []))
        except Exception as exc:
            logger.warning("读取工作记忆失败，使用空上下文：%s", exc)
    return MemoryContext()


async def _recognize_request(
    recognizer: Any,
    message: str,
    history: List[Any],
) -> Any:
    """调用注入的意图识别器，兼容 Step7.1 适配器和离线替身。"""

    method = getattr(recognizer, "recognize_intent", None)
    if not callable(method):
        method = getattr(recognizer, "recognize", None)
    if not callable(method):
        return await recognize_intent(message, history=history)
    return await _maybe_await(method(message, history))


def _intent_value(intent: Any, default: str = "other") -> str:
    """将枚举或兼容字符串转换为 API 可序列化值。"""

    raw = getattr(intent, "value", intent)
    return str(raw or default)


def _should_use_knowledge(
    intent: Any,
    message: str = "",
    secondary_intents: Optional[List[Any]] = None,
    sub_requests: Optional[List[Mapping[str, Any]]] = None,
) -> bool:
    """按本轮全部子意图的显式知识模式决定是否预检索。"""
    intent_ids = [_intent_value(intent)]
    intent_ids.extend(_intent_value(value, "") for value in secondary_intents or [])
    for item in sub_requests or []:
        value = item.get("intent") or item.get("intent_id")
        if value:
            intent_ids.append(_intent_value(value, ""))
    return any(should_prefetch_knowledge(value, message) for value in dict.fromkeys(intent_ids))


def _active_contract_intent_ids(
    intent: Any,
    secondary_intents: Optional[List[Any]] = None,
    sub_requests: Optional[List[Mapping[str, Any]]] = None,
) -> List[str]:
    values = [_intent_value(intent, "")]
    values.extend(_intent_value(item, "") for item in secondary_intents or [])
    for item in sub_requests or []:
        if isinstance(item, Mapping):
            values.append(_intent_value(item.get("intent") or item.get("intent_id"), ""))
    return list(dict.fromkeys(value for value in values if value in BEAUTY_INTENT_TICKET_BY_ID))


def _has_entity_value(value: Any) -> bool:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    unanswered = {"", "unknown", "null", "none", "未知", "不清楚", "不确定", "不知道", "未提供", "待确认"}
    return any(item is not None and str(item).strip().casefold() not in unanswered for item in values)


def _consumer_confirms_resolution(message: str) -> bool:
    """仅识别明确的肯定式结案表达；否定、追问和转折一律不关闭诉求。"""
    text = "".join(str(message or "").lower().split())
    if not text:
        return False
    unresolved = (
        "没解决", "没有解决", "未解决", "还没解决", "没处理好", "没有处理好",
        "还没好", "没好", "仍未", "依旧", "还是不行", "还是有问题",
        "解决了吗", "解决了没", "好了吗", "有没有解决", "是否解决",
    )
    if any(marker in text for marker in unresolved):
        return False
    if any(marker in text for marker in (
        "但是还", "不过还", "但是有", "不过有", "但还有", "不过还有",
        "但仍", "但又", "但还是", "又出现", "再次出现",
    )):
        return False
    resolved = (
        "问题解决了", "问题已经解决", "问题已解决", "事情解决了", "事情已经解决",
        "已经解决了", "已解决了", "已经处理好了", "已处理好了", "已经好了",
        "已经恢复正常", "已恢复正常", "不用再处理了", "可以关闭了",
    )
    return any(marker in text for marker in resolved)


def _pending_questions(
    intent: Any,
    entities: Mapping[str, Any],
    secondary_intents: Optional[List[Any]] = None,
    sub_requests: Optional[List[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """按所有当前子意图的阻塞字段生成选项；已知字段与非必需字段不会发问。"""
    intent_ids = _active_contract_intent_ids(intent, secondary_intents, sub_requests)
    blocking_fields = policy_for_intents(intent_ids).get("blocking_fields", [])
    questions: List[Dict[str, Any]] = []
    for slot_id in blocking_fields:
        if _has_entity_value(entities.get(slot_id)):
            continue
        slot = get_slot(slot_id)
        # 自由文本槽位由模型按字段状态用一句话询问；只有枚举字段需要渲染选项。
        if not slot or slot.get("type") != "enum":
            continue
        questions.append({
            "question_id": f"{slot_id}_question",
            "slot_id": slot_id,
            "text": str(slot.get("question_text") or f"{slot['display_name']}？"),
            "options": list(slot.get("options", [])),
        })
    return questions[:3]


def _follow_up_options(
    intent: Any,
    entities: Mapping[str, Any],
    secondary_intents: Optional[List[Any]] = None,
    sub_requests: Optional[List[Mapping[str, Any]]] = None,
) -> List[str]:
    """保留旧响应字段，但选项统一从结构化必需槽位生成。"""
    return [
        str(option.get("label") or option.get("value") or "")
        for question in _pending_questions(intent, entities, secondary_intents, sub_requests)
        for option in question.get("options", [])
        if option.get("label") or option.get("value")
    ]


def _pending_text_fields_for_response(
    response: str,
    field_state: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """只持久化最终答复实际询问的首个文本阻塞字段及其业务上下文。"""
    active_intent_ids = [
        str(value) for value in field_state.get("intent_ids", [])
        if str(value) in BEAUTY_INTENT_TICKET_BY_ID
    ]
    asked_fields = asked_fields_in_text(
        response,
        field_state.get("askable_text_fields", []),
    )
    pending: List[Dict[str, Any]] = []
    for field_id in asked_fields[:1]:
        field_intents = [
            contract["intent_id"]
            for contract in BEAUTY_INTENT_TICKET_BY_ID.values()
            if field_id in contract.get("blocking_fields", [])
            and contract["intent_id"] in active_intent_ids
        ]
        pending.append({
            "field_id": field_id,
            "intent_ids": field_intents,
            "context_intent_ids": active_intent_ids,
        })
    return pending


def _explicit_handoff_requested(message: str) -> bool:
    return is_explicit_handoff_requested(
        message, load_handoff_urgency_config()
    )


async def _bind_consumer_request(
    store: SQLiteCaseStore,
    chat_request: ChatRequest,
    authorization: Optional[str],
) -> ChatRequest:
    """Authenticate a consumer and bind the request to the token's user/session.

    Args:
        store: SQLite identity and session store.
        chat_request: Untrusted API request body.
        authorization: Bearer token from the request header.

    Returns:
        A copied request whose user ID is taken from the authenticated identity.

    Raises:
        HTTPException: If the identity, supplied user ID, or session owner is invalid.
    """
    identity = await _require_identity(store, authorization, role="consumer")
    requested_user_id = str(chat_request.user_id or "").strip()
    if requested_user_id not in {"", "anonymous", identity["user_id"]}:
        raise HTTPException(status_code=403, detail="只能操作自己的消费者会话")
    if chat_request.conv_id:
        session = await asyncio.to_thread(store.get_session, chat_request.conv_id)
        if session is not None and session["user_id"] != identity["user_id"]:
            raise HTTPException(status_code=403, detail="无权操作该消费者会话")
    return chat_request.model_copy(update={"user_id": identity["user_id"]})


async def _require_management_identity(authorization: Optional[str]) -> Dict[str, Any]:
    """Require an authenticated agent identity for knowledge and operations APIs."""
    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="SQLite 账号存储尚未就绪")
    return await _require_identity(store, authorization, role="agent")


async def _start_atomic_handoff(
    chat_request: ChatRequest,
    store: Optional[SQLiteCaseStore],
    authorization: Optional[str],
    *,
    conv_id: Optional[str] = None,
    handoff_reason: str = "consumer_requested_human",
    intent: Optional[str] = None,
    intents: Optional[Iterable[str]] = None,
    temporary_summary_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist the complete immediate handoff contract before exposing its ACK."""
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="ShopCare 尚未就绪")
    if store is None:
        raise HTTPException(status_code=503, detail="转人工存储尚未就绪，未发送确认")
    identity = await _require_identity(store, authorization, role="consumer")
    if identity["user_id"] != chat_request.user_id:
        raise HTTPException(status_code=403, detail="只能操作自己的消费者会话")
    existing_for_owner = await asyncio.to_thread(store.get_session, conv_id or chat_request.conv_id or "")
    if existing_for_owner is not None and existing_for_owner["user_id"] != identity["user_id"]:
        raise HTTPException(status_code=403, detail="无权转接该消费者会话")

    # Why：必须先验证完整配置，再做任何会话写入；缺文件时不能留下空会话，
    # 也不能把运行时配置错误延迟到消费者点击转人工时才暴露。
    validate_handoff_configuration()
    copy_config = load_handoff_copy_config()
    urgency = score_temporary_urgency(
        chat_request.message,
        load_handoff_urgency_config(),
        intent=intent,
        intents=intents,
    )
    conversation_id = conv_id or chat_request.conv_id or uuid.uuid4().hex
    existing_session = await asyncio.to_thread(store.get_session, conversation_id)
    if existing_session is None:
        conversation = {"messages": []}
        active_requests = []
    else:
        conversation = await asyncio.to_thread(
            store.get_conversation, conversation_id, viewer_role="agent"
        )
        active_requests = [
            item for item in await asyncio.to_thread(store.list_case_sub_requests, conversation_id)
            if item.get("status") != "resolved"
        ]
    if temporary_summary_text:
        temporary_text = temporary_summary_text
    elif active_requests:
        temporary_text = copy_config["temporary_summary_open_requests"]
    elif conversation.get("messages"):
        temporary_text = copy_config["temporary_summary_prior"]
    else:
        temporary_text = copy_config["temporary_summary_direct"]
    summary_sources = sorted({
        int(item["source_message_seq"])
        for item in active_requests
        if item.get("source_message_seq") is not None
    })
    temporary_summary = {
        "text": temporary_text,
        "requests": [
            {
                "intent_id": str(item.get("intent_id") or ""),
                "status": str(item.get("status") or "open"),
                "source_message_seq": item.get("source_message_seq"),
            }
            for item in active_requests
        ],
        "sources": summary_sources,
        "temporary": True,
    }
    temporary_priority_reasons = [
        f"{item['rule_id']}: {item['text']}" + (
            f"（证据：{item['evidence']}）" if item.get("evidence") else ""
        )
        for item in urgency["reasons"]
    ]
    idempotency_key = f"handoff:{conversation_id}:{uuid.uuid4().hex}"
    return await asyncio.to_thread(
        store.start_handoff,
        conversation_id,
        user_id=chat_request.user_id,
        message=chat_request.message,
        idempotency_key=idempotency_key,
        reason=handoff_reason,
        temporary_summary=temporary_summary,
        temporary_priority=str(urgency["priority"]),
        temporary_priority_score=int(urgency["score"]),
        temporary_priority_reasons=temporary_priority_reasons,
        temporary_priority_rule_version=str(urgency["rule_version"]),
        online_ack=str(copy_config["online_ack"]),
        unknown_presence_ack=str(copy_config["unknown_presence_ack"]),
        callback_due_at=callback_due_at(copy_config),
        callback_reason=str(copy_config["callback_reason"]),
    )


def _atomic_handoff_response(
    conv_id: str, result: Mapping[str, Any], *, latency_ms: float
) -> ChatResponse:
    return ChatResponse(
        conv_id=conv_id,
        request_id=str(result.get("handoff_id") or "")[:8],
        response=str(result["ack_text"]),
        intent=IntentCategory.HUMAN_HANDOFF.value,
        intent_group="handoff",
        agent_type=AgentType.ESCALATION.value,
        agent_types=[AgentType.ESCALATION.value],
        primary_agent=AgentType.ESCALATION.value,
        routing_reason="规则命中明确人工请求",
        routing_confidence=1.0,
        escalated=True,
        agent_recommended_handoff=True,
        latency_ms=max(0.0, latency_ms),
        handoff_summary={
            "summary": str((result.get("temporary_summary") or {}).get("text") or ""),
            "temporary": True,
            "urgency": str(result.get("temporary_priority") or "P3"),
            "urgency_score": int(result.get("temporary_priority_score") or 0),
            "urgency_reasons": list(result.get("temporary_priority_reasons") or []),
            "urgency_rule_version": str(result.get("temporary_priority_rule_version") or ""),
        },
        intent_diagnostics={
            "handoff_mode": "atomic",
            "handoff_id": str(result["handoff_id"]),
            "ack_message_id": str(result["ack_message_id"]),
            "handoff_created": bool(result.get("created")),
            "background_processing": True,
        },
    )


def _agent_stats(orchestrator: Any) -> Dict[str, Any]:
    """读取编排器公开统计；旧版编排器无 get_stats 时从 Agent 池汇总。"""

    get_stats = getattr(orchestrator, "get_stats", None)
    if callable(get_stats):
        try:
            stats = get_stats()
            if isinstance(stats, Mapping):
                return dict(stats)
        except Exception as exc:
            logger.warning("读取 Agent 统计失败：%s", exc)
    pool = getattr(orchestrator, "agent_pool", {})
    result: Dict[str, Any] = {}
    if isinstance(pool, Mapping):
        for agent_type, agent in pool.items():
            stats = getattr(agent, "stats", None)
            if stats is None:
                continue
            key = _intent_value(agent_type, str(agent_type))
            result[key] = {
                "total": int(getattr(stats, "total", 0)),
                "success": int(getattr(stats, "success", 0)),
                "success_rate": float(getattr(stats, "success_rate", 0.0)),
                "avg_ms": float(getattr(stats, "avg_ms", 0.0)),
            }
    return result


async def _knowledge_metadata(knowledge_base: Any) -> Dict[str, Any]:
    """读取知识库 collection/model/doc_count 元数据。"""

    collection = getattr(knowledge_base, "_collection", None)
    count = getattr(collection, "count", None)
    doc_count = 0
    if callable(count):
        try:
            doc_count = int(await asyncio.to_thread(count))
        except Exception as exc:
            logger.warning("读取知识库文档数失败：%s", exc)
    return {
        "collection_name": str(getattr(knowledge_base, "collection_name", "")),
        "embedding_model": str(getattr(knowledge_base, "embedding_model", "")),
        "doc_count": doc_count,
    }


def _consume_profile_task(task: asyncio.Task[Any]) -> None:
    """消费后台画像任务异常，避免任务异常冒泡到事件循环。"""

    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as exc:
        # Why：画像更新是非阻塞增强能力，异常不能冒泡导致用户已完成请求变成白屏。
        logger.warning("后台用户画像更新失败，已隔离：%s", exc)


def _schedule_profile_update(memory_manager: Any, user_id: str, conv_id: str) -> None:
    """创建并登记画像异步更新任务，不让后台异常改变 HTTP 响应。"""

    update_profile = getattr(memory_manager, "update_profile", None)
    if not callable(update_profile):
        return
    try:
        coroutine = update_profile(user_id, conv_id)
        task = asyncio.create_task(coroutine)
        task.add_done_callback(_consume_profile_task)
    except Exception as exc:
        logger.warning("创建后台用户画像任务失败，已隔离：%s", exc)
        close = getattr(coroutine, "close", None) if "coroutine" in locals() else None
        if callable(close):
            close()


def _consume_handoff_task(task: asyncio.Task[Any]) -> None:
    """记录交接后台任务异常，消费者侧固定确认不受其影响。"""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as exc:
        logger.error("转人工后台事实补齐失败，保留待恢复状态 error_type=%s", type(exc).__name__)


def _schedule_handoff_enrichment(
    store: Optional[SQLiteCaseStore], conv_id: str, result: Mapping[str, Any]
) -> None:
    """事务提交后启动幂等事实补齐，不运行整条 Agent 流程。"""
    handoff_id = str(result.get("handoff_id") or "")
    if store is None or not handoff_id:
        return
    tasks = getattr(app.state, "handoff_tasks", None)
    if tasks is None:
        tasks = {}
        app.state.handoff_tasks = tasks
    active = tasks.get(handoff_id)
    if active is not None and not active.done():
        return

    async def run() -> None:
        await enrich_handoff(
            store,
            conv_id,
            handoff_id,
            emotion_client=getattr(app.state, "anthropic_client", None),
        )

    try:
        task = asyncio.create_task(run(), name=f"handoff-enrichment:{handoff_id}")
    except RuntimeError as exc:
        logger.warning("无法排程转人工后台任务 handoff_id=%s error_type=%s", handoff_id, type(exc).__name__)
        return
    tasks[handoff_id] = task

    def cleanup(done: asyncio.Task[Any]) -> None:
        _consume_handoff_task(done)
        if tasks.get(handoff_id) is done:
            tasks.pop(handoff_id, None)

    task.add_done_callback(cleanup)


@app.get("/health")
async def health() -> Dict[str, Any]:
    """返回核心组件就绪状态、Agent 统计和产品知识库元数据。

    Returns:
        健康状态 JSON；组件尚未完成 lifespan 装配时返回 HTTP 503。

    Raises:
        HTTPException: 应用未就绪时返回 503，避免伪造 healthy 状态。
    """

    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="ShopCare 尚未就绪")
    orchestrator = getattr(app.state, "agent_orchestrator", None)
    knowledge_base = getattr(app.state, "knowledge_base", None)
    if orchestrator is None or knowledge_base is None:
        raise HTTPException(status_code=503, detail="核心组件未完整装配")
    return {
        "status": "healthy",
        "agent_stats": _agent_stats(orchestrator),
        "knowledge_base": await _knowledge_metadata(knowledge_base),
    }


def _require_ready() -> None:
    """校验生命周期是否完成装配，避免支持性路由伪造可用状态。

    Raises:
        HTTPException: 应用未进入 ready 状态时返回 503。
    """

    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="ShopCare 尚未就绪")


def _require_knowledge_base() -> Any:
    """取得已装配的知识库组件，否则返回明确的 503。"""

    _require_ready()
    knowledge_base = getattr(app.state, "knowledge_base", None)
    if knowledge_base is None:
        raise HTTPException(status_code=503, detail="知识库尚未装配")
    return knowledge_base


def _normalize_search_result(item: Any) -> Dict[str, Any]:
    """将 KnowledgeBase 检索结果收敛为前端稳定字段。"""

    row = item if isinstance(item, Mapping) else {}
    chunk_id = row.get("source_chunk_id") or row.get("chunk_id") or row.get("id") or ""
    return {
        "title": str(row.get("title", "")),
        "chunk_id": str(chunk_id),
        "content": str(row.get("content", row.get("document", ""))),
        "score": float(row.get("score", 0.0)),
    }


@app.post("/search")
async def search_knowledge(
    request: SearchRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """执行产品知识库只读检索。

    Args:
        request: 包含自然语言 query 与 top_k 的检索请求。

    Returns:
        ``results`` 列表，每项包含 title、chunk_id、content 与 score。

    Raises:
        HTTPException: 服务未就绪或知识库检索异常时返回 503。

    仅允许已认证客服账号调用。
    """

    await _require_management_identity(authorization)
    knowledge_base = _require_knowledge_base()
    try:
        raw = await asyncio.to_thread(
            knowledge_base.search,
            request.query,
            request.top_k,
            domain=request.domain,
        )
        raw = await _maybe_await(raw)
    except Exception as exc:
        logger.warning("知识库检索路由失败：%s", exc)
        raise HTTPException(status_code=503, detail="知识库检索暂不可用") from exc
    return {
        "query": request.query,
        "top_k": request.top_k,
        "results": [_normalize_search_result(item) for item in (raw or [])],
    }


@app.post("/knowledge/add")
async def add_knowledge(
    request: KnowledgeAddRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """接收纯文本并调用 KnowledgeBase 切片入库。

    Args:
        request: 文档标题与正文。

    Returns:
        ``success`` 与实际新增 chunk 数量；不执行外部订单或支付写操作。

    Raises:
        HTTPException: 服务未就绪或知识库写入失败时返回 503。

    仅允许已认证客服账号调用。
    """

    await _require_management_identity(authorization)
    if request.domain is None:
        raise HTTPException(
            status_code=422,
            detail="domain is required: consult, order, or after_sales",
        )
    knowledge_base = _require_knowledge_base()
    try:
        added = await asyncio.to_thread(
            knowledge_base.add_documents,
            [{"title": request.title, "content": request.content, "domain": request.domain}],
        )
        added = await _maybe_await(added)
    except Exception as exc:
        logger.warning("知识库文本入库失败：%s", exc)
        raise HTTPException(status_code=503, detail="知识库写入暂不可用") from exc
    return {"success": True, "title": request.title, "added": int(added or 0)}


@app.post("/knowledge/upload")
async def upload_knowledge(
    file: UploadFile = File(...),
    domain: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """上传 txt/md/json 文档并交由 KnowledgeBase 切片入库。

    Args:
        file: multipart 上传文件；扩展名和大小均在 API 边界校验。

    Returns:
        文件名、入库 chunk 数和成功标志。

    Raises:
        HTTPException: 扩展名不允许、超过 10MB、JSON 无法解析或知识库不可用。

    仅允许已认证客服账号调用；本接口不连接外部业务系统。
    """

    await _require_management_identity(authorization)
    knowledge_base = _require_knowledge_base()
    if domain is None or domain not in BUSINESS_KB_DOMAINS:
        raise HTTPException(
            status_code=422,
            detail="domain is required: consult, order, or after_sales",
        )
    filename = str(file.filename or "")
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in ALLOWED_KNOWLEDGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="仅支持 .txt、.md、.json 文件")
    try:
        # Why：读取上限+1 字节即可判断 10MB 越界，避免无界内存占用。
        payload = await file.read(MAX_KNOWLEDGE_UPLOAD_BYTES + 1)
    except Exception as exc:
        logger.warning("知识文件读取失败：%s", exc)
        raise HTTPException(status_code=400, detail="上传文件读取失败") from exc
    if len(payload) > MAX_KNOWLEDGE_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="单文件大小不能超过 10MB")
    try:
        # Why：非 JSON 文档按约定忽略非法 UTF-8 字节，保证客服文本导入不中断。
        content = payload.decode("utf-8", errors="ignore")
        if suffix == ".json":
            json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="JSON 文件内容无效") from exc
    try:
        added = await asyncio.to_thread(
            knowledge_base.add_documents,
            [{"title": Path(filename).stem, "content": content, "domain": domain}],
        )
        added = await _maybe_await(added)
    except Exception as exc:
        logger.warning("知识文件入库失败：%s", exc)
        raise HTTPException(status_code=503, detail="知识库写入暂不可用") from exc
    return {"success": True, "filename": filename, "added": int(added or 0)}


@app.get("/knowledge/stats")
async def knowledge_stats(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """返回产品知识库当前片段数量与模型配置。"""

    await _require_management_identity(authorization)
    knowledge_base = _require_knowledge_base()
    try:
        count_value = getattr(knowledge_base, "doc_count", 0)
        if callable(count_value):
            count_value = await asyncio.to_thread(count_value)
            count_value = await _maybe_await(count_value)
        elif count_value == 0:
            collection = getattr(knowledge_base, "_collection", None)
            count_method = getattr(collection, "count", None)
            if callable(count_method):
                count_value = await asyncio.to_thread(count_method)
                count_value = await _maybe_await(count_value)
        return {
            "collection_name": str(getattr(knowledge_base, "collection_name", "")),
            "embedding_model": str(getattr(knowledge_base, "embedding_model", "")),
            "doc_count": int(count_value or 0),
        }
    except Exception as exc:
        logger.warning("读取知识库统计失败：%s", exc)
        raise HTTPException(status_code=503, detail="知识库统计暂不可用") from exc


def _get_trace_snapshot() -> List[Dict[str, Any]]:
    """读取编排器公开的内存 Trace 快照，最多保留 200 条。"""

    _require_ready()
    orchestrator = getattr(app.state, "agent_orchestrator", None)
    getter = getattr(orchestrator, "get_recent_traces", None)
    if not callable(getter):
        raise HTTPException(status_code=503, detail="Trace 组件尚未装配")
    try:
        traces = getter(limit=TRACE_MEMORY_LIMIT)
        return [dict(item) for item in (traces or []) if isinstance(item, Mapping)]
    except Exception as exc:
        logger.warning("读取内存 Trace 失败：%s", exc)
        raise HTTPException(status_code=503, detail="Trace 暂不可用") from exc


@app.get("/trace/tool/{request_id}")
async def get_tool_trace(
    request_id: str,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """按 request_id 返回工具调用 Trace。

    Trace 当前只驻留进程内存，服务重启后会丢失；该接口不提供审计级持久化。
    """

    await _require_management_identity(authorization)
    traces = _get_trace_snapshot()
    for trace in traces:
        if str(trace.get("request_id", "")) == request_id:
            return {
                "request_id": request_id,
                "trace": trace,
                "storage": "memory",
                "restart_loses_data": True,
            }
    raise HTTPException(status_code=404, detail="未找到对应 request_id 的 Trace")


@app.get("/trace/tools")
async def get_tool_traces(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """返回最近的内存工具 Trace，并标记服务重启丢失边界。"""

    await _require_management_identity(authorization)
    return {
        "traces": _get_trace_snapshot(),
        "storage": "memory",
        "max_items": TRACE_MEMORY_LIMIT,
        "restart_loses_data": True,
    }


@app.get("/trace/token/{request_id}")
async def get_token_trace(
    request_id: str,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """返回单条请求的分阶段 Token 用量；供应商未返回时明确标记 unavailable。"""

    await _require_management_identity(authorization)
    _require_ready()
    ledger = getattr(app.state, "token_usage", None)
    item = ledger.get(request_id) if ledger is not None else None
    if item is None:
        raise HTTPException(status_code=404, detail="未找到对应 request_id 的 Token Trace")
    return {"trace": item, "storage": "memory", "restart_loses_data": True}


@app.get("/trace/tokens")
async def get_token_traces(
    limit: int = TRACE_MEMORY_LIMIT,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """返回最近请求的 Token 汇总与阶段明细，不保存原始 Prompt 或响应正文。"""

    await _require_management_identity(authorization)
    _require_ready()
    ledger = getattr(app.state, "token_usage", None)
    if ledger is None:
        raise HTTPException(status_code=503, detail="Token 观测组件尚未装配")
    return ledger.summary(limit=max(1, min(int(limit), TRACE_MEMORY_LIMIT)))


@app.get("/skills")
async def list_skills(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """返回本地 SkillManager 当前已加载的技能列表。"""

    await _require_management_identity(authorization)
    _require_ready()
    manager = getattr(app.state, "skill_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="SkillManager 尚未装配")
    skills = list(getattr(manager, "skills", []) or [])
    return {"skills": skills, "count": len(skills)}


@app.post("/skills/reload")
async def reload_skills(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """重新扫描本地技能目录并返回热加载结果。

    仅允许已认证客服账号调用。
    """

    await _require_management_identity(authorization)
    _require_ready()
    manager = getattr(app.state, "skill_manager", None)
    reload_method = getattr(manager, "reload_skills", None)
    if not callable(reload_method):
        raise HTTPException(status_code=503, detail="SkillManager 不支持热加载")
    try:
        result = await _maybe_await(reload_method())
    except Exception as exc:
        logger.warning("技能热加载失败：%s", exc)
        raise HTTPException(status_code=503, detail="技能热加载失败") from exc
    if isinstance(result, Mapping):
        return dict(result)
    skills = list(getattr(manager, "skills", []) or [])
    return {"success": True, "skills": skills, "count": len(skills)}


async def _require_identity(store: SQLiteCaseStore, authorization: Optional[str], *, role: str) -> Dict[str, Any]:
    parts = str(authorization or "").strip().split(None, 1)
    token = parts[1].strip() if len(parts) == 2 and parts[0].casefold() == "bearer" else ""
    identity = await asyncio.to_thread(store.get_identity, token) if token else None
    if not identity or identity["role"] != role:
        raise HTTPException(status_code=401, detail="登录已失效或身份不匹配")
    return identity


def _is_assigned(store: SQLiteCaseStore, agent_id: str, consumer_id: str) -> bool:
    return store.is_assigned(agent_id, consumer_id)


@app.post("/auth/login")
async def login(payload: LoginRequest) -> Dict[str, Any]:
    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="账号存储尚未就绪")
    identity = await asyncio.to_thread(store.authenticate, payload.username, payload.password)
    if identity is None:
        raise HTTPException(status_code=401, detail="账号或密码错误")
    return identity


@app.post("/auth/register")
async def register_consumer(payload: RegisterConsumerRequest) -> Dict[str, Any]:
    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="账号存储尚未就绪")
    try:
        account = await asyncio.to_thread(store.register_consumer, payload.username, payload.password, payload.display_name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    identity = await asyncio.to_thread(store.authenticate, payload.username, payload.password)
    return identity or account


@app.get("/service/sessions")
async def list_service_sessions(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """客服端读取最近会话列表。当前为演示接口，返回 SQLite 中的会话索引。"""

    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="SQLite 会话存储尚未就绪")
    identity = await _require_identity(store, authorization, role="agent")
    return await asyncio.to_thread(store.list_sessions, identity["user_id"])


@app.get("/service/dashboard")
async def service_dashboard(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """客服风险看板：统计与用户列表均来自 SQLite 当前快照。"""
    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="SQLite 会话存储尚未就绪")
    identity = await _require_identity(store, authorization, role="agent")
    return await asyncio.to_thread(store.get_risk_dashboard, identity["user_id"])


@app.get("/service/sessions/{conv_id}")
async def get_service_session(
    conv_id: str,
    after_seq: Optional[int] = Query(default=None, ge=0),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """客服端读取完整会话，包括内部 CaseState。"""

    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="SQLite 会话存储尚未就绪")
    identity = await _require_identity(store, authorization, role="agent")
    session = await asyncio.to_thread(store.get_session, conv_id)
    if session is None or not await asyncio.to_thread(_is_assigned, store, identity["user_id"], session["user_id"]):
        raise HTTPException(status_code=403, detail="无权查看该消费者会话")
    return await asyncio.to_thread(
        store.get_conversation, conv_id, viewer_role="agent", after_message_seq=after_seq
    )


@app.post("/service/sessions/{conv_id}/refresh")
async def refresh_service_session(conv_id: str, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """客服显式刷新时，增量处理交接后的消费者消息并更新工单草稿和交接内容。"""
    if not handoff_enabled():
        raise HTTPException(status_code=404, detail="转人工交接功能未启用")
    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="SQLite 会话存储尚未就绪")
    identity = await _require_identity(store, authorization, role="agent")
    session = await asyncio.to_thread(store.get_session, conv_id)
    if session is None or not await asyncio.to_thread(_is_assigned, store, identity["user_id"], session["user_id"]):
        raise HTTPException(status_code=403, detail="无权刷新该消费者会话")
    snapshot = await asyncio.to_thread(store.get_conversation, conv_id, viewer_role="agent")
    handoff = snapshot.get("handoff")
    if not isinstance(handoff, Mapping):
        return {"success": True, "refreshed": False, "processed_messages": 0}

    handoff_id = str(handoff.get("handoff_id") or "")
    active_task = getattr(app.state, "handoff_tasks", {}).get(handoff_id)
    if active_task is not None and not active_task.done():
        await asyncio.shield(active_task)
    else:
        await enrich_handoff(
            store,
            conv_id,
            handoff_id,
            emotion_client=getattr(app.state, "anthropic_client", None),
        )
    result = await refresh_handoff_after_messages(
        store,
        conv_id,
        handoff_id,
        emotion_client=getattr(app.state, "anthropic_client", None),
    )
    return {"success": True, "refreshed": True, **result}


@app.get("/chat/{conv_id}/messages")
async def get_consumer_messages(
    conv_id: str,
    after_seq: Optional[int] = Query(default=None, ge=0),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """消费者按消息序号增量读取本人会话中的公开消息。"""
    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="SQLite 会话存储尚未就绪")
    identity = await _require_identity(store, authorization, role="consumer")
    session = await asyncio.to_thread(store.get_session, conv_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if session["user_id"] != identity["user_id"]:
        raise HTTPException(status_code=403, detail="无权查看该消费者会话")
    conversation = await asyncio.to_thread(
        store.get_conversation,
        conv_id,
        viewer_role="consumer",
        after_message_seq=after_seq,
    )
    return {
        "conv_id": conv_id,
        "messages": [
            {
                "message_id": item["message_id"],
                "message_seq": item["message_seq"],
                "sender_type": item["sender_type"],
                "content": item["content"],
                "created_at": item["created_at"],
            }
            for item in conversation["messages"]
        ],
        "last_message_seq": conversation.get("last_message_seq", 0),
    }


@app.get("/service/sessions/{conv_id}/handoff/events")
async def stream_service_handoff_events(
    conv_id: str,
    authorization: Optional[str] = Header(default=None),
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
    idle_timeout_seconds: float = Query(default=25.0, gt=0.0, le=60.0),
) -> StreamingResponse:
    """Stream assigned staff's durable handoff events, replaying after Last-Event-ID."""
    if not handoff_enabled():
        raise HTTPException(status_code=404, detail="转人工交接功能未启用")
    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="SQLite 会话存储尚未就绪")
    identity = await _require_identity(store, authorization, role="agent")
    session = await asyncio.to_thread(store.get_session, conv_id)
    if session is None or not await asyncio.to_thread(_is_assigned, store, identity["user_id"], session["user_id"]):
        raise HTTPException(status_code=403, detail="无权订阅该消费者会话")
    try:
        cursor = max(0, int(last_event_id or "0"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Last-Event-ID 必须是非负整数") from exc

    async def events():
        nonlocal cursor
        last_activity = time.monotonic()
        last_heartbeat = last_activity
        while True:
            rows = await asyncio.to_thread(store.list_handoff_events, conv_id, after_event_id=cursor)
            for item in rows:
                cursor = int(item["event_id"])
                last_activity = time.monotonic()
                yield (
                    f"id: {cursor}\n"
                    f"event: {item['event_type']}\n"
                    f"data: {json.dumps(item['data'], ensure_ascii=False, separators=(',', ':'))}\n\n"
                )
            now = time.monotonic()
            if now - last_activity >= idle_timeout_seconds:
                return
            if now - last_heartbeat >= 10.0:
                last_heartbeat = now
                yield ": keepalive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/service/sessions/{conv_id}/messages")
async def append_service_message(conv_id: str, payload: ServiceMessageRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """客服端写入一条公开文字回复，消费者可在同一会话中读取。"""

    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="SQLite 会话存储尚未就绪")
    identity = await _require_identity(store, authorization, role="agent")
    session = await asyncio.to_thread(store.get_session, conv_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if not await asyncio.to_thread(_is_assigned, store, identity["user_id"], session["user_id"]):
        raise HTTPException(status_code=403, detail="无权回复该消费者会话")
    message = await asyncio.to_thread(
        store.append_message, conv_id, "customer_service", identity["user_id"], payload.content
    )
    return {"success": True, "message": message}


def _ticket_departments(ticket: Mapping[str, Any]) -> list[dict[str, str]]:
    sub_tasks = ticket.get("sub_tasks")
    if isinstance(sub_tasks, list) and sub_tasks:
        return [
            {
                "department": str(item.get("department") or "售后"),
                "task": str(item.get("task") or "核对并处理相关诉求"),
                "status": str(item.get("status") or "待处理"),
            }
            for item in sub_tasks
            if isinstance(item, Mapping)
        ]
    intent = str(ticket.get("intent") or "")
    mapping = {
        "adverse_reaction": ("质量", "核对不良反应风险与商品批次"),
        "return_refund": ("财务", "核对退货资格与退款处理"),
        "refund_not_received": ("财务", "核对退款状态与到账进度"),
        "missing_item": ("仓储", "核对订单明细与补发"),
        "wrong_item": ("仓储", "核对错发商品与换货"),
        "damaged_on_arrival": ("质量", "核对破损情况与责任"),
        "logistics_stalled": ("物流", "核对物流节点并催查"),
        "signed_not_received": ("物流", "核对签收与代收信息"),
    }
    department, task = mapping.get(intent, ("售后", "核对诉求并反馈处理结果"))
    return [{"department": department, "task": task, "status": "待处理"}]


async def _get_agent_ticket(store: SQLiteCaseStore, conv_id: str, identity: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    session = await asyncio.to_thread(store.get_session, conv_id)
    if session is None or not await asyncio.to_thread(_is_assigned, store, identity["user_id"], session["user_id"]):
        raise HTTPException(status_code=403, detail="无权操作该会话")
    conversation = await asyncio.to_thread(store.get_conversation, conv_id, viewer_role="agent")
    state = dict(conversation.get("case_state") or {})
    ticket = dict(state.get("ticket_draft") or {})
    if not ticket:
        raise HTTPException(status_code=404, detail="当前会话没有工单")
    return state, ticket


@app.patch("/service/sessions/{conv_id}/ticket")
async def update_service_ticket(conv_id: str, payload: TicketUpdateRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="SQLite 会话存储尚未就绪")
    identity = await _require_identity(store, authorization, role="agent")
    state, ticket = await _get_agent_ticket(store, conv_id, identity)
    allowed = set(ticket.get("fields") or {})
    fields = dict(ticket.get("fields") or {})
    case_facts = CaseFacts.from_dict(state.get("case_facts"))
    changed_fields: Dict[str, Any] = {}
    for key, value in payload.fields.items():
        if key in allowed:
            fields[key] = value
            changed_fields[key] = value
            if value is None or (isinstance(value, str) and not value.strip()):
                case_facts.clear(key, source_type="agent_edit", source_ref=identity["user_id"])
            else:
                case_facts.record(
                    key, value, source_type="agent_edit", source_ref=identity["user_id"],
                    confidence=1.0, confirmation_status="confirmed",
                )
    ticket["fields"] = fields
    contract = BEAUTY_INTENT_TICKET_BY_ID.get(str(ticket.get("intent") or ""), {})
    blocking_fields = list(ticket.get("blocking_fields") or contract.get("blocking_fields", []))
    ticket["missing_blocking_fields"] = [field for field in blocking_fields if not fields.get(field)]
    try:
        await asyncio.to_thread(
            store.update_case_state,
            conv_id,
            {"ticket_draft": ticket, "case_facts": case_facts.to_dict()},
            payload.expected_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="工单已被其他客服更新，请刷新后重试") from exc
    if handoff_enabled() and changed_fields:
        await asyncio.to_thread(
            store.append_case_action,
            conv_id,
            action_type="ticket",
            description="客服已修改工单字段",
            source_ref=str(ticket.get("ticket_id") or conv_id),
            idempotency_key=f"ticket:update:{ticket.get('ticket_id', conv_id)}:{state.get('version', 0)}",
            status="success",
        )
    await asyncio.to_thread(store.append_audit, conv_id, identity["user_id"], "ticket.update", {"fields": list(payload.fields), "version": ticket.get("version")})
    for key, value in changed_fields.items():
        await asyncio.to_thread(
            store.record_slot_value,
            conv_id,
            key,
            "" if value is None else str(value),
            "" if value is None else str(value),
            source_type="agent_edit",
            source_ref=identity["user_id"],
            confidence=1.0,
            confirmation_status="cleared" if value is None or (isinstance(value, str) and not value.strip()) else "confirmed",
        )
    return {"success": True, "ticket": ticket}


@app.post("/service/sessions/{conv_id}/ticket/confirm")
async def confirm_service_ticket(conv_id: str, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """客服提交本地工单供人工核对；消费者通知由客服采纳答复后手动发送。

    Args:
        conv_id: 当前工单所属会话 ID。
        authorization: 已登录客服的 Bearer token。

    Returns:
        已提交的本地工单状态；本接口不会写入消费者可见消息，也不代表申请获批。
    """
    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="SQLite 会话存储尚未就绪")
    identity = await _require_identity(store, authorization, role="agent")
    state, ticket = await _get_agent_ticket(store, conv_id, identity)
    already_submitted = str(ticket.get("status") or "draft") == "submitted"
    ticket_id = str((ticket.get("system_fields") or {}).get("ticket_id") or ticket.get("ticket_id") or conv_id)
    if not already_submitted:
        # Recompute against the current registry instead of trusting the snapshot
        # persisted when the ticket was first created. This keeps old tickets
        # compatible after required-field rules are relaxed or changed.
        fields = ticket.get("fields") or {}
        contract = BEAUTY_INTENT_TICKET_BY_ID.get(str(ticket.get("intent") or ""), {})
        blocking_fields = list(ticket.get("blocking_fields") or contract.get("blocking_fields", []))
        missing = [field for field in blocking_fields if not fields.get(field)]
        ticket["missing_blocking_fields"] = missing
        if missing:
            raise HTTPException(status_code=409, detail={"message": "请先补充必填字段", "missing_fields": missing})
        ticket["status"] = "submitted"
        for task in ticket.get("sub_tasks") or []:
            if isinstance(task, dict):
                task["status"] = "已提交"
        ticket["department_tasks"] = _ticket_departments(ticket)
        submitted_departments = list(dict.fromkeys(
            item["department"] for item in ticket["department_tasks"]
        ))
        ticket["submitted_departments"] = submitted_departments
        ticket["submitted_department"] = submitted_departments[0] if submitted_departments else "售后"
        version = int(state.get("version", 0) or 0)
        try:
            await asyncio.to_thread(store.update_case_state, conv_id, {"ticket_draft": ticket, "ticket_status": "submitted", "department_tasks": ticket["department_tasks"]}, version)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="工单已被其他客服更新，请刷新后重试") from exc
        if handoff_enabled():
            await asyncio.to_thread(
                store.append_case_action,
                conv_id,
                action_type="ticket",
                description="客服已确认提交工单",
                source_ref=ticket_id,
                idempotency_key=f"ticket:confirm:{ticket_id}:{version}",
                status="success",
            )
        await asyncio.to_thread(store.append_audit, conv_id, identity["user_id"], "ticket.confirm", {"department_tasks": ticket["department_tasks"]})

    return {"success": True, "status": "submitted", "ticket": ticket}


@app.post("/chat/{conv_id}/slots")
async def answer_slot(conv_id: str, payload: SlotAnswerRequest, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """写入追问槽位；按钮回答只更新 case_state，不调用意图识别或模型。"""
    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="SQLite 会话存储尚未就绪")
    identity = await _require_identity(store, authorization, role="consumer")
    session = await asyncio.to_thread(store.get_session, conv_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if session["user_id"] != identity["user_id"]:
        raise HTTPException(status_code=403, detail="无权修改该会话")
    try:
        value, label = normalize_value(payload.slot_id, payload.value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    current = await asyncio.to_thread(store.get_conversation, conv_id, viewer_role="agent")
    state = dict(current.get("case_state") or {})
    slots = dict(state.get("slots") or {})
    slots[payload.slot_id] = value
    pending = [item for item in list(state.get("pending_slots") or []) if item != payload.slot_id]
    case_facts = CaseFacts.from_dict(state.get("case_facts"))
    case_facts.record(
        payload.slot_id,
        value,
        source_type="button",
        source_ref=payload.question_id,
        confidence=1.0,
        confirmation_status="confirmed",
    )
    updated = await asyncio.to_thread(
        store.update_case_state,
        conv_id,
        {"slots": slots, "pending_slots": pending, "last_slot_answer": {"question_id": payload.question_id, "slot_id": payload.slot_id, "value": value, "label": label}, "case_facts": case_facts.to_dict()},
    )
    await asyncio.to_thread(store.record_slot_value, conv_id, payload.slot_id, value, label, source_type="button", source_ref=payload.question_id)
    if handoff_enabled():
        action_hash = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"slot-answer:{conv_id}:{payload.question_id}:{payload.slot_id}:{value}",
        ).hex
        await asyncio.to_thread(
            store.append_case_action,
            conv_id,
            action_type="followup",
            description="顾客已补充追问字段",
            source_ref=payload.question_id,
            idempotency_key=f"slot-answer:{action_hash}",
            status="success",
        )
    return {"success": True, "conv_id": conv_id, "slot_id": payload.slot_id, "value": value, "label": label, "case_state": updated}


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, authorization: Optional[str] = Header(default=None)) -> ChatResponse:
    _require_ready()
    active_store = getattr(app.state, "sqlite_store", None)
    if active_store is None:
        raise HTTPException(status_code=503, detail="SQLite 会话存储尚未就绪")
    chat_request = await _bind_consumer_request(active_store, chat_request, authorization)
    if handoff_enabled() and _explicit_handoff_requested(chat_request.message):
        started = time.perf_counter()
        store = active_store
        conv_id = chat_request.conv_id or uuid.uuid4().hex
        result = await _start_atomic_handoff(
            chat_request, store, authorization, conv_id=conv_id
        )
        _schedule_handoff_enrichment(store, conv_id, result)
        return _atomic_handoff_response(
            conv_id, result, latency_ms=(time.perf_counter() - started) * 1000
        )
    if (
        handoff_enabled()
        and chat_request.conv_id
        and await asyncio.to_thread(active_store.is_waiting_human, chat_request.conv_id)
    ):
        raise HTTPException(status_code=409, detail="会话已由人工接管，AI 回复已取消")
    result = await _chat_core(chat_request, authorization)
    store = active_store
    atomic_rule_handoff = str(
        (result.intent_diagnostics or {}).get("handoff_mode") or ""
    ) == "atomic_rule"
    if (
        handoff_enabled()
        and store is not None
        and await asyncio.to_thread(store.is_waiting_human, result.conv_id)
        and not atomic_rule_handoff
    ):
        raise HTTPException(status_code=409, detail="会话已转交人工，AI 回复已取消")
    await _record_reply_action(
        store,
        result.conv_id,
        result.request_id,
        "success",
    )
    return result


async def _chat_core(
    chat_request: ChatRequest,
    authorization: Optional[str] = None,
    progress_callback: Optional[Any] = None,
    persisted_user_message_id: Optional[str] = None,
) -> ChatResponse:
    """执行一次上下文读取、意图识别、编排、记忆写回和画像后台更新。

    Args:
        chat_request: 已通过 Pydantic 边界校验的用户请求。

    Returns:
        完整的 ``ChatResponse``；画像后台任务失败不改变已生成响应。

    Raises:
        HTTPException: lifespan 未装配核心组件时返回 503。
    """

    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="ShopCare 尚未就绪")
    sqlite_store = getattr(app.state, "sqlite_store", None)
    if authorization and sqlite_store is not None:
        identity = await _require_identity(sqlite_store, authorization, role="consumer")
        if identity["user_id"] != chat_request.user_id:
            raise HTTPException(status_code=403, detail="只能操作自己的消费者会话")
    started = time.perf_counter()
    conv_id = chat_request.conv_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())[:8]
    existing_case_state: Dict[str, Any] = {}
    if sqlite_store is not None:
        await asyncio.to_thread(sqlite_store.ensure_session, chat_request.user_id, conv_id)
        # 在耗时的上下文/Agent 处理之前登记当前请求，后台情绪分析只允许写回最新一轮。
        await asyncio.to_thread(
            sqlite_store.update_case_state,
            conv_id,
            {"emotion_analysis_request_id": request_id},
        )
        existing_conversation = await asyncio.to_thread(
            sqlite_store.get_conversation, conv_id, viewer_role="agent"
        )
        existing_case_state = dict(existing_conversation.get("case_state") or {})
    memory_manager = app.state.memory_manager
    context = await _load_memory_context(
        memory_manager,
        chat_request.user_id,
        conv_id,
        chat_request.message,
    )
    if progress_callback:
        await progress_callback("context", "读取会话上下文")
    # 只把最近 5 条交给意图识别，避免历史上下文挤占分类输入预算。
    history = list(context.recent_messages[-5:])
    intent_result = await _recognize_request(
        app.state.intent_recognizer,
        chat_request.message,
        history,
    )
    if progress_callback:
        await progress_callback("intent", "意图识别")
    intent = getattr(intent_result, "intent", IntentCategory.OTHER)
    diagnostics = dict(getattr(intent_result, "intent_diagnostics", {}) or {})
    diagnostics.setdefault("emotion", getattr(intent_result, "emotion", "neutral"))
    diagnostics.setdefault("emotion_intensity", getattr(intent_result, "emotion_intensity", 0.0))
    diagnostics.setdefault("emotion_evidence", list(getattr(intent_result, "emotion_evidence", []) or []))
    sub_requests = list(getattr(intent_result, "sub_requests", []) or diagnostics.get("sub_requests", []) or [])
    secondary_intents = list(getattr(intent_result, "secondary_intents", []) or [])
    known_secondary = {_intent_value(value, "") for value in secondary_intents}
    primary_intent_id = _intent_value(intent, "")
    for item in sub_requests:
        if not isinstance(item, Mapping):
            continue
        sub_intent = item.get("intent") or item.get("intent_id")
        sub_intent_id = _intent_value(sub_intent, "")
        if sub_intent_id and sub_intent_id != primary_intent_id and sub_intent_id not in known_secondary:
            secondary_intents.append(sub_intent)
            known_secondary.add(sub_intent_id)

    # 规则要求升级的风险与明确“转人工”共用原子交接入口；
    # 在运行领域 Agent 或业务工具前短路，避免只把会话改成 waiting_human。
    risk_intent_ids = list(dict.fromkeys(
        [primary_intent_id]
        + [_intent_value(value, "") for value in secondary_intents]
        + [
            _intent_value(item.get("intent") or item.get("intent_id"), "")
            for item in sub_requests if isinstance(item, Mapping)
        ]
    ))
    if (
        handoff_enabled()
        and sqlite_store is not None
        and not _explicit_handoff_requested(chat_request.message)
    ):
        urgency_config = load_handoff_urgency_config()
        risk = score_temporary_urgency(
            chat_request.message,
            urgency_config,
            intent=primary_intent_id,
            intents=risk_intent_ids,
        )
        if risk.get("automatic_handoff"):
            reasons = list(risk.get("reasons") or [])
            reason = reasons[0] if reasons and isinstance(reasons[0], Mapping) else {}
            rule_id = str(reason.get("rule_id") or "automatic_handoff")
            reason_text = str(reason.get("text") or risk.get("action") or "需要人工进一步核查")
            handoff_policy = urgency_config.get("handoff_policy", {})
            automatic_intents = set(handoff_policy.get("automatic_handoff_intents") or [])
            summary_text = (
                str(risk.get("action") or reason_text)
                if automatic_intents.intersection(risk_intent_ids)
                else reason_text
            )
            handoff_result = await _start_atomic_handoff(
                chat_request,
                sqlite_store,
                authorization,
                conv_id=conv_id,
                handoff_reason=rule_id,
                intent=primary_intent_id,
                intents=risk_intent_ids,
                temporary_summary_text=summary_text,
            )
            _schedule_handoff_enrichment(sqlite_store, conv_id, handoff_result)
            automatic_response = _atomic_handoff_response(
                conv_id,
                handoff_result,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            automatic_response.intent = primary_intent_id or automatic_response.intent
            automatic_response.intent_group = str(
                getattr(intent_result, "intent_group", "other") or "other"
            )
            automatic_response.secondary_intents = [
                _intent_value(value, "") for value in secondary_intents
            ]
            automatic_response.routing_reason = f"命中转人工规则：{reason_text}"
            automatic_response.routing_confidence = float(
                getattr(intent_result, "confidence", 0.0) or 0.0
            )
            automatic_response.intent_diagnostics.update({
                "handoff_mode": "atomic_rule",
                "handoff_id": str(handoff_result.get("handoff_id") or ""),
                "handoff_reason": rule_id,
                "handoff_rule_text": reason_text,
                "background_processing": True,
            })
            return automatic_response

    raw_current_entities = dict(getattr(intent_result, "entities", {}) or {})
    pending_text_fields = list(existing_case_state.get("pending_text_fields") or [])
    pending_answer_field = pending_text_answer_field_id(
        chat_request.message,
        pending_text_fields,
        current_intent_id=primary_intent_id,
    )
    current_entities = apply_pending_text_answer(
        chat_request.message,
        raw_current_entities,
        pending_text_fields,
        current_intent_id=primary_intent_id,
    )
    mapped_pending_fields = {
        pending_answer_field
    } if pending_answer_field and current_entities.get(pending_answer_field) else set()
    # 若用户是在回答上轮追问，即使本轮单句无法独立识别为原业务意图，
    # 也恢复该字段所属意图，保证后续字段状态、工单及工具策略仍在原诉求上下文中。
    resumed_intent_ids: List[str] = []
    for item in pending_text_fields:
        if not isinstance(item, Mapping) or str(item.get("field_id") or "") not in mapped_pending_fields:
            continue
        for value in item.get("intent_ids", []):
            normalized_value = str(value)
            if normalized_value in BEAUTY_INTENT_TICKET_BY_ID and normalized_value not in resumed_intent_ids:
                resumed_intent_ids.append(normalized_value)
    if primary_intent_id == IntentCategory.OTHER.value and resumed_intent_ids:
        # 简短字段答案常无法独立分类；用上轮明确等待的业务字段恢复主意图，
        # 避免被低置信度 OTHER 澄清分支截断。
        primary_intent_id = resumed_intent_ids[0]
        intent = IntentCategory(primary_intent_id)
    for resumed_intent_id in resumed_intent_ids:
        if resumed_intent_id != primary_intent_id and resumed_intent_id not in known_secondary:
            secondary_intents.append(IntentCategory(resumed_intent_id))
            known_secondary.add(resumed_intent_id)
    # 知识库是否调用由意图注册表硬绑定，不能被模型诊断字段覆盖。
    # 对复合请求汇总各子意图知识策略，不让主意图掩盖其他领域的检索需要。
    knowledge_required = _should_use_knowledge(
        intent, chat_request.message, secondary_intents, sub_requests
    )
    diagnostics["knowledge_required"] = knowledge_required
    if sqlite_store is not None and not persisted_user_message_id:
        saved_user_message = await asyncio.to_thread(
            sqlite_store.append_message,
            conv_id,
            "consumer",
            chat_request.user_id,
            chat_request.message,
        )
        persisted_user_message_id = str(saved_user_message.get("message_id") or "")
    case_facts = CaseFacts.from_dict(existing_case_state.get("case_facts"))
    message_ref = persisted_user_message_id or f"request:{request_id}:user"
    handoff_mode = handoff_enabled()
    handoff_sub_requests: list[Dict[str, Any]] = []
    if handoff_mode and sqlite_store is not None:
        if _consumer_confirms_resolution(chat_request.message):
            await asyncio.to_thread(
                sqlite_store.resolve_active_case_sub_requests,
                conv_id,
                source_ref=message_ref,
            )
        known_sub_request_intents: set[str] = set()
        for index, candidate in enumerate([*sub_requests, *secondary_intents], start=1):
            if isinstance(candidate, Mapping):
                candidate_intent = candidate.get("intent") or candidate.get("intent_id")
                candidate_id = str(candidate.get("id") or candidate.get("sub_request_key") or "")
            else:
                candidate_intent = candidate
                candidate_id = ""
            candidate_id_value = _intent_value(candidate_intent, "")
            if candidate_id_value not in BEAUTY_INTENT_TICKET_BY_ID:
                continue
            known_sub_request_intents.add(candidate_id_value)
            handoff_sub_requests.append({
                "id": candidate_id or f"{candidate_id_value}:{index}",
                "intent": candidate_id_value,
            })
        if primary_intent_id in BEAUTY_INTENT_TICKET_BY_ID and primary_intent_id not in known_sub_request_intents:
            handoff_sub_requests.insert(0, {
                "id": f"{primary_intent_id}:primary", "intent": primary_intent_id,
            })
        message_sequence = await asyncio.to_thread(
            sqlite_store.get_message_sequence, conv_id, message_ref
        ) if persisted_user_message_id else None
        stored_sub_requests = await asyncio.to_thread(
            sqlite_store.sync_case_sub_requests,
            conv_id,
            handoff_sub_requests,
            source_ref=message_ref,
            source_message_seq=message_sequence,
        )
        await asyncio.to_thread(
            sqlite_store.append_case_action,
            conv_id,
            action_type="rule",
            description=("识别到转人工控制信号" if primary_intent_id == IntentCategory.HUMAN_HANDOFF.value else f"意图识别完成：{primary_intent_id}"),
            source_ref=message_ref,
            idempotency_key=f"intent:{request_id}",
            status="success",
        )
    else:
        stored_sub_requests = []

    case_action_logger = None
    if handoff_mode and sqlite_store is not None:
        async def _log_case_action(
            action_type: str,
            description: str,
            source_ref: str,
            idempotency_key: str,
            status: str,
        ) -> None:
            await asyncio.to_thread(
                sqlite_store.append_case_action,
                conv_id,
                action_type=action_type,
                description=description,
                source_ref=source_ref,
                idempotency_key=idempotency_key,
                status=status,
            )
            # 工具与复合子诉求没有可靠的一一映射时，只记录工具事实；
            # 不能因单个工具失败就把所有子诉求错误标为 blocked。
            if action_type == "tool" and status == "pending" and stored_sub_requests:
                for sub_request in stored_sub_requests:
                    if sub_request["status"] in {"open", "blocked"}:
                        try:
                            updated = await asyncio.to_thread(
                                sqlite_store.set_case_sub_request_status,
                                conv_id,
                                sub_request["sub_request_key"],
                                "in_progress",
                                source_ref=idempotency_key,
                            )
                            sub_request["status"] = updated["status"]
                        except ValueError:
                            logger.warning("案件子诉求状态更新失败：%s", sub_request.get("intent_id"))

    entity_confidence = float(diagnostics.get("entity_confidence", 0.8) or 0.8)
    extracted_facts = CaseFacts.from_entities(
        current_entities,
        source_ref=message_ref,
        confidence=entity_confidence,
    )
    case_facts.merge(extracted_facts)
    prior_slots = dict(existing_case_state.get("slots") or {})
    last_slot_answer = existing_case_state.get("last_slot_answer") or {}
    for slot_name, slot_value in prior_slots.items():
        case_facts.record(
            str(slot_name),
            slot_value,
            source_type="button" if last_slot_answer.get("slot_id") == slot_name else "legacy_slot",
            source_ref=str(last_slot_answer.get("question_id") or ""),
            confirmation_status="confirmed",
        )
    case_facts.record(
        "intent", _intent_value(intent), source_type="intent_recognizer",
        source_ref=request_id, confidence=float(getattr(intent_result, "confidence", 0.8) or 0.8),
        confirmation_status="inferred",
    )
    case_facts.record(
        "emotion", str(diagnostics.get("emotion", "neutral")),
        source_type="emotion_rule", source_ref=request_id,
        confidence=float(diagnostics.get("emotion_intensity", 0.0) or 0.0),
        confirmation_status="inferred",
    )
    case_facts.record(
        "emotion_intensity", float(diagnostics.get("emotion_intensity", 0.0) or 0.0),
        source_type="emotion_analysis", source_ref=request_id,
        confidence=1.0, confirmation_status="inferred",
    )
    urgency_value = getattr(getattr(intent_result, "urgency", None), "name", getattr(intent_result, "urgency", None))
    if urgency_value is not None:
        case_facts.record(
            "urgency", str(urgency_value), source_type="intent_recognizer",
            source_ref=request_id, confidence=float(getattr(intent_result, "confidence", 0.8) or 0.8),
            confirmation_status="inferred",
        )
    for evidence in diagnostics.get("emotion_evidence", []) or []:
        case_facts.record(
            "emotion_evidence", str(evidence), source_type="emotion_analysis",
            source_ref=request_id, confirmation_status="inferred",
        )
    for extracted_slot, extracted_values in current_entities.items():
        slot_values = extracted_values if isinstance(extracted_values, (list, tuple, set)) else [extracted_values]
        for slot_value in slot_values:
            if slot_value is None or isinstance(slot_value, (dict, list, tuple, set)):
                continue
            if sqlite_store is not None:
                await asyncio.to_thread(
                    sqlite_store.record_slot_value,
                    conv_id,
                    str(extracted_slot),
                    str(slot_value),
                    str(slot_value),
                    source_type="message",
                    source_ref=message_ref,
                    confidence=entity_confidence,
                    confirmation_status="unconfirmed",
                )
    internal_request = Request(
        message=chat_request.message,
        user_id=chat_request.user_id,
        conv_id=conv_id,
        context=context.to_prompt_text(),
        history=history,
        entities=current_entities,
        intent=intent,
        intent_group=getattr(intent_result, "intent_group", None),
        urgency=getattr(intent_result, "urgency", None),
        intent_confidence=float(getattr(intent_result, "confidence", 0.0)),
        request_id=request_id,
        intent_diagnostics=diagnostics,
        secondary_intents=secondary_intents,
        knowledge_required=knowledge_required,
        sub_requests=sub_requests,
        case_facts=case_facts.to_dict(),
        source_message_ref=message_ref,
        case_action_logger=case_action_logger,
    )
    result = await _maybe_await(app.state.agent_orchestrator.run(internal_request))
    case_facts.merge(getattr(result, "case_facts", None))
    result_intent = getattr(result, "intent", intent)
    result_secondary = list(getattr(result, "secondary_intents", []) or secondary_intents)
    result_ticket_draft = getattr(result, "ticket_draft", None)
    response_intent_candidates = [
        primary_intent_id,
        _intent_value(result_intent, ""),
        *[_intent_value(item, "") for item in result_secondary],
        *[_intent_value(item, "") for item in secondary_intents],
        *[
            _intent_value(item.get("intent") or item.get("intent_id"), "")
            for item in sub_requests
            if isinstance(item, Mapping)
        ],
    ]
    response_intent_ids = {
        value for value in response_intent_candidates
        if value in BEAUTY_INTENT_TICKET_BY_ID
    }
    is_refund_request = "return_refund" in response_intent_ids
    is_pure_refund_request = bool(response_intent_ids) and response_intent_ids <= {"return_refund"}
    explicit_handoff = _explicit_handoff_requested(chat_request.message)
    response_escalated = bool(
        getattr(result, "escalated", False)
        or getattr(result, "agent_recommended_handoff", False)
    )
    # Why：退款申请由代码保证形成待审核草稿，不能依赖模型是否自行调用建单工具。
    if (
        is_refund_request
        and not result_ticket_draft
        and sqlite_store is not None
    ):
        result_ticket_draft = build_handoff_payload(
            message=chat_request.message,
            intent=result_intent,
            entities=current_entities,
            should_handoff=False,
            conv_id=conv_id,
            buyer_nickname=chat_request.user_id,
            case_facts=case_facts,
            secondary_intents=result_secondary,
            sub_requests=sub_requests,
        ).get("ticket_draft")
    if progress_callback and knowledge_required:
        await progress_callback("knowledge", "知识库调用")
    if progress_callback and getattr(result, "tools_used", None):
        await progress_callback("tools", "工具调用")
    if progress_callback:
        await progress_callback("response", "整理回复")
    await _safe_add_message(
        memory_manager,
        chat_request.user_id,
        conv_id,
        MsgRole.USER,
        chat_request.message,
    )
    customer_visible_response = (
        "为您转接人工，请您稍等。"
        if explicit_handoff
        else str(getattr(result, "response", ""))
    )
    customer_visible_response = guard_response(customer_visible_response)
    if is_refund_request and not explicit_handoff:
        if is_pure_refund_request and not response_escalated:
            customer_visible_response = build_return_refund_request_reply(
                result_ticket_draft,
                existing_case_state.get("ticket_draft"),
            )
        else:
            customer_visible_response, removed_external_guidance = filter_external_refund_instructions(
                customer_visible_response
            )
            if removed_external_guidance and result_ticket_draft and not response_escalated:
                refund_status_reply = build_return_refund_request_reply(
                    result_ticket_draft,
                    existing_case_state.get("ticket_draft"),
                )
                customer_visible_response = "\n".join(
                    part for part in (customer_visible_response, refund_status_reply) if part
                )
        customer_visible_response = guard_response(customer_visible_response)
    case_facts.record(
        "ai_response", customer_visible_response, source_type="assistant_message",
        source_ref=request_id, confidence=1.0, confirmation_status="confirmed",
    )
    persist_refund_response_after_case_state = bool(
        is_refund_request and result_ticket_draft and sqlite_store is not None
    )
    if not persist_refund_response_after_case_state:
        await _safe_add_message(
            memory_manager,
            chat_request.user_id,
            conv_id,
            MsgRole.ASSISTANT,
            customer_visible_response,
        )
        if sqlite_store is not None:
            await asyncio.to_thread(
                sqlite_store.append_message,
                conv_id,
                "ai",
                "shopcare-ai",
                customer_visible_response,
            )
        _schedule_profile_update(memory_manager, chat_request.user_id, conv_id)

        # 让刚创建的后台任务获得一个事件循环调度机会；只让协程开始执行，不等待画像写入完成，
        # 这样测试与真实请求都能观察到任务已入队，同时不把画像更新重新变成同步链路。
        await asyncio.sleep(0)

    # API 边界采用端到端计时，覆盖上下文读取、意图识别、编排与同步记忆写回，
    # 避免把仅包含 Agent 编排阶段的 latency_ms 误报成用户实际等待时间。
    end_to_end_latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
    orchestrator_latency_ms = max(0.0, float(getattr(result, "latency_ms", 0.0) or 0.0))
    result_intent = getattr(result, "intent", intent)
    result_secondary = list(getattr(result, "secondary_intents", []) or [])
    if not result_secondary:
        result_secondary = list(secondary_intents)
    result_agents = list(getattr(result, "agent_types", []) or [])
    primary_agent = getattr(result, "primary_agent", None)
    if not result_agents and primary_agent is not None:
        result_agents = [primary_agent]
    tool_traces = list(getattr(result, "tool_traces", []) or [])
    response_diagnostics = dict(diagnostics)
    response_diagnostics.update(dict(getattr(result, "intent_diagnostics", {}) or {}))
    response_diagnostics["memory_context_loaded"] = bool(context.to_prompt_text())
    response_diagnostics["knowledge_required"] = knowledge_required
    response_diagnostics["orchestrator_latency_ms"] = round(orchestrator_latency_ms, 3)
    response_diagnostics["end_to_end_latency_ms"] = round(end_to_end_latency_ms, 3)
    response_diagnostics.setdefault("cache_hits", {})
    response_diagnostics.setdefault("degradation_reason", "")
    usage = dict(getattr(result, "usage", {}) or {})
    usage_by_stage = {
        str(stage): dict(stage_usage)
        for stage, stage_usage in dict(getattr(result, "usage_by_stage", {}) or {}).items()
        if isinstance(stage_usage, Mapping)
    }
    token_ledger = getattr(app.state, "token_usage", None)
    if token_ledger is not None:
        token_ledger.record_request(
            str(getattr(result, "request_id", "") or request_id),
            usage_by_stage,
            latency_ms=end_to_end_latency_ms,
        )
    knowledge_used = knowledge_required or any(
        isinstance(trace, Mapping) and trace.get("retrieval_stage")
        for trace in tool_traces
    )
    if sqlite_store is not None:
        await asyncio.to_thread(
            sqlite_store.record_observability,
            request_id,
            conv_id,
            {"context": response_diagnostics.get("context_latency_ms", 0), "orchestrator": orchestrator_latency_ms, "end_to_end": end_to_end_latency_ms},
            dict(response_diagnostics.get("cache_hits") or {}),
            str(response_diagnostics.get("degradation_reason") or ""),
        )
    response_entities: Dict[str, List[str]] = {}
    internal_fact_fields = {
        "intent", "emotion", "emotion_intensity", "emotion_evidence",
        "ai_response", "knowledge_evidence",
    }
    for field_name, fact_values in case_facts.to_prompt_dict().items():
        if field_name in internal_fact_fields:
            continue
        display_values = [str(value) for value in fact_values if not isinstance(value, (dict, list, tuple, set))]
        if display_values:
            response_entities[field_name] = list(dict.fromkeys(display_values))
    for slot_id, slot_value in dict(existing_case_state.get("slots") or {}).items():
        response_entities.setdefault(slot_id, [str(slot_value)])
    chat_response = ChatResponse(
        conv_id=conv_id,
        request_id=str(getattr(result, "request_id", "") or request_id),
        response=customer_visible_response,
        intent=_intent_value(result_intent),
        secondary_intents=[_intent_value(value) for value in result_secondary],
        intent_group=str(getattr(intent_result, "intent_group", "other") or "other"),
        agent_type=_intent_value(getattr(result, "agent_type", AgentType.GENERAL)),
        agent_types=[_intent_value(value) for value in result_agents],
        primary_agent=_intent_value(primary_agent, "") if primary_agent is not None else "",
        supporting_agents=[
            _intent_value(value)
            for value in list(getattr(result, "supporting_agents", []) or [])
        ],
        tools_used=list(getattr(result, "tools_used", []) or []),
        routing_reason=str(getattr(result, "routing_reason", "") or ""),
        routing_confidence=float(getattr(result, "routing_confidence", 0.0) or 0.0),
        escalated=bool(getattr(result, "escalated", False)),
        agent_recommended_handoff=bool(
            getattr(result, "agent_recommended_handoff", False)
        ),
        latency_ms=round(end_to_end_latency_ms, 3),
        knowledge_used=knowledge_used,
        entities=response_entities,
        intent_confidence=float(getattr(intent_result, "confidence", 0.0) or 0.0),
        intent_source_scores=dict(getattr(intent_result, "source_scores", {}) or {}),
        intent_diagnostics=response_diagnostics,
        emotion=str(getattr(intent_result, "emotion", "neutral") or "neutral"),
        emotion_intensity=float(getattr(intent_result, "emotion_intensity", 0.0) or 0.0),
        emotion_evidence=list(getattr(intent_result, "emotion_evidence", []) or []),
        handoff_summary=getattr(result, "handoff_summary", None),
        ticket_draft=result_ticket_draft,
        suggested_replies=list(getattr(result, "suggested_replies", []) or []),
        follow_up_options=_follow_up_options(
            result_intent,
            response_entities,
            result_secondary,
            sub_requests,
        ),
        pending_questions=_pending_questions(
            result_intent,
            response_entities,
            result_secondary,
            sub_requests,
        ),
        usage=usage,
        usage_by_stage=usage_by_stage,
    )
    if explicit_handoff:
        # 明确转人工是规则分支：消费者端不继续追问，也不展示槽位按钮。
        chat_response.response = customer_visible_response
        chat_response.follow_up_options = []
        chat_response.pending_questions = []
        chat_response.agent_recommended_handoff = True
    if handoff_mode and sqlite_store is not None:
        if chat_response.pending_questions:
            await asyncio.to_thread(
                sqlite_store.append_case_action,
                conv_id,
                action_type="followup",
                description="必要字段追问已向顾客展示",
                source_ref=message_ref,
                idempotency_key=f"followup:{request_id}",
                status="success",
            )
            primary_key = next(
                (item["sub_request_key"] for item in stored_sub_requests if item["intent_id"] == primary_intent_id),
                None,
            )
            if primary_key:
                try:
                    await asyncio.to_thread(
                        sqlite_store.set_case_sub_request_status,
                        conv_id,
                        primary_key,
                        "blocked",
                        source_ref=message_ref,
                    )
                except ValueError:
                    logger.warning("案件子诉求缺字段状态更新失败：%s", primary_intent_id)
    if sqlite_store is not None and chat_response.ticket_draft:
        system_fields = dict(chat_response.ticket_draft.get("system_fields") or {})
        system_fields["conv_id"] = conv_id
        system_fields["buyer_nickname"] = await asyncio.to_thread(sqlite_store.get_user_display_name, chat_request.user_id)
        previous_ticket = dict(existing_case_state.get("ticket_draft") or {})
        if previous_ticket.get("status") == "draft":
            previous_system_fields = dict(previous_ticket.get("system_fields") or {})
            if previous_system_fields.get("ticket_id"):
                system_fields["ticket_id"] = previous_system_fields["ticket_id"]
            if previous_system_fields.get("created_at"):
                system_fields["created_at"] = previous_system_fields["created_at"]
            previous_fields = dict(previous_ticket.get("fields") or {})
            current_fields = dict(chat_response.ticket_draft.get("fields") or {})
            chat_response.ticket_draft["fields"] = {
                key: value if value not in (None, "") else previous_fields.get(key)
                for key, value in current_fields.items()
            }
            blocking_fields = chat_response.ticket_draft.get("blocking_fields") or []
            chat_response.ticket_draft["missing_blocking_fields"] = [
                field for field in blocking_fields
                if not chat_response.ticket_draft["fields"].get(field)
            ]
        chat_response.ticket_draft["system_fields"] = system_fields
    if sqlite_store is not None:
        state_updates: Dict[str, Any] = {
            "case_facts": case_facts.to_dict(),
            "intent": chat_response.intent,
            "emotion": chat_response.emotion,
            "emotion_intensity": chat_response.emotion_intensity,
            "emotion_evidence": chat_response.emotion_evidence,
            "last_request_id": request_id,
        }
        field_state_request = Request(
            message=chat_request.message,
            intent=result_intent,
            secondary_intents=result_secondary,
            sub_requests=sub_requests,
            entities=response_entities,
            case_facts=case_facts.to_dict(),
        )
        field_state = build_business_field_state(field_state_request)
        pending_text_fields_next = (
            [] if explicit_handoff else _pending_text_fields_for_response(
                chat_response.response, field_state
            )
        )
        # 只保留回复里实际问出的一个文本字段；用户下一轮可直接用自然语言回答。
        state_updates["pending_text_fields"] = pending_text_fields_next
        if chat_response.ticket_draft:
            state_updates.update({
                "ticket_draft": chat_response.ticket_draft,
                "ticket_status": str(chat_response.ticket_draft.get("status") or "draft"),
                "department_tasks": chat_response.ticket_draft.get("department_tasks", []),
                "urgency": chat_response.ticket_draft.get("risk_level", "P3"),
                "risk_level": chat_response.ticket_draft.get("risk_level", "P3"),
            })
        if chat_response.escalated or chat_response.agent_recommended_handoff:
            state_updates.update({
                "handoff_required": True,
                "status": "waiting_human",
                "sub_requests": sub_requests,
                "handoff_summary": chat_response.handoff_summary,
                "next_steps": (chat_response.handoff_summary or {}).get("next_steps", []),
                "next_steps_sources": (chat_response.handoff_summary or {}).get("next_steps_sources", []),
                "suggested_replies": chat_response.suggested_replies,
                "urgency": (chat_response.handoff_summary or {}).get("urgency", "P3"),
                "risk_level": (chat_response.ticket_draft or {}).get("risk_level", (chat_response.handoff_summary or {}).get("urgency", "P3")),
            })
        await asyncio.to_thread(sqlite_store.update_case_state, conv_id, state_updates)
    if persist_refund_response_after_case_state:
        # 退款确认必须排在本地草稿成功写入之后；写库失败时不向消费者留下“已登记”消息。
        await _safe_add_message(
            memory_manager,
            chat_request.user_id,
            conv_id,
            MsgRole.ASSISTANT,
            customer_visible_response,
        )
        await asyncio.to_thread(
            sqlite_store.append_message,
            conv_id,
            "ai",
            "shopcare-ai",
            customer_visible_response,
        )
        _schedule_profile_update(memory_manager, chat_request.user_id, conv_id)
        await asyncio.sleep(0)
    if sqlite_store is not None and (chat_response.escalated or chat_response.agent_recommended_handoff):
        await asyncio.to_thread(
            sqlite_store.record_risk_event,
            chat_request.user_id,
            conv_id,
            str((chat_response.handoff_summary or {}).get("urgency", "P3")),
            chat_response.intent,
            str((chat_response.handoff_summary or {}).get("handoff_reason", "需要人工接管")),
        )
    emotion_client = getattr(app.state, "anthropic_client", None)
    if sqlite_store is not None and emotion_client is not None:
        recent_emotion_context = "\n".join(
            f"{str(getattr(getattr(item, 'role', ''), 'value', getattr(item, 'role', '')))}: "
            f"{str(getattr(item, 'content', ''))}"
            for item in history[-3:]
            if getattr(item, "content", None)
        )
        async def _persist_emotion() -> None:
            await asyncio.to_thread(
                sqlite_store.update_emotion_analysis,
                conv_id,
                request_id,
                await analyze_emotion(
                    emotion_client,
                    chat_request.message,
                    recent=recent_emotion_context,
                    fallback_emotion=str(diagnostics.get("emotion", "平静")),
                ),
            )
        asyncio.create_task(_persist_emotion())
    return chat_response


@app.post("/chat/stream")
async def chat_stream(
    chat_request: ChatRequest,
    authorization: Optional[str] = Header(default=None),
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """以 SSE 推送处理阶段，最终事件携带与 /chat 相同的完整响应。"""

    _require_ready()
    store = getattr(app.state, "sqlite_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="SQLite 会话存储尚未就绪")
    chat_request = await _bind_consumer_request(store, chat_request, authorization)

    async def events():
        replay_conv_id = chat_request.conv_id
        if store is not None and replay_conv_id and last_event_id:
            try:
                replay_after = int(last_event_id)
            except (TypeError, ValueError):
                replay_after = 0
            for item in await asyncio.to_thread(store.list_stream_events, replay_conv_id, replay_after):
                yield f"id: {item['id']}\nevent: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
            if replay_after:
                return

        async def packet(event_type: str, payload: Dict[str, Any], conv_id: Optional[str] = None) -> str:
            event_id = None
            if store is not None and conv_id:
                event_id = await asyncio.to_thread(store.append_stream_event, conv_id, event_type, payload)
            prefix = f"id: {event_id}\n" if event_id else ""
            return f"{prefix}event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        direct_handoff = _explicit_handoff_requested(chat_request.message)
        if direct_handoff and handoff_enabled():
            immediate_conv_id = chat_request.conv_id or str(uuid.uuid4())
            started = time.perf_counter()
            try:
                handoff_result = await _start_atomic_handoff(
                    chat_request,
                    store,
                    authorization,
                    conv_id=immediate_conv_id,
                )
            except HTTPException as exc:
                yield f"event: error\ndata: {json.dumps({'message': str(exc.detail)}, ensure_ascii=False)}\n\n"
                return
            except Exception:
                logger.exception("原子转人工事务失败；消费者未收到确认")
                yield f"event: error\ndata: {json.dumps({'message': '转人工登记失败，请稍后重试。'}, ensure_ascii=False)}\n\n"
                return

            # start_handoff 已返回，SQLite 事务已提交；只有此后才发送消费者 ACK。
            _schedule_handoff_enrichment(store, immediate_conv_id, handoff_result)
            ack = str(handoff_result["ack_text"])
            yield await packet(
                "ack",
                {"message_id": handoff_result["ack_message_id"], "text": ack},
                immediate_conv_id,
            )
            yield f"event: status\ndata: {json.dumps({'message': ack, 'stage': 'handoff_ack'}, ensure_ascii=False)}\n\n"
            yield f"event: delta\ndata: {json.dumps({'text': ack}, ensure_ascii=False)}\n\n"
            yield f"event: segment_start\ndata: {json.dumps({'seg_id': 'handoff_ack', 'title': '转人工'}, ensure_ascii=False)}\n\n"
            yield f"event: segment_delta\ndata: {json.dumps({'seg_id': 'handoff_ack', 'text': ack}, ensure_ascii=False)}\n\n"
            yield f"event: segment_end\ndata: {json.dumps({'seg_id': 'handoff_ack'}, ensure_ascii=False)}\n\n"
            immediate = _atomic_handoff_response(
                immediate_conv_id,
                handoff_result,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            yield await packet(
                "done", json.loads(immediate.model_dump_json()), immediate_conv_id
            )
            return

        if direct_handoff:
            # 关键词命中后立即结束消费者侧请求；摘要和工单在后台继续整理，
            # 不让上下文读取、LLM 意图识别或知识库查询阻塞转人工确认。
            ack = "为您转接人工，请您稍等。"
            immediate_conv_id = chat_request.conv_id or str(uuid.uuid4())
            if store is not None:
                await asyncio.to_thread(store.ensure_session, chat_request.user_id, immediate_conv_id)
            yield await packet("ack", {'message_id': str(uuid.uuid4()), 'text': ack}, immediate_conv_id)
            yield f"event: status\ndata: {json.dumps({'message': ack, 'stage': 'handoff_ack'}, ensure_ascii=False)}\n\n"
            yield f"event: delta\ndata: {json.dumps({'text': ack}, ensure_ascii=False)}\n\n"
            yield f"event: segment_start\ndata: {json.dumps({'seg_id': 'handoff_ack', 'title': '转人工'}, ensure_ascii=False)}\n\n"
            yield f"event: segment_delta\ndata: {json.dumps({'seg_id': 'handoff_ack', 'text': ack}, ensure_ascii=False)}\n\n"
            yield f"event: segment_end\ndata: {json.dumps({'seg_id': 'handoff_ack'}, ensure_ascii=False)}\n\n"
            # 先落消费者原话，客服端无需等待 Chroma/LLM 后台整理即可看到会话内容。
            if store is not None:
                saved_user_message = await asyncio.to_thread(
                    store.append_message,
                    immediate_conv_id,
                    "consumer",
                    chat_request.user_id,
                    chat_request.message,
                )
            async def background_handoff() -> None:
                try:
                    await _chat_core(
                        chat_request.model_copy(update={"conv_id": immediate_conv_id}),
                        authorization,
                        persisted_user_message_id=(saved_user_message.get("message_id") if store is not None else None),
                    )
                except Exception:
                    logger.exception("后台转人工上下文整理失败")

            asyncio.create_task(background_handoff())
            immediate = ChatResponse(
                conv_id=immediate_conv_id,
                request_id=str(uuid.uuid4())[:8],
                response=ack,
                intent=IntentCategory.HUMAN_HANDOFF.value,
                intent_group="handoff",
                agent_type=AgentType.ESCALATION.value,
                agent_types=[AgentType.ESCALATION.value],
                primary_agent=AgentType.ESCALATION.value,
                routing_reason="规则命中明确人工请求",
                routing_confidence=1.0,
                escalated=True,
                agent_recommended_handoff=True,
                latency_ms=0.0,
                intent_diagnostics={"handoff_mode": "rule", "background_processing": True},
            )
            yield await packet("done", json.loads(immediate.model_dump_json()), immediate_conv_id)
            return
        if (
            handoff_enabled()
            and store is not None
            and chat_request.conv_id
            and await asyncio.to_thread(store.is_waiting_human, chat_request.conv_id)
        ):
            # 转人工后的新消息进入同一会话，但不再启动 Agent 或向消费者生成答复。
            await asyncio.to_thread(
                store.append_message,
                chat_request.conv_id,
                "consumer",
                chat_request.user_id,
                chat_request.message,
            )
            suppressed = ChatResponse(
                conv_id=chat_request.conv_id,
                request_id=str(uuid.uuid4())[:8],
                response="",
                intent=IntentCategory.HUMAN_HANDOFF.value,
                intent_group="handoff",
                agent_type=AgentType.ESCALATION.value,
                escalated=True,
                intent_diagnostics={"handoff_suppressed": True, "reason": "waiting_human"},
                latency_ms=0.0,
            )
            yield await packet(
                "done", json.loads(suppressed.model_dump_json()), chat_request.conv_id
            )
            return
        else:
            yield f"event: ack\ndata: {json.dumps({'message_id': str(uuid.uuid4()), 'text': '已收到您的消息。'}, ensure_ascii=False)}\n\n"
            yield f"event: status\ndata: {json.dumps({'message': '已收到，正在处理您的问题，请稍候…'}, ensure_ascii=False)}\n\n"
        queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

        async def emit(event: str, message: str) -> None:
            await queue.put((event, message))

        task = asyncio.create_task(_chat_core(chat_request, authorization, progress_callback=emit))
        try:
            while not task.done() or not queue.empty():
                try:
                    event, label = await asyncio.wait_for(queue.get(), timeout=0.2)
                    yield f"event: {event}\ndata: {json.dumps({'message': label}, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    continue
            result = await task
            if str((result.intent_diagnostics or {}).get("handoff_mode") or "") == "atomic_rule":
                ack = result.response
                yield await packet(
                    "ack",
                    {
                        "message_id": (result.intent_diagnostics or {}).get("ack_message_id"),
                        "text": ack,
                    },
                    result.conv_id,
                )
                yield f"event: status\ndata: {json.dumps({'message': ack, 'stage': 'handoff_ack'}, ensure_ascii=False)}\n\n"
                yield f"event: delta\ndata: {json.dumps({'text': ack}, ensure_ascii=False)}\n\n"
                yield await packet("done", json.loads(result.model_dump_json()), result.conv_id)
                return
            if store is not None and await asyncio.to_thread(store.is_waiting_human, result.conv_id):
                await _record_reply_action(store, result.conv_id, result.request_id, "failed")
                suppressed_payload = result.model_dump()
                suppressed_payload["response"] = ""
                suppressed_payload["intent_diagnostics"] = {
                    **dict(suppressed_payload.get("intent_diagnostics") or {}),
                    "handoff_suppressed": True,
                }
                yield await packet("done", suppressed_payload, result.conv_id)
                return
            if direct_handoff:
                # 消费者明确请求人工时，最终消费者消息只保留转接确认；
                # 上下文整理结果仍已由后台任务写入客服侧 case_state。
                result.response = "为您转接人工，请您稍等。"
                result.follow_up_options = []
                result.pending_questions = []
            # 结果已经生成后按短片段推送，避免长回复一次性“砸”到页面；
            # 这改善的是可感知等待，真正的后端耗时仍由阶段事件和 latency_ms 记录。
            response_text = str(result.response or "")
            await _record_reply_action(store, result.conv_id, result.request_id, "pending")
            yield f"event: segment_start\ndata: {json.dumps({'seg_id': 'response', 'title': '回复'}, ensure_ascii=False)}\n\n"
            for index in range(0, len(response_text), 12):
                if store is not None and await asyncio.to_thread(store.is_waiting_human, result.conv_id):
                    await _record_reply_action(store, result.conv_id, result.request_id, "failed")
                    suppressed_payload = result.model_dump()
                    suppressed_payload["response"] = ""
                    suppressed_payload["intent_diagnostics"] = {
                        **dict(suppressed_payload.get("intent_diagnostics") or {}),
                        "handoff_suppressed": True,
                    }
                    yield await packet("done", suppressed_payload, result.conv_id)
                    return
                yield f"event: delta\ndata: {json.dumps({'text': response_text[index:index + 12]}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
            yield f"event: segment_delta\ndata: {json.dumps({'seg_id': 'response', 'text': response_text}, ensure_ascii=False)}\n\n"
            yield f"event: segment_end\ndata: {json.dumps({'seg_id': 'response'}, ensure_ascii=False)}\n\n"
            await _record_reply_action(store, result.conv_id, result.request_id, "success")
            yield f"event: done\ndata: {result.model_dump_json()}\n\n"
        except Exception as exc:
            logger.exception("SSE chat failed")
            yield f"event: error\ndata: {json.dumps({'message': '当前处理遇到问题，请稍后重试。'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _record_reply_action(
    store: Optional[SQLiteCaseStore], conv_id: str, request_id: str, status: str
) -> None:
    """在对外答复的边界留下不含答复正文的可幂等动作记录。"""
    if not handoff_enabled() or store is None or not conv_id or not request_id:
        return
    await asyncio.to_thread(
        store.append_case_action,
        conv_id,
        action_type="reply",
        description=(
            "AI 客服答复已发送" if status == "success"
            else "AI 客服答复被转人工抢占" if status == "failed"
            else "AI 客服答复发送中"
        ),
        source_ref=request_id,
        idempotency_key=f"reply:return:{request_id}",
        status=status,
    )


async def _safe_add_message(
    memory_manager: Any,
    user_id: str,
    conv_id: str,
    role: MsgRole,
    content: str,
) -> None:
    """按时序写入一条消息，并隔离底层存储异常。"""

    add_message = getattr(memory_manager, "add_message", None)
    if not callable(add_message):
        return
    try:
        await _maybe_await(add_message(user_id, conv_id, role, content))
    except Exception as exc:
        logger.warning("%s 消息写入失败，继续返回响应：%s", role.value, exc)
