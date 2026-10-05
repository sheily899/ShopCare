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
from typing import Any, Dict, List, Mapping, Optional
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentType,
    BillingAgent,
    EscalationAgent,
    GeneralAgent,
    Request,
    TechnicalAgent,
)
from intent.intent_recognizer import IntentCategory, recognize_intent
from mcp.knowledge_base import KnowledgeBase
from mcp.tool_manager import MCPToolManager, Tool
from memory.conversation_memory import MemoryContext, MemoryManager, MsgRole
from skills.skill_manager import SkillManager


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

    async def recognize(self, message: str, history: Any = None) -> Any:
        """调用核心三路意图识别，不在 API 层复制识别算法。

        Args:
            message: 待识别的用户消息。
            history: 可选历史，仅由核心识别器决定是否使用。

        Returns:
            核心模块返回的 ``IntentResult`` dataclass。
        """

        return await recognize_intent(message, history=history)

    async def recognize_intent(self, message: str, history: Any = None) -> Any:
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
        "EVALUATION_DATASET": "evaluation/datasets/golden_benchmark_v2.1_100.json",
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
    return await asyncio.to_thread(knowledge_base.search, query, top_k)


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

        load_dotenv()
    except Exception as exc:  # pragma: no cover - python-dotenv 可选安装差异
        logger.warning(".env 加载失败，继续使用系统环境变量：%s", exc)

    settings = _read_settings()
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
        "general": GeneralAgent(client=anthropic_client, skill_manager=skill_manager),
        "technical": TechnicalAgent(client=anthropic_client, skill_manager=skill_manager),
        "billing": BillingAgent(client=anthropic_client, skill_manager=skill_manager),
        "escalation": EscalationAgent(client=anthropic_client, skill_manager=skill_manager),
    }
    typed_agents = {
        AgentType.GENERAL: agents["general"],
        AgentType.TECHNICAL: agents["technical"],
        AgentType.BILLING: agents["billing"],
        AgentType.ESCALATION: agents["escalation"],
    }
    orchestrator = AgentOrchestrator(
        agent_pool=typed_agents,
        knowledge_search=knowledge_base,
    )
    performance_monitor = PerformanceMonitor()
    evaluator = EndToEndEvaluator(settings["EVALUATION_DATASET"])

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
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False
        await _close_resource(performance_monitor)
        await _close_resource(skill_manager)
        await _close_resource(anthropic_client)
        await _close_resource(getattr(knowledge_base, "_client", None))
        await _close_resource(redis_client)


app = FastAPI(title="ShopCare", version="0.1.0", lifespan=lifespan)
# Why：保留开发期前端联调基线；管理路由当前无鉴权，生产前必须补齐鉴权网关。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


class SearchRequest(BaseModel):
    """知识库只读检索的 API 输入契约。

    Attributes:
        query: 待检索的自然语言问题。
        top_k: 请求的最大召回条数，必须为正整数。
    """

    query: str
    top_k: int = Field(default=5, ge=1)

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

    @field_validator("title", "content")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """拒绝没有标题或正文的入库请求。"""

        if not value.strip():
            raise ValueError("title/content must not be blank")
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


def _should_use_knowledge(intent: Any) -> bool:
    """按业务意图门控是否需要知识库证据。"""

    value = _intent_value(intent)
    return value not in {
        IntentCategory.GREETING.value,
        IntentCategory.FEEDBACK.value,
        IntentCategory.OTHER.value,
        IntentCategory.HUMAN_HANDOFF.value,
    }


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
async def search_knowledge(request: SearchRequest) -> Dict[str, Any]:
    """执行产品知识库只读检索。

    Args:
        request: 包含自然语言 query 与 top_k 的检索请求。

    Returns:
        ``results`` 列表，每项包含 title、chunk_id、content 与 score。

    Raises:
        HTTPException: 服务未就绪或知识库检索异常时返回 503。

    说明：当前管理/检索路由开放且没有鉴权，生产前必须接入鉴权网关。
    """

    knowledge_base = _require_knowledge_base()
    try:
        raw = await asyncio.to_thread(knowledge_base.search, request.query, request.top_k)
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
async def add_knowledge(request: KnowledgeAddRequest) -> Dict[str, Any]:
    """接收纯文本并调用 KnowledgeBase 切片入库。

    Args:
        request: 文档标题与正文。

    Returns:
        ``success`` 与实际新增 chunk 数量；不执行外部订单或支付写操作。

    Raises:
        HTTPException: 服务未就绪或知识库写入失败时返回 503。

    说明：这是开发期开放管理接口，正式环境必须由鉴权网关保护。
    """

    knowledge_base = _require_knowledge_base()
    try:
        added = await asyncio.to_thread(
            knowledge_base.add_documents,
            [{"title": request.title, "content": request.content}],
        )
        added = await _maybe_await(added)
    except Exception as exc:
        logger.warning("知识库文本入库失败：%s", exc)
        raise HTTPException(status_code=503, detail="知识库写入暂不可用") from exc
    return {"success": True, "title": request.title, "added": int(added or 0)}


@app.post("/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...)) -> Dict[str, Any]:
    """上传 txt/md/json 文档并交由 KnowledgeBase 切片入库。

    Args:
        file: multipart 上传文件；扩展名和大小均在 API 边界校验。

    Returns:
        文件名、入库 chunk 数和成功标志。

    Raises:
        HTTPException: 扩展名不允许、超过 10MB、JSON 无法解析或知识库不可用。

    说明：当前管理路由无鉴权，生产前需补齐鉴权网关；本接口不连接外部业务系统。
    """

    knowledge_base = _require_knowledge_base()
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
            [{"title": Path(filename).stem, "content": content}],
        )
        added = await _maybe_await(added)
    except Exception as exc:
        logger.warning("知识文件入库失败：%s", exc)
        raise HTTPException(status_code=503, detail="知识库写入暂不可用") from exc
    return {"success": True, "filename": filename, "added": int(added or 0)}


@app.get("/knowledge/stats")
async def knowledge_stats() -> Dict[str, Any]:
    """返回产品知识库当前片段数量与模型配置。"""

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
async def get_tool_trace(request_id: str) -> Dict[str, Any]:
    """按 request_id 返回工具调用 Trace。

    Trace 当前只驻留进程内存，服务重启后会丢失；该接口不提供审计级持久化。
    """

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
async def get_tool_traces() -> Dict[str, Any]:
    """返回最近的内存工具 Trace，并标记服务重启丢失边界。"""

    return {
        "traces": _get_trace_snapshot(),
        "storage": "memory",
        "max_items": TRACE_MEMORY_LIMIT,
        "restart_loses_data": True,
    }


@app.get("/skills")
async def list_skills() -> Dict[str, Any]:
    """返回本地 SkillManager 当前已加载的技能列表。"""

    _require_ready()
    manager = getattr(app.state, "skill_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="SkillManager 尚未装配")
    skills = list(getattr(manager, "skills", []) or [])
    return {"skills": skills, "count": len(skills)}


@app.post("/skills/reload")
async def reload_skills() -> Dict[str, Any]:
    """重新扫描本地技能目录并返回热加载结果。

    说明：该开放管理接口当前无鉴权，正式生产前必须接入鉴权网关。
    """

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


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest) -> ChatResponse:
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
    started = time.perf_counter()
    conv_id = chat_request.conv_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())[:8]
    memory_manager = app.state.memory_manager
    context = await _load_memory_context(
        memory_manager,
        chat_request.user_id,
        conv_id,
        chat_request.message,
    )
    # 只把最近 5 条交给意图识别，避免历史上下文挤占分类输入预算。
    history = list(context.recent_messages[-5:])
    intent_result = await _recognize_request(
        app.state.intent_recognizer,
        chat_request.message,
        history,
    )
    intent = getattr(intent_result, "intent", IntentCategory.OTHER)
    diagnostics = dict(getattr(intent_result, "intent_diagnostics", {}) or {})
    knowledge_required = bool(diagnostics.get("knowledge_required", _should_use_knowledge(intent)))
    internal_request = Request(
        message=chat_request.message,
        user_id=chat_request.user_id,
        conv_id=conv_id,
        context=context.to_prompt_text(),
        history=history,
        entities=dict(getattr(intent_result, "entities", {}) or {}),
        intent=intent,
        intent_group=getattr(intent_result, "intent_group", None),
        urgency=getattr(intent_result, "urgency", None),
        intent_confidence=float(getattr(intent_result, "confidence", 0.0)),
        request_id=request_id,
        intent_diagnostics=diagnostics,
        secondary_intents=list(getattr(intent_result, "secondary_intents", []) or []),
        knowledge_required=knowledge_required,
    )
    result = await _maybe_await(app.state.agent_orchestrator.run(internal_request))
    await _safe_add_message(
        memory_manager,
        chat_request.user_id,
        conv_id,
        MsgRole.USER,
        chat_request.message,
    )
    await _safe_add_message(
        memory_manager,
        chat_request.user_id,
        conv_id,
        MsgRole.ASSISTANT,
        str(getattr(result, "response", "")),
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
        result_secondary = list(getattr(intent_result, "secondary_intents", []) or [])
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
    knowledge_used = knowledge_required or any(
        isinstance(trace, Mapping) and trace.get("retrieval_stage")
        for trace in tool_traces
    )
    return ChatResponse(
        conv_id=conv_id,
        request_id=str(getattr(result, "request_id", "") or request_id),
        response=str(getattr(result, "response", "")),
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
        entities=dict(getattr(intent_result, "entities", {}) or {}),
        intent_confidence=float(getattr(intent_result, "confidence", 0.0) or 0.0),
        intent_source_scores=dict(getattr(intent_result, "source_scores", {}) or {}),
        intent_diagnostics=response_diagnostics,
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
