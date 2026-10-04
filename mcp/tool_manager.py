"""ShopCare 工具与 RAG 召回前处理基础设施。

本模块属于 MCP/RAG 治理层，负责工具契约、参数校验、超时隔离、内存缓存、
熔断状态机、fallback、Query Rewrite、并行召回和稳定去重。LLM Rerank、
真实订单/支付/退款写操作和人工工单系统不属于本模块当前阶段边界。
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """工具断路器的三种运行状态。"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# 兼容设计文档和后续调用方可能使用的更完整枚举名称。
CircuitBreakerState = CircuitState


@dataclass(init=False)
class ToolResult:
    """表示一次工具调用的结构化结果。

    Args:
        name: 工具名。
        success: 本次调用是否得到可用结果；fallback 成功时为 True，但 error
            仍保留主 handler 的失败原因。
        data: 工具结果或 fallback 提供的只读建议数据。
        error: 主调用或 fallback 失败信息。
        cached: 是否直接命中内存缓存。
        latency_ms: 主调用耗时，单位为毫秒。
        metadata: 供 trace/monitor 扩展的非业务元数据。

    Returns:
        无；该类只承载工具调用结果。
    """

    name: str
    success: bool
    data: Any
    error: Optional[str]
    cached: bool
    latency_ms: float
    metadata: Dict[str, Any]
    reranked: bool
    rerank_degraded: bool

    def __init__(
        self,
        name: Optional[str] = None,
        success: bool = False,
        data: Any = None,
        error: Optional[str] = None,
        cached: bool = False,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        reranked: bool = False,
        rerank_degraded: bool = False,
        *,
        tool_name: Optional[str] = None,
    ) -> None:
        """初始化结果，并兼容旧代码中的 ``tool_name`` 关键字。"""
        self.name = str(name or tool_name or "")
        self.success = bool(success)
        self.data = data
        self.error = error
        self.cached = bool(cached)
        self.latency_ms = float(latency_ms)
        self.metadata = dict(metadata or {})
        self.reranked = bool(reranked)
        self.rerank_degraded = bool(rerank_degraded)

    @property
    def tool_name(self) -> str:
        """返回兼容旧接口的工具名属性。"""
        return self.name


@dataclass
class ToolStats:
    """记录工具调用统计，供后续 monitor 层读取。"""

    total: int = 0
    success: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0
    consecutive_fails: int = 0

    @property
    def success_rate(self) -> float:
        """返回成功率；尚无调用时按 1.0 表示没有失败证据。"""
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        """返回平均调用耗时。"""
        return self.total_latency_ms / self.total if self.total else 0.0


@dataclass(init=False)
class Tool:
    """定义一个可由 MCPToolManager 执行的异步工具契约。

    Args:
        name: 工具唯一名称。
        description: 工具职责说明。
        schema: JSON Schema 参数契约，支持 required、properties 和 type。
        handler: 主异步处理函数，接收 ``params``，也兼容 ``params, context``。
        fallback: 可选降级函数，接收 ``params, context, error``。
        cache_ttl: 成功结果缓存秒数；默认 0 表示不缓存，知识工具由注册方设为 300。
        timeout: 主 handler 的超时时间，默认 30 秒。

    Returns:
        无；该类保存工具定义及其运行时统计状态。
    """

    name: str
    description: str
    schema: Dict[str, Any]
    handler: Callable[..., Any]
    fallback: Optional[Callable[..., Any]]
    cache_ttl: int
    timeout: float
    stats: ToolStats
    breaker: "CircuitBreaker"

    def __init__(
        self,
        name: str,
        description: str,
        schema: Optional[Dict[str, Any]] = None,
        handler: Optional[Callable[..., Any]] = None,
        fallback: Optional[Callable[..., Any]] = None,
        cache_ttl: int = 0,
        timeout: float = 30.0,
        *,
        input_schema: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        """初始化工具，并兼容旧版 ``handler, schema`` 位置参数顺序。"""
        # 兼容旧工程 Tool(name, description, handler, schema) 的构造方式。
        if callable(schema) and isinstance(handler, dict):
            schema, handler = handler, schema
        self.name = str(name)
        self.description = str(description)
        self.schema = dict(input_schema or schema or {})
        self.handler = handler  # type: ignore[assignment]
        self.fallback = fallback
        self.cache_ttl = int(cache_ttl)
        self.timeout = float(timeout_s if timeout_s is not None else timeout)
        self.stats = ToolStats()
        self.breaker = CircuitBreaker()

    @property
    def input_schema(self) -> Dict[str, Any]:
        """返回 JSON Schema 的语义别名。"""
        return self.schema

    @property
    def timeout_s(self) -> float:
        """返回旧代码使用的 timeout_s 别名。"""
        return self.timeout


class CircuitBreaker:
    """实现 CLOSED、OPEN、HALF_OPEN 三态断路器。

    Args:
        failure_threshold: 连续失败次数阈值，默认 5。
        recovery_s: OPEN 状态冷却秒数，默认 60。
        clock: 可注入时钟，生产环境使用 ``time.monotonic``，测试可控时间推进。

    Returns:
        无；状态通过 ``state``、``failure_count`` 和 ``opened_at`` 暴露。
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = int(failure_threshold)
        self.recovery_s = float(recovery_s)
        self._clock = clock
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.opened_at: Optional[float] = None
        self._probe_in_flight = False

    @property
    def threshold(self) -> int:
        """返回旧代码兼容的失败阈值属性。"""
        return self.failure_threshold

    @property
    def fail_count(self) -> int:
        """返回旧代码兼容的失败计数属性。"""
        return self.failure_count

    def allow(self) -> bool:
        """判断当前请求是否可以访问主 handler。

        Returns:
            CLOSED 放行；OPEN 冷却不足时拒绝；冷却完成后只放行一次 HALF_OPEN 探测。
        """
        if self.state is CircuitState.CLOSED:
            return True
        if self.state is CircuitState.OPEN:
            # Why：60 秒冷却避免故障依赖被连续重试压垮，同时给服务恢复窗口。
            if self.opened_at is not None and self._clock() - self.opened_at >= self.recovery_s:
                self.state = CircuitState.HALF_OPEN
                self._probe_in_flight = True
                return True
            return False
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def record_success(self) -> None:
        """记录主 handler 或半开探测成功并关闭断路器。"""
        self.failure_count = 0
        self.opened_at = None
        self._probe_in_flight = False
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """记录一次主 handler 失败并按阈值推进状态机。"""
        if self.state is CircuitState.HALF_OPEN:
            # Why：探测失败说明依赖尚未恢复，必须重新开始完整冷却周期。
            self.state = CircuitState.OPEN
            self.opened_at = self._clock()
            self._probe_in_flight = False
            return
        self.failure_count += 1
        # Why：连续 5 次失败才打开，过滤单次抖动并防止故障依赖造成雪崩。
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = self._clock()
            self._probe_in_flight = False
            logger.warning("工具熔断器打开：连续失败 %s 次", self.failure_count)


@dataclass
class _CacheEntry:
    """内存缓存的内部条目。"""

    data: Any
    expires_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPToolManager:
    """编排工具注册、治理能力和 RAG 召回前处理。

    Args:
        api_key: Anthropic API key；仅在实际执行 Query Rewrite 时惰性创建客户端。
        base_url: 为兼容后续 LLM 扩展保留的可选参数。
        model: 为 Query Rewrite 使用的可选模型名。
        clock: 可注入的单调时钟，用于稳定测试 TTL 和熔断冷却。
        client: 可选的异步 Anthropic 客户端，主要用于离线测试和依赖注入。

    Returns:
        无；通过 ``call``/``execute`` 执行已注册工具。
    """

    MAX_CACHE_SIZE = 5000
    CACHE_EVICT_COUNT = 1250
    KNOWLEDGE_CACHE_TTL = 300

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        clock: Callable[[], float] = time.monotonic,
        client: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client = client
        self._clock = clock
        self._tools: Dict[str, Tool] = {}
        self._cache: "OrderedDict[str, _CacheEntry]" = OrderedDict()

    def register(self, tool: Tool) -> None:
        """注册或覆盖同名工具。"""
        if not isinstance(tool, Tool):
            raise TypeError("tool 必须是 Tool 实例")
        self._tools[tool.name] = tool
        logger.info("注册工具：%s", tool.name)

    def register_knowledge_tool(self, tool: Tool) -> None:
        """以知识工具约定的 300 秒 TTL 注册工具。"""
        tool.cache_ttl = self.KNOWLEDGE_CACHE_TTL
        self.register(tool)

    def unregister(self, name: str) -> None:
        """注销工具及其后续可调用入口。"""
        self._tools.pop(name, None)

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
    ) -> ToolResult:
        """按固定流水线执行工具并返回结构化结果。

        Args:
            name: 已注册的工具名。
            params: 待校验的 JSON 参数对象。
            context: 当前请求上下文，不参与缓存 key。
            use_cache: 是否允许读取和写入成功结果缓存。

        Returns:
            ``ToolResult``；任何主 handler 异常都会被隔离并进入 fallback 或失败结果。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name=name, success=False, data=None, error=f"工具不存在: {name}")

        # 1. Cache 查询：只有成功结果且 TTL 大于 0 时才允许命中。
        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params)
            if cached is not None:
                tool.stats.total += 1
                tool.stats.success += 1
                return ToolResult(
                    name=name,
                    success=True,
                    data=copy.deepcopy(cached.data),
                    cached=True,
                    metadata=dict(cached.metadata),
                )

        # 2. CircuitBreaker 状态判定：OPEN 冷却内不接触主 handler。
        if not tool.breaker.allow():
            tool.stats.total += 1
            tool.stats.failed += 1
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, context, error, 0.0)

        # 3. required/type 校验失败属于输入错误，不污染连续失败计数。
        try:
            self._validate_params(tool, params)
        except ValueError as exc:
            tool.stats.total += 1
            tool.stats.failed += 1
            return ToolResult(name=name, success=False, data=None, error=str(exc))

        started_at = self._clock()
        tool.stats.total += 1
        try:
            # 4. Why：所有主 handler 都必须受 timeout 保护，避免单个依赖阻塞事件循环。
            data = await asyncio.wait_for(
                self._run_handler(tool, params, context),
                timeout=tool.timeout,
            )
        except asyncio.TimeoutError:
            latency_ms = self._elapsed_ms(started_at)
            error = f"执行超时: {tool.timeout} 秒"
            self._record_failure(tool, latency_ms)
            logger.warning("工具超时：%s", name)
            return await self._fallback_result(tool, params, context, error, latency_ms)
        except Exception as exc:
            latency_ms = self._elapsed_ms(started_at)
            error = str(exc) or exc.__class__.__name__
            self._record_failure(tool, latency_ms)
            logger.warning("工具异常：%s，%s", name, error)
            return await self._fallback_result(tool, params, context, error, latency_ms)

        latency_ms = self._elapsed_ms(started_at)
        self._record_success(tool, latency_ms)
        result = self._normalize_result(name, data, latency_ms)

        # 5. 成功结果才允许写入缓存；fallback 和失败结果永不缓存。
        if use_cache and tool.cache_ttl > 0 and result.success:
            self._set_cache(name, params, result.data, tool.cache_ttl, result.metadata)
        return result

    async def execute(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
    ) -> ToolResult:
        """``call`` 的语义别名，便于后续 Agent 调度层使用。"""
        return await self.call(name, params, context, use_cache=use_cache)

    async def call_tool(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
    ) -> ToolResult:
        """``call`` 的兼容别名。"""
        return await self.call(name, params, context, use_cache=use_cache)

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        """使用真实单调时钟计算耗时，避免测试时钟倒退造成负数。"""
        return max(0.0, (time.monotonic() - started_at) * 1000.0)

    @staticmethod
    def _normalize_result(name: str, data: Any, latency_ms: float) -> ToolResult:
        """将 handler 返回值统一成 ToolResult。"""
        if isinstance(data, ToolResult):
            data.name = name
            data.latency_ms = latency_ms
            return data
        return ToolResult(name=name, success=True, data=data, latency_ms=latency_ms)

    def _record_success(self, tool: Tool, latency_ms: float) -> None:
        """记录主 handler 成功并复位该工具熔断连续失败数。"""
        tool.stats.success += 1
        tool.stats.total_latency_ms += latency_ms
        tool.stats.consecutive_fails = 0
        tool.breaker.record_success()

    def _record_failure(self, tool: Tool, latency_ms: float) -> None:
        """记录主 handler 失败并推进熔断器。"""
        tool.stats.failed += 1
        tool.stats.total_latency_ms += latency_ms
        tool.stats.consecutive_fails += 1
        tool.breaker.record_failure()

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
        latency_ms: float,
    ) -> ToolResult:
        """执行 fallback，并保留主调用错误，不把建议伪装成真实成功。"""
        if tool.fallback is None:
            return ToolResult(name=tool.name, success=False, data=None, error=error, latency_ms=latency_ms)
        try:
            data = await self._run_fallback(tool.fallback, params, context, error)
            return ToolResult(
                name=tool.name,
                success=True,
                data=data,
                error=error,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            fallback_error = f"{error}; fallback失败: {exc}"
            logger.warning("工具 fallback 失败：%s，%s", tool.name, fallback_error)
            return ToolResult(
                name=tool.name,
                success=False,
                data=None,
                error=fallback_error,
                latency_ms=latency_ms,
            )

    async def _run_handler(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """调用 async/sync handler；同步 handler 放入线程避免阻塞事件循环。"""
        return await self._invoke_callable(tool.handler, (params, context), context_name="context")

    async def _run_fallback(
        self,
        fallback: Callable[..., Any],
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
    ) -> Any:
        """调用兼容一到三参数形式的 fallback。"""
        return await self._invoke_callable(
            fallback,
            (params, context, error),
            context_name="context",
            error_name="error",
        )

    @staticmethod
    async def _invoke_callable(
        callback: Callable[..., Any],
        args: tuple[Any, ...],
        *,
        context_name: str,
        error_name: str = "",
    ) -> Any:
        """根据函数签名选择兼容参数数量，并等待异步结果。"""
        if callback is None:
            raise RuntimeError("handler 未配置")
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            # 无法解析签名时保守地按完整参数调用，真实 handler 异常不在这里吞掉。
            result = callback(*args)
        else:
            positional = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_varargs = any(
                parameter.kind is inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
            count = len(args) if has_varargs else min(len(args), len(positional))
            selected = args[:count]
            if error_name and count >= 2 and positional[1].name == error_name:
                selected = (args[0], args[2])
            # 只在签名已确定后调用，避免吞掉 handler 自己抛出的 TypeError。
            result = callback(*selected)
        if inspect.isawaitable(result):
            return await result
        return result

    def _cache_key(self, name: str, params: Dict[str, Any]) -> str:
        """生成稳定缓存 key；context 不进入 key，避免同一工具重复缓存。"""
        try:
            payload = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            payload = repr(params)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{name}:{digest}"

    def _get_cache(self, name: str, params: Dict[str, Any]) -> Optional[_CacheEntry]:
        """读取未过期缓存；过期条目会被即时删除。"""
        key = self._cache_key(name, params)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if self._clock() >= entry.expires_at:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return entry

    def _set_cache(
        self,
        name: str,
        params: Dict[str, Any],
        data: Any,
        ttl: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """写入成功结果并按 5000/1250 容量策略 FIFO 批量淘汰。"""
        if ttl <= 0:
            return
        key = self._cache_key(name, params)
        self._cache[key] = _CacheEntry(
            data=copy.deepcopy(data),
            expires_at=self._clock() + float(ttl),
            metadata=dict(metadata or {}),
        )
        self._cache.move_to_end(key)
        if len(self._cache) > self.MAX_CACHE_SIZE:
            # Why：删除最早 1250 条可避免每次超限只删一条造成频繁抖动。
            for _ in range(self.CACHE_EVICT_COUNT):
                self._cache.popitem(last=False)

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
        """执行 JSON Schema 的 required、type 和 additionalProperties 校验。"""
        if not isinstance(params, dict):
            raise ValueError(f"工具 {tool.name} 参数必须是 object")
        schema = tool.schema if isinstance(tool.schema, dict) else {}
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for field_name in required:
            if field_name not in params:
                raise ValueError(f"工具 {tool.name} 缺少必需参数: {field_name}")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(params) - set(properties))
            if unknown:
                raise ValueError(f"工具 {tool.name} 包含未声明参数: {unknown[0]}")
        for key, value in params.items():
            if key not in properties:
                continue
            expected_type = properties[key].get("type") if isinstance(properties[key], dict) else None
            if expected_type and not self._matches_type(value, expected_type):
                raise ValueError(
                    f"工具 {tool.name} 参数 {key} 类型错误: 期望 {expected_type}，实际 {type(value).__name__}"
                )

    @staticmethod
    def _matches_type(value: Any, expected_type: str) -> bool:
        """判断 JSON Schema 基础类型，避免 bool 被误判为 number/integer。"""
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type == "array":
            return isinstance(value, list)
        if expected_type == "object":
            return isinstance(value, dict)
        if expected_type == "null":
            return value is None
        return True

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """返回所有工具的统计快照。"""
        return {
            name: {
                "total": tool.stats.total,
                "success": tool.stats.success,
                "failed": tool.stats.failed,
                "success_rate": round(tool.stats.success_rate, 3),
                "avg_latency_ms": round(tool.stats.avg_latency_ms, 1),
                "consecutive_fails": tool.stats.consecutive_fails,
                "circuit_state": tool.breaker.state.value,
            }
            for name, tool in self._tools.items()
        }

    async def rewrite_query(self, query: str, n: int = 3) -> List[str]:
        """将原始知识检索 query 改写为多个角度并保留原 query。

        Args:
            query: 用户原始检索问题；该文本始终作为返回列表第一项。
            n: LLM 最多生成的改写数量，默认 3。

        Returns:
            去重后的 query 列表；模型不可用或输出非法时只返回原始 query。

        业务逻辑：改写只使用当前 query，不携带会话历史或用户画像，避免
        将私有上下文扩散到知识库检索；失败时确定性回退，保证主链路可继续。
        """
        original_query = str(query or "")
        try:
            rewrite_count = int(n)
        except (TypeError, ValueError):
            rewrite_count = 0
        if rewrite_count <= 0:
            return [original_query]

        client = self._get_llm_client()
        if client is None:
            logger.warning("查询改写不可用，使用原始 query")
            return [original_query]

        # Why：固定 n=3 是改写召回覆盖率与额外 LLM/检索成本之间的边界，
        # 不携带 history/context 可避免会话隐私污染公共知识检索。
        prompt = (
            f"将以下用户查询改写为 {rewrite_count} 个不同角度的搜索子查询，用于检索知识库。\n"
            "要求：每个子查询角度不同，覆盖原始问题的不同方面；只返回 JSON 对象，"
            f"格式为 {{\"queries\": [\"子查询1\", \"子查询2\"]}}，最多返回 {rewrite_count} 条。\n"
            f"原始查询：{original_query}"
        )
        try:
            response = await client.messages.create(
                model=self.model or "claude-3-5-sonnet-20241022",
                max_tokens=256,
                temperature=0.3,
                extra_body={
                    "thinking": {"type": "disabled"},
                    "response_format": {"type": "json_object"},
                },
                messages=[{"role": "user", "content": prompt}],
            )
            raw = self._extract_text_content(getattr(response, "content", None))
            payload = self._parse_json_text(raw)
            queries = payload.get("queries") if isinstance(payload, dict) else None
            if (
                not isinstance(queries, list)
                or not queries
                or len(queries) > rewrite_count
                or any(not isinstance(item, str) or not item.strip() for item in queries)
            ):
                raise ValueError("查询改写结果必须是非空字符串数组，且不超过指定数量")

            # Why：保留原始 query 可防止改写遗漏关键实体；dict 顺序同时保证结果稳定。
            normalized = [item.strip() for item in queries]
            return list(dict.fromkeys([original_query] + normalized))
        except Exception as exc:
            logger.warning("查询改写失败，使用原始 query：%s", exc)
            return [original_query]

    async def search_with_rewrite(
        self,
        tool_name: str,
        query: str,
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """执行 Query Rewrite、并行召回和稳定去重，不执行 Rerank。

        Args:
            tool_name: 已注册的知识检索工具名。
            query: 用户原始检索问题。
            top_k: 下游希望的结果规模；每路召回至少使用 5 条。
            context: 传给底层 Tool handler 的可选上下文。

        Returns:
            ``ToolResult``；成功时 data 是完整去重候选集，失败时 data 为空列表。

        业务逻辑：所有改写 query 通过 ``asyncio.gather`` 并行调用基础 Tool，
        单路错误局部隔离；本阶段不按 score 截断，最终排序和 Top-K 留给后续 Rerank。
        """
        sub_queries = await self.rewrite_query(query, n=3)
        # Why：至少召回 5 条候选才能为后续去重和 Rerank 留出足够覆盖面。
        recall_k = max(int(top_k), 5)
        tasks = [
            self.call(
                tool_name,
                {"query": sub_query, "top_k": recall_k},
                context,
                use_cache=True,
            )
            for sub_query in sub_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        merged_order: List[Tuple[str, str]] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("子查询召回异常，跳过该路：%s", result)
                continue
            if not isinstance(result, ToolResult) or not result.success or not isinstance(result.data, list):
                continue
            for item in result.data:
                if not isinstance(item, dict):
                    continue
                key = self._retrieval_item_key(item, len(merged_order))
                if key not in merged_by_key:
                    merged_order.append(key)
                    merged_by_key[key] = item
                elif self._item_score(item) > self._item_score(merged_by_key[key]):
                    merged_by_key[key] = item

        merged = [merged_by_key[key] for key in merged_order]
        if not merged:
            return ToolResult(
                name=tool_name,
                success=False,
                data=[],
                error="所有子查询均无结果",
                metadata={"original_query": query, "queries": sub_queries, "recall_k": recall_k},
            )

        return ToolResult(
            name=tool_name,
            success=True,
            data=merged,
            reranked=False,
            rerank_degraded=False,
            metadata={
                "original_query": query,
                "queries": sub_queries,
                "recall_k": recall_k,
                "candidate_count": len(merged),
            },
        )

    def _get_llm_client(self) -> Optional[Any]:
        """惰性获取 Anthropic client；无 key 时不创建网络客户端。"""
        if self._client is not None:
            return self._client
        if not self.api_key:
            return None
        try:
            from anthropic import AsyncAnthropic

            kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncAnthropic(**kwargs)
            return self._client
        except Exception as exc:
            logger.warning("Anthropic client 初始化失败：%s", exc)
            return None

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        """从 Anthropic 文本 block 或测试替身中提取文本。"""
        if isinstance(content, str):
            return content
        blocks = content if isinstance(content, list) else [content]
        texts: List[str] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block.get("text") or ""))
            elif getattr(block, "type", None) == "text":
                texts.append(str(getattr(block, "text", "") or ""))
        return "\n".join(texts).strip()

    @staticmethod
    def _parse_json_text(raw: str) -> Any:
        """解析普通 JSON 或 Markdown fenced JSON 文本。"""
        text = str(raw or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)

    @staticmethod
    def _retrieval_item_key(item: Dict[str, Any], fallback_index: int) -> Tuple[str, str]:
        """按稳定字段优先级生成检索候选 identity。"""
        for field_name in ("source_chunk_id", "content_hash", "runtime_id"):
            value = item.get(field_name)
            if value not in (None, ""):
                return field_name, str(value)
        content = item.get("content")
        if content not in (None, ""):
            digest = hashlib.sha256(str(content).encode("utf-8")).hexdigest()
            return "content_hash", digest
        # 无 identity 的候选不能互相合并，否则可能误删不同来源的结果。
        return "unidentified", str(fallback_index)

    @staticmethod
    def _item_score(item: Dict[str, Any]) -> float:
        """读取候选 score，仅用于同一 identity 的较高分代表选择。"""
        try:
            return float(item.get("score", float("-inf")))
        except (TypeError, ValueError):
            return float("-inf")
