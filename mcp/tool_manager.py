"""ShopCare 工具执行基础设施。

本模块属于 MCP/RAG 治理层，仅负责工具契约、参数校验、超时隔离、内存缓存、
熔断状态机和 fallback 语义。它不实现 Query Rewrite、并行召回、LLM Rerank，
也不连接真实订单、支付、退款或人工工单系统。
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
from typing import Any, Callable, Dict, Optional


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
    """编排工具注册、校验、缓存、超时、熔断、统计和 fallback。

    Args:
        api_key: 为兼容后续 LLM 扩展保留的可选参数；Step 4.2 不创建 LLM 客户端。
        base_url: 为兼容后续 LLM 扩展保留的可选参数。
        model: 为兼容后续 LLM 扩展保留的可选模型名。
        clock: 可注入的单调时钟，用于稳定测试 TTL 和熔断冷却。

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
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
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
