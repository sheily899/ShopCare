"""Step 4.2 工具管理器、内存缓存与熔断器行为测试。"""

from __future__ import annotations

import asyncio
from functools import wraps

from mcp.tool_manager import (
    CircuitBreaker,
    CircuitState,
    MCPToolManager,
    Tool,
)


def _schema(required: list[str] | None = None) -> dict:
    """构造测试用的最小 JSON Schema。"""
    return {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": required or [],
        "additionalProperties": False,
    }


def _run_async(test_function):
    """使用标准库运行异步测试，避免引入未锁定的 pytest-asyncio。"""

    @wraps(test_function)
    def wrapper(*args, **kwargs):
        return asyncio.run(test_function(*args, **kwargs))

    return wrapper


@_run_async
async def test_five_failures_open_breaker_and_successful_probe_closes_it() -> None:
    """连续五次主处理失败后应熔断，冷却后一次探测成功应恢复。"""
    now = [100.0]
    calls = [0]
    probing = [False]

    async def handler(params):
        calls[0] += 1
        if not probing[0]:
            raise RuntimeError("primary failed")
        return {"ok": True}

    breaker = CircuitBreaker(clock=lambda: now[0])
    tool = Tool("demo", "演示工具", _schema(), handler, timeout=1.0)
    tool.breaker = breaker
    manager = MCPToolManager()
    manager.register(tool)

    for _ in range(5):
        result = await manager.call("demo", {})
        assert result.success is False
    assert breaker.state is CircuitState.OPEN
    assert calls[0] == 5

    blocked = await manager.call("demo", {})
    assert blocked.success is False
    assert calls[0] == 5

    now[0] += 60.0
    probing[0] = True
    recovered = await manager.call("demo", {})
    assert recovered.success is True
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0


@_run_async
async def test_half_open_probe_failure_reopens_breaker() -> None:
    """半开探测失败必须重新打开熔断器并重置冷却起点。"""
    now = [0.0]

    async def handler(params):
        raise RuntimeError("probe failed")

    breaker = CircuitBreaker(clock=lambda: now[0])
    tool = Tool("probe", "探测工具", _schema(), handler)
    tool.breaker = breaker
    manager = MCPToolManager()
    manager.register(tool)

    for _ in range(5):
        await manager.call("probe", {})
    first_opened_at = breaker.opened_at
    now[0] += 60.0
    result = await manager.call("probe", {})

    assert result.success is False
    assert breaker.state is CircuitState.OPEN
    assert breaker.opened_at == now[0]
    assert breaker.opened_at != first_opened_at


@_run_async
async def test_timeout_counts_as_failure_and_does_not_escape() -> None:
    """wait_for 超时应返回失败结果并增加连续失败计数。"""

    async def slow_handler(params):
        await asyncio.sleep(0.05)
        return "late"

    breaker = CircuitBreaker()
    tool = Tool("slow", "慢工具", _schema(), slow_handler, timeout=0.001)
    tool.breaker = breaker
    manager = MCPToolManager()
    manager.register(tool)

    result = await manager.call("slow", {})

    assert result.success is False
    assert "超时" in (result.error or "")
    assert breaker.failure_count == 1


@_run_async
async def test_invalid_required_or_type_parameters_do_not_touch_breaker() -> None:
    """参数不合法时应在 handler 前失败，且不计入熔断连续失败。"""

    async def handler(params):
        raise AssertionError("参数校验失败时不应执行 handler")

    breaker = CircuitBreaker()
    tool = Tool("validate", "校验工具", _schema(["query"]), handler)
    tool.breaker = breaker
    manager = MCPToolManager()
    manager.register(tool)

    missing = await manager.call("validate", {})
    wrong_type = await manager.call("validate", {"query": 42})

    assert missing.success is False
    assert "缺少必需参数" in (missing.error or "")
    assert wrong_type.success is False
    assert "类型错误" in (wrong_type.error or "")
    assert breaker.failure_count == 0


@_run_async
async def test_successful_result_is_cached_until_ttl_expires() -> None:
    """成功结果在 TTL 内命中缓存，过期后重新执行 handler。"""
    now = [10.0]
    calls = [0]

    async def handler(params):
        calls[0] += 1
        return {"calls": calls[0]}

    tool = Tool("cached", "缓存工具", _schema(), handler, cache_ttl=5)
    manager = MCPToolManager(clock=lambda: now[0])
    manager.register(tool)

    first = await manager.call("cached", {})
    second = await manager.call("cached", {})
    now[0] += 6
    third = await manager.call("cached", {})

    assert first.cached is False
    assert second.cached is True
    assert second.data == {"calls": 1}
    assert third.cached is False
    assert calls[0] == 2


def test_cache_evicts_oldest_1250_after_capacity_5000() -> None:
    """缓存超过 5000 条时应批量删除最早写入的 1250 条。"""
    manager = MCPToolManager(clock=lambda: 0.0)

    for index in range(5001):
        manager._set_cache("bulk", {"index": index}, index, ttl=100)

    assert len(manager._cache) == 3751
    assert manager._cache_key("bulk", {"index": 0}) not in manager._cache
    assert manager._cache_key("bulk", {"index": 1250}) in manager._cache


@_run_async
async def test_fallback_success_preserves_primary_error() -> None:
    """fallback 成功也必须保留主 handler 的异常信息，不能伪装真实成功。"""

    async def handler(params):
        raise RuntimeError("外部服务不可用")

    async def fallback(params, context, error):
        assert error == "外部服务不可用"
        return {"advice": "请稍后重试"}

    tool = Tool(
        "fallback",
        "降级工具",
        _schema(),
        handler,
        fallback=fallback,
    )
    manager = MCPToolManager()
    manager.register(tool)

    result = await manager.call("fallback", {})

    assert result.success is True
    assert result.data == {"advice": "请稍后重试"}
    assert result.error == "外部服务不可用"
    assert result.cached is False
