"""Step 4.3 查询改写、并行召回与稳定去重测试。

本模块只验证 RAG 召回前处理，不测试或调用 Step 4.4 的 LLM Rerank。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from mcp.tool_manager import MCPToolManager, ToolResult


class FakeMessages:
    """记录 Anthropic 请求并返回预设内容的异步消息替身。"""

    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        if isinstance(self.response, dict):
            return SimpleNamespace(content=[self.response])
        return SimpleNamespace(content=[{"type": "text", "text": self.response}])


class FakeClient:
    """只实现 ``messages.create`` 的 Anthropic 客户端替身。"""

    def __init__(self, response):
        self.messages = FakeMessages(response)


def build_manager(response):
    """构造注入 fake client 的工具管理器。"""
    return MCPToolManager(
        api_key="test-key",
        model="test-model",
        client=FakeClient(response),
    )


def test_query_rewrite_keeps_original_and_sends_bounded_llm_request() -> None:
    """改写必须保留原 query，并使用固定的模型调用参数。"""
    manager = build_manager('{"queries":["退款审核时效", "退款到账周期"]}')

    result = asyncio.run(manager.rewrite_query("退款多久到账？"))

    assert result == ["退款多久到账？", "退款审核时效", "退款到账周期"]
    call = manager._client.messages.calls[0]
    assert call["model"] == "test-model"
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 256
    assert call["extra_body"] == {
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    prompt = call["messages"][0]["content"]
    assert "退款多久到账？" in prompt
    assert "history" not in prompt
    assert "user_profile" not in prompt


def test_query_rewrite_accepts_fenced_json_and_deduplicates() -> None:
    """Markdown fenced JSON 可解析，重复改写不能重复召回。"""
    manager = build_manager('```json\n{"queries":["原始问题", "改写问题", "改写问题"]}\n```')

    result = asyncio.run(manager.rewrite_query("原始问题"))

    assert result == ["原始问题", "改写问题"]


def test_query_rewrite_invalid_shape_always_falls_back_to_original() -> None:
    """非法 JSON 结构、空文本和超量改写都必须安全降级。"""
    invalid_responses = [
        "not-json",
        '{"queries": []}',
        '{"queries": ["有效", 42]}',
        '{"queries": ["一", "二", "三", "四"]}',
        '{"queries": ["   "]}',
    ]

    for response in invalid_responses:
        manager = build_manager(response)
        assert asyncio.run(manager.rewrite_query("原始问题")) == ["原始问题"]


def test_query_rewrite_exception_falls_back_without_raising() -> None:
    """Anthropic 网络异常不能冒泡到用户对话主链路。"""
    manager = build_manager(RuntimeError("network down"))

    result = asyncio.run(manager.rewrite_query("原始问题"))

    assert result == ["原始问题"]


def test_parallel_recall_starts_all_queries_and_uses_minimum_recall_k() -> None:
    """所有改写 query 应并发召回，并把 top_k=2 提升为 recall_k=5。"""
    manager = build_manager("unused")
    started = 0
    release = asyncio.Event()
    ready = asyncio.Event()
    calls = []

    async def fake_rewrite(query: str, n: int = 3):
        return ["q1", "q2", "q3"]

    async def fake_call(name, params, context=None, *, use_cache=True):
        nonlocal started
        started += 1
        calls.append((name, params, context, use_cache))
        if started == 3:
            ready.set()
        await release.wait()
        return ToolResult(name=name, success=True, data=[{"source_chunk_id": params["query"]}])

    manager.rewrite_query = fake_rewrite
    manager.call = fake_call

    async def run():
        task = asyncio.create_task(
            manager.search_with_rewrite("knowledge", "原始问题", top_k=2, context={"trace": "t1"})
        )
        await asyncio.wait_for(ready.wait(), timeout=1.0)
        assert started == 3
        release.set()
        return await task

    result = asyncio.run(run())

    assert result.success is True
    assert [item[1]["query"] for item in calls] == ["q1", "q2", "q3"]
    assert all(item[1]["top_k"] == 5 for item in calls)
    assert all(item[3] is True for item in calls)
    assert result.reranked is False
    assert result.rerank_degraded is False


def test_recall_isolates_partial_failures_and_returns_successful_candidates() -> None:
    """单路异常或失败不能阻断其他子查询的有效召回。"""
    manager = build_manager("unused")

    async def fake_rewrite(query: str, n: int = 3):
        return ["broken", "failed", "healthy"]

    async def fake_call(name, params, context=None, *, use_cache=True):
        if params["query"] == "broken":
            raise RuntimeError("one query failed")
        if params["query"] == "failed":
            return ToolResult(name=name, success=False, data=[], error="no result")
        return ToolResult(name=name, success=True, data=[{"source_chunk_id": "healthy"}])

    manager.rewrite_query = fake_rewrite
    manager.call = fake_call

    result = asyncio.run(manager.search_with_rewrite("knowledge", "原始问题"))

    assert result.success is True
    assert result.data == [{"source_chunk_id": "healthy"}]


def test_all_recall_failures_return_explicit_empty_failure() -> None:
    """全部子查询没有有效结果时不得伪造知识库内容。"""
    manager = build_manager("unused")

    async def fake_rewrite(query: str, n: int = 3):
        return ["q1", "q2"]

    async def fake_call(name, params, context=None, *, use_cache=True):
        return ToolResult(name=name, success=False, data=[], error="unavailable")

    manager.rewrite_query = fake_rewrite
    manager.call = fake_call

    result = asyncio.run(manager.search_with_rewrite("knowledge", "原始问题"))

    assert result.success is False
    assert result.data == []
    assert "所有子查询均无结果" in (result.error or "")


def test_dedup_uses_identity_priority_and_keeps_highest_score() -> None:
    """多级稳定身份去重时保留相同 chunk 的最高分代表。"""
    manager = build_manager("unused")

    async def fake_rewrite(query: str, n: int = 3):
        return ["q1", "q2"]

    async def fake_call(name, params, context=None, *, use_cache=True):
        if params["query"] == "q1":
            data = [
                {"source_chunk_id": "chunk-1", "score": 0.2, "content": "source"},
                {"content_hash": "hash-1", "score": 0.2, "content": "content-hash"},
                {"runtime_id": "runtime-1", "score": 0.1, "content": "runtime"},
                {"content": "same-content", "score": 0.1},
                {"score": 0.1, "title": "unidentified-1"},
            ]
        else:
            data = [
                {"source_chunk_id": "chunk-1", "score": 0.9, "content": "source"},
                {"content_hash": "hash-1", "score": 0.8, "content": "content-hash"},
                {"runtime_id": "runtime-1", "score": 0.7, "content": "runtime"},
                {"content": "same-content", "score": 0.6},
                {"score": 0.2, "title": "unidentified-2"},
            ]
        return ToolResult(name=name, success=True, data=data)

    manager.rewrite_query = fake_rewrite
    manager.call = fake_call

    result = asyncio.run(manager.search_with_rewrite("knowledge", "原始问题", top_k=1))

    assert result.success is True
    assert [item.get("title", item.get("source_chunk_id", item.get("content"))) for item in result.data] == [
        "chunk-1",
        "content-hash",
        "runtime",
        "same-content",
        "unidentified-1",
        "unidentified-2",
    ]
    assert [item["score"] for item in result.data[:4]] == [0.9, 0.8, 0.7, 0.6]
    assert len(result.data) == 6


def test_recall_does_not_rerank_or_truncate_deduplicated_candidates() -> None:
    """Step 4.3 必须保留完整候选集，把 Rerank 留给 Step 4.4。"""
    manager = build_manager("unused")

    async def fake_rewrite(query: str, n: int = 3):
        return ["q1"]

    async def fake_call(name, params, context=None, *, use_cache=True):
        return ToolResult(
            name=name,
            success=True,
            data=[
                {"source_chunk_id": "a"},
                {"source_chunk_id": "b"},
                {"source_chunk_id": "c"},
            ],
        )

    manager.rewrite_query = fake_rewrite
    manager.call = fake_call

    result = asyncio.run(manager.search_with_rewrite("knowledge", "原始问题", top_k=1))

    assert result.success is True
    assert len(result.data) == 3
    assert result.reranked is False
    assert result.rerank_degraded is False
