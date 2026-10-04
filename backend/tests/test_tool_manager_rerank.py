"""Step 4.4 LLM Rerank 与降级状态机测试。

本模块只测试候选切片的结构化重排，不访问真实 Anthropic 服务，
也不引入 Cross-Encoder 或 score threshold。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from mcp.tool_manager import MCPToolManager


class FakeMessages:
    """记录 Tool Use 请求并返回预设响应的异步消息替身。"""

    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeClient:
    """只实现 ``messages.create`` 的 Anthropic 客户端替身。"""

    def __init__(self, response):
        self.messages = FakeMessages(response)


def tool_use_response(indices):
    """构造合法或非法的 rerank_results Tool Use 响应。"""
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="rerank_results",
                input={"indices": indices},
            )
        ]
    )


def candidates():
    """构造具有稳定身份的候选切片。"""
    return [
        {"source_chunk_id": "chunk-0", "content": "零", "score": 0.70},
        {"source_chunk_id": "chunk-1", "content": "一", "score": 0.90},
        {"source_chunk_id": "chunk-2", "content": "二", "score": 0.80},
    ]


def test_rerank_skips_llm_when_candidate_count_does_not_exceed_top_k() -> None:
    """候选数量不超过 top_k 时直接保留候选集且不触发 LLM。"""
    client = FakeClient(RuntimeError("LLM should not be called"))
    manager = MCPToolManager(api_key="test-key", client=client, model="test-model")
    rows = candidates()[:2]

    result = asyncio.run(manager.rerank_results("查询", rows, top_k=2))

    assert result.success is True
    assert result.data == rows
    assert result.reranked is False
    assert result.rerank_degraded is False
    assert client.messages.calls == []


def test_rerank_applies_valid_index_permutation_and_returns_top_k() -> None:
    """合法完整置换应按模型顺序重排并截取前 top_k。"""
    client = FakeClient(tool_use_response([1, 0, 2]))
    manager = MCPToolManager(api_key="test-key", client=client, model="test-model")

    result = asyncio.run(manager.rerank_results("查询", candidates(), top_k=2))

    assert result.success is True
    assert [item["source_chunk_id"] for item in result.data] == ["chunk-1", "chunk-0"]
    assert result.reranked is True
    assert result.rerank_degraded is False
    call = client.messages.calls[0]
    assert call["model"] == "test-model"
    assert call["temperature"] == 0
    assert call["max_tokens"] == 256
    assert call["tool_choice"] == {"type": "tool", "name": "rerank_results"}
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert call["tools"][0]["name"] == "rerank_results"
    assert call["tools"][0]["input_schema"]["required"] == ["indices"]


def test_rerank_api_error_falls_back_to_original_physical_order() -> None:
    """LLM 网络异常不能中断检索，应返回原顺序前 top_k 并标记降级。"""
    client = FakeClient(RuntimeError("network down"))
    manager = MCPToolManager(api_key="test-key", client=client)

    result = asyncio.run(manager.rerank_results("查询", candidates(), top_k=2))

    assert result.success is True
    assert [item["source_chunk_id"] for item in result.data] == ["chunk-0", "chunk-1"]
    assert result.reranked is False
    assert result.rerank_degraded is True
    assert "network down" in (result.error or "")


def test_rerank_invalid_permutation_falls_back_without_crashing() -> None:
    """越界、重复或缺失 index 都必须走确定性原顺序降级。"""
    invalid_permutations = ([1, 0, 99], [1, 1, 2], [1, 0])

    for indices in invalid_permutations:
        client = FakeClient(tool_use_response(indices))
        manager = MCPToolManager(api_key="test-key", client=client)

        result = asyncio.run(manager.rerank_results("查询", candidates(), top_k=2))

        assert result.success is True
        assert [item["source_chunk_id"] for item in result.data] == ["chunk-0", "chunk-1"]
        assert result.reranked is False
        assert result.rerank_degraded is True
        assert result.error


def test_rerank_missing_tool_use_falls_back_and_does_not_add_score_threshold() -> None:
    """缺少 Tool Use 时降级，候选 score 不参与硬过滤。"""
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="{}")])
    client = FakeClient(response)
    manager = MCPToolManager(api_key="test-key", client=client)
    rows = [
        {"source_chunk_id": "low-score", "content": "仍保留", "score": -10.0},
        {"source_chunk_id": "second", "content": "也保留", "score": 0.0},
        {"source_chunk_id": "third", "content": "候选", "score": 0.2},
    ]

    result = asyncio.run(manager.rerank_results("查询", rows, top_k=2))

    assert [item["source_chunk_id"] for item in result.data] == ["low-score", "second"]
    assert result.rerank_degraded is True
