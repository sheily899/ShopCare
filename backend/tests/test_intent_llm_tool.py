from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

try:
    from httpx import TimeoutException
except ModuleNotFoundError:  # 依赖基线尚未安装时仍保持异常路径可测
    TimeoutException = TimeoutError

from intent.intent_recognizer import (
    CLASSIFY_INTENT_TOOL,
    IntentCategory,
    _recognize_with_llm,
)


def _client_with_response(content):
    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock())
    client.messages.create.return_value = SimpleNamespace(content=content)
    return client


def _run(coroutine):
    return asyncio.run(coroutine)


def test_llm_tool_use_success_returns_structured_intent_result() -> None:
    client = _client_with_response(
        [
            SimpleNamespace(
                type="tool_use",
                name="classify_intent",
                input={
                    "intent": "refund",
                    "confidence": 0.91,
                    "reasoning": "用户明确询问退款流程",
                },
            )
        ]
    )

    result = _run(
        _recognize_with_llm(
            "我想申请退款",
            history=[{"role": "user", "content": "订单刚刚送达"}],
            client=client,
            model="test-model",
        )
    )

    assert result.intent is IntentCategory.REFUND
    assert result.confidence == 0.91
    assert result.reasoning == "用户明确询问退款流程"
    assert result.success is True


def test_llm_call_uses_strict_classify_tool_contract() -> None:
    client = _client_with_response(
        [
            {
                "type": "tool_use",
                "name": "classify_intent",
                "input": {
                    "intent": "query",
                    "confidence": 0.8,
                    "reasoning": "用户在询问信息",
                },
            }
        ]
    )

    _run(_recognize_with_llm("怎么查询", client=client, model="test-model"))

    call = client.messages.create.await_args.kwargs
    assert call["tool_choice"] == {"type": "tool", "name": "classify_intent"}
    assert call["tools"] == [CLASSIFY_INTENT_TOOL]
    schema = call["tools"][0]["input_schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["intent", "confidence", "reasoning"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]["intent"]["enum"]) == {
        category.value for category in IntentCategory
    }
    assert call["system"]
    assert call["messages"] == [{"role": "user", "content": "怎么查询"}]


def test_unknown_intent_degrades_to_other_without_raising() -> None:
    client = _client_with_response(
        [
            {
                "type": "tool_use",
                "name": "classify_intent",
                "input": {
                    "intent": "unknown_intent",
                    "confidence": 0.9,
                    "reasoning": "无法确定",
                },
            }
        ]
    )

    result = _run(_recognize_with_llm("未知请求", client=client))

    assert result.intent is IntentCategory.OTHER
    assert result.confidence == 0.0
    assert result.success is False
    assert "LLM 失败" in result.reasoning


def test_out_of_range_confidence_is_clamped_to_safe_range() -> None:
    low_client = _client_with_response(
        [
            {
                "type": "tool_use",
                "name": "classify_intent",
                "input": {
                    "intent": "query",
                    "confidence": -0.5,
                    "reasoning": "用户在询问信息",
                },
            }
        ]
    )
    high_client = _client_with_response(
        [
            {
                "type": "tool_use",
                "name": "classify_intent",
                "input": {
                    "intent": "query",
                    "confidence": 1.5,
                    "reasoning": "用户在询问信息",
                },
            }
        ]
    )

    low_result = _run(_recognize_with_llm("怎么查询", client=low_client))
    high_result = _run(_recognize_with_llm("怎么查询", client=high_client))

    assert low_result.success is True
    assert low_result.confidence == 0.0
    assert high_result.success is True
    assert high_result.confidence == 1.0


def test_missing_tool_use_or_required_field_degrades_to_other() -> None:
    no_tool_client = _client_with_response(
        [SimpleNamespace(type="text", text="query")]
    )
    missing_field_client = _client_with_response(
        [
            {
                "type": "tool_use",
                "name": "classify_intent",
                "input": {"intent": "query", "confidence": 0.5},
            }
        ]
    )

    no_tool_result = _run(_recognize_with_llm("查询", client=no_tool_client))
    missing_field_result = _run(
        _recognize_with_llm("查询", client=missing_field_client)
    )

    assert no_tool_result.intent is IntentCategory.OTHER
    assert no_tool_result.success is False
    assert missing_field_result.intent is IntentCategory.OTHER
    assert missing_field_result.success is False


def test_api_timeout_is_caught_and_never_escapes() -> None:
    client = SimpleNamespace()
    client.messages = SimpleNamespace(
        create=AsyncMock(side_effect=TimeoutException("timed out"))
    )

    result = _run(_recognize_with_llm("页面打不开", client=client))

    assert result.intent is IntentCategory.OTHER
    assert result.confidence == 0.0
    assert result.success is False
    assert "LLM 失败" in result.reasoning
