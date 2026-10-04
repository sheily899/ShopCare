from __future__ import annotations

import asyncio
import json
from datetime import datetime

from memory.conversation_memory import (
    COMPRESS_AT,
    KEEP_AFTER_COMPRESS,
    TTL_SECONDS,
    WORKING_MAX,
    MemoryContext,
    MemoryManager,
    Message,
    MsgRole,
)


class FakeAsyncRedis:
    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.calls: list[tuple] = []
        self.expirations: dict[str, int] = {}

    async def lpush(self, key: str, value: str) -> int:
        self.calls.append(("lpush", key, value))
        self.lists.setdefault(key, []).insert(0, value)
        return len(self.lists[key])

    async def expire(self, key: str, seconds: int) -> bool:
        self.calls.append(("expire", key, seconds))
        self.expirations[key] = seconds
        return True

    async def llen(self, key: str) -> int:
        self.calls.append(("llen", key))
        return len(self.lists.get(key, []))

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        self.calls.append(("lrange", key, start, end))
        return self.lists.get(key, [])[start : end + 1]


class FailingRedis:
    async def lrange(self, *args, **kwargs):
        raise TimeoutError("redis timeout")


def _run(coroutine):
    return asyncio.run(coroutine)


def test_message_round_trips_timestamp_and_metadata() -> None:
    timestamp = datetime(2026, 10, 4, 12, 30, 45)
    message = Message(
        role=MsgRole.USER,
        content="退款进度",
        timestamp=timestamp,
        metadata={"source": "web", "attempt": 1},
    )

    payload = message.to_dict()
    restored = Message.from_dict(payload)

    assert payload["role"] == "user"
    assert payload["content"] == "退款进度"
    assert restored.role is MsgRole.USER
    assert restored.timestamp == timestamp
    assert restored.metadata == {"source": "web", "attempt": 1}


def test_add_message_uses_working_memory_key_lpush_and_ttl() -> None:
    redis = FakeAsyncRedis()
    manager = MemoryManager(redis_client=redis)

    _run(manager.add_message("u-1", "c-9", MsgRole.USER, "你好"))

    key = "wm:u-1:c-9"
    assert redis.calls[0][0:2] == ("lpush", key)
    stored = json.loads(redis.calls[0][2])
    assert stored["role"] == "user"
    assert stored["content"] == "你好"
    assert ("expire", key, TTL_SECONDS) in redis.calls
    assert manager._wm_key("u-1", "c-9") == key


def test_recent_messages_reverse_lpush_order_back_to_time_order() -> None:
    redis = FakeAsyncRedis()
    manager = MemoryManager(redis_client=redis)

    async def write_messages() -> None:
        await manager.add_message("u", "c", MsgRole.USER, "first")
        await manager.add_message("u", "c", MsgRole.ASSISTANT, "second")
        await manager.add_message("u", "c", MsgRole.USER, "third")

    _run(write_messages())
    messages = _run(manager.get_recent_messages("u", "c"))

    assert [item.content for item in messages] == ["first", "second", "third"]
    assert redis.calls[-1] == ("lrange", "wm:u:c", 0, WORKING_MAX - 1)


def test_reaching_compress_threshold_sets_trigger_signal_without_compressing() -> None:
    redis = FakeAsyncRedis()
    manager = MemoryManager(redis_client=redis)

    async def write_messages() -> None:
        for index in range(COMPRESS_AT):
            await manager.add_message("u", "c", MsgRole.USER, f"m-{index}")

    _run(write_messages())

    assert manager.last_compression_triggered is True
    assert manager.compression_trigger_key == "wm:u:c"
    assert len(redis.lists["wm:u:c"]) == COMPRESS_AT
    assert KEEP_AFTER_COMPRESS == 5


def test_redis_failure_returns_empty_messages_and_logs_safe_path(caplog) -> None:
    manager = MemoryManager(redis_client=FailingRedis())

    with caplog.at_level("ERROR"):
        messages = _run(manager.get_recent_messages("u", "c"))

    assert messages == []
    assert "工作记忆读取失败" in caplog.text


def test_memory_context_prompt_limits_history_and_recent_messages() -> None:
    context = MemoryContext(
        summary="已知用户正在跟进一个售后问题。",
        relevant_history=[f"history-{index}" for index in range(5)],
        recent_messages=[
            Message(MsgRole.USER, f"message-{index}") for index in range(10)
        ],
        user_profile={"preferences": ["中文"]},
    )

    prompt = context.to_prompt_text()

    assert prompt.index("[会话摘要]") < prompt.index("[相关历史]")
    assert "history-0" in prompt and "history-2" in prompt
    assert "history-3" not in prompt and "history-4" not in prompt
    assert "message-2" in prompt and "message-9" in prompt
    assert "message-0" not in prompt and "message-1" not in prompt
