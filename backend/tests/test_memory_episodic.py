import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from memory.conversation_memory import (
    COMPRESS_AT,
    EPISODIC_FULL_TEXT_MAX_CHARS,
    HISTORY_TOP_K,
    KEEP_AFTER_COMPRESS,
    SUMMARY_MAX_CHARS,
    MemoryManager,
    MsgRole,
)


class EpisodicRedis:
    def __init__(self):
        self.lists = {}
        self.strings = {}
        self.expirations = {}

    async def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)
        return len(self.lists[key])

    async def expire(self, key, seconds):
        self.expirations[key] = seconds
        return True

    async def llen(self, key):
        return len(self.lists.get(key, []))

    async def lrange(self, key, start, end):
        values = self.lists.get(key, [])
        return values[start : end + 1 if end >= 0 else None]

    async def setex(self, key, seconds, value):
        self.strings[key] = value
        self.expirations[key] = seconds
        return True

    async def delete(self, key):
        self.lists.pop(key, None)
        self.strings.pop(key, None)
        return 1


class FakeLLM:
    def __init__(self, text=None, error=None):
        self.text = text
        self.error = error
        self.calls = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)])


class FakeCollection:
    def __init__(self):
        self.add_calls = []
        self.query_calls = []
        self.responses = {}

    def add(self, **kwargs):
        self.add_calls.append(kwargs)

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        where = kwargs.get("where", {})
        if "conv_id" in where:
            return self.responses.get("same", {"ids": [[]], "documents": [[]], "metadatas": [[]]})
        return self.responses.get("user", {"ids": [[]], "documents": [[]], "metadatas": [[]]})


async def _seed(manager, user="u", conv="c", count=16, prefix="m"):
    for index in range(count):
        await manager.add_message(user, conv, MsgRole.USER, f"{prefix}-{index}")


def _run(coroutine):
    return asyncio.run(coroutine)


@pytest.fixture(autouse=True)
def _use_deterministic_test_embedding(monkeypatch):
    """Memory unit tests must not load a real embedding model or hit the network."""
    monkeypatch.setattr(
        "memory.conversation_memory._local_memory_embedding",
        lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
    )


def test_compress_splits_old_and_keep_messages_and_resets_redis():
    redis = EpisodicRedis()
    llm = FakeLLM("摘要内容")
    manager = MemoryManager(redis_client=redis, llm_client=llm, episodic_collection=FakeCollection())
    _run(_seed(manager, count=16))

    summary = _run(manager.compress_working_memory("u", "c"))

    assert summary == "摘要内容"
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["messages"][0]["content"]
    assert all(f"m-{i}" in prompt for i in range(11))
    assert all(f"m-{i}" not in prompt for i in range(11, 16))
    assert llm.calls[0]["temperature"] == 0
    assert llm.calls[0]["max_tokens"] == 256
    key = manager._wm_key("u", "c")
    remaining = _run(manager.get_recent_messages("u", "c"))
    assert len(remaining) == KEEP_AFTER_COMPRESS
    assert [item.content for item in remaining] == [f"m-{i}" for i in range(11, 16)]
    assert redis.expirations[key] == 86400
    assert redis.strings[manager._summary_key("u", "c")] == summary


def test_summary_is_truncated_and_failure_uses_fixed_fallback():
    redis = EpisodicRedis()
    llm = FakeLLM("x" * (SUMMARY_MAX_CHARS + 200))
    manager = MemoryManager(redis_client=redis, llm_client=llm, episodic_collection=FakeCollection())
    _run(_seed(manager, count=16))
    summary = _run(manager.compress_working_memory("u", "c"))
    assert len(summary) == SUMMARY_MAX_CHARS

    redis2 = EpisodicRedis()
    failed_llm = FakeLLM(error=RuntimeError("timeout"))
    manager2 = MemoryManager(
        redis_client=redis2,
        llm_client=failed_llm,
        episodic_collection=FakeCollection(),
    )
    _run(_seed(manager2, count=16))
    fallback = _run(manager2.compress_working_memory("u", "c"))
    assert fallback == "对话包含 11 条消息（摘要生成失败）"


def test_compress_writes_episodic_metadata_and_limits_full_text():
    redis = EpisodicRedis()
    collection = FakeCollection()
    llm = FakeLLM("摘要")
    manager = MemoryManager(
        redis_client=redis,
        llm_client=llm,
        episodic_collection=collection,
    )
    _run(_seed(manager, count=16, prefix="z" * 600))

    _run(manager.compress_working_memory("u", "c"))

    assert len(collection.add_calls) == 1
    call = collection.add_calls[0]
    assert call["documents"] == ["摘要"]
    metadata = call["metadatas"][0]
    assert metadata["user_id"] == "u"
    assert metadata["conv_id"] == "c"
    assert len(metadata["full_text"]) == EPISODIC_FULL_TEXT_MAX_CHARS


def test_episodic_context_prioritizes_same_conversation_then_same_user():
    redis = EpisodicRedis()
    collection = FakeCollection()
    collection.responses["same"] = {
        "ids": [["same-1", "same-2"]],
        "documents": [["same doc 1", "same doc 2"]],
        "metadatas": [[
            {"user_id": "u", "conv_id": "c", "ts": "1", "full_text": "a"},
            {"user_id": "u", "conv_id": "c", "ts": "2", "full_text": "b"},
        ]],
    }
    collection.responses["user"] = {
        "ids": [["other-1", "other-2", "other-3", "other-4"]],
        "documents": [["other 1", "other 2", "other 3", "other 4"]],
        "metadatas": [[
            {"user_id": "u", "conv_id": "x", "ts": "3", "full_text": "c"},
            {"user_id": "u", "conv_id": "y", "ts": "4", "full_text": "d"},
            {"user_id": "u", "conv_id": "z", "ts": "5", "full_text": "e"},
            {"user_id": "u", "conv_id": "w", "ts": "6", "full_text": "f"},
        ]],
    }
    manager = MemoryManager(redis_client=redis, episodic_collection=collection)

    result = _run(manager.get_episodic_context("u", "c", query="退款", limit=HISTORY_TOP_K))

    assert len(result) == HISTORY_TOP_K
    assert [item["id"] for item in result] == [
        "same-1", "same-2", "other-1", "other-2", "other-3"
    ]
    assert collection.query_calls[0]["where"] == {"user_id": "u", "conv_id": "c"}
    assert collection.query_calls[1]["where"] == {"user_id": "u"}


def test_episodic_access_uses_to_thread_and_isolates_chroma_errors(monkeypatch):
    redis = EpisodicRedis()
    collection = FakeCollection()
    calls = []
    original_to_thread = asyncio.to_thread

    async def tracking_to_thread(func, /, *args, **kwargs):
        calls.append(func)
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr("memory.conversation_memory.asyncio.to_thread", tracking_to_thread)
    manager = MemoryManager(redis_client=redis, episodic_collection=collection)
    _run(manager._store_episodic("u", "c", "摘要", "full text"))
    assert calls

    class BrokenCollection:
        def query(self, **kwargs):
            raise RuntimeError("chroma unavailable")

    broken = MemoryManager(redis_client=redis, episodic_collection=BrokenCollection())
    assert _run(broken.get_episodic_context("u", "c")) == []


def test_memory_collections_disable_chroma_default_embedding(monkeypatch, tmp_path):
    class CollectionClient:
        def __init__(self):
            self.calls = []
            self.collection = FakeCollection()

        def get_or_create_collection(self, **kwargs):
            self.calls.append(kwargs)
            return self.collection

    monkeypatch.delenv("CHROMA_HOST", raising=False)
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path))

    injected_client = CollectionClient()
    injected_manager = MemoryManager(
        redis_client=EpisodicRedis(), chroma_client=injected_client
    )
    assert injected_manager._get_episodic_collection() is not None
    assert injected_manager._get_profile_collection() is not None
    assert [call["embedding_function"] for call in injected_client.calls] == [None, None]

    local_client = CollectionClient()
    chromadb = ModuleType("chromadb")
    chromadb.PersistentClient = lambda **kwargs: local_client
    monkeypatch.setitem(sys.modules, "chromadb", chromadb)
    local_manager = MemoryManager(redis_client=EpisodicRedis())
    assert local_manager._get_episodic_collection() is not None
    assert local_manager._get_profile_collection() is not None
    assert [call["embedding_function"] for call in local_client.calls] == [None, None]

    remote_client = CollectionClient()
    http_calls = []
    chromadb_http = ModuleType("chromadb")

    def make_http_client(*, host, port):
        http_calls.append((host, port))
        return remote_client

    chromadb_http.HttpClient = make_http_client
    monkeypatch.setenv("CHROMA_HOST", "chroma.test")
    monkeypatch.setenv("CHROMA_PORT", "8000")
    monkeypatch.setitem(sys.modules, "chromadb", chromadb_http)
    remote_manager = MemoryManager(redis_client=EpisodicRedis())
    assert remote_manager._get_episodic_collection() is not None
    assert remote_manager._get_profile_collection() is not None
    assert [call["embedding_function"] for call in remote_client.calls] == [None, None]
    assert http_calls == [("chroma.test", 8000), ("chroma.test", 8000)]

    _run(remote_manager._store_episodic("u", "c", "summary", "full text"))
    assert remote_client.collection.add_calls[0]["embeddings"] == [[0.1, 0.2, 0.3]]
