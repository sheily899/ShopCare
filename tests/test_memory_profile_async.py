import asyncio
import json
from types import SimpleNamespace

from memory.conversation_memory import MemoryManager, MsgRole


class ProfileRedis:
    def __init__(self):
        self.lists = {}

    async def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)
        return len(self.lists[key])

    async def expire(self, key, seconds):
        return True

    async def llen(self, key):
        return len(self.lists.get(key, []))

    async def lrange(self, key, start, end):
        values = self.lists.get(key, [])
        return values[start : end + 1 if end >= 0 else None]


class ProfileLLM:
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


class ProfileCollection:
    def __init__(self, existing=None, error=None):
        self.existing = existing
        self.error = error
        self.upsert_calls = []
        self.add_calls = []
        self.get_calls = []

    def upsert(self, **kwargs):
        if self.error:
            raise self.error
        self.upsert_calls.append(kwargs)
        self.existing = kwargs["documents"][0]

    def add(self, **kwargs):
        if self.error:
            raise self.error
        self.add_calls.append(kwargs)
        self.existing = kwargs["documents"][0]

    def get(self, **kwargs):
        if self.error:
            raise self.error
        self.get_calls.append(kwargs)
        if self.existing is None:
            return {"ids": [], "documents": [], "metadatas": []}
        return {
            "ids": ["profile:u-1"],
            "documents": [self.existing],
            "metadatas": [{"user_id": "u-1", "conv_id": "c-1"}],
        }


def _run(coroutine):
    return asyncio.run(coroutine)


async def _seed(manager):
    await manager.add_message("u-1", "c-1", MsgRole.USER, "我只用顺丰快递")
    await manager.add_message("u-1", "c-1", MsgRole.USER, "订单 ORD-1001 请尽快送达")


def test_update_profile_extracts_json_and_persists_stable_document():
    redis = ProfileRedis()
    collection = ProfileCollection()
    payload = {
        "preferences": ["只用顺丰快递"],
        "entities": {
            "order_id": ["ORD-1001"],
            "product": [],
            "date": [],
            "amount": [],
            "error_code": [],
        },
    }
    llm = ProfileLLM(json.dumps(payload, ensure_ascii=False))
    manager = MemoryManager(
        redis_client=redis,
        llm_client=llm,
        profile_collection=collection,
    )
    _run(_seed(manager))

    task = _run(_create_task(manager.update_profile("u-1", "c-1")))

    assert task is None
    assert len(collection.upsert_calls) == 1
    call = collection.upsert_calls[0]
    assert call["ids"] == ["profile:u-1"]
    assert json.loads(call["documents"][0]) == payload
    metadata = call["metadatas"][0]
    assert metadata["user_id"] == "u-1"
    assert metadata["conv_id"] == "c-1"
    assert "T" in metadata["updated_at"]
    assert llm.calls[0]["temperature"] == 0
    assert llm.calls[0]["max_tokens"] == 512


async def _create_task(coroutine):
    task = asyncio.create_task(coroutine)
    return await task


def test_malformed_profile_does_not_overwrite_collection():
    redis = ProfileRedis()
    collection = ProfileCollection(existing='{"preferences":["旧偏好"],"entities":{}}')
    llm = ProfileLLM("这不是 JSON")
    manager = MemoryManager(redis_client=redis, llm_client=llm, profile_collection=collection)
    _run(_seed(manager))

    _run(manager.update_profile("u-1", "c-1"))

    assert collection.upsert_calls == []
    assert collection.add_calls == []
    assert collection.existing == '{"preferences":["旧偏好"],"entities":{}}'


def test_profile_background_task_swallows_llm_and_chroma_failures():
    redis = ProfileRedis()
    llm_failed = ProfileLLM(error=RuntimeError("network interrupted"))
    manager_llm = MemoryManager(
        redis_client=redis,
        llm_client=llm_failed,
        profile_collection=ProfileCollection(),
    )
    _run(_seed(manager_llm))
    assert _run(_create_task(manager_llm.update_profile("u-1", "c-1"))) is None

    payload = {"preferences": ["中文"], "entities": {}}
    manager_chroma = MemoryManager(
        redis_client=redis,
        llm_client=ProfileLLM(json.dumps(payload)),
        profile_collection=ProfileCollection(error=RuntimeError("chroma unavailable")),
    )
    _run(_seed(manager_chroma))
    assert _run(_create_task(manager_chroma.update_profile("u-1", "c-1"))) is None


def test_get_user_profile_deserializes_existing_document_and_returns_default():
    payload = {"preferences": ["顺丰"], "entities": {"product": ["耳机"]}}
    existing = ProfileCollection(existing=json.dumps(payload, ensure_ascii=False))
    manager = MemoryManager(redis_client=ProfileRedis(), profile_collection=existing)

    assert _run(manager.get_user_profile("u-1")) == payload
    assert existing.get_calls[0]["ids"] == ["profile:u-1"]

    missing = MemoryManager(redis_client=ProfileRedis(), profile_collection=ProfileCollection())
    assert _run(missing.get_user_profile("unknown")) == {
        "preferences": [],
        "entities": {},
    }
