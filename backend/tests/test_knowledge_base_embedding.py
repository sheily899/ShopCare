"""KnowledgeBase 初始化、Embedding 配置和旧 collection 迁移测试。"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import mcp.knowledge_base as knowledge_base_module


class FakeCollection:
    """记录 collection 初始化和写入参数的 Chroma 测试替身。"""

    def __init__(self, count: int = 1) -> None:
        self._count = count
        self.add_calls = []

    def count(self) -> int:
        return self._count

    def add(self, **kwargs):
        self.add_calls.append(kwargs)
        self._count += len(kwargs.get("ids", []))

    def get(self, **kwargs):
        return {"ids": [], "documents": [], "metadatas": []}


class FakeClient:
    """支持 collection 创建和旧库读取的最小 Chroma client。"""

    def __init__(self, collection: FakeCollection, legacy=None) -> None:
        self.collection = collection
        self.legacy = legacy or FakeCollection(0)
        self.collection_kwargs = None

    def get_or_create_collection(self, **kwargs):
        self.collection_kwargs = kwargs
        return self.collection

    def get_collection(self, name: str):
        assert name == "knowledge_base"
        return self.legacy


def test_http_failure_falls_back_to_persistent_client_and_logs(monkeypatch, caplog):
    """HTTP 服务失败时必须显式记录 fallback 并创建 PersistentClient。"""

    collection = FakeCollection(count=1)
    calls = {}

    class FakeHttpClient:
        def __init__(self, **kwargs):
            calls["http"] = kwargs
            # 模拟真实 Chroma HttpClient 对 Settings 的原地 FastAPI 改写。
            kwargs["settings"]["chroma_api_impl"] = "chromadb.api.fastapi.FastAPI"
            raise ConnectionError("http unavailable")

    class FakePersistentClient:
        def __init__(self, **kwargs):
            calls["persistent"] = kwargs
            assert kwargs["settings"]["chroma_api_impl"] == (
                "chromadb.api.segment.SegmentAPI"
            )

        def get_or_create_collection(self, **kwargs):
            calls["collection"] = kwargs
            return collection

    fake_chromadb = SimpleNamespace(
        HttpClient=FakeHttpClient,
        PersistentClient=FakePersistentClient,
        Settings=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(knowledge_base_module, "chromadb", fake_chromadb)
    monkeypatch.setattr(
        knowledge_base_module,
        "build_embedding_function",
        lambda model: {"model": model},
    )

    with caplog.at_level(logging.WARNING):
        knowledge_base_module.KnowledgeBase(
            chroma_host="bad-host",
            chroma_port=8000,
            chroma_path=".data/test-chroma",
        )

    assert "http" in calls
    assert calls["persistent"]["path"] == ".data/test-chroma"
    assert "fallback 到 PersistentClient" in caplog.text


def test_embedding_cpu_normalization_and_hnsw_configuration(monkeypatch):
    """模型名、CPU、归一化和 HNSW l2 配置必须精确传递。"""

    captured = {}

    class FakeEmbeddingFunction:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = SimpleNamespace(
        utils=SimpleNamespace(
            embedding_functions=SimpleNamespace(
                SentenceTransformerEmbeddingFunction=FakeEmbeddingFunction
            )
        )
    )
    monkeypatch.setattr(knowledge_base_module, "chromadb", fake_module)

    embedding = knowledge_base_module.build_embedding_function()

    assert isinstance(embedding, FakeEmbeddingFunction)
    assert captured == {
        "model_name": "BAAI/bge-small-zh-v1.5",
        "device": "cpu",
        "normalize_embeddings": True,
    }

    collection = FakeCollection(count=1)
    client = FakeClient(collection)
    monkeypatch.setattr(
        knowledge_base_module,
        "build_embedding_function",
        lambda model: embedding,
    )
    knowledge_base_module.KnowledgeBase(client=client)

    assert client.collection_kwargs["name"] == "knowledge_base_zh"
    assert client.collection_kwargs["metadata"]["hnsw:space"] == "l2"
    assert client.collection_kwargs["metadata"]["embedding_model"] == (
        "BAAI/bge-small-zh-v1.5"
    )


def test_empty_new_collection_migrates_legacy_documents(monkeypatch):
    """新 collection 为空时应将旧 collection 的内容迁移过来。"""

    new_collection = FakeCollection(count=0)

    class LegacyCollection:
        def get(self, **kwargs):
            return {
                "ids": ["legacy-1"],
                "documents": ["旧知识"],
                "metadatas": [{"title": "旧文档"}],
            }

    client = FakeClient(new_collection, legacy=LegacyCollection())
    monkeypatch.setattr(
        knowledge_base_module,
        "build_embedding_function",
        lambda model: object(),
    )

    knowledge_base_module.KnowledgeBase(client=client)

    assert len(new_collection.add_calls) == 1
    migrated_metadata = new_collection.add_calls[0]["metadatas"][0]
    assert migrated_metadata["source_chunk_id"] == "legacy_chunk_0"
    assert migrated_metadata["content_hash"].startswith("sha256:")
