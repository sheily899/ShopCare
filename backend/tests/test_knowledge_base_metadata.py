"""KnowledgeBase 文档切片、元数据和基础检索测试。"""

from __future__ import annotations

import pytest

import mcp.knowledge_base as knowledge_base_module


class SearchCollection:
    """保存写入数据并返回固定距离的 Chroma collection 测试替身。"""

    def __init__(self) -> None:
        self.add_calls = []
        self.query_calls = []
        self.fail_filtered_query = False

    def count(self) -> int:
        return 1

    def add(self, **kwargs):
        self.add_calls.append(kwargs)

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        if kwargs.get("where") and self.fail_filtered_query:
            raise RuntimeError("domain filter unavailable")
        rows = [
            (
                "runtime-1",
                "相关片段",
                {
                    "title": "退款政策",
                    "source_document_id": "refund-policy",
                    "source_chunk_id": "refund-policy#chunk_001",
                    "content_hash": "sha256:a",
                    "chunk_index": 0,
                    "total_chunks": 2,
                    "corpus_version": "kb-v2",
                    "domain": "after_sales",
                },
                0.12345,
            ),
            (
                "runtime-2",
                "低分片段",
                {
                    "title": "其他",
                    "source_document_id": "other",
                    "source_chunk_id": "other#chunk_001",
                    "content_hash": "sha256:b",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "corpus_version": "kb-v2",
                    "domain": "consult",
                },
                0.95,
            ),
        ]
        domain_filter = kwargs.get("where", {}).get("domain")
        if domain_filter:
            rows = [row for row in rows if row[2]["domain"] == domain_filter]
        limit = kwargs.get("n_results", len(rows))
        rows = rows[:limit]
        return {
            "ids": [[row[0] for row in rows]],
            "documents": [[row[1] for row in rows]],
            "metadatas": [[row[2] for row in rows]],
            "distances": [[row[3] for row in rows]],
        }


class InjectedClient:
    """向 KnowledgeBase 提供固定 collection 的 Chroma client。"""

    def __init__(self, collection) -> None:
        self.collection = collection

    def get_or_create_collection(self, **kwargs):
        return self.collection

    def get_collection(self, name: str):
        raise RuntimeError("没有旧库")


@pytest.fixture()
def knowledge_base(monkeypatch):
    """创建不触发真实模型下载或网络连接的 KnowledgeBase。"""

    collection = SearchCollection()
    monkeypatch.setattr(
        knowledge_base_module,
        "build_embedding_function",
        lambda model: lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
    )
    return knowledge_base_module.KnowledgeBase(client=InjectedClient(collection)), collection


def test_document_chunks_are_bounded_and_metadata_has_domain(knowledge_base):
    """每个 chunk 不超过 500 字符，并写入业务域 metadata。"""

    base, collection = knowledge_base
    text = "。".join(["退款规则说明" + "字" * 120 for _ in range(10)])

    written = base.add_documents([
        {
            "title": "退款政策",
            "content": text,
            "source_document_id": "refund-policy",
            "corpus_version": "kb-v2",
            "domain": "after_sales",
        }
    ])

    assert written > 1
    payload = collection.add_calls[-1]
    assert all(len(chunk) <= 500 for chunk in payload["documents"])
    required = {
        "title",
        "source_document_id",
        "source_chunk_id",
        "content_hash",
        "chunk_index",
        "total_chunks",
        "corpus_version",
        "domain",
    }
    assert all(set(metadata) == required for metadata in payload["metadatas"])
    assert payload["metadatas"][0]["source_chunk_id"] == "refund-policy_chunk_0"
    assert payload["metadatas"][0]["total_chunks"] == written


def test_search_returns_stable_identity_and_rounds_distance_score(knowledge_base):
    """检索保留低分结果，不设 threshold，并按 1-distance 四舍五入。"""

    base, _ = knowledge_base

    rows = base.search("退款", top_k=2)

    assert rows[0]["id"] == "runtime-1"
    assert rows[0]["runtime_id"] == "runtime-1"
    assert rows[0]["source_chunk_id"] == "refund-policy#chunk_001"
    assert rows[0]["score"] == round(1 - 0.12345, 4)
    assert rows[1]["score"] == round(1 - 0.95, 4)
    assert len(rows) == 2


def test_domain_search_passes_filter_and_returns_domain_metadata(knowledge_base):
    base, collection = knowledge_base

    rows = base.search("退款", top_k=2, domain="after_sales")

    assert collection.query_calls[0]["where"] == {"domain": "after_sales"}
    assert all(row["domain"] == "after_sales" for row in rows)


def test_domain_filter_failure_never_retries_unfiltered_search(knowledge_base):
    base, collection = knowledge_base
    collection.fail_filtered_query = True

    with pytest.raises(RuntimeError, match="domain filter unavailable"):
        base.search("退款", top_k=2, domain="consult")

    assert len(collection.query_calls) == 1
    assert collection.query_calls[0]["where"] == {"domain": "consult"}


def test_unknown_domain_is_rejected_before_query(knowledge_base):
    base, collection = knowledge_base

    with pytest.raises(ValueError, match="不支持的知识业务域"):
        base.search("退款", top_k=2, domain="legacy")

    assert collection.query_calls == []
