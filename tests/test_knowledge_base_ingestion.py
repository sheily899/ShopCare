"""产品知识源文本读取、切片、metadata 和基础召回契约测试。"""

from __future__ import annotations

import mcp.knowledge_base as knowledge_base_module
from mcp.knowledge_base import KnowledgeBase
from scripts.init_knowledge_base import (
    DEFAULT_SOURCE_DIR,
    REQUIRED_METADATA_FIELDS,
    load_source_documents,
    validate_chunks,
)


class InMemoryCollection:
    """提供最小 add/query 行为的内存 Chroma collection。"""

    def __init__(self) -> None:
        """初始化空的内存记录。"""
        self.rows = []

    def count(self) -> int:
        """返回当前内存记录数量。"""
        return len(self.rows)

    def add(self, *, ids, documents, metadatas):
        """保存 KnowledgeBase 写入的 chunk 和 metadata。"""
        self.rows.extend(
            {
                "id": item_id,
                "document": document,
                "metadata": metadata,
            }
            for item_id, document, metadata in zip(ids, documents, metadatas)
        )

    def query(self, *, query_texts, n_results, include):
        """按关键词模拟 Chroma 查询并返回固定距离。"""
        query = query_texts[0]
        matches = [row for row in self.rows if query in row["document"]]
        matches = matches[:n_results]
        return {
            "ids": [[row["id"] for row in matches]],
            "documents": [[row["document"] for row in matches]],
            "metadatas": [[row["metadata"] for row in matches]],
            "distances": [[0.2 for _ in matches]],
        }


class InMemoryClient:
    """将 KnowledgeBase 指向内存 collection，避免真实 Chroma 依赖。"""

    def __init__(self, collection: InMemoryCollection) -> None:
        """绑定内存 collection。"""
        self.collection = collection

    def get_or_create_collection(self, **kwargs):
        """返回测试用 collection。"""
        return self.collection

    def get_collection(self, name: str):
        """显式报告测试中不提供旧 collection。"""
        raise RuntimeError("测试中不存在旧 collection")


def make_test_knowledge_base(monkeypatch):
    """创建不下载模型、不访问网络的 KnowledgeBase。"""
    collection = InMemoryCollection()
    monkeypatch.setattr(
        knowledge_base_module,
        "build_embedding_function",
        lambda model: {"model": model, "device": "cpu"},
    )
    kb = KnowledgeBase(client=InMemoryClient(collection))
    return kb, collection


def test_six_source_documents_are_read_and_chunks_are_bounded(monkeypatch):
    """6 篇源文档均能读取，且所有切片不超过 500 字符。"""
    documents = load_source_documents(DEFAULT_SOURCE_DIR)
    assert len(documents) == 6
    assert {item["source_document_id"] for item in documents} == {
        "refund_policy",
        "order_logistics",
        "account_security",
        "technical_troubleshooting",
        "membership_points",
        "invoice_and_payment",
    }

    kb, _ = make_test_knowledge_base(monkeypatch)
    expected_chunks = validate_chunks(kb, documents)

    assert expected_chunks > 6
    assert all(
        len(chunk) <= 500
        for document in documents
        for chunk in kb._chunk_text(document["content"])
    )


def test_ingested_chunks_have_all_seven_metadata_fields(monkeypatch):
    """实际写入 collection 的每个 chunk 都具备完整七项 metadata。"""
    documents = load_source_documents(DEFAULT_SOURCE_DIR)
    kb, collection = make_test_knowledge_base(monkeypatch)

    written = kb.add_documents(documents)

    assert written == len(collection.rows)
    assert written > 0
    for row in collection.rows:
        metadata = row["metadata"]
        assert set(metadata) == REQUIRED_METADATA_FIELDS
        assert len(row["document"]) <= 500
        assert metadata["source_chunk_id"].startswith(
            f"{metadata['source_document_id']}_chunk_"
        )
        assert metadata["content_hash"].startswith("sha256:")


def test_keyword_query_recalls_matching_document_with_distance_score(monkeypatch):
    """退款关键词应召回退款文档，score 必须遵循 1-distance 规则。"""
    documents = load_source_documents(DEFAULT_SOURCE_DIR)
    kb, _ = make_test_knowledge_base(monkeypatch)
    kb.add_documents(documents)

    rows = kb.search("原路退回", top_k=3)

    assert rows
    assert rows[0]["source_document_id"] == "refund_policy"
    assert rows[0]["source_chunk_id"].startswith("refund_policy_chunk_")
    assert rows[0]["score"] == round(1 - 0.2, 4)
