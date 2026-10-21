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

    def __init__(self, embedding_texts):
        self.embedding_texts = embedding_texts
        self.rows = []

    def count(self) -> int:
        """返回当前内存记录数量。"""
        return len(self.rows)

    def add(self, *, ids, documents, metadatas, embeddings):
        """保存 KnowledgeBase 写入的 chunk 和 metadata。"""
        self.rows.extend(
            {
                "id": item_id,
                "document": document,
                "metadata": metadata,
            }
            for item_id, document, metadata in zip(ids, documents, metadatas)
        )

    def query(self, *, query_embeddings, n_results, include, where=None):
        """按显式向量反查测试查询文本，模拟关键词命中并返回固定距离。"""
        query = self.embedding_texts[tuple(query_embeddings[0])]
        terms = query.split()
        matches = [
            row for row in self.rows
            if all(term in row["document"] for term in terms)
            and (where is None or row["metadata"].get("domain") == where.get("domain"))
        ]
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
    embedding_texts = {}
    collection = InMemoryCollection(embedding_texts)

    def embed(texts):
        vectors = []
        for text in texts:
            vector = [float(len(embedding_texts) + 1)]
            embedding_texts[tuple(vector)] = text
            vectors.append(vector)
        return vectors

    monkeypatch.setattr(
        knowledge_base_module,
        "build_embedding_function",
        lambda model: embed,
    )
    kb = KnowledgeBase(client=InMemoryClient(collection))
    return kb, collection


def test_beauty_source_documents_are_read_and_chunks_are_bounded(monkeypatch):
    """美妆知识源至少覆盖 50 篇，且切片不超过 500 字符。"""
    documents = load_source_documents(DEFAULT_SOURCE_DIR)
    assert len(documents) >= 50
    assert all(item["source_document_id"] for item in documents)

    kb, _ = make_test_knowledge_base(monkeypatch)
    expected_chunks = validate_chunks(kb, documents)

    assert expected_chunks > 6
    assert all(
        len(chunk) <= 500
        for document in documents
        for chunk in kb._chunk_text(document["content"])
    )


def test_ingested_chunks_have_domain_metadata(monkeypatch):
    """实际写入 collection 的每个 chunk 都具备业务域 metadata。"""
    documents = load_source_documents(DEFAULT_SOURCE_DIR)
    kb, collection = make_test_knowledge_base(monkeypatch)

    written = kb.add_documents(documents)

    assert written == len(collection.rows)
    assert written > 0
    for row in collection.rows:
        metadata = row["metadata"]
        assert set(metadata) == REQUIRED_METADATA_FIELDS
        assert metadata["domain"] in {"consult", "order", "after_sales"}
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

    rows = kb.search("退款", top_k=3)

    assert rows
    assert "退款" in rows[0]["content"]
    assert rows[0]["source_chunk_id"]
    assert rows[0]["score"] == round(1 - 0.2, 4)


def test_domain_query_never_returns_other_business_domain_chunks(monkeypatch):
    """指定业务域后，embedding 召回只能返回同域 metadata 的切片。"""
    documents = load_source_documents(DEFAULT_SOURCE_DIR)
    kb, _ = make_test_knowledge_base(monkeypatch)
    kb.add_documents(documents)

    rows = kb.search("退款", top_k=10, domain="consult")

    assert rows
    assert all(row["domain"] == "consult" for row in rows)


def test_all_source_documents_have_one_explicit_business_domain():
    """当前语料每篇文档都必须被且只被一个业务域映射。"""
    documents = load_source_documents(DEFAULT_SOURCE_DIR)

    assert len(documents) == 52
    assert all(item["domain"] in {"consult", "order", "after_sales"} for item in documents)
    assert len({item["source_document_id"] for item in documents}) == len(documents)
