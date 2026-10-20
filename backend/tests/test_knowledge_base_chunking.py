"""验证产品知识库的 Markdown Header-Aware 语义切片契约。"""

from __future__ import annotations

import mcp.knowledge_base as knowledge_base_module
from mcp.knowledge_base import KnowledgeBase
from scripts.init_knowledge_base import (
    DEFAULT_SOURCE_DIR,
    REQUIRED_METADATA_FIELDS,
    load_source_documents,
)


class ChunkCollection:
    """保存 chunk 并按查询词模拟 Chroma 召回。"""

    def __init__(self, embedding_texts) -> None:
        self.embedding_texts = embedding_texts
        self.rows = []

    def count(self) -> int:
        """返回已写入 chunk 数量。"""
        return len(self.rows)

    def add(self, *, ids, documents, metadatas, embeddings) -> None:
        """保存 KnowledgeBase 写入的文本、ID 和 metadata。"""
        self.rows.extend(
            {
                "id": item_id,
                "document": document,
                "metadata": metadata,
            }
            for item_id, document, metadata in zip(ids, documents, metadatas)
        )

    def query(self, *, query_embeddings, n_results, include):
        """根据显式向量还原测试查询，再模拟关键词召回。"""
        query_text = self.embedding_texts[tuple(query_embeddings[0])]
        query_terms = [term for term in query_text.split() if term]
        matches = [
            row
            for row in self.rows
            if all(term in row["document"] for term in query_terms)
        ][:n_results]
        return {
            "ids": [[row["id"] for row in matches]],
            "documents": [[row["document"] for row in matches]],
            "metadatas": [[row["metadata"] for row in matches]],
            "distances": [[0.2 for _ in matches]],
        }


class ChunkClient:
    """将 KnowledgeBase 绑定到内存 collection，避免外部 Chroma。"""

    def __init__(self, collection: ChunkCollection) -> None:
        """保存测试 collection。"""
        self.collection = collection

    def get_or_create_collection(self, **kwargs):
        """返回内存 collection。"""
        return self.collection

    def get_collection(self, name: str):
        """测试环境不提供 legacy collection。"""
        raise RuntimeError("测试中不存在旧 collection")


def make_knowledge_base(monkeypatch) -> tuple[KnowledgeBase, ChunkCollection]:
    """创建不下载模型、不访问网络的 KnowledgeBase。"""
    embedding_texts = {}
    collection = ChunkCollection(embedding_texts)

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
    return KnowledgeBase(client=ChunkClient(collection)), collection


def test_current_beauty_corpus_expands_to_header_aware_chunks(monkeypatch):
    """当前 50+ 篇美妆知识源应展开为聚焦语义 chunk。"""
    documents = load_source_documents(DEFAULT_SOURCE_DIR)
    knowledge_base, collection = make_knowledge_base(monkeypatch)

    written = knowledge_base.add_documents(documents)

    assert len(documents) >= 50
    assert written > len(documents)
    assert written == len(collection.rows)


def test_chunks_have_breadcrumbs_bounds_and_complete_metadata(monkeypatch):
    """每片都要有面包屑、500 字符硬上限和七项 metadata。"""
    documents = load_source_documents(DEFAULT_SOURCE_DIR)
    knowledge_base, collection = make_knowledge_base(monkeypatch)

    knowledge_base.add_documents(documents)

    assert all("[" in row["document"] and ">" in row["document"] for row in collection.rows)
    assert all(len(row["document"]) <= 500 for row in collection.rows)
    assert all(set(row["metadata"]) == REQUIRED_METADATA_FIELDS for row in collection.rows)
    assert all(
        row["metadata"]["source_chunk_id"].startswith(
            f"{row['metadata']['source_document_id']}_chunk_"
        )
        for row in collection.rows
    )


def test_sensitive_skin_query_recalls_matching_beauty_product_chunk(monkeypatch):
    """敏感肌咨询应召回对应产品资料，而不是旧的通用技术文档。"""
    documents = [
        item for item in load_source_documents(DEFAULT_SOURCE_DIR)
        if "澄润屏障修护精华水" in item["title"]
    ]
    assert documents
    knowledge_base, _ = make_knowledge_base(monkeypatch)
    knowledge_base.add_documents(documents)

    rows = knowledge_base.search("神经酰胺", top_k=5)

    assert rows
    assert "澄润屏障修护精华水" in rows[0]["title"]
    assert "神经酰胺" in rows[0]["content"]
