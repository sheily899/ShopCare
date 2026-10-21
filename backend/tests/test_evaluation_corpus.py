"""评测 corpus 隔离、稳定身份和不重新切片契约测试。"""

from pathlib import Path

from evaluation.corpus import (
    EVALUATION_COLLECTION_NAME,
    PRODUCT_COLLECTION_NAME,
    load_cases,
    load_corpus,
    upsert_stable_chunks,
    validate_collection_name,
)
from mcp.knowledge_base import KnowledgeBase


ROOT = Path(__file__).parents[1]


class FakeCollection:
    def __init__(self):
        self.calls = []

    def upsert(self, **kwargs):
        self.calls.append(kwargs)


def test_adapted_corpus_has_100_cases_and_100_stable_chunks():
    cases = load_cases(ROOT / "evaluation/datasets/shopcare_evaluation_cases.json")
    chunks = load_corpus(ROOT / "evaluation/datasets/shopcare_evaluation_corpus.json")
    assert len(cases) == 100
    assert len(chunks) == 100
    assert len({str(item["id"]) for item in chunks}) == 100
    assert all(all(field in item for field in (
        "title", "source_document_id", "source_chunk_id", "content_hash",
        "chunk_index", "total_chunks", "corpus_version",
    )) for item in chunks)


def test_eval_collection_isolated_from_product_collection():
    assert validate_collection_name(EVALUATION_COLLECTION_NAME) == EVALUATION_COLLECTION_NAME
    try:
        validate_collection_name(PRODUCT_COLLECTION_NAME)
    except ValueError as exc:
        assert "knowledge_base_zh" in str(exc)
    else:
        raise AssertionError("产品 collection 不应被评测准备函数接受")


def test_upsert_preserves_stable_ids_and_does_not_reslice():
    chunks = load_corpus(ROOT / "evaluation/datasets/shopcare_evaluation_corpus.json")[:1]
    collection = FakeCollection()
    assert upsert_stable_chunks(
        collection,
        chunks,
        lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
    ) == 1
    assert collection.calls[0]["ids"] == [chunks[0]["id"]]
    assert collection.calls[0]["documents"] == [chunks[0]["content"]]
    assert collection.calls[0]["metadatas"][0]["source_chunk_id"] == chunks[0]["source_chunk_id"]
    assert collection.calls[0]["embeddings"] == [[0.1, 0.2, 0.3]]


def test_stable_writer_requires_explicit_embedding_function():
    chunks = load_corpus(ROOT / "evaluation/datasets/shopcare_evaluation_corpus.json")[:1]
    try:
        upsert_stable_chunks(FakeCollection(), chunks, None)
    except TypeError as exc:
        assert "显式本地 embedding" in str(exc)
    else:
        raise AssertionError("评测写入不可依赖 Chroma 默认 embedding")


def test_knowledge_base_stable_writer_keeps_original_chunk_identity():
    collection = FakeCollection()
    kb = object.__new__(KnowledgeBase)
    kb._collection = collection
    kb._embedding_function = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
    chunks = load_corpus(ROOT / "evaluation/datasets/shopcare_evaluation_corpus.json")[:1]
    assert kb.add_stable_chunks(chunks) == 1
    assert collection.calls[0]["ids"] == [chunks[0]["id"]]
    assert collection.calls[0]["documents"][0] == chunks[0]["content"]


def test_evaluation_collection_skips_legacy_product_migration():
    class ClientThatMustNotBeRead:
        def get_or_create_collection(self, **kwargs):
            return FakeCollection()

        def get_collection(self, **kwargs):
            raise AssertionError("评测 collection 不得读取旧产品 collection")

    KnowledgeBase(
        client=ClientThatMustNotBeRead(),
        collection_name=EVALUATION_COLLECTION_NAME,
        embedding_function=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
    )
