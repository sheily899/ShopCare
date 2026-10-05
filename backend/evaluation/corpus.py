"""评测案例与稳定知识片段的加载、校验和准备辅助函数。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


REQUIRED_METADATA_FIELDS = (
    "title",
    "source_document_id",
    "source_chunk_id",
    "content_hash",
    "chunk_index",
    "total_chunks",
    "corpus_version",
)
EVALUATION_COLLECTION_NAME = "evaluation_benchmark"
PRODUCT_COLLECTION_NAME = "knowledge_base_zh"


def load_json(path: str | Path) -> Dict[str, Any]:
    """读取 JSON 对象，并在入口处给出清晰错误。

    Args:
        path: JSON 文件路径。

    Returns:
        JSON 顶层对象。
    """

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"评测文件顶层必须是对象: {source}")
    return value


def load_cases(path: str | Path) -> List[Dict[str, Any]]:
    """加载并校验 100 条 ShopCare 适配案例。

    Args:
        path: 案例 JSON 路径。

    Returns:
        按文件顺序排列的案例字典列表。
    """

    payload = load_json(path)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("评测案例缺少非空 cases 数组")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("每条评测案例必须是对象")
        case_id = str(case.get("case_id") or "")
        if not case_id or case_id in seen:
            raise ValueError(f"评测 case_id 缺失或重复: {case_id}")
        if not str(case.get("message") or "").strip():
            raise ValueError(f"评测消息为空: {case_id}")
        seen.add(case_id)
    expected_count = int(payload.get("case_count") or len(cases))
    if expected_count != len(cases):
        raise ValueError(f"case_count 与实际数量不一致: {expected_count} != {len(cases)}")
    return cases


def load_corpus(path: str | Path) -> List[Dict[str, Any]]:
    """加载并校验不重新切片的评测知识片段。

    Args:
        path: 评测 corpus JSON 路径。

    Returns:
        保持原始 ID、正文和 metadata 的切片列表。
    """

    payload = load_json(path)
    chunks = payload.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("评测 corpus 缺少非空 chunks 数组")
    seen = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("每个评测 chunk 必须是对象")
        chunk_id = str(chunk.get("id") or chunk.get("source_chunk_id") or "")
        if not chunk_id or chunk_id in seen:
            raise ValueError(f"评测 chunk ID 缺失或重复: {chunk_id}")
        missing = [field for field in REQUIRED_METADATA_FIELDS if field not in chunk]
        if missing:
            raise ValueError(f"评测 chunk {chunk_id} 缺少字段: {missing}")
        if not str(chunk.get("content") or "").strip():
            raise ValueError(f"评测 chunk 正文为空: {chunk_id}")
        expected_hash = f"sha256:{hashlib.sha256(str(chunk['content']).encode('utf-8')).hexdigest()}"
        if str(chunk.get("content_hash")) != expected_hash:
            raise ValueError(f"评测 chunk content_hash 不匹配: {chunk_id}")
        seen.add(chunk_id)
    expected_count = int(payload.get("chunk_count") or len(chunks))
    if expected_count != len(chunks):
        raise ValueError(f"chunk_count 与实际数量不一致: {expected_count} != {len(chunks)}")
    return chunks


def validate_collection_name(collection_name: str) -> str:
    """阻止评测语料误写入产品 collection。

    Args:
        collection_name: 目标 Chroma collection 名称。

    Returns:
        清理后的 collection 名称。

    Raises:
        ValueError: 名称不是独立评测集合。
    """

    value = str(collection_name or "").strip()
    if value == PRODUCT_COLLECTION_NAME:
        raise ValueError("评测语料严禁写入产品 collection knowledge_base_zh")
    if value != EVALUATION_COLLECTION_NAME:
        raise ValueError("评测 collection 必须固定为 evaluation_benchmark")
    return value


def stable_chunk_metadata(chunk: Mapping[str, Any]) -> Dict[str, Any]:
    """提取 Chroma 可接受的稳定 metadata，保留评测标签。

    Args:
        chunk: 已通过 ``load_corpus`` 校验的切片。

    Returns:
        不含 ``id``/``content`` 且只含标量值的 metadata。
    """

    metadata: Dict[str, Any] = {}
    for key, value in chunk.items():
        if key in {"id", "content"}:
            continue
        if value is None:
            metadata[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            metadata[key] = str(value)
    return metadata


def upsert_stable_chunks(collection: Any, chunks: Sequence[Mapping[str, Any]]) -> int:
    """直接 upsert 评测切片，确保不会复用产品切片器。

    Args:
        collection: 支持 ``upsert`` 或 ``add`` 的 Chroma collection 替身/实例。
        chunks: 稳定评测切片列表。

    Returns:
        写入数量。
    """

    ids = [str(item.get("id") or item.get("source_chunk_id")) for item in chunks]
    documents = [str(item.get("content") or "") for item in chunks]
    metadatas = [stable_chunk_metadata(item) for item in chunks]
    if not ids:
        return 0
    writer = getattr(collection, "upsert", None)
    if not callable(writer):
        writer = getattr(collection, "add", None)
    if not callable(writer):
        raise TypeError("评测 collection 缺少 upsert/add 方法")
    writer(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def corpus_summary(cases: Sequence[Mapping[str, Any]], chunks: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """统计评测规模，供准备命令和报告复用。"""

    compound = [case for case in cases if bool((case.get("compound") or {}).get("is_compound"))]
    cross_domain = [case for case in cases if case.get("expected_secondary_intents")]
    knowledge = [case for case in cases if bool(case.get("knowledge_required"))]
    return {
        "cases": len(cases),
        "chunks": len(chunks),
        "knowledge_required": len(knowledge),
        "compound": len(compound),
        "explicit_cross_domain": len(cross_domain),
    }
