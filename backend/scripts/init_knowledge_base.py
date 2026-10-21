"""ShopCare 产品知识库源文本入库脚本。

本脚本扫描 ``data/knowledge_drafts/*.md``，调用当前 KnowledgeBase 重新切片并写入
产品 collection。它不导入 benchmark 语料，不复制旧 Chroma 二进制，也不包含
ToolManager、缓存或 CircuitBreaker 逻辑。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
ROOT_STRING = str(ROOT)
if ROOT_STRING in sys.path:
    sys.path.remove(ROOT_STRING)
sys.path.insert(0, ROOT_STRING)

from mcp.knowledge_base import DEFAULT_COLLECTION_NAME, KnowledgeBase


DEFAULT_SOURCE_DIR = ROOT / "data" / "knowledge_drafts"
DEFAULT_DOMAIN_MAP_PATH = ROOT.parent / "config" / "knowledge_domains.json"
DEFAULT_CORPUS_VERSION = "shopcare-kb-v2"
REQUIRED_METADATA_FIELDS = {
    "title",
    "source_document_id",
    "source_chunk_id",
    "content_hash",
    "chunk_index",
    "total_chunks",
    "corpus_version",
    "domain",
}


def load_source_documents(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    domain_map_path: Path = DEFAULT_DOMAIN_MAP_PATH,
) -> List[Dict[str, Any]]:
    """读取并规范化产品知识 Markdown 源文档。

    Args:
        source_dir: 包含 ``*.md`` 源文档的目录。
        domain_map_path: source_document_id 到三类业务域的唯一映射配置。

    Returns:
        按文件名排序的文档字典列表，每项包含标题、正文、稳定来源 ID 和语料版本。

    Raises:
        FileNotFoundError: 源目录不存在或没有 Markdown 文件时抛出。
        ValueError: 文档缺少标题、正文或业务域映射时抛出。
    """
    files = sorted(source_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"知识源目录没有 Markdown 文档: {source_dir}")
    domain_config = json.loads(domain_map_path.read_text(encoding="utf-8"))
    document_domains = domain_config.get("document_domains")
    if not isinstance(document_domains, dict):
        raise ValueError(f"知识业务域配置缺少 document_domains: {domain_map_path}")
    source_ids = {path.stem for path in files}
    mapped_ids = set(document_domains)
    missing = sorted(source_ids - mapped_ids)
    unknown = sorted(mapped_ids - source_ids)
    if missing or unknown:
        raise ValueError(
            f"知识业务域映射必须与源文件一一对应: 缺失={missing}, 多余={unknown}"
        )
    valid_domains = {"consult", "order", "after_sales"}
    invalid = {
        source_id: domain
        for source_id, domain in document_domains.items()
        if domain not in valid_domains
    }
    if invalid:
        raise ValueError(f"知识业务域配置包含无效领域: {invalid}")
    documents: List[Dict[str, Any]] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        lines = text.splitlines()
        title_line = next((line.strip() for line in lines if line.strip().startswith("# ")), "")
        title = title_line[2:].strip()
        body = "\n".join(line for line in lines if line.strip() != title_line).strip()
        if not title or not body:
            raise ValueError(f"知识源文档必须包含标题和正文: {path}")
        documents.append(
            {
                "title": title,
                "content": body,
                "source_document_id": path.stem,
                "domain": document_domains[path.stem],
                "corpus_version": os.getenv("CUSTOMEROPS_CORPUS_VERSION", DEFAULT_CORPUS_VERSION),
                "source_path": str(path),
            }
        )
    return documents


def validate_chunks(kb: KnowledgeBase, documents: Sequence[Dict[str, Any]]) -> int:
    """在写入前验证切片长度和含业务域的 metadata 契约。

    Args:
        kb: 当前 KnowledgeBase 实例，用于复用同一切片规则。
        documents: 待导入的源文档。

    Returns:
        预计写入的 chunk 总数。

    Raises:
        ValueError: 任一切片超长或 metadata 契约不完整时抛出。
    """
    total = 0
    for document in documents:
        chunks = kb._chunk_text(
            str(document["content"]), title=str(document["title"])
        )
        for index, chunk in enumerate(chunks):
            if len(chunk) > 500:
                raise ValueError(
                    f"切片超过 500 字符: {document['source_document_id']}#{index}"
                )
            metadata = kb._metadata_for_chunk(
                title=str(document["title"]),
                source_document_id=str(document["source_document_id"]),
                source_chunk_id=(
                    f"{document['source_document_id']}_chunk_{index}"
                ),
                content=chunk,
                chunk_index=index,
                total_chunks=len(chunks),
                corpus_version=str(document["corpus_version"]),
                domain=str(document["domain"]),
            )
            if set(metadata) != REQUIRED_METADATA_FIELDS:
                raise ValueError(f"切片 metadata 字段不完整: {metadata}")
            total += 1
    return total


def build_knowledge_base() -> KnowledgeBase:
    """按环境变量创建产品 collection 的 KnowledgeBase。

    Returns:
        默认指向 ``knowledge_base_zh`` 的 KnowledgeBase 实例。
    """
    collection_name = os.getenv("CUSTOMEROPS_KB_COLLECTION", DEFAULT_COLLECTION_NAME)
    persist_directory = os.getenv(
        "CHROMA_PERSIST_DIRECTORY", str(ROOT / ".data" / "chroma")
    )
    return KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=persist_directory,
        collection_name=collection_name,
    )


def run_ingestion(source_dir: Path = DEFAULT_SOURCE_DIR) -> Dict[str, Any]:
    """执行一次产品知识源文本入库并返回统计信息。

    Args:
        source_dir: 源文档目录。

    Returns:
        包含文档数、chunk 数、写入数、collection 和耗时的统计字典。
    """
    started = time.perf_counter()
    documents = load_source_documents(source_dir)
    kb = build_knowledge_base()
    expected_chunks = validate_chunks(kb, documents)
    max_chunk_length = max(
        len(chunk)
        for document in documents
        for chunk in kb._chunk_text(
            str(document["content"]), title=str(document["title"])
        )
    )
    written_chunks = kb.add_documents(documents)
    if written_chunks != expected_chunks:
        raise RuntimeError(
            f"入库 chunk 数量异常: expected={expected_chunks}, actual={written_chunks}"
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    stats = {
        "documents": len(documents),
        "chunks": written_chunks,
        "max_chunk_length": max_chunk_length,
        "collection": kb.collection_name,
        "elapsed_ms": elapsed_ms,
    }
    logger.info(
        "知识库入库完成: documents=%s chunks=%s max_chunk_length=%s collection=%s elapsed_ms=%s",
        stats["documents"],
        stats["chunks"],
        stats["max_chunk_length"],
        stats["collection"],
        stats["elapsed_ms"],
    )
    return stats


def main() -> None:
    """解析命令行参数并打印 JSON 入库统计。"""
    parser = argparse.ArgumentParser(description="初始化 ShopCare 产品知识库")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    stats = run_ingestion(args.source_dir)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
