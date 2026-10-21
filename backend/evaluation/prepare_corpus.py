"""将独立评测 corpus 写入 evaluation_benchmark collection。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evaluation.corpus import (
    EVALUATION_COLLECTION_NAME,
    load_corpus,
    upsert_stable_chunks,
    validate_collection_name,
)
from mcp.knowledge_base import KnowledgeBase


logger = logging.getLogger(__name__)


def prepare_corpus(corpus_path: str, collection_name: str = EVALUATION_COLLECTION_NAME) -> int:
    """加载评测切片并写入独立 Chroma collection。

    Args:
        corpus_path: 稳定评测 corpus 文件。
        collection_name: 只能为 ``evaluation_benchmark``。

    Returns:
        写入数量。
    """

    target = validate_collection_name(collection_name)
    chunks = load_corpus(corpus_path)
    if len(chunks) != 100:
        raise ValueError(f"正式评测 corpus 必须包含 100 个稳定片段，当前为 {len(chunks)} 个")
    knowledge_base = KnowledgeBase(collection_name=target)
    count = upsert_stable_chunks(
        knowledge_base._collection,
        chunks,
        knowledge_base._embedding_function,
    )
    logger.info(
        "评测语料准备完成: collection=%s chunks=%s",
        target,
        len(chunks),
    )
    return count


def main() -> None:
    """解析命令行参数并准备评测 collection。"""

    # Why：本地 Chroma 映射端口和模型路径通常写在 backend/.env；准备命令若不加载，
    # 会把 FastAPI 的 8000 误当成 Chroma 端口并触发错误 fallback。
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception as exc:  # pragma: no cover - 依赖缺失时沿用环境变量
        logger.warning(".env 加载失败，继续使用系统环境变量：%s", exc)

    parser = argparse.ArgumentParser(description="准备 ShopCare 独立评测知识库")
    parser.add_argument(
        "--corpus-path",
        default="evaluation/datasets/shopcare_evaluation_corpus.json",
        help="稳定评测 corpus JSON 路径",
    )
    parser.add_argument("--collection-name", default=EVALUATION_COLLECTION_NAME)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    count = prepare_corpus(args.corpus_path, args.collection_name)
    print(f"evaluation_benchmark prepared: chunks={count} path={Path(args.corpus_path)}")


if __name__ == "__main__":
    main()
