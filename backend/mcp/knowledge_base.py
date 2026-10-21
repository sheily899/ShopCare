"""ShopCare 产品知识库基础设施。

本模块属于 MCP/RAG 基础层，负责 ChromaDB 客户端初始化、知识文档切片、
带业务域的切片元数据写入和基础向量检索。它不负责 ToolManager、缓存、熔断、
Query Rewrite 或 Rerank，也不连接任何真实订单或支付系统。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)

try:  # 依赖可能只在运行环境安装，保持纯切片测试可导入。
    import chromadb as chromadb  # type: ignore
except Exception:  # pragma: no cover - 由运行环境决定
    chromadb = None  # type: ignore


# 集合名包含 embedding 版本，避免复用旧的 384 维默认集合并触发 Chroma
# 的 all-MiniLM-L6-v2 provider；新集合只接受项目本地 BGE 向量。
DEFAULT_COLLECTION_NAME = "knowledge_base_zh_bge_small_zh"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_PERSIST_DIRECTORY = ".data/chroma"
DEFAULT_CORPUS_VERSION = "kb-v1-default"
BUSINESS_KB_DOMAINS = frozenset({"consult", "order", "after_sales"})
CHUNK_SIZE = 500
HNSW_SPACE = "l2"


def resolve_embedding_model_name(model_name: str = DEFAULT_EMBEDDING_MODEL) -> str:
    """只解析已存在的本地模型目录；缺失时失败关闭，绝不回退到远程模型名。"""
    candidate = os.path.abspath(model_name)
    if os.path.isdir(candidate):
        return candidate
    project_model = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", model_name)
    )
    if os.path.isdir(project_model):
        return project_model
    raise FileNotFoundError(
        f"本地 embedding 模型不存在: {model_name}; 已检查 {candidate} 和 {project_model}"
    )


def _load_chromadb() -> Any:
    """惰性加载 ChromaDB SDK，避免切片和契约测试被可选依赖阻塞。

    Returns:
        已加载的 ``chromadb`` 模块。

    Raises:
        RuntimeError: 当前环境无法导入 ChromaDB 时抛出。
    """
    global chromadb
    if chromadb is not None:
        return chromadb
    try:
        import chromadb as chromadb_module  # type: ignore
    except Exception as exc:  # pragma: no cover - 由运行环境决定
        raise RuntimeError(f"ChromaDB SDK 不可用: {exc}") from exc
    chromadb = chromadb_module
    return chromadb_module


def _chroma_settings(module: Any) -> Any:
    """构造关闭匿名 telemetry 的 Chroma 设置对象（若 SDK 支持）。

    Args:
        module: 已加载的 ChromaDB 模块或测试替身。

    Returns:
        SDK ``Settings`` 实例；不支持该类时返回 ``None``。
    """
    settings_cls = getattr(module, "Settings", None)
    if settings_cls is None:
        return None
    try:
        return settings_cls(anonymized_telemetry=False)
    except TypeError:
        return settings_cls()


def _persistent_chroma_settings(module: Any) -> Any:
    """为本地持久化客户端创建独立且明确的 Chroma 设置。

    ``chromadb.HttpClient`` 会原地修改传入 Settings 的 ``chroma_api_impl``。
    因此本地 fallback 必须重新创建 Settings，并固定为 SegmentAPI，避免把
    HTTP 配置带入 ``PersistentClient`` 后再次请求远端服务。

    Args:
        module: 已加载的 ChromaDB 模块或测试替身。

    Returns:
        供 ``PersistentClient`` 使用的独立 Settings 实例；SDK 不支持 Settings
        时返回 ``None``。
    """
    settings = _chroma_settings(module)
    if settings is None:
        return None

    # Why：本地 fallback 必须走 SegmentAPI，不能复用 HTTP 客户端污染过的 FastAPI 配置。
    try:
        settings.chroma_api_impl = "chromadb.api.segment.SegmentAPI"
    except (AttributeError, TypeError):
        # 测试替身可能使用不可变或字典式 Settings；构造参数是第二条兼容路径。
        settings_cls = getattr(module, "Settings", None)
        if settings_cls is None:
            return settings
        try:
            return settings_cls(
                anonymized_telemetry=False,
                chroma_api_impl="chromadb.api.segment.SegmentAPI",
            )
        except TypeError:
            return settings
    return settings


def build_embedding_function(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    """构建固定 CPU、归一化的 BGE SentenceTransformer embedding function。

    Args:
        model_name: 使用的 SentenceTransformer 模型名。

    Returns:
        Chroma 可接受的 ``SentenceTransformerEmbeddingFunction``。

    Raises:
        RuntimeError: Chroma embedding function 依赖不可用时抛出。
    """
    try:
        module = _load_chromadb()
        embedding_functions = module.utils.embedding_functions
        resolved_model = resolve_embedding_model_name(model_name)
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=resolved_model,
            device="cpu",
            normalize_embeddings=True,
        )
    except Exception as exc:
        raise RuntimeError(f"SentenceTransformer embedding 初始化失败: {exc}") from exc


class KnowledgeBase:
    """提供 ShopCare 产品知识库的初始化、切片写入和基础检索。

    HTTP Chroma 是首选连接方式；HTTP 服务不可用时才退回本地
    ``PersistentClient``，并通过 warning 日志显式暴露该降级。

    Args:
        chroma_host: HTTP Chroma 主机；未传入时读取 ``CHROMA_HOST``，默认 localhost。
        chroma_port: HTTP Chroma 端口；未传入时读取 ``CHROMA_PORT``，默认 8000。
        chroma_path: 本地持久化目录；未传入时读取 ``CHROMA_PERSIST_DIRECTORY``。
        embedding_model: BGE 模型名；未传入时读取 ``CUSTOMEROPS_EMBEDDING_MODEL``。
        collection_name: 产品知识库 collection；未传入时读取 ``CUSTOMEROPS_KB_COLLECTION``。
        embedding_function: 可选注入的 embedding function，便于离线测试。
        client: 可选注入的 Chroma client，便于离线测试。
    """

    COLLECTION_NAME = DEFAULT_COLLECTION_NAME
    EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
    CHUNK_SIZE = CHUNK_SIZE

    def __init__(
        self,
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
        chroma_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_function: Any = None,
        client: Any = None,
    ) -> None:
        """初始化 Chroma client、embedding function 和产品 collection。

        Args:
            chroma_host: HTTP Chroma 主机覆盖值。
            chroma_port: HTTP Chroma 端口覆盖值。
            chroma_path: PersistentClient 路径覆盖值。
            embedding_model: embedding 模型覆盖值。
            collection_name: collection 名称覆盖值。
            embedding_function: 测试或调用方注入的 embedding function。
            client: 注入后跳过 HTTP/PersistentClient 创建的 Chroma client。
        """
        self.chroma_host = (chroma_host or os.getenv("CHROMA_HOST", "localhost")).strip()
        self.chroma_port = int(chroma_port or os.getenv("CHROMA_PORT", "8000"))
        self.chroma_path = (
            chroma_path
            or os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_PERSIST_DIRECTORY)
        ).strip()
        self.embedding_model = (
            embedding_model
            or os.getenv("CUSTOMEROPS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        ).strip()
        self.collection_name = (
            collection_name
            or os.getenv("CUSTOMEROPS_KB_COLLECTION", DEFAULT_COLLECTION_NAME)
        ).strip()
        self._embedding_function = embedding_function or build_embedding_function(self.embedding_model)
        self._use_server = False
        self._client = client or self._connect_client()
        collection_kwargs = {
            "name": self.collection_name,
            "metadata": {
                "description": "ShopCare 产品知识库",
                "embedding_model": self.embedding_model,
                "hnsw:space": HNSW_SPACE,
            },
        }
        # 禁止 Chroma 默认 embedding provider（all-MiniLM）。无论 HTTP 还是本地
        # collection，写入和查询都显式传入项目 BGE 向量，因此关闭隐式 provider。
        collection_kwargs["embedding_function"] = None
        self._collection = self._client.get_or_create_collection(**collection_kwargs)

    def _connect_client(self) -> Any:
        """按 HTTP 优先、PersistentClient fallback 的顺序创建 Chroma client。

        Returns:
            已连接的 Chroma client。

        Raises:
            RuntimeError: Chroma SDK 不可用或 HTTP 与本地客户端均创建失败。
        """
        module = _load_chromadb()
        settings = _chroma_settings(module)
        http_kwargs: Dict[str, Any] = {
            "host": self.chroma_host,
            "port": self.chroma_port,
        }
        if settings is not None:
            http_kwargs["settings"] = settings
        try:
            http_client = module.HttpClient(**http_kwargs)
            heartbeat = getattr(http_client, "heartbeat", None)
            if callable(heartbeat):
                heartbeat()
            self._use_server = True
            logger.info(
                "知识库 Chroma HTTP 已连接: %s:%s",
                self.chroma_host,
                self.chroma_port,
            )
            return http_client
        except Exception as exc:
            # HTTP 失败必须显式暴露，PersistentClient 只作为本地可用性降级而非静默替代。
            logger.warning(
                "知识库 Chroma HTTP 连接失败，fallback 到 PersistentClient: "
                "host=%s port=%s path=%s error=%s",
                self.chroma_host,
                self.chroma_port,
                self.chroma_path,
                exc,
            )
        # Why：HTTP Settings 会被 HttpClient 原地改写为 FastAPI；fallback 必须使用
        # 全新本地 Settings，才能真正落盘而不是再次访问失败的 HTTP 端点。
        persistent_settings = _persistent_chroma_settings(module)
        persistent_kwargs: Dict[str, Any] = {"path": self.chroma_path}
        if persistent_settings is not None:
            persistent_kwargs["settings"] = persistent_settings
        try:
            return module.PersistentClient(**persistent_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Chroma HTTP 与 PersistentClient 均不可用: {exc}") from exc

    @staticmethod
    def _slugify(title: str) -> str:
        """为缺失来源 ID 的文档生成稳定可读的来源标识。

        Args:
            title: 文档标题。

        Returns:
            由标题生成的安全来源标识。
        """
        value = re.sub(r"[^\w\u4e00-\u9fff]+", "_", str(title).strip())
        return value.strip("_") or "doc"

    @staticmethod
    def _runtime_id(title: str, index: int, content: str) -> str:
        """生成同一标题、序号和内容下稳定不变的 Chroma runtime ID。

        Args:
            title: 文档标题。
            index: chunk 序号。
            content: chunk 正文。

        Returns:
            稳定的 MD5 文档 ID。
        """
        return hashlib.md5(f"{title}_{index}_{content[:50]}".encode("utf-8")).hexdigest()

    @staticmethod
    def _metadata_for_chunk(
        *,
        title: str,
        source_document_id: str,
        source_chunk_id: str,
        content: str,
        chunk_index: int,
        total_chunks: int,
        corpus_version: str,
        domain: str = "",
    ) -> Dict[str, Any]:
        """构造含业务域与稳定身份的 chunk metadata。

        Args:
            title: 原文标题。
            source_document_id: 稳定的原文 ID。
            source_chunk_id: 稳定的 chunk ID。
            content: chunk 正文。
            chunk_index: 从 0 开始的 chunk 序号。
            total_chunks: 原文切片总数。
            corpus_version: 语料版本。
            domain: consult、order 或 after_sales；未分类旧片段使用空字符串。

        Returns:
            含来源、内容哈希和业务域的 metadata 字典。
        """
        return {
            "title": title,
            "source_document_id": source_document_id,
            "source_chunk_id": source_chunk_id,
            "content_hash": f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}",
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "corpus_version": corpus_version,
            "domain": domain,
        }

    @staticmethod
    def _split_long_body(text: str, max_length: int) -> List[str]:
        """将超过可用长度的正文按段落、句号和硬上限拆开。

        Args:
            text: 待拆分的正文。
            max_length: 正文可占用的最大字符数，已扣除面包屑前缀。

        Returns:
            不超过 ``max_length`` 的正文片段列表。
        """
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n{2,}", text)
            if paragraph.strip()
        ]
        units: List[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= max_length:
                units.append(paragraph)
                continue
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[。！？!?；;])|\n+", paragraph)
                if sentence.strip()
            ]
            for sentence in sentences:
                units.extend(
                    sentence[start : start + max_length]
                    for start in range(0, len(sentence), max_length)
                )

        pieces: List[str] = []
        current = ""
        for unit in units:
            candidate = f"{current}\n\n{unit}" if current else unit
            if len(candidate) <= max_length:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = unit
        if current:
            pieces.append(current)
        return pieces

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = CHUNK_SIZE,
        title: str = "",
    ) -> List[str]:
        """按 Markdown 标题层级切片，注入面包屑并执行 500 字符硬兜底。

        Args:
            text: 待切片的原始文档正文。
            chunk_size: 单片最大字符数，默认锁定为 500。
            title: 文档一级标题，用于构造面包屑根节点。

        Returns:
            带独立上下文前缀且不超过 ``chunk_size`` 的文本列表。
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正数")
        normalized = str(text or "").replace("\r\n", "\n").strip()
        if not normalized:
            return []

        sections: List[Tuple[List[str], str]] = []
        heading_stack: List[Tuple[int, str]] = []
        body_lines: List[str] = []

        def flush_section() -> None:
            """将当前标题路径下累积的正文保存为一个待拆分小节。"""
            body = "\n".join(body_lines).strip()
            if body:
                sections.append(([item[1] for item in heading_stack], body))
            body_lines.clear()

        for line in normalized.splitlines():
            match = re.match(r"^(#{2,6})\s+(.+?)\s*$", line)
            if match:
                flush_section()
                level = len(match.group(1))
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, match.group(2).strip()))
            else:
                body_lines.append(line)
        flush_section()

        if not sections:
            sections = [([], normalized)]

        chunks: List[str] = []
        for headings, body in sections:
            breadcrumb = [str(title).strip()] if str(title).strip() else []
            breadcrumb.extend(headings)
            prefix = f"[{' > '.join(breadcrumb)}]\n" if breadcrumb else ""
            available = chunk_size - len(prefix)
            if available <= 0:
                raise ValueError("文档标题面包屑已超过 chunk_size")
            for piece in self._split_long_body(body, available):
                chunks.append(f"{prefix}{piece}")
        return [chunk for chunk in chunks if chunk.strip()]

    def add_documents(self, documents: Sequence[Dict[str, Any]]) -> int:
        """切片并批量写入产品知识文档。

        Args:
            documents: 每项至少包含 ``title`` 和 ``content``，可选来源 ID 与语料版本。

        Returns:
            写入的 chunk 数量。
        """
        ids: List[str] = []
        contents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for document in documents:
            title = str(document.get("title") or "")
            content = str(document.get("content") or "")
            chunks = self._chunk_text(content, title=title)
            if not chunks:
                continue
            source_document_id = str(document.get("source_document_id") or self._slugify(title))
            corpus_version = str(document.get("corpus_version") or DEFAULT_CORPUS_VERSION)
            domain = str(document.get("domain") or "").strip()
            if domain not in BUSINESS_KB_DOMAINS:
                raise ValueError(
                    f"知识文档必须标记一个有效业务域 {sorted(BUSINESS_KB_DOMAINS)}: "
                    f"{source_document_id}"
                )
            for index, chunk in enumerate(chunks):
                ids.append(self._runtime_id(title, index, chunk))
                contents.append(chunk)
                metadatas.append(
                    self._metadata_for_chunk(
                        title=title,
                        source_document_id=source_document_id,
                        source_chunk_id=f"{source_document_id}_chunk_{index}",
                        content=chunk,
                        chunk_index=index,
                        total_chunks=len(chunks),
                        corpus_version=corpus_version,
                        domain=domain,
                    )
                )
        if ids:
            # HTTP Chroma 服务端不能复用本地 embedding function；显式传入本地
            # BGE 向量，避免服务端回退到 all-MiniLM-L6-v2 并触发下载。
            embeddings = self._embedding_function(contents)
            upsert = getattr(self._collection, "upsert", None)
            write = upsert if callable(upsert) else self._collection.add
            write(ids=ids, documents=contents, metadatas=metadatas, embeddings=embeddings)
            logger.info("知识库写入 %s 个文档片段", len(ids))
        return len(ids)

    def add_stable_chunks(self, chunks: Sequence[Dict[str, Any]]) -> int:
        """按既有稳定身份写入评测切片，不重新执行产品文档切片。

        Args:
            chunks: 已完成切片的评测片段，必须包含 ``id``、``content`` 和
                七项基础 metadata；允许附带 domain、agent 等评测字段。

        Returns:
            实际写入或更新的切片数量。

        Raises:
            ValueError: 片段缺少稳定身份或七项 metadata。

        说明：评测语料必须与产品知识库物理隔离，且保留原始 chunk ID；因此
        本方法只做校验与 upsert，不调用 ``_chunk_text``，也不改变正文内容。
        """

        required = {
            "title",
            "source_document_id",
            "source_chunk_id",
            "content_hash",
            "chunk_index",
            "total_chunks",
            "corpus_version",
        }
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError("评测 chunk 必须是对象")
            chunk_id = str(chunk.get("id") or chunk.get("source_chunk_id") or "").strip()
            content = str(chunk.get("content") or "")
            if not chunk_id or not content:
                raise ValueError("评测 chunk 必须包含非空 id 与 content")
            metadata = dict(chunk)
            metadata.pop("id", None)
            metadata.pop("content", None)
            missing = required.difference(metadata)
            if missing:
                raise ValueError(f"评测 chunk 缺少 metadata: {sorted(missing)}")
            # Chroma metadata 只接受标量；评测字段也统一转为稳定字符串或数值。
            clean_metadata: Dict[str, Any] = {}
            for key, value in metadata.items():
                if value is None:
                    clean_metadata[key] = ""
                elif isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)
            ids.append(chunk_id)
            documents.append(content)
            metadatas.append(clean_metadata)
        if not ids:
            return 0
        embeddings = self._embedding_function(documents)
        upsert = getattr(self._collection, "upsert", None)
        if callable(upsert):
            upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        logger.info(
            "评测知识库写入 %s 个稳定文档片段: collection=%s",
            len(ids),
            getattr(self, "collection_name", "unknown"),
        )
        return len(ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """执行基础 Chroma 向量检索并返回稳定 chunk 身份。

        Args:
            query: 用户检索文本。
            top_k: 最多返回的结果数。
            domain: 可选业务域过滤。指定后只查该域；过滤异常直接失败，不回退全库。

        Returns:
            包含 ``title``、``content``、``score``、``id``、``runtime_id`` 和
            ``source_chunk_id`` 的结果列表；score 是距离转换值而非概率。
        """
        bounded_top_k = max(0, int(top_k))
        if bounded_top_k == 0:
            return []
        query_text = str(query or "")
        # HttpClient 不会把本地 embedding_function 传给 Chroma 服务端；
        # 如果使用 query_texts，Chroma 会自动下载默认 all-MiniLM-L6-v2。
        # 这里强制本地 BGE 生成向量，再传 query_embeddings，避免错误模型和网络下载。
        query_embedding = self._embedding_function([query_text])
        query_kwargs = {
            "query_embeddings": query_embedding,
            "n_results": bounded_top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if domain:
            if domain not in BUSINESS_KB_DOMAINS:
                raise ValueError(f"不支持的知识业务域: {domain}")
            results = self._collection.query(**query_kwargs, where={"domain": domain})
        else:
            results = self._collection.query(**query_kwargs)
        documents = self._first_nested(results.get("documents"))
        metadatas = self._first_nested(results.get("metadatas"))
        distances = self._first_nested(results.get("distances"))
        ids = self._first_nested(results.get("ids"))
        rows: List[Dict[str, Any]] = []
        for index, content in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            metadata = metadata if isinstance(metadata, dict) else {}
            runtime_id = str(ids[index]) if index < len(ids) and ids[index] else ""
            distance = distances[index] if index < len(distances) else 1.0
            rows.append(
                {
                    "title": str(metadata.get("title", "")),
                    "content": str(content),
                    "score": round(1.0 - float(distance), 4),
                    "id": runtime_id,
                    "runtime_id": runtime_id,
                    "source_chunk_id": str(metadata.get("source_chunk_id", "")),
                    "source_document_id": str(metadata.get("source_document_id", "")),
                    "content_hash": str(metadata.get("content_hash", "")),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "total_chunks": metadata.get("total_chunks", 0),
                    "corpus_version": str(metadata.get("corpus_version", "")),
                    "domain": str(metadata.get("domain", "")),
                }
            )
        return rows

    @staticmethod
    def _first_nested(value: Any) -> List[Any]:
        """统一 Chroma 返回的二维列表与测试替身的扁平列表。

        Args:
            value: Chroma 返回字段。

        Returns:
            第一组结果列表。
        """
        if isinstance(value, list) and value and isinstance(value[0], list):
            return value[0]
        return value if isinstance(value, list) else []

    @property
    def doc_count(self) -> int:
        """返回当前 collection 中的 chunk 数量。"""
        return int(self._collection.count())
