"""ShopCare Step 7.3 支持性 API 的离线契约测试。

测试只验证知识库管理、内存 Trace 和本地 Skill 热加载边界；所有外部存储、
模型与业务系统均使用替身，不访问真实网络。
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.main as api_main


class FakeRedis:
    """提供 lifespan 所需的异步关闭接口。"""

    async def aclose(self) -> None:
        return None


class FakeCollection:
    """提供健康检查与统计所需的 chunk 数量。"""

    def count(self) -> int:
        return 18


class FakeKnowledgeBase:
    """记录管理 API 对知识库的调用，不加载 embedding 或 Chroma。"""

    def __init__(self, **kwargs) -> None:
        self.collection_name = kwargs.get("collection_name", "knowledge_base_zh")
        self.embedding_model = kwargs.get("embedding_model", "BAAI/bge-small-zh-v1.5")
        self._collection = FakeCollection()
        self._client = SimpleNamespace()
        self.search_calls: list[tuple[str, int, str | None]] = []
        self.added_documents: list[dict] = []

    def search(self, query: str, top_k: int = 5, domain: str | None = None):
        self.search_calls.append((query, top_k, domain))
        return [
            {
                "title": "技术排障",
                "source_chunk_id": "technical_troubleshooting_chunk_0",
                "content": "401 表示认证失败。",
                "score": 0.91,
                "id": "runtime-1",
            }
        ]

    def add_documents(self, documents):
        self.added_documents.extend(documents)
        return len(documents)

    @property
    def doc_count(self) -> int:
        return 18


class FakeSkillManager:
    """验证技能列表读取与 reload 调用。"""

    def __init__(self) -> None:
        self.skills = ["default"]
        self.reload_calls = 0

    def reload_skills(self):
        self.reload_calls += 1
        self.skills = ["default", "refund"]
        return {"success": True, "skills": list(self.skills), "count": len(self.skills)}


def _prepare_app(monkeypatch):
    """以离线依赖启动应用，并替换状态中的知识库与技能管理器。"""

    monkeypatch.setattr(api_main, "KnowledgeBase", FakeKnowledgeBase)
    monkeypatch.setattr(api_main, "_create_redis_client", lambda url: FakeRedis())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = TestClient(api_main.app)
    client.__enter__()
    knowledge_base = FakeKnowledgeBase(collection_name="knowledge_base_zh")
    skill_manager = FakeSkillManager()
    api_main.app.state.knowledge_base = knowledge_base
    api_main.app.state.skill_manager = skill_manager
    api_main.app.state.agent_orchestrator._recent_traces.clear()
    store = api_main.app.state.sqlite_store
    identity = store.authenticate("agent-demo", "shopcare")
    assert identity is not None
    client.headers.update({"Authorization": f"Bearer {identity['token']}"})
    return client, knowledge_base, skill_manager


def test_search_and_knowledge_add_return_structured_results(monkeypatch):
    """/search 与 /knowledge/add 应复用知识库契约并返回结构化数据。"""

    client, knowledge_base, _ = _prepare_app(monkeypatch)
    try:
        response = client.post(
            "/search",
            json={"query": "401 认证失败", "top_k": 3, "domain": "consult"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["results"][0]["chunk_id"] == "technical_troubleshooting_chunk_0"
        assert body["results"][0]["title"] == "技术排障"
        assert knowledge_base.search_calls == [("401 认证失败", 3, "consult")]

        add_response = client.post(
            "/knowledge/add",
            json={
                "title": "临时规则",
                "content": "请先检查网络。",
                "domain": "order",
            },
        )
        assert add_response.status_code == 200
        assert add_response.json()["added"] == 1
        assert knowledge_base.added_documents == [
            {"title": "临时规则", "content": "请先检查网络。", "domain": "order"}
        ]
    finally:
        client.__exit__(None, None, None)


def test_knowledge_upload_enforces_suffix_and_ten_mb_limit(monkeypatch):
    """上传只允许 txt/md/json，并在 10MB 后立即拒绝。"""

    client, knowledge_base, _ = _prepare_app(monkeypatch)
    try:
        response = client.post(
            "/knowledge/upload",
            files={"file": ("guide.md", "标题\n正文".encode("utf-8"), "text/markdown")},
            data={"domain": "consult"},
        )
        assert response.status_code == 200
        assert response.json()["filename"] == "guide.md"
        assert knowledge_base.added_documents[-1]["content"] == "标题\n正文"
        assert knowledge_base.added_documents[-1]["domain"] == "consult"

        rejected = client.post(
            "/knowledge/upload",
            files={"file": ("payload.exe", b"MZ", "application/octet-stream")},
            data={"domain": "consult"},
        )
        assert rejected.status_code == 400

        oversized = client.post(
            "/knowledge/upload",
            files={
                "file": (
                    "oversized.md",
                    b"x" * (10 * 1024 * 1024 + 1),
                    "text/markdown",
                )
            },
            data={"domain": "consult"},
        )
        assert oversized.status_code == 400
    finally:
        client.__exit__(None, None, None)


def test_knowledge_writes_require_an_explicit_business_domain(monkeypatch):
    """认证通过后，新增和上传知识都不能生成未标业务域的数据。"""

    client, knowledge_base, _ = _prepare_app(monkeypatch)
    try:
        add_response = client.post(
            "/knowledge/add",
            json={"title": "未分类规则", "content": "不能进入知识库。"},
        )
        upload_response = client.post(
            "/knowledge/upload",
            files={"file": ("unclassified.md", "不能进入知识库。".encode("utf-8"), "text/markdown")},
        )

        assert add_response.status_code == 422
        assert upload_response.status_code == 422
        assert knowledge_base.added_documents == []
    finally:
        client.__exit__(None, None, None)


def test_knowledge_stats_and_trace_memory_routes(monkeypatch):
    """统计和 Trace API 应返回内存数据，并明确重启会丢失。"""

    client, _, _ = _prepare_app(monkeypatch)
    try:
        stats = client.get("/knowledge/stats")
        assert stats.status_code == 200
        assert stats.json()["doc_count"] == 18
        assert stats.json()["collection_name"] == "knowledge_base_zh"

        api_main.app.state.agent_orchestrator._recent_traces.append(
            {
                "request_id": "req-1",
                "tool_traces": [{"tool_name": "knowledge_search", "status": "success"}],
            }
        )
        all_traces = client.get("/trace/tools")
        assert all_traces.status_code == 200
        assert all_traces.json()["storage"] == "memory"
        assert all_traces.json()["restart_loses_data"] is True
        assert all_traces.json()["traces"][0]["request_id"] == "req-1"

        one_trace = client.get("/trace/tool/req-1")
        assert one_trace.status_code == 200
        assert one_trace.json()["trace"]["request_id"] == "req-1"
        assert client.get("/trace/tool/not-found").status_code == 404
    finally:
        client.__exit__(None, None, None)


def test_token_trace_routes_expose_stage_usage_without_prompt_content(monkeypatch):
    """Token 台账只返回阶段用量和内存边界，不暴露原始请求文本。"""

    client, _, _ = _prepare_app(monkeypatch)
    try:
        api_main.app.state.token_usage.record_request(
            "token-1",
            {"intent": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}},
            latency_ms=12.3,
        )
        one = client.get("/trace/token/token-1")
        assert one.status_code == 200
        assert one.json()["trace"]["total"]["total_tokens"] == 5
        assert "message" not in one.json()["trace"]

        summary = client.get("/trace/tokens")
        assert summary.status_code == 200
        assert summary.json()["by_stage"]["intent"]["total_tokens"] == 5
        assert summary.json()["restart_loses_data"] is True
    finally:
        client.__exit__(None, None, None)


def test_skills_list_and_reload(monkeypatch):
    """技能列表与热加载路由应调用本地 SkillManager。"""

    client, _, skill_manager = _prepare_app(monkeypatch)
    try:
        listed = client.get("/skills")
        assert listed.status_code == 200
        assert listed.json()["skills"] == ["default"]

        reloaded = client.post("/skills/reload")
        assert reloaded.status_code == 200
        assert reloaded.json()["success"] is True
        assert reloaded.json()["skills"] == ["default", "refund"]
        assert skill_manager.reload_calls == 1
    finally:
        client.__exit__(None, None, None)
