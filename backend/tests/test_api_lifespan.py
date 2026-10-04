"""FastAPI lifespan 组件装配的离线契约测试。

测试只验证启动时的依赖注入、工具注册和资源释放，不访问真实 Redis、Chroma、
Anthropic 或任何外部业务系统。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import api.main as api_main


class FakeRedis:
    """记录关闭动作的异步 Redis 替身。"""

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeKnowledgeBase:
    """不加载模型、不联网的知识库替身。"""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self._client = object()
        self.collection_name = kwargs.get("collection_name", "knowledge_base_zh")
        self.embedding_model = kwargs.get("embedding_model", "BAAI/bge-small-zh-v1.5")

    def search(self, query: str, top_k: int = 5):
        return []


def _patch_dependencies(monkeypatch, *, api_key: str | None = "test-key"):
    redis = FakeRedis()
    monkeypatch.setattr(api_main, "KnowledgeBase", FakeKnowledgeBase)
    monkeypatch.setattr(api_main, "_create_redis_client", lambda url: redis)
    monkeypatch.setenv("ANTHROPIC_API_KEY", api_key or "")
    return redis


def test_lifespan_mounts_core_components_and_knowledge_tool(monkeypatch):
    """进入 TestClient 时应装配所有 Step7.1 核心状态。"""

    redis = _patch_dependencies(monkeypatch)
    with TestClient(api_main.app):
        state = api_main.app.state
        for name in (
            "settings",
            "redis_client",
            "knowledge_base",
            "tool_manager",
            "memory_manager",
            "intent_recognizer",
            "agent_orchestrator",
            "skill_manager",
            "performance_monitor",
            "evaluator",
        ):
            assert hasattr(state, name)
        assert state.ready is True
        assert set(state.agents) == {"general", "technical", "billing", "escalation"}
        tool = state.tool_manager._tools["knowledge_search"]
        assert tool.cache_ttl == 300
        assert tool.supports_rerank is True
    assert redis.closed is True


def test_lifespan_warns_when_anthropic_key_is_missing(monkeypatch, caplog):
    """缺少 API key 时保留离线可启动能力，同时留下明确警告。"""

    _patch_dependencies(monkeypatch, api_key=None)
    caplog.set_level("WARNING")
    with TestClient(api_main.app):
        assert api_main.app.state.anthropic_client is None
        assert api_main.app.state.ready is True
    assert "ANTHROPIC_API_KEY" in caplog.text


def test_lifespan_closes_async_resources(monkeypatch):
    """退出 lifespan 必须调用可用资源的异步关闭方法。"""

    redis = _patch_dependencies(monkeypatch)
    with TestClient(api_main.app):
        assert redis.closed is False
    assert redis.closed is True
