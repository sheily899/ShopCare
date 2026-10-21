"""身份、会话归属与管理接口访问边界的离线回归测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from storage.sqlite_case_store import SQLiteCaseStore


class _FakeRedis:
    async def aclose(self) -> None:
        return None


class _FakeKnowledgeBase:
    collection_name = "knowledge_base_zh"
    embedding_model = "local-test-model"
    _collection = SimpleNamespace(count=lambda: 0)
    _client = SimpleNamespace()

    def __init__(self, **kwargs) -> None:
        self.collection_name = str(kwargs.get("collection_name", self.collection_name))
        self.embedding_model = str(kwargs.get("embedding_model", self.embedding_model))

    def search(self, query: str, top_k: int = 5):
        return []

    def add_documents(self, documents):
        return len(documents)


@pytest.fixture
def secured_app(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "KnowledgeBase", _FakeKnowledgeBase)
    monkeypatch.setattr(api_main, "_create_redis_client", lambda url: _FakeRedis())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("HANDOFF", "false")
    with TestClient(api_main.app) as client:
        store = SQLiteCaseStore(tmp_path / "security.sqlite3")
        api_main.app.state.sqlite_store = store
        consumer = store.authenticate("consumer-demo", "shopcare")
        agent = store.authenticate("agent-demo", "shopcare")
        store.register_consumer("other-consumer", "shopcare", "另一位消费者")
        other = store.authenticate("other-consumer", "shopcare")
        yield client, store, consumer, agent, other


def test_consumer_chat_and_slot_writes_require_identity(secured_app):
    client, store, consumer, _, _ = secured_app
    store.ensure_session("consumer-demo", "owned-case")

    assert client.post("/chat", json={"message": "你好"}).status_code == 401
    assert client.post("/chat/stream", json={"message": "你好"}).status_code == 401
    assert client.post(
        "/chat/owned-case/slots",
        json={"question_id": "q1", "slot_id": "return_reason", "value": "不合适"},
    ).status_code == 401

    # The SSE route must reject before returning a successful streaming response.
    assert client.post("/chat/stream", json={"message": "转人工"}).status_code == 401
    assert client.get("/chat/owned-case/messages").status_code == 401


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("get", "/service/sessions", {}),
        ("get", "/service/dashboard", {}),
        ("get", "/service/sessions/owned-case", {}),
        ("post", "/service/sessions/owned-case/refresh", {}),
        ("get", "/service/sessions/owned-case/handoff/events", {}),
        ("post", "/service/sessions/owned-case/messages", {"json": {"content": "reply"}}),
        ("patch", "/service/sessions/owned-case/ticket", {"json": {"fields": {}, "expected_version": 0}}),
        ("post", "/service/sessions/owned-case/ticket/confirm", {}),
    ],
)
def test_service_session_management_routes_require_identity(secured_app, monkeypatch, method, path, kwargs):
    monkeypatch.setenv("HANDOFF", "true")
    client, store, _, _, _ = secured_app
    store.ensure_session("consumer-demo", "owned-case")
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 401


def test_consumer_chat_and_slot_writes_are_bound_to_authenticated_owner(secured_app):
    client, store, consumer, _, other = secured_app
    store.ensure_session("consumer-demo", "owned-case")
    store.ensure_session("other-consumer", "other-case")

    headers = {"Authorization": f"Bearer {consumer['token']}"}
    mismatch = client.post(
        "/chat",
        headers=headers,
        json={"message": "你好", "user_id": "other-consumer"},
    )
    foreign_chat = client.post(
        "/chat",
        headers=headers,
        json={"message": "你好", "user_id": "consumer-demo", "conv_id": "other-case"},
    )
    foreign_slot = client.post(
        "/chat/other-case/slots",
        headers=headers,
        json={"question_id": "q1", "slot_id": "return_reason", "value": "不合适"},
    )

    assert mismatch.status_code == 403
    assert foreign_chat.status_code == 403
    assert foreign_slot.status_code == 403


def test_consumer_read_routes_are_bound_to_authenticated_owner(secured_app):
    client, store, consumer, _, _ = secured_app
    store.ensure_session("consumer-demo", "owned-case")
    store.ensure_session("other-consumer", "other-case")
    headers = {"Authorization": f"Bearer {consumer['token']}"}

    own = client.get("/chat/owned-case/messages", headers=headers)
    foreign = client.get("/chat/other-case/messages", headers=headers)

    assert own.status_code == 200
    assert foreign.status_code == 403


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/search", {"json": {"query": "成分", "top_k": 2}}),
        ("post", "/knowledge/add", {"json": {"title": "test", "content": "test"}}),
        ("post", "/knowledge/upload", {"files": {"file": ("test.md", b"test", "text/markdown")}}),
        ("get", "/knowledge/stats", {}),
        ("get", "/trace/tool/missing", {}),
        ("get", "/trace/tools", {}),
        ("get", "/trace/token/missing", {}),
        ("get", "/trace/tokens", {}),
        ("get", "/skills", {}),
        ("post", "/skills/reload", {}),
    ],
)
def test_management_routes_require_agent_identity(secured_app, method, path, kwargs):
    client, _, _, _, _ = secured_app
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 401


def test_management_routes_reject_consumer_identity(secured_app):
    client, _, consumer, _, _ = secured_app
    headers = {"Authorization": f"Bearer {consumer['token']}"}

    assert client.get("/knowledge/stats", headers=headers).status_code == 401
    assert client.get("/trace/tools", headers=headers).status_code == 401
    assert client.get("/skills", headers=headers).status_code == 401


def test_malformed_bearer_scheme_is_rejected(secured_app):
    client, _, _, _, _ = secured_app
    assert client.get("/knowledge/stats", headers={"Authorization": "Basic invalid"}).status_code == 401


def test_management_and_session_routes_remain_available_to_authenticated_agent(secured_app):
    client, store, _, agent, _ = secured_app
    store.ensure_session("consumer-demo", "assigned-case")
    headers = {"Authorization": f"Bearer {agent['token']}"}

    assert client.get("/knowledge/stats", headers=headers).status_code == 200
    assert client.get("/service/sessions", headers=headers).status_code == 200
    assert client.get("/service/sessions/assigned-case", headers=headers).status_code == 200


def test_agent_reply_uses_authenticated_agent_id_not_client_supplied_id(secured_app):
    client, store, _, agent, _ = secured_app
    store.ensure_session("consumer-demo", "assigned-case")
    headers = {"Authorization": f"Bearer {agent['token']}"}

    response = client.post(
        "/service/sessions/assigned-case/messages",
        headers=headers,
        json={"agent_id": "forged-agent", "content": "已收到"},
    )

    assert response.status_code == 200
    assert response.json()["message"]["sender_id"] == "agent-demo"


def test_health_remains_public_for_container_readiness(secured_app):
    client, _, _, _, _ = secured_app
    response = client.get("/health")
    assert response.status_code == 200
