"""ShopCare /health 与 /chat 主链路的离线集成测试。

测试替换 Redis、Chroma、意图识别、编排和画像更新依赖，只验证 API 时序、契约
映射和后台任务异常隔离，不访问真实外部服务。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.main as api_main
from agents.agent_orchestrator import AgentType, OrchestratorResult
from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel
from memory.conversation_memory import MemoryContext, Message, MsgRole


class FakeRedis:
    """支持生命周期关闭的异步 Redis 替身。"""

    async def aclose(self) -> None:
        return None


class FakeCollection:
    """提供 health 所需 count 的 Chroma collection 替身。"""

    def count(self) -> int:
        return 18


class FakeKnowledgeBase:
    """不加载 embedding 模型的知识库替身。"""

    def __init__(self, **kwargs) -> None:
        self.collection_name = kwargs.get("collection_name", "knowledge_base_zh")
        self.embedding_model = kwargs.get("embedding_model", "BAAI/bge-small-zh-v1.5")
        self._collection = FakeCollection()
        self._client = SimpleNamespace()

    def search(self, query: str, top_k: int = 5):
        return []


class FakeMemoryManager:
    """记录上下文读取、消息写回和画像后台任务。"""

    def __init__(self, *, update_error: bool = False) -> None:
        self.calls: list[tuple] = []
        self.update_error = update_error

    async def get_context(self, user_id: str, conv_id: str, query: str) -> MemoryContext:
        self.calls.append(("get_context", user_id, conv_id, query))
        return MemoryContext(
            recent_messages=[Message(MsgRole.USER, "之前询问过订单")],
        )

    async def add_message(self, user_id: str, conv_id: str, role: MsgRole, content: str, metadata=None):
        self.calls.append(("add_message", role, content))
        return False

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        self.calls.append(("update_profile", user_id, conv_id))
        if self.update_error:
            raise RuntimeError("模拟画像更新失败")


class FakeIntentRecognizer:
    """固定返回技术意图并记录最近 5 条历史。"""

    def __init__(self) -> None:
        self.histories = []

    async def recognize_intent(self, message: str, history):
        self.histories.append((message, history))
        return IntentResult(
            intent=IntentCategory.TECHNICAL,
            confidence=0.91,
            urgency=UrgencyLevel.LOW,
            intent_group="technical",
            entities={"order_id": [], "product": [], "date": [], "amount": [], "error_code": ["500"]},
            reasoning="测试识别",
            latency_ms=1.0,
            source_scores={"llm": 0.91},
            source_intents={"llm": "technical"},
            intent_diagnostics={"test": True},
        )


class FakeOrchestrator:
    """返回可序列化的编排结果并提供 health 统计。"""

    def __init__(self) -> None:
        self.requests = []

    def get_stats(self):
        return {"technical": {"total": 1, "success": 1}}

    async def run(self, request):
        self.requests.append(request)
        return OrchestratorResult(
            request_id=request.request_id,
            response="已生成只读技术排查建议。",
            agent_type=AgentType.TECHNICAL,
            intent=request.intent,
            latency_ms=4.5,
            agent_types=[AgentType.TECHNICAL],
            primary_agent=AgentType.TECHNICAL,
            tools_used=["lookup_error_code"],
            tool_traces=[{"tool_name": "lookup_error_code", "status": "success"}],
            routing_reason="technical intent",
            routing_confidence=0.91,
            intent_diagnostics={"orchestrator": True},
            secondary_intents=[],
            knowledge_required=False,
        )


def _prepare_app(monkeypatch, *, update_error: bool = False):
    """以 fake 基础设施启动应用，并返回进入上下文后的状态替身。"""

    monkeypatch.setattr(api_main, "KnowledgeBase", FakeKnowledgeBase)
    monkeypatch.setattr(api_main, "_create_redis_client", lambda url: FakeRedis())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = TestClient(api_main.app)
    client.__enter__()
    memory = FakeMemoryManager(update_error=update_error)
    recognizer = FakeIntentRecognizer()
    orchestrator = FakeOrchestrator()
    api_main.app.state.memory_manager = memory
    api_main.app.state.intent_recognizer = recognizer
    api_main.app.state.agent_orchestrator = orchestrator
    return client, memory, recognizer, orchestrator


def test_health_returns_readiness_stats_and_knowledge_metadata(monkeypatch):
    """健康检查应返回 Agent 统计与产品知识库元数据。"""

    client, _, _, _ = _prepare_app(monkeypatch)
    try:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"healthy", "ready"}
        assert body["agent_stats"]["technical"]["success"] == 1
        assert body["knowledge_base"] == {
            "collection_name": "knowledge_base_zh",
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "doc_count": 18,
        }
    finally:
        client.__exit__(None, None, None)


def test_chat_runs_context_intent_orchestration_memory_and_profile_task(monkeypatch):
    """完整对话流应按上下文、识别、编排、写回、画像任务顺序执行。"""

    client, memory, recognizer, orchestrator = _prepare_app(monkeypatch)
    try:
        response = client.post("/chat", json={"message": "电脑报 500，怎么排查？", "user_id": "u-1"})
        assert response.status_code == 200
        body = response.json()
        assert body["conv_id"]
        assert len(body["request_id"]) == 8
        assert body["intent"] == "technical"
        assert body["agent_type"] == "technical"
        assert body["response"]
        assert body["tools_used"] == ["lookup_error_code"]
        assert body["knowledge_used"] is True
        assert recognizer.histories[0][1][-1].content == "之前询问过订单"
        assert orchestrator.requests[0].request_id == body["request_id"]
        assert [item[0] for item in memory.calls[:3]] == [
            "get_context",
            "add_message",
            "add_message",
        ]
        assert memory.calls[1][1] is MsgRole.USER
        assert memory.calls[2][1] is MsgRole.ASSISTANT
        assert any(item[0] == "update_profile" for item in memory.calls)
    finally:
        client.__exit__(None, None, None)


def test_chat_profile_task_failure_does_not_change_http_success(monkeypatch):
    """画像后台协程失败时，已完成的 /chat 响应仍保持 200。"""

    client, memory, _, _ = _prepare_app(monkeypatch, update_error=True)
    try:
        response = client.post("/chat", json={"message": "请帮我排查登录问题"})
        assert response.status_code == 200
        assert response.json()["response"] == "已生成只读技术排查建议。"
        assert any(item[0] == "update_profile" for item in memory.calls)
    finally:
        client.__exit__(None, None, None)


def test_chat_reports_end_to_end_latency_and_preserves_orchestrator_latency(monkeypatch):
    """ChatResponse 的 latency_ms 应覆盖 API 全链路，并保留编排阶段诊断值。"""

    client, _, _, orchestrator = _prepare_app(monkeypatch)
    original_run = orchestrator.run

    async def delayed_run(request):
        await asyncio.sleep(0.03)
        return await original_run(request)

    orchestrator.run = delayed_run
    try:
        response = client.post("/chat", json={"message": "请帮我排查登录问题"})
        assert response.status_code == 200
        body = response.json()
        assert body["latency_ms"] > 10.0
        assert body["intent_diagnostics"]["orchestrator_latency_ms"] == 4.5
    finally:
        client.__exit__(None, None, None)


def test_health_returns_503_when_app_is_not_ready(monkeypatch):
    """未就绪状态必须使用明确的 503，而不是伪造 healthy。"""

    client, _, _, _ = _prepare_app(monkeypatch)
    try:
        api_main.app.state.ready = False
        response = client.get("/health")
        assert response.status_code == 503
    finally:
        client.__exit__(None, None, None)
