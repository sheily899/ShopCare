"""ShopCare /health 与 /chat 主链路的离线集成测试。

测试替换 Redis、Chroma、意图识别、编排和画像更新依赖，只验证 API 时序、契约
映射和后台任务异常隔离，不访问真实外部服务。
"""

from __future__ import annotations

import asyncio
import sqlite3
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from agents.agent_orchestrator import AgentOrchestrator, AgentType, OrchestratorResult, TechnicalAgent
from agents.tools import BEAUTY_LOOKUP_ORDER_TOOL
from intent.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel
from memory.conversation_memory import MemoryContext, Message, MsgRole
from storage.sqlite_case_store import SQLiteCaseStore


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
    """固定返回美妆履约意图并记录最近 5 条历史。"""

    def __init__(self) -> None:
        self.histories = []

    async def recognize_intent(self, message: str, history):
        self.histories.append((message, history))
        return IntentResult(
            intent=IntentCategory.ORDER_STATUS_BEAUTY,
            confidence=0.91,
            urgency=UrgencyLevel.LOW,
            intent_group="order",
            entities={"order_id": ["SC-DEMO-4821"], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="测试识别",
            latency_ms=1.0,
            source_scores={"llm": 0.91},
            source_intents={"llm": "order_status"},
            intent_diagnostics={"test": True},
        )


class FakeOrchestrator:
    """返回可序列化的编排结果并提供 health 统计。"""

    def __init__(self) -> None:
        self.requests = []

    def get_stats(self):
        return {"order": {"total": 1, "success": 1}}

    async def run(self, request):
        self.requests.append(request)
        return OrchestratorResult(
            request_id=request.request_id,
            response="SC-DEMO-4821 的物流状态已核对。",
            agent_type=AgentType.FULFILLMENT,
            intent=request.intent,
            latency_ms=4.5,
            agent_types=[AgentType.FULFILLMENT],
            primary_agent=AgentType.FULFILLMENT,
            tools_used=["lookup_order", "lookup_logistics"],
            tool_traces=[
                {"tool_name": "lookup_order", "status": "success"},
                {"tool_name": "lookup_logistics", "status": "success"},
            ],
            routing_reason="beauty order status intent",
            routing_confidence=0.91,
            intent_diagnostics={"orchestrator": True},
            secondary_intents=[],
            knowledge_required=False,
            usage={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            usage_by_stage={
                "intent": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
                "agent_order": {"input_tokens": 8, "output_tokens": 6, "total_tokens": 14},
            },
        )


def _prepare_app(monkeypatch, *, update_error: bool = False):
    """以 fake 基础设施启动应用，并返回进入上下文后的状态替身。"""

    monkeypatch.setattr(api_main, "KnowledgeBase", FakeKnowledgeBase)
    monkeypatch.setattr(api_main, "_create_redis_client", lambda url: FakeRedis())
    # 这些旧集成用例只验证 HTTP/Agent 契约；交接后台由独立测试覆盖，
    # 避免在 TestClient 中对默认本地数据库触发异步恢复或真实识别。
    monkeypatch.setattr(api_main, "_schedule_handoff_enrichment", lambda *args, **kwargs: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = TestClient(api_main.app)
    client.__enter__()
    memory = FakeMemoryManager(update_error=update_error)
    recognizer = FakeIntentRecognizer()
    orchestrator = FakeOrchestrator()
    api_main.app.state.memory_manager = memory
    api_main.app.state.intent_recognizer = recognizer
    api_main.app.state.agent_orchestrator = orchestrator
    _set_consumer_auth(client, api_main.app.state.sqlite_store)
    return client, memory, recognizer, orchestrator


def _set_consumer_auth(client: TestClient, store: SQLiteCaseStore) -> None:
    """Bind integration requests to the consumer identity in the active test DB."""
    identity = store.authenticate("consumer-demo", "shopcare")
    assert identity is not None
    client.headers.update({"Authorization": f"Bearer {identity['token']}"})


def test_health_returns_readiness_stats_and_knowledge_metadata(monkeypatch):
    """健康检查应返回 Agent 统计与产品知识库元数据。"""

    client, _, _, _ = _prepare_app(monkeypatch)
    try:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"healthy", "ready"}
        assert body["agent_stats"]["order"]["success"] == 1
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
        response = client.post("/chat", json={"message": "订单 SC-DEMO-4821 的物流到哪里了？", "user_id": "consumer-demo"})
        assert response.status_code == 200
        body = response.json()
        assert body["conv_id"]
        assert len(body["request_id"]) == 8
        assert body["intent"] == "order_status"
        assert body["agent_type"] == "order"
        assert body["response"]
        assert body["tools_used"] == ["lookup_order", "lookup_logistics"]
        assert body["knowledge_used"] is False
        assert body["usage"] == {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
        assert body["usage_by_stage"] == {
            "intent": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            "agent_order": {"input_tokens": 8, "output_tokens": 6, "total_tokens": 14},
        }
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
        assert response.json()["response"] == "SC-DEMO-4821 的物流状态已核对。"
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


def test_compound_handoff_flows_from_chat_into_editable_submitted_ticket(monkeypatch, tmp_path):
    """复合风险请求应贯穿消费者答复、共享事实、客服摘要、工单与提交状态。"""

    monkeypatch.setenv("HANDOFF", "false")
    client, memory, _, _ = _prepare_app(monkeypatch)
    conv_id = "e2e-compound-handoff"
    store = SQLiteCaseStore(tmp_path / "shopcare-e2e.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)
    api_main.app.state.agent_orchestrator = AgentOrchestrator()
    agent_token = store.authenticate("agent-demo", "shopcare")["token"]

    class CompoundHandoffRecognizer:
        async def recognize_intent(self, message, history):
            return IntentResult(
                intent=IntentCategory.HUMAN_HANDOFF,
                confidence=0.99,
                urgency=UrgencyLevel.HIGH,
                intent_group="after_sales",
                entities={
                    "order_id": ["SC-DEMO-4821"],
                    "product_name": ["轻透粉底液"],
                    "symptom": ["使用后发红"],
                },
                reasoning="复合风险流程集成测试",
                latency_ms=1.0,
                secondary_intents=[IntentCategory.RETURN_REFUND, IntentCategory.MISSING_ITEM],
                sub_requests=[
                    {"intent": "return_refund", "message": "申请退货退款"},
                    {"intent": "missing_item", "message": "直播赠品小样没收到"},
                ],
                emotion="焦虑",
                emotion_intensity=0.7,
                emotion_evidence=["用了两天后发红"],
            )

    api_main.app.state.intent_recognizer = CompoundHandoffRecognizer()
    user_message = (
        "我的订单 SC-DEMO-4821 的轻透粉底液用了两天后发红，我要求转人工并申请退货退款，"
        "直播赠品小样也没收到。"
    )
    try:
        chat_response = client.post(
            "/chat",
            json={"message": user_message, "user_id": "consumer-demo", "conv_id": conv_id},
        )
        assert chat_response.status_code == 200
        chat_body = chat_response.json()
        assert chat_body["response"] == "为您转接人工，请您稍等。"
        assert chat_body["escalated"] is True
        assert chat_body["ticket_draft"]["status"] == "draft"

        service_response = client.get(
            f"/service/sessions/{conv_id}",
            headers={"Authorization": f"Bearer {agent_token}"},
        )
        assert service_response.status_code == 200
        service_body = service_response.json()
        state = service_body["case_state"]
        assert service_body["status"] == "waiting_human"
        assert state["handoff_summary"]["request_items"]
        assert state["handoff_summary"]["commitments"] == []
        assert state["ticket_draft"]["fields"]["order_id"] == "SC-DEMO-4821"
        assert state["ticket_draft"]["fields"]["product_name"] == "轻透粉底液"
        assert set(state["ticket_draft"]["missing_blocking_fields"]) == {
            "return_reason", "opened_or_used", "missing_item", "quantity"
        }
        user_row = next(item for item in service_body["messages"] if item["sender_type"] == "consumer")
        order_fact = state["case_facts"]["facts"]["order_id"][-1]
        assert order_fact["source_ref"] == user_row["message_id"]
        assert order_fact["confirmation_status"] == "unconfirmed"
        assert any(item["sender_type"] == "ai" and item["content"] == chat_body["response"] for item in service_body["messages"])

        current_version = state["version"]
        edited = client.patch(
            f"/service/sessions/{conv_id}/ticket",
            headers={"Authorization": f"Bearer {agent_token}"},
            json={
                "expected_version": current_version,
                "fields": {
                    "return_reason": "使用后不适",
                    "opened_or_used": "已开封并使用",
                    "missing_item": "直播赠品小样",
                    "quantity": "2件",
                },
            },
        )
        assert edited.status_code == 200
        assert edited.json()["ticket"]["missing_blocking_fields"] == []

        submitted = client.post(
            f"/service/sessions/{conv_id}/ticket/confirm",
            headers={"Authorization": f"Bearer {agent_token}"},
        )
        assert submitted.status_code == 200
        assert submitted.json()["status"] == "submitted"
        assert set(submitted.json()["ticket"]["submitted_departments"]) == {"质量", "售后", "财务", "仓储"}

        dashboard = client.get(
            "/service/dashboard",
            headers={"Authorization": f"Bearer {agent_token}"},
        )
        assert dashboard.status_code == 200
        dashboard_body = dashboard.json()
        assert dashboard_body["metrics"]["total_sessions"] == 1
        assert dashboard_body["metrics"]["handoff_rate"] == 100
        assert dashboard_body["metrics"]["ai_resolution_rate"] == 0
        assert dashboard_body["metrics"]["pending_tickets"] == 0
        assert any(item["ticket_status"] == "submitted" for item in dashboard_body["users"])
    finally:
        client.__exit__(None, None, None)


def test_non_handoff_refund_draft_is_persisted_for_service_review(monkeypatch, tmp_path):
    """退款诉求即使由 AI 继续处理，也应在客服端落为待审核草稿。"""
    monkeypatch.setenv("HANDOFF", "false")
    conv_id = "e2e-refund-review-draft"
    store = SQLiteCaseStore(tmp_path / "shopcare-refund-draft.sqlite3")
    persistence_order = []
    original_append_message = store.append_message
    original_update_case_state = store.update_case_state

    def observe_append_message(*args, **kwargs):
        role = args[1] if len(args) > 1 else kwargs.get("role")
        if role == "ai":
            persistence_order.append("assistant_message")
        return original_append_message(*args, **kwargs)

    def observe_update_case_state(*args, **kwargs):
        updates = args[1] if len(args) > 1 else kwargs.get("updates", {})
        if isinstance(updates, dict) and updates.get("ticket_draft"):
            persistence_order.append("ticket_state")
        return original_update_case_state(*args, **kwargs)

    store.append_message = observe_append_message
    store.update_case_state = observe_update_case_state
    memory = FakeMemoryManager()
    consumer_token = store.authenticate("consumer-demo", "shopcare")["token"]
    agent_token = store.authenticate("agent-demo", "shopcare")["token"]
    store.ensure_session("consumer-demo", conv_id)

    class RefundRecognizer:
        async def recognize_intent(self, message, history):
            entities = {"order_id": ["SC-DEMO-4821"]}
            if "包装破损" in message:
                entities = {"return_reason": ["包装破损"], "opened_or_used": ["已拆封"]}
            return IntentResult(
                intent=IntentCategory.RETURN_REFUND,
                confidence=0.98,
                urgency=UrgencyLevel.LOW,
                intent_group="after_sales",
                entities=entities,
                reasoning="退款诉求测试",
                latency_ms=1.0,
            )

    class RefundOrchestrator:
        async def run(self, request):
            return OrchestratorResult(
                request_id=request.request_id,
                response="请在订单页选择退货退款；如果是使用不适，请补充症状和首次使用时间。",
                agent_type=AgentType.REFUND_RISK,
                intent=IntentCategory.RETURN_REFUND,
                latency_ms=1.0,
                agent_types=[AgentType.REFUND_RISK],
                primary_agent=AgentType.REFUND_RISK,
            )

    monkeypatch.setattr(api_main.app.state, "ready", True, raising=False)
    monkeypatch.setattr(api_main.app.state, "sqlite_store", store, raising=False)
    monkeypatch.setattr(api_main.app.state, "memory_manager", memory, raising=False)
    monkeypatch.setattr(api_main.app.state, "intent_recognizer", RefundRecognizer(), raising=False)
    monkeypatch.setattr(api_main.app.state, "agent_orchestrator", RefundOrchestrator(), raising=False)
    monkeypatch.setattr(api_main.app.state, "token_usage", None, raising=False)
    monkeypatch.setattr(api_main.app.state, "anthropic_client", None, raising=False)

    async def exercise_refund_request():
        response = await api_main._chat_core(
            api_main.ChatRequest(message="我想申请退款", user_id="consumer-demo", conv_id=conv_id),
            f"Bearer {consumer_token}",
        )
        service_response = await api_main.get_service_session(
            conv_id, after_seq=None, authorization=f"Bearer {agent_token}"
        )
        service_sessions = await api_main.list_service_sessions(
            authorization=f"Bearer {agent_token}"
        )
        first_ticket = service_response["case_state"]["ticket_draft"]
        first_id = first_ticket["system_fields"]["ticket_id"]

        updated_response = await api_main._chat_core(
            api_main.ChatRequest(
                message="退款原因是包装破损，商品已经拆封",
                user_id="consumer-demo",
                conv_id=conv_id,
            ),
            f"Bearer {consumer_token}",
        )
        updated = await api_main.get_service_session(
            conv_id, after_seq=None, authorization=f"Bearer {agent_token}"
        )
        updated_ticket = updated["case_state"]["ticket_draft"]

        await api_main.confirm_service_ticket(
            conv_id, authorization=f"Bearer {agent_token}"
        )
        await api_main._chat_core(
            api_main.ChatRequest(
                message="我还有一个新的退货退款问题",
                user_id="consumer-demo",
                conv_id=conv_id,
            ),
            f"Bearer {consumer_token}",
        )
        after_review = await api_main.get_service_session(
            conv_id, after_seq=None, authorization=f"Bearer {agent_token}"
        )
        new_ticket = after_review["case_state"]["ticket_draft"]
        return response, updated_response, service_response, service_sessions, first_id, updated_ticket, new_ticket

    response, updated_response, service_response, service_sessions, first_id, updated_ticket, new_ticket = asyncio.run(
        exercise_refund_request()
    )
    assert response.escalated is False
    assert persistence_order.index("ticket_state") < persistence_order.index("assistant_message")
    assert "订单页" not in response.response
    assert "使用不适" not in response.response
    assert "这次退款的原因" in response.response
    assert "待客服审核" in updated_response.response
    assert "包装破损" in updated_response.response
    assert "订单页" not in updated_response.response
    assert "使用不适" not in updated_response.response
    assert service_response["status"] != "waiting_human"
    assert service_response["case_state"]["ticket_status"] == "draft"
    assert service_response["case_state"]["ticket_draft"]["status"] == "draft"
    assert service_response["case_state"]["ticket_draft"]["needs_review"] is True
    assert conv_id in {item["conv_id"] for item in service_sessions["sessions"]}
    assert updated_ticket["status"] == "draft"
    assert updated_ticket["system_fields"]["ticket_id"] == first_id
    assert updated_ticket["fields"]["order_id"] == "SC-DEMO-4821"
    assert updated_ticket["fields"]["return_reason"] == "包装破损"
    assert updated_ticket["fields"]["opened_or_used"] == "已拆封"
    assert new_ticket["status"] == "draft"
    assert new_ticket["needs_review"] is True
    assert new_ticket["system_fields"]["ticket_id"] != first_id


def test_chat_persists_and_consumes_one_pending_refund_reason(monkeypatch, tmp_path):
    """真实 API 两轮请求把自由文本原因落到同一字段，第二轮不再等待该原因。"""
    monkeypatch.setenv("HANDOFF", "false")
    client, _, _, _ = _prepare_app(monkeypatch)
    conv_id = "pending-refund-reason-roundtrip"
    store = SQLiteCaseStore(tmp_path / "shopcare-pending-reason.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)

    class RefundFollowupRecognizer:
        async def recognize_intent(self, message, history):
            if message == "因为用后脸红":
                return IntentResult(
                    intent=IntentCategory.OTHER,
                    confidence=0.2,
                    urgency=UrgencyLevel.LOW,
                    intent_group="other",
                    entities={"symptom": ["脸红"]},
                    reasoning="简短补充答案",
                    latency_ms=1.0,
                )
            return IntentResult(
                intent=IntentCategory.RETURN_REFUND,
                confidence=0.98,
                urgency=UrgencyLevel.LOW,
                intent_group="after_sales",
                entities={
                    "order_id": ["SC-DEMO-4821"],
                    "opened_or_used": ["已使用"],
                },
                reasoning="退款诉求",
                latency_ms=1.0,
            )

    class RefundFollowupOrchestrator:
        def __init__(self):
            self.requests = []

        async def run(self, request):
            self.requests.append(request)
            return OrchestratorResult(
                request_id=request.request_id,
                response=(
                    "您这次退款的原因是什么？"
                    if len(self.requests) == 1
                    else "已收到您补充的退款原因。"
                ),
                agent_type=AgentType.REFUND_RISK,
                intent=request.intent,
                latency_ms=1.0,
                agent_types=[AgentType.REFUND_RISK],
                primary_agent=AgentType.REFUND_RISK,
                secondary_intents=list(request.secondary_intents),
            )

    orchestrator = RefundFollowupOrchestrator()
    monkeypatch.setattr(api_main.app.state, "intent_recognizer", RefundFollowupRecognizer(), raising=False)
    monkeypatch.setattr(api_main.app.state, "agent_orchestrator", orchestrator, raising=False)

    try:
        first = client.post("/chat", json={
            "message": "订单 SC-DEMO-4821 已签收，我想退款",
            "user_id": "consumer-demo",
            "conv_id": conv_id,
        })
        assert first.status_code == 200
        first_state = store.get_conversation(conv_id, viewer_role="agent")["case_state"]
        assert first_state["pending_text_fields"][0]["field_id"] == "return_reason"

        second = client.post("/chat", json={
            "message": "因为用后脸红",
            "user_id": "consumer-demo",
            "conv_id": conv_id,
        })
        assert second.status_code == 200
        assert orchestrator.requests[1].intent is IntentCategory.RETURN_REFUND
        assert orchestrator.requests[1].entities["return_reason"] == ["因为用后脸红"]
        final_state = store.get_conversation(conv_id, viewer_role="agent")["case_state"]
        assert final_state["pending_text_fields"] == []
    finally:
        client.__exit__(None, None, None)


def test_consumer_can_incrementally_read_assigned_agent_replies(monkeypatch, tmp_path):
    """消费者只能按游标读取自己的公开会话消息，包括人工客服回复。"""
    monkeypatch.setenv("HANDOFF", "false")
    store = SQLiteCaseStore(tmp_path / "shopcare-consumer-messages.sqlite3")
    monkeypatch.setattr(api_main.app.state, "sqlite_store", store, raising=False)
    consumer_token = store.authenticate("consumer-demo", "shopcare")["token"]
    agent_token = store.authenticate("agent-demo", "shopcare")["token"]
    store.register_consumer("other-consumer", "shopcare", "其他消费者")
    other_token = store.authenticate("other-consumer", "shopcare")["token"]
    conv_id = "consumer-visible-messages"
    store.ensure_session("consumer-demo", conv_id)
    initial = store.append_message(conv_id, "consumer", "consumer-demo", "想申请退款")
    store.append_message(conv_id, "ai", "shopcare-ai", "已记录您的申请")
    internal = store.append_message(
        conv_id, "system", "shopcare", "内部处理字段", visible_to_consumer=False
    )

    async def exchange_messages():
        service_reply = await api_main.append_service_message(
            conv_id,
            api_main.ServiceMessageRequest(agent_id="agent-demo", content="请补充退款原因"),
            authorization=f"Bearer {agent_token}",
        )
        response = await api_main.get_consumer_messages(
            conv_id,
            after_seq=initial["message_seq"],
            authorization=f"Bearer {consumer_token}",
        )
        return service_reply, response

    service_reply, response = asyncio.run(exchange_messages())
    reply = service_reply["message"]
    assert service_reply["success"] is True
    assert [item["sender_type"] for item in response["messages"]] == ["ai", "customer_service"]
    assert response["messages"][-1]["message_id"] == reply["message_id"]
    assert response["messages"][-1]["content"] == "请补充退款原因"
    assert "sender_id" not in response["messages"][-1]
    assert all(item["message_id"] != internal["message_id"] for item in response["messages"])

    with pytest.raises(api_main.HTTPException) as forbidden:
        asyncio.run(api_main.get_consumer_messages(
            conv_id, after_seq=0, authorization=f"Bearer {other_token}"
        ))
    assert forbidden.value.status_code == 403


def test_handoff_keeps_consumer_and_agent_messages_in_one_public_thread(monkeypatch, tmp_path):
    """转人工后消费者消息进入客服会话，人工回复再回到消费者端，AI 保持静默。"""
    monkeypatch.setenv("HANDOFF", "true")
    client, _, _, orchestrator = _prepare_app(monkeypatch)
    store = SQLiteCaseStore(tmp_path / "handoff-bidirectional.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)
    agent_token = store.authenticate("agent-demo", "shopcare")["token"]
    conv_id = "handoff-bidirectional"

    try:
        handoff = client.post(
            "/chat",
            json={"message": "请转人工", "user_id": "consumer-demo", "conv_id": conv_id},
        )
        assert handoff.status_code == 200
        assert handoff.json()["response"].startswith("已为您转接人工")
        initial = client.get(f"/chat/{conv_id}/messages?after_seq=0")
        assert initial.status_code == 200
        initial_cursor = initial.json()["last_message_seq"]

        consumer_followup = client.post(
            "/chat/stream",
            json={
                "message": "补充一下，商品还没有拆封。",
                "user_id": "consumer-demo",
                "conv_id": conv_id,
            },
        )
        assert consumer_followup.status_code == 200
        assert "handoff_suppressed" in consumer_followup.text
        assert orchestrator.requests == []

        service_snapshot = client.get(
            f"/service/sessions/{conv_id}?after_seq={initial_cursor}",
            headers={"Authorization": f"Bearer {agent_token}"},
        )
        assert service_snapshot.status_code == 200
        assert [item["content"] for item in service_snapshot.json()["messages"]] == [
            "补充一下，商品还没有拆封。"
        ]
        consumer_cursor = service_snapshot.json()["messages"][-1]["message_seq"]

        service_reply = client.post(
            f"/service/sessions/{conv_id}/messages",
            json={"agent_id": "agent-demo", "content": "收到，我会按未拆封情况核对退款条件。"},
            headers={"Authorization": f"Bearer {agent_token}"},
        )
        assert service_reply.status_code == 200

        consumer_snapshot = client.get(
            f"/chat/{conv_id}/messages?after_seq={consumer_cursor}"
        )
        assert consumer_snapshot.status_code == 200
        assert [item["sender_type"] for item in consumer_snapshot.json()["messages"]] == [
            "customer_service"
        ]
        assert consumer_snapshot.json()["messages"][0]["content"] == (
            "收到，我会按未拆封情况核对退款条件。"
        )
        assert store.get_session(conv_id)["status"] == "waiting_human"
    finally:
        client.__exit__(None, None, None)


@pytest.mark.parametrize(
    ("message", "intent", "secondary_intents", "expected_priority", "expected_rule"),
    [
        (
            "这笔退款记录疑似重复",
            IntentCategory.SUSPECTED_ABNORMAL_REFUND,
            [],
            "P1",
            "intent.priority_floor.suspected_abnormal_refund",
        ),
        (
            "我现在呼吸困难",
            IntentCategory.ADVERSE_REACTION,
            [],
            "P0",
            "safety.breathing_or_emergency",
        ),
        (
            "我要向12315投诉这款粉底液",
            IntentCategory.OTHER,
            [],
            "P0",
            "escalation.external_action",
        ),
        (
            "我用了以后脸红肿，想退款",
            IntentCategory.ADVERSE_REACTION,
            [IntentCategory.RETURN_REFUND],
            None,
            None,
        ),
    ],
)
def test_rule_triggered_handoff_uses_atomic_record_before_agent_execution(
    monkeypatch, tmp_path, message, intent, secondary_intents,
    expected_priority, expected_rule,
):
    """安全/投诉/异常款项共用原子交接；普通红肿退款仍由 AI 处理。"""
    monkeypatch.setenv("HANDOFF", "true")
    client, _, recognizer, orchestrator = _prepare_app(monkeypatch)
    store = SQLiteCaseStore(tmp_path / "rule-handoff-atomic.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)

    async def recognize_scenario(message, history):
        return IntentResult(
            intent=intent,
            confidence=0.99,
            urgency=UrgencyLevel.HIGH,
            intent_group="after_sales",
            entities={"order_id": [], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="测试用例预设识别结果",
            latency_ms=1.0,
            secondary_intents=secondary_intents,
        )

    recognizer.recognize_intent = recognize_scenario
    conv_id = "rule-handoff-atomic"
    try:
        response = client.post(
            "/chat",
            json={
                "message": message,
                "user_id": "consumer-demo",
                "conv_id": conv_id,
            },
        )

        assert response.status_code == 200
        body = response.json()
        conversation = store.get_conversation(conv_id, viewer_role="agent")
        if expected_priority:
            assert body["intent_diagnostics"]["handoff_mode"] == "atomic_rule"
            assert body["intent"] == intent.value
            assert body["handoff_summary"]["urgency"] == expected_priority
            assert body["intent_diagnostics"]["handoff_id"]
            assert body["intent_diagnostics"]["handoff_reason"] == expected_rule
            assert orchestrator.requests == []
            assert conversation["status"] == "waiting_human"
            assert conversation["handoff"]["handoff_id"] == body["intent_diagnostics"]["handoff_id"]
            assert [item["sender_type"] for item in conversation["messages"]] == ["consumer", "ai"]
        else:
            assert "handoff_mode" not in body["intent_diagnostics"]
            assert conversation["status"] != "waiting_human"
            assert conversation.get("handoff") is None
            assert len(orchestrator.requests) == 1
    finally:
        client.__exit__(None, None, None)


def test_stream_rule_handoff_returns_ack_instead_of_suppressing_it(monkeypatch, tmp_path):
    """流式接口遇到规则升级应推送原子交接确认，不误当成并发抢占丢弃。"""
    monkeypatch.setenv("HANDOFF", "true")
    client, _, recognizer, orchestrator = _prepare_app(monkeypatch)
    store = SQLiteCaseStore(tmp_path / "rule-handoff-stream.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)

    async def recognize_safety_risk(message, history):
        return IntentResult(
            intent=IntentCategory.ADVERSE_REACTION,
            confidence=0.99,
            urgency=UrgencyLevel.HIGH,
            intent_group="after_sales",
            entities={"order_id": [], "product": [], "date": [], "amount": [], "error_code": []},
            reasoning="测试严重安全风险",
            latency_ms=1.0,
        )

    recognizer.recognize_intent = recognize_safety_risk
    conv_id = "rule-handoff-stream"
    try:
        response = client.post(
            "/chat/stream",
            json={
                "message": "我现在呼吸困难",
                "user_id": "consumer-demo",
                "conv_id": conv_id,
            },
        )

        assert response.status_code == 200
        assert '"handoff_mode": "atomic_rule"' in response.text
        assert "handoff_suppressed" not in response.text
        assert "已为您转接人工" in response.text
        assert orchestrator.requests == []
        assert store.get_session(conv_id)["status"] == "waiting_human"
    finally:
        client.__exit__(None, None, None)


def test_handoff_flag_uses_atomic_ack_without_running_agent_flow(monkeypatch, tmp_path):
    """HANDOFF 开启后，明确转人工必须先原子落库，再返回固定 ACK。"""
    monkeypatch.setenv("HANDOFF", "true")
    client, _, recognizer, orchestrator = _prepare_app(monkeypatch)
    store = SQLiteCaseStore(tmp_path / "handoff-api.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)
    conv_id = "atomic-handoff-api"
    try:
        response = client.post(
            "/chat",
            json={"message": "请帮我转人工", "user_id": "consumer-demo", "conv_id": conv_id},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["response"].endswith("专员上线后会联系您。")
        assert body["intent_diagnostics"]["handoff_mode"] == "atomic"
        assert body["intent_diagnostics"]["handoff_id"]
        assert body["handoff_summary"]["urgency"] == "P3"
        assert body["handoff_summary"]["urgency_rule_version"] == "handoff-urgency-v2"
        assert body["handoff_summary"]["urgency_reasons"][0].startswith("default.no_higher_signal:")
        assert recognizer.histories == []
        assert orchestrator.requests == []
        conversation = store.get_conversation(conv_id, viewer_role="agent")
        assert [item["sender_type"] for item in conversation["messages"]] == ["consumer", "ai"]
        assert conversation["status"] == "waiting_human"
        assert conversation["case_state"]["handoff_id"] == body["intent_diagnostics"]["handoff_id"]
    finally:
        client.__exit__(None, None, None)


def test_service_handoff_sse_replays_events_after_last_event_id(monkeypatch, tmp_path):
    monkeypatch.setenv("HANDOFF", "true")
    client, _, _, _ = _prepare_app(monkeypatch)
    store = SQLiteCaseStore(tmp_path / "service-handoff-events.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)
    store.ensure_session("consumer-demo", "service-events-case")
    handoff = store.start_handoff(
        "service-events-case", user_id="consumer-demo", message="请转人工",
        idempotency_key="handoff:service-events-case:one", reason="consumer_requested_human",
        temporary_summary={"text": "顾客直接要求转人工", "sources": []},
        temporary_priority="P3", temporary_priority_score=0,
        temporary_priority_reasons=["转人工本身不加分"], temporary_priority_rule_version="test-v1",
        online_ack="已为您转接人工，请您稍等。",
        unknown_presence_ack="已登记，专员上线后会联系您。", callback_due_at=None,
    )
    token = store.authenticate("agent-demo", "shopcare")["token"]
    initial_messages = store.get_conversation("service-events-case", viewer_role="agent")["messages"]
    initial_message_seq = max(item["message_seq"] for item in initial_messages)
    snapshot_delta = client.get(
        "/service/sessions/service-events-case",
        params={"after_seq": initial_message_seq},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert snapshot_delta.status_code == 200
    assert snapshot_delta.json()["messages"] == []
    store.append_message("service-events-case", "consumer", "consumer-demo", "补充：目前还有刺痛。")
    snapshot_delta = client.get(
        "/service/sessions/service-events-case",
        params={"after_seq": initial_message_seq},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert len(snapshot_delta.json()["messages"]) == 1
    assert snapshot_delta.json()["messages"][0]["content"] == "补充：目前还有刺痛。"
    store.save_handoff_generated_block(
        handoff["handoff_id"], block_type="summary", status="ready", state_hash="summary-v1",
        payload={"text": "顾客补充目前有刺痛。", "statements": []}, latency_ms=3,
        source_message_seq=handoff["source_message_seq"],
    )
    incremental_events = store.list_handoff_events("service-events-case", after_event_id=handoff["event_id"])
    assert [item["event_type"] for item in incremental_events] == [
        "handoff.message.incremental", "handoff.block.updated",
    ]
    try:
        with client.stream(
            "GET", f"/service/sessions/service-events-case/handoff/events?idle_timeout_seconds=0.1",
            headers={"Authorization": f"Bearer {token}", "Last-Event-ID": str(incremental_events[0]["event_id"])},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            lines = response.iter_lines()
            assert next(lines) == f"id: {incremental_events[1]['event_id']}"
            assert next(lines) == "event: handoff.block.updated"
            assert next(lines).startswith("data: {")
    finally:
        client.__exit__(None, None, None)


def test_service_handoff_sse_is_disabled_with_feature_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("HANDOFF", "false")
    client, _, _, _ = _prepare_app(monkeypatch)
    store = SQLiteCaseStore(tmp_path / "service-handoff-disabled.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)
    store.ensure_session("consumer-demo", "service-handoff-disabled")
    token = store.authenticate("agent-demo", "shopcare")["token"]
    try:
        response = client.get(
            "/service/sessions/service-handoff-disabled/handoff/events",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404
    finally:
        client.__exit__(None, None, None)


def test_atomic_handoff_schedules_only_fact_enrichment_after_commit(monkeypatch, tmp_path):
    """ACK 返回前只排后台事实任务，不等待识别/情绪，也不调用 Agent。"""
    monkeypatch.setenv("HANDOFF", "true")
    real_scheduler = api_main._schedule_handoff_enrichment
    client, _, recognizer, orchestrator = _prepare_app(monkeypatch)
    monkeypatch.setattr(api_main, "_schedule_handoff_enrichment", real_scheduler)
    store = SQLiteCaseStore(tmp_path / "handoff-scheduled.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)
    scheduled = []
    scheduled_event = Event()
    finished_event = Event()

    async def fake_enrich(store_arg, conv_id, handoff_id, *, emotion_client=None):
        scheduled.append((store_arg, conv_id, handoff_id, emotion_client))
        scheduled_event.set()
        await asyncio.sleep(0.25)
        finished_event.set()

    monkeypatch.setattr(api_main, "enrich_handoff", fake_enrich)
    try:
        response = client.post(
            "/chat",
            json={"message": "请帮我转人工", "user_id": "consumer-demo", "conv_id": "handoff-scheduled"},
        )
        assert response.status_code == 200
        body = response.json()
        handoff_id = body["intent_diagnostics"]["handoff_id"]
        assert body["intent_diagnostics"]["background_processing"] is True
        assert scheduled_event.wait(1)
        assert not finished_event.is_set()
        assert scheduled[0][2] == handoff_id
        assert handoff_id in api_main.app.state.handoff_tasks
        assert recognizer.histories == []
        assert orchestrator.requests == []
    finally:
        client.__exit__(None, None, None)


def test_atomic_handoff_scoring_uses_latest_message_not_transfer_signal(monkeypatch, tmp_path):
    monkeypatch.setenv("HANDOFF", "true")
    client, _, recognizer, orchestrator = _prepare_app(monkeypatch)
    store = SQLiteCaseStore(tmp_path / "handoff-priority-api.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)
    try:
        response = client.post(
            "/chat",
            json={
                "message": "用了之后脸红肿，请帮我转人工",
                "user_id": "consumer-demo",
                "conv_id": "atomic-handoff-priority",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["handoff_summary"]["urgency"] == "P1"
        assert any("证据：红肿" in reason for reason in body["handoff_summary"]["urgency_reasons"])
        assert recognizer.histories == []
        assert orchestrator.requests == []
    finally:
        client.__exit__(None, None, None)


def test_repeated_stream_handoff_returns_same_ack_identity_and_persists_one_ack(monkeypatch, tmp_path):
    """多次点击复用开放交接和 ACK 消息，不创建重复消费者确认。"""
    monkeypatch.setenv("HANDOFF", "true")
    client, _, recognizer, orchestrator = _prepare_app(monkeypatch)
    store = SQLiteCaseStore(tmp_path / "handoff-stream.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)
    conv_id = "atomic-handoff-stream"
    request_body = {"message": "我要找人工", "user_id": "consumer-demo", "conv_id": conv_id}
    try:
        first = client.post("/chat/stream", json=request_body)
        second = client.post("/chat/stream", json=request_body)
        assert first.status_code == second.status_code == 200
        assert "event: ack" in first.text
        assert "专员上线后会联系您" in first.text
        assert "AI 客服" not in first.text
        assert recognizer.histories == []
        assert orchestrator.requests == []
        conversation = store.get_conversation(conv_id, viewer_role="agent")
        assert len([item for item in conversation["messages"] if item["sender_type"] == "ai"]) == 1
        assert conversation["messages"][1]["message_id"] in second.text
        with sqlite3.connect(store.path) as db:
            assert db.execute("SELECT COUNT(*) FROM handoff_records").fetchone()[0] == 1
            assert db.execute("SELECT COUNT(*) FROM handoff_events").fetchone()[0] == 1
    finally:
        client.__exit__(None, None, None)


def test_handoff_event_write_failure_never_streams_ack(monkeypatch, tmp_path):
    """原子事务中事件写入失败时，SSE 不得先显示消费者确认。"""
    monkeypatch.setenv("HANDOFF", "true")
    client, _, _, _ = _prepare_app(monkeypatch)
    store = SQLiteCaseStore(tmp_path / "handoff-event-failure.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)
    conv_id = "atomic-handoff-failure"
    store.ensure_session("consumer-demo", conv_id)
    with sqlite3.connect(store.path) as db:
        db.execute(
            "CREATE TRIGGER reject_handoff_event BEFORE INSERT ON handoff_events "
            "BEGIN SELECT RAISE(ABORT, 'event write rejected'); END"
        )
    try:
        response = client.post(
            "/chat/stream",
            json={"message": "请人工客服处理", "user_id": "consumer-demo", "conv_id": conv_id},
        )
        assert response.status_code == 200
        assert "event: error" in response.text
        assert "event: ack" not in response.text
        assert store.get_conversation(conv_id, viewer_role="agent")["messages"] == []
        assert store.get_session(conv_id)["status"] == "ai_processing"
    finally:
        client.__exit__(None, None, None)


def test_in_flight_ai_response_is_suppressed_after_handoff_commit(monkeypatch, tmp_path):
    """正在运行的 AI 回复在 SSE 输出前复查 waiting_human 并被丢弃。"""
    monkeypatch.setenv("HANDOFF", "true")
    client, _, _, orchestrator = _prepare_app(monkeypatch)
    store = SQLiteCaseStore(tmp_path / "handoff-race.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)

    async def handoff_during_generation(request):
        await asyncio.to_thread(
            store.start_handoff,
            request.conv_id,
            user_id=request.user_id,
            message="请转人工",
            idempotency_key="handoff:race:request",
            reason="consumer_requested_human",
            temporary_summary={"text": "顾客直接要求转人工，暂无其他诉求", "sources": []},
            temporary_priority="P3",
            temporary_priority_score=0,
            temporary_priority_reasons=["转人工本身不加分"],
            online_ack="为您转接人工，请您稍等。",
            unknown_presence_ack="为您转接人工，请您稍等。已为您登记，专员上线后会联系您。",
            callback_due_at=None,
        )
        return OrchestratorResult(
            request_id=request.request_id,
            response="这段 AI 回复必须被抢占丢弃。",
            agent_type=AgentType.FULFILLMENT,
            intent=request.intent,
            latency_ms=1.0,
        )

    orchestrator.run = handoff_during_generation
    try:
        response = client.post(
            "/chat/stream",
            json={
                "message": "订单 SC-DEMO-4821 到哪了？",
                "user_id": "consumer-demo",
                "conv_id": "handoff-race-stream",
            },
        )
        assert response.status_code == 200
        assert "handoff_suppressed" in response.text
        assert "这段 AI 回复必须被抢占丢弃" not in response.text
    finally:
        client.__exit__(None, None, None)


def test_successful_order_tool_result_reaches_consumer_answer_and_service_case_facts(
    monkeypatch, tmp_path
):
    """真实工具执行结果应经 Agent 模型答复并落入同一客服会话事实。"""

    client, _, _, _ = _prepare_app(monkeypatch)
    conv_id = "e2e-order-tool-answer"
    store = SQLiteCaseStore(tmp_path / "shopcare-order-e2e.sqlite3")
    api_main.app.state.sqlite_store = store
    _set_consumer_auth(client, store)
    agent_token = store.authenticate("agent-demo", "shopcare")["token"]

    class ScriptedAgentClient:
        """模拟 Tool Use 与最终消费者答复，不模拟工具本身。"""

        def __init__(self):
            self.messages = self
            self.calls = []
            self.responses = [
                SimpleNamespace(content=[SimpleNamespace(
                    type="tool_use",
                    id="toolu-order-1",
                    name="lookup_order",
                    input={"order_id": "SC-DEMO-4821"},
                )]),
                SimpleNamespace(content=[SimpleNamespace(
                    type="text",
                    text="订单 SC-DEMO-4821 对应的轻透粉底液（N02）已签收。",
                )]),
            ]

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return self.responses.pop(0)

    agent_client = ScriptedAgentClient()
    api_main.app.state.agent_orchestrator = AgentOrchestrator(
        agent_pool={
            AgentType.FULFILLMENT: TechnicalAgent(
                client=agent_client,
                tools=[BEAUTY_LOOKUP_ORDER_TOOL],
            )
        }
    )

    try:
        chat_response = client.post(
            "/chat",
            json={
                "message": "SC-DEMO-4821 的订单到哪里了？",
                "user_id": "consumer-demo",
                "conv_id": conv_id,
            },
        )
        assert chat_response.status_code == 200
        chat_body = chat_response.json()
        assert chat_body["response"] == "订单 SC-DEMO-4821 对应的轻透粉底液（N02）已签收。"
        assert chat_body["tools_used"] == ["lookup_order"]
        assert len(agent_client.calls) == 2

        service_response = client.get(
            f"/service/sessions/{conv_id}",
            headers={"Authorization": f"Bearer {agent_token}"},
        )
        assert service_response.status_code == 200
        service_body = service_response.json()
        state = service_body["case_state"]
        facts = state["case_facts"]["facts"]
        order_fact = facts["order_id"][-1]
        assert order_fact["value"] == "SC-DEMO-4821"
        assert order_fact["source_type"] == "tool:lookup_order"
        assert order_fact["source_ref"] == chat_body["request_id"]
        assert facts["product_name"][-1]["value"] == "轻透粉底液"
        assert facts["order_status"][-1]["value"] == "已签收"
        assert any(
            item["sender_type"] == "ai" and item["content"] == chat_body["response"]
            for item in service_body["messages"]
        )
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
