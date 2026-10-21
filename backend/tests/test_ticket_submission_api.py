import asyncio

from fastapi import HTTPException
import pytest

import api.main as api_main
from agents.case_facts import CaseFacts
from storage.sqlite_case_store import SQLiteCaseStore


def _ticket_state():
    return {
        "ticket_draft": {
            "status": "draft",
            "intent": "adverse_reaction",
            "system_fields": {
                "ticket_id": "T-system-owned",
                "conv_id": "conv-ticket",
                "created_at": "2026-10-20T00:00:00+00:00",
            },
            "blocking_fields": ["order_id", "symptom"],
            "fields": {"order_id": "SC-OLD-1", "symptom": "发红"},
            "sub_tasks": [
                {"department": "质量", "task": "核对风险", "status": "待处理"},
                {"department": "售后", "task": "核对售后", "status": "待处理"},
            ],
        },
        "case_facts": CaseFacts.from_entities(
            {"order_id": ["SC-OLD-1"], "symptom": ["发红"]}, source_ref="message-1"
        ).to_dict(),
    }


def _refund_ticket_state():
    return {
        "ticket_draft": {
            "status": "draft",
            "intent": "return_refund",
            "ticket_type": "return_refund",
            "system_fields": {
                "ticket_id": "T-refund-1",
                "conv_id": "conv-refund-ticket",
                "created_at": "2026-10-20T00:00:00+00:00",
            },
            "blocking_fields": ["order_id", "return_reason", "opened_or_used"],
            "fields": {
                "order_id": "SC-DEMO-4821",
                "return_reason": "买错了",
                "opened_or_used": "正装未拆封",
            },
            "sub_tasks": [{"department": "售后", "task": "审核退款申请", "status": "待处理"}],
        },
        "case_facts": CaseFacts.from_entities(
            {"order_id": ["SC-DEMO-4821"]}, source_ref="message-refund"
        ).to_dict(),
    }


def _setup_agent_store(monkeypatch, tmp_path, conv_id="conv-ticket"):
    store = SQLiteCaseStore(tmp_path / f"{conv_id}.sqlite3")
    store.ensure_session("consumer-demo", conv_id)
    store.update_case_state(conv_id, _ticket_state())
    monkeypatch.setattr(api_main.app.state, "sqlite_store", store, raising=False)
    token = store.authenticate("agent-demo", "shopcare")["token"]
    return store, token


def test_agent_ticket_edit_updates_case_facts_and_confirmation_submits_all_departments(monkeypatch, tmp_path):
    store, token = _setup_agent_store(monkeypatch, tmp_path)
    conv_id = "conv-ticket"

    edited = asyncio.run(api_main.update_service_ticket(
        conv_id,
        api_main.TicketUpdateRequest(fields={"order_id": "SC-DEMO-4821"}, expected_version=1),
        authorization=f"Bearer {token}",
    ))
    edited_fact = edited["ticket"]["fields"]["order_id"]
    assert edited_fact == "SC-DEMO-4821"
    state = store.get_conversation(conv_id, viewer_role="agent")["case_state"]
    assert state["case_facts"]["facts"]["order_id"][-1]["source_type"] == "agent_edit"
    assert state["case_facts"]["facts"]["order_id"][-1]["confirmation_status"] == "confirmed"

    submitted = asyncio.run(api_main.confirm_service_ticket(conv_id, authorization=f"Bearer {token}"))

    assert submitted["status"] == "submitted"
    assert submitted["ticket"]["submitted_departments"] == ["质量", "售后"]
    assert {task["status"] for task in submitted["ticket"]["department_tasks"]} == {"已提交"}
    assert {task["status"] for task in submitted["ticket"]["sub_tasks"]} == {"已提交"}


def test_ticket_confirmation_returns_409_for_missing_blocking_fields(monkeypatch, tmp_path):
    store, token = _setup_agent_store(monkeypatch, tmp_path, "conv-incomplete")
    state = store.get_conversation("conv-incomplete", viewer_role="agent")["case_state"]
    state["ticket_draft"]["fields"]["symptom"] = ""
    store.update_case_state("conv-incomplete", {"ticket_draft": state["ticket_draft"]}, state["version"])

    with pytest.raises(HTTPException) as error:
        asyncio.run(api_main.confirm_service_ticket("conv-incomplete", authorization=f"Bearer {token}"))

    assert error.value.status_code == 409
    assert error.value.detail["missing_fields"] == ["symptom"]


def test_refund_ticket_submission_waits_for_agent_to_send_progress_reply(monkeypatch, tmp_path):
    store, agent_token = _setup_agent_store(monkeypatch, tmp_path, "conv-refund-ticket")
    store.update_case_state("conv-refund-ticket", _refund_ticket_state(), expected_version=1)
    consumer_token = store.authenticate("consumer-demo", "shopcare")["token"]

    first = asyncio.run(api_main.confirm_service_ticket(
        "conv-refund-ticket", authorization=f"Bearer {agent_token}"
    ))
    version_after_first = store.get_conversation("conv-refund-ticket", viewer_role="agent")["case_state"]["version"]
    second = asyncio.run(api_main.confirm_service_ticket(
        "conv-refund-ticket", authorization=f"Bearer {agent_token}"
    ))
    version_after_second = store.get_conversation("conv-refund-ticket", viewer_role="agent")["case_state"]["version"]
    visible = asyncio.run(api_main.get_consumer_messages(
        "conv-refund-ticket", after_seq=None, authorization=f"Bearer {consumer_token}"
    ))
    assert first["status"] == second["status"] == "submitted"
    assert version_after_second == version_after_first
    assert not any(item["sender_type"] == "customer_service" for item in visible["messages"])
    assert not any("已通过人工审核" in item["content"] for item in visible["messages"])

    progress_reply = "您的工单已提交，相关专员会核对后在这里回复您。"
    asyncio.run(api_main.append_service_message(
        "conv-refund-ticket",
        api_main.ServiceMessageRequest(content=progress_reply),
        authorization=f"Bearer {agent_token}",
    ))
    after_agent_send = asyncio.run(api_main.get_consumer_messages(
        "conv-refund-ticket", after_seq=None, authorization=f"Bearer {consumer_token}"
    ))
    assert [item["content"] for item in after_agent_send["messages"] if item["sender_type"] == "customer_service"] == [progress_reply]


def test_non_refund_ticket_confirmation_does_not_claim_application_approval(monkeypatch, tmp_path):
    store, agent_token = _setup_agent_store(monkeypatch, tmp_path)
    consumer_token = store.authenticate("consumer-demo", "shopcare")["token"]

    asyncio.run(api_main.confirm_service_ticket("conv-ticket", authorization=f"Bearer {agent_token}"))
    visible = asyncio.run(api_main.get_consumer_messages(
        "conv-ticket", after_seq=None, authorization=f"Bearer {consumer_token}"
    ))

    assert not any(item["sender_type"] == "customer_service" for item in visible["messages"])


def test_staff_ticket_edit_cannot_change_backend_system_fields(monkeypatch, tmp_path):
    store, token = _setup_agent_store(monkeypatch, tmp_path)

    edited = asyncio.run(api_main.update_service_ticket(
        "conv-ticket",
        api_main.TicketUpdateRequest(
            fields={"order_id": "SC-DEMO-4821", "ticket_id": "T-forged"},
            expected_version=1,
        ),
        authorization=f"Bearer {token}",
    ))
    state = store.get_conversation("conv-ticket", viewer_role="agent")["case_state"]

    assert edited["ticket"]["fields"]["order_id"] == "SC-DEMO-4821"
    assert "ticket_id" not in edited["ticket"]["fields"]
    assert state["ticket_draft"]["system_fields"]["ticket_id"] == "T-system-owned"
