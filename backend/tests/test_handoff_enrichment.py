"""交接后台补齐意图/槽位与情绪，不能启动领域 Agent 或业务工具。"""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from handoff.config import load_handoff_copy_config
from handoff.processor import enrich_handoff
from handoff.urgency import score_temporary_urgency
from storage.sqlite_case_store import SQLiteCaseStore
from intent.intent_recognizer import IntentCategory


def _start_handoff(store: SQLiteCaseStore, message: str) -> dict:
    copy = load_handoff_copy_config()
    from handoff.config import load_handoff_urgency_config

    urgency = score_temporary_urgency(message, load_handoff_urgency_config())
    return store.start_handoff(
        "enrich-case",
        user_id="consumer-demo",
        message=message,
        idempotency_key="handoff:enrich-case:one",
        reason="consumer_requested_human",
        temporary_summary={"text": copy["temporary_summary_direct"], "sources": []},
        temporary_priority=urgency["priority"],
        temporary_priority_score=urgency["score"],
        temporary_priority_reasons=[item["text"] for item in urgency["reasons"]],
        temporary_priority_rule_version=urgency["rule_version"],
        online_ack=copy["online_ack"],
        unknown_presence_ack=copy["unknown_presence_ack"],
        callback_due_at=None,
    )


def test_intent_and_emotion_enrichment_run_in_parallel_and_persist_final_urgency(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "handoff-enrichment.sqlite3")
    store.ensure_session("consumer-demo", "enrich-case")
    message = "一直没处理，我要投诉平台，请帮我转人工"
    handoff = _start_handoff(store, message)
    store.sync_case_sub_requests(
        "enrich-case",
        [{"id": "refund:one", "intent": "return_refund"},
         {"id": "abnormal-refund:one", "intent": "suspected_abnormal_refund"}],
        source_ref="message-1",
        source_message_seq=1,
    )
    enrichment_input = store.get_handoff_enrichment_input(handoff["handoff_id"])
    assert enrichment_input["intents"] == ["return_refund", "suspected_abnormal_refund"]
    both_started = __import__("asyncio").Event()
    calls = []

    async def intent_analyzer(message, history=None):
        calls.append("intent")
        if len(calls) == 2:
            both_started.set()
        await both_started.wait()
        return SimpleNamespace(
            intent=IntentCategory.HUMAN_HANDOFF,
            confidence=0.99,
            intent_group="handoff",
            entities={"order_id": ["SC-DEMO-4821"]},
            secondary_intents=[IntentCategory.RETURN_REFUND],
            sub_requests=[{"id": "return_refund:one", "intent": "return_refund"}],
            intent_diagnostics={"test": True},
        )

    async def emotion_analyzer(client, message, recent, config):
        calls.append("emotion")
        if len(calls) == 2:
            both_started.set()
        await both_started.wait()
        return {
            "emotion": "威胁升级",
            "answers": {"threat": True, "anger": False, "dissatisfaction": True, "anxiety": False},
            "evidence": {"threat": "我要投诉平台"},
            "source": "llm",
            "degraded": False,
        }

    import asyncio
    result = asyncio.run(enrich_handoff(
        store,
        "enrich-case",
        handoff["handoff_id"],
        emotion_client=object(),
        intent_analyzer=intent_analyzer,
        emotion_analyzer=emotion_analyzer,
    ))

    assert set(calls) == {"intent", "emotion"}
    assert result["intent_status"] == "ready"
    assert result["emotion_status"] == "ready"
    conversation = store.get_conversation("enrich-case", viewer_role="agent")
    state = conversation["case_state"]
    assert state["handoff_intent_enrichment"]["intent"] == "human_handoff"
    assert state["temporary_urgency"]["value"] == "P2"
    assert state["final_urgency"]["value"] == "P0"
    assert state["final_urgency"]["reasons"][0]["evidence"] == "我要投诉平台"
    assert "return_refund" in {
        item["intent_id"] for item in store.list_case_sub_requests("enrich-case")
    }
    assert all(item["action_type"] != "tool" for item in store.list_case_action_logs("enrich-case"))
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM tickets WHERE conv_id=?", ("enrich-case",)).fetchone()[0] == 0
        blocks = dict(db.execute(
            "SELECT block_type,status FROM handoff_blocks WHERE handoff_id=?",
            (handoff["handoff_id"],),
        ).fetchall())
        slots = db.execute(
            "SELECT slot_id,value,source_type,confirmation_status FROM slot_values WHERE conv_id=?",
            ("enrich-case",),
        ).fetchall()
        assert db.execute(
            "SELECT COUNT(*) FROM handoff_events WHERE handoff_id=? AND event_type='handoff.block.updated'",
            (handoff["handoff_id"],),
        ).fetchone()[0] == 3
    assert blocks == {"summary": "degraded", "emotion_urgency": "ready", "next_steps": "degraded"}
    with sqlite3.connect(store.path) as db:
        generated = {
            kind: json.loads(payload)
            for kind, payload in db.execute(
                "SELECT block_type,payload_json FROM handoff_blocks WHERE handoff_id=?",
                (handoff["handoff_id"],),
            ).fetchall()
        }
    assert generated["summary"]["text"]
    assert generated["next_steps"]["items"]
    assert ("order_id", "SC-DEMO-4821", "handoff_intent", "unconfirmed") in slots

    cached = asyncio.run(enrich_handoff(
        store,
        "enrich-case",
        handoff["handoff_id"],
        emotion_client=object(),
        intent_analyzer=intent_analyzer,
        emotion_analyzer=emotion_analyzer,
    ))
    assert cached["status"] == "cached"
    assert len(calls) == 2


def test_emotion_block_completes_when_intent_enrichment_fails(tmp_path) -> None:
    import asyncio

    store = SQLiteCaseStore(tmp_path / "handoff-independent.sqlite3")
    store.ensure_session("consumer-demo", "enrich-case")
    handoff = _start_handoff(store, "谢谢，帮我转人工")

    async def fail_intent(message, history=None):
        raise RuntimeError("intent unavailable")

    async def emotion_ok(client, message, recent, config):
        return {"emotion": "平静", "answers": {}, "evidence": {}, "source": "fallback_rules", "degraded": True}

    result = asyncio.run(enrich_handoff(
        store,
        "enrich-case",
        handoff["handoff_id"],
        emotion_client=None,
        intent_analyzer=fail_intent,
        emotion_analyzer=emotion_ok,
    ))

    assert result["intent_status"] == "failed"
    assert result["emotion_status"] == "degraded"
    state = store.get_conversation("enrich-case", viewer_role="agent")["case_state"]
    assert state["handoff_intent_enrichment"]["status"] == "failed"
    with sqlite3.connect(store.path) as db:
        status = db.execute(
            "SELECT status FROM handoff_blocks WHERE handoff_id=? AND block_type='emotion_urgency'",
            (handoff["handoff_id"],),
        ).fetchone()[0]
    assert status == "degraded"


def test_emotion_block_storage_failure_is_isolated_from_summary_and_next_steps(tmp_path, monkeypatch) -> None:
    import asyncio

    store = SQLiteCaseStore(tmp_path / "handoff-emotion-block-failure.sqlite3")
    store.ensure_session("consumer-demo", "enrich-case")
    handoff = _start_handoff(store, "谢谢，帮我转人工")

    async def intent_ok(message, history=None):
        return SimpleNamespace(
            intent=IntentCategory.HUMAN_HANDOFF, confidence=0.9, intent_group="handoff",
            entities={}, secondary_intents=[], sub_requests=[], intent_diagnostics={},
        )

    async def emotion_ok(client, message, recent, config):
        return {"emotion": "平静", "answers": {}, "evidence": {}, "source": "fallback_rules", "degraded": True}

    def fail_emotion_save(*args, **kwargs):
        raise RuntimeError("emotion_storage_failed")

    monkeypatch.setattr(store, "save_handoff_emotion_enrichment", fail_emotion_save)
    result = asyncio.run(enrich_handoff(
        store, "enrich-case", handoff["handoff_id"], emotion_client=None,
        intent_analyzer=intent_ok, emotion_analyzer=emotion_ok,
    ))

    assert result["emotion_status"] == "failed"
    assert result["summary_status"] == "degraded"
    assert result["next_steps_status"] == "degraded"
    conversation = store.get_conversation("enrich-case", viewer_role="agent")
    assert conversation["handoff"]["blocks"]["emotion_urgency"]["status"] == "failed"
    with sqlite3.connect(store.path) as db:
        events = db.execute(
            "SELECT block_type,status FROM handoff_blocks WHERE handoff_id=? ORDER BY block_type",
            (handoff["handoff_id"],),
        ).fetchall()
    assert dict(events)["emotion_urgency"] == "failed"
