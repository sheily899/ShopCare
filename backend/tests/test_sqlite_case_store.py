from __future__ import annotations

from storage.sqlite_case_store import SQLiteCaseStore


def test_store_shares_messages_and_case_state_between_roles(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "shopcare.sqlite3")
    store.ensure_session("consumer-demo", "conv-1")

    message = store.append_message(
        "conv-1", "consumer", "consumer-demo", "用了精华后脸发红，想退货"
    )
    store.update_case_state(
        "conv-1",
        {"sub_requests": ["adverse_reaction", "return_refund"], "risk_level": "P1"},
    )

    service_view = store.get_conversation("conv-1", viewer_role="agent")
    consumer_view = store.get_conversation("conv-1", viewer_role="consumer")

    assert service_view["messages"][0]["message_id"] == message["message_id"]
    assert service_view["case_state"]["risk_level"] == "P1"
    assert consumer_view["messages"][0]["content"] == "用了精华后脸发红，想退货"
    assert "case_state" not in consumer_view


def test_consumer_message_cursor_advances_across_hidden_rows_without_skipping_visible_rows(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "consumer-message-cursor.sqlite3")
    store.ensure_session("consumer-demo", "consumer-cursor")
    for index in range(105):
        store.append_message(
            "consumer-cursor", "system", "shopcare", f"internal-{index}",
            visible_to_consumer=False,
        )
    visible = store.append_message(
        "consumer-cursor", "customer_service", "agent-demo", "请补充订单号"
    )

    first_page = store.get_conversation(
        "consumer-cursor", viewer_role="consumer", after_message_seq=0
    )
    second_page = store.get_conversation(
        "consumer-cursor",
        viewer_role="consumer",
        after_message_seq=first_page["last_message_seq"],
    )

    assert first_page["messages"] == []
    assert first_page["last_message_seq"] == 100
    assert [item["message_id"] for item in second_page["messages"]] == [visible["message_id"]]
    assert second_page["last_message_seq"] == visible["message_seq"]


def test_store_creates_one_open_ticket_and_appends_related_request(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "shopcare.sqlite3")
    store.ensure_session("consumer-demo", "conv-2")

    first = store.create_ticket(
        "consumer-demo", "conv-2", "adverse_reaction", "不良反应", ["adverse_reaction"]
    )
    second = store.create_or_append_ticket(
        "consumer-demo", "conv-2", "return_refund", "退货", ["return_refund"]
    )

    assert second["ticket_id"] == first["ticket_id"]
    assert second["items"] == ["adverse_reaction", "return_refund"]


def test_dashboard_counts_handoffs_and_reads_risk_tags_from_shared_case_facts(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "shopcare.sqlite3")
    store.ensure_session("consumer-demo", "conv-risk")
    store.ensure_session("consumer-demo", "conv-ai")
    store.update_case_state(
        "conv-risk",
        {
            "status": "waiting_human",
            "intent": "adverse_reaction",
            "urgency": "P0",
            "emotion": "anxious",
            "case_facts": {
                "schema_version": 1,
                "facts": {"symptom": [{"value": "脸部发红", "source_type": "message", "source_ref": "m-1"}]},
            },
        },
    )
    store.update_case_state(
        "conv-ai",
        {"intent": "ingredient_fit", "emotion": "calm", "case_facts": {"schema_version": 1, "facts": {}}},
    )

    dashboard = store.get_risk_dashboard()
    users = {item["conv_id"]: item for item in dashboard["users"]}

    assert dashboard["metrics"]["total_sessions"] == 2
    assert dashboard["metrics"]["handoff_rate"] == 50
    assert dashboard["metrics"]["ai_resolution_rate"] == 50
    assert users["conv-risk"]["status"] == "waiting_human"
    assert users["conv-risk"]["risk_tags"] == ["不良反应"]


def test_async_emotion_update_keeps_summary_and_case_facts_in_sync(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "emotion.sqlite3")
    store.ensure_session("consumer-demo", "conv-emotion")
    store.update_case_state(
        "conv-emotion",
        {
            "emotion": "平静",
            "emotion_analysis_request_id": "request-3",
            "handoff_summary": {"emotion": "平静", "next_steps": ["核对退款条件"]},
            "ticket_draft": {"status": "draft"},
            "case_facts": {"schema_version": 1, "facts": {}},
        },
    )

    updated = store.update_emotion_analysis(
        "conv-emotion",
        "request-3",
        {
            "emotion": "威胁升级",
            "answers": {"threat": True, "anger": False, "dissatisfaction": False, "anxiety": False},
            "evidence": {"threat": "明天再没结果我就投诉"},
            "source": "llm",
            "degraded": False,
        },
    )

    assert updated["emotion"] == "威胁升级"
    assert updated["handoff_summary"]["emotion"] == "威胁升级"
    assert updated["handoff_summary"]["next_steps"] == ["核对退款条件"]
    assert updated["ticket_draft"] == {"status": "draft"}
    emotion_record = updated["case_facts"]["facts"]["emotion"][-1]
    assert emotion_record["source_type"] == "emotion_llm"
    assert emotion_record["source_ref"] == "request-3"
    assert updated["case_facts"]["facts"]["emotion_evidence"][-1]["value"] == "明天再没结果我就投诉"


def test_stale_async_emotion_result_cannot_override_newer_request(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "stale-emotion.sqlite3")
    store.ensure_session("consumer-demo", "conv-stale-emotion")
    store.update_case_state(
        "conv-stale-emotion",
        {
            "emotion": "不满",
            "emotion_analysis_request_id": "request-new",
            "handoff_summary": {"emotion": "不满", "next_steps": ["核对退款条件"]},
            "case_facts": {"schema_version": 1, "facts": {}},
        },
    )

    updated = store.update_emotion_analysis(
        "conv-stale-emotion",
        "request-old",
        {
            "emotion": "威胁升级",
            "answers": {"threat": True},
            "evidence": {"threat": "上一条消息的威胁语句"},
            "source": "llm",
            "degraded": False,
        },
    )

    assert updated["emotion"] == "不满"
    assert updated["handoff_summary"]["emotion"] == "不满"
    assert "emotion_llm" not in updated
    assert updated["case_facts"]["facts"] == {}

