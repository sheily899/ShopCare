"""转人工动作、摘要和建议回复必须来自结构化案件事实。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from types import SimpleNamespace
from typing import Any

from handoff.generation import (
    _generation_hash,
    generate_handoff_blocks,
    load_handoff_action_library,
    select_handoff_actions,
)
from handoff.processor import _detect_exclusive_slot_conflicts
from agents.response_guard import guard_handoff_suggested_reply
from intent.beauty_intent_registry import BEAUTY_INTENT_BY_ID
from storage.sqlite_case_store import SQLiteCaseStore


def _source() -> dict[str, Any]:
    return {
        "handoff_id": "handoff-1",
        "conv_id": "case-1",
        "source_message_seq": 10,
        "handoff_reason": "consumer_requested_human",
        "intent_enrichment": {"intent": "human_handoff", "secondary_intents": []},
        "sub_requests": [
            {"intent_id": "adverse_reaction", "status": "open", "source_message_seq": 7},
            {"intent_id": "return_refund", "status": "open", "source_message_seq": 8},
            {"intent_id": "missing_item", "status": "open", "source_message_seq": 9},
        ],
        "slots": {
            "order_id": {"value": "SC-DEMO-4821", "source_message_seq": 8},
            "symptom": {"value": "脸部发红", "source_message_seq": 7},
            "current_symptoms": {"value": "脸红、肿胀", "source_message_seq": 7},
            "medical_visit": {"value": None, "source_message_seq": 7},
            "opened_or_used": {"value": "已使用", "source_message_seq": 8},
            "return_reason": {"value": "使用后不适", "source_message_seq": 8},
            "missing_item": {"value": "直播赠品小样", "source_message_seq": 9},
            "quantity": {"value": "2", "source_message_seq": 9},
        },
        "action_logs": [
            {"action_id": "action-1", "action_type": "tool", "description": "已核对订单号", "status": "success", "source_ref": "m-8", "source_message_seq": 8},
            {"action_id": "action-tool-call", "action_type": "tool", "description": "工具 lookup_order 返回成功", "status": "success", "source_message_seq": 8},
            {"action_id": "action-intent", "action_type": "rule", "description": "意图识别完成：adverse_reaction", "status": "success", "source_message_seq": 7},
            {"action_id": "action-internal", "action_type": "system", "description": "转人工后台已更新情绪与最终紧急度", "status": "success"},
        ],
        "conversation_messages": [
            {"message_seq": 7, "role": "消费者", "content": "用后脸红肿，想退货。"},
            {"message_seq": 8, "role": "AI客服", "content": "我先核对订单信息。"},
            {"message_seq": 9, "role": "消费者", "content": "订单号 SC-DEMO-4821，赠品小样没有收到。"},
            {"message_seq": 10, "role": "消费者", "content": "转人工"},
            {"message_seq": 11, "role": "AI客服", "content": "已为您转接人工，请您稍等。"},
        ],
        "ticket": {"status": "draft", "fields": {"medical_visit": None}},
        "callback_task": None,
        "contradictions": [],
        # 这是隔离测试用的原始聊天陷阱，任何模型输入都不应携带它。
        "raw_messages": ["RAW_CHAT_MUST_NOT_REACH_MODEL_92a1"],
    }


def test_action_library_covers_exactly_nineteen_intents_with_three_to_five_rules() -> None:
    library = load_handoff_action_library()

    assert set(library) == set(BEAUTY_INTENT_BY_ID)
    assert all(3 <= len(actions) <= 5 for actions in library.values())


def test_only_opposing_values_of_exclusive_slots_are_marked_as_conflicts() -> None:
    conflicts = _detect_exclusive_slot_conflicts({
        "medical_visit": ["已就医", "没就医"],
        "symptom": ["发红", "刺痛"],
        "opened_or_used": ["未拆封"],
    }, 17)

    assert conflicts == [{
        "field_id": "medical_visit",
        "values": ["已就医", "没就医"],
        "source_message_seqs": [17],
    }]


def test_action_selection_orders_safety_then_missing_fields_then_ticket_actions() -> None:
    actions = select_handoff_actions(_source())

    assert [item["id"] for item in actions] == [
        "adverse_reaction.confirm_medical_visit",
        "return_refund.confirm_return_conditions",
        "missing_item.verify_gift_fulfillment",
    ]
    assert [item["type"] for item in actions] == ["safety", "ticket", "ticket"]
    assert all(item["source_refs"] for item in actions)
    assert "R-04" in actions[1]["reason"]
    assert "R-11" in actions[2]["reason"]


def test_completed_safety_field_and_resolved_request_are_not_repeated() -> None:
    source = _source()
    source["slots"]["medical_visit"]["value"] = "已就医"
    source["sub_requests"] = [source["sub_requests"][0]]
    actions = select_handoff_actions(source)

    assert all(item["id"] != "adverse.confirm_medical_visit" for item in actions)


def test_ticket_facts_satisfy_slot_actions_without_reasking() -> None:
    source = _source()
    source["sub_requests"] = [source["sub_requests"][1]]
    source["slots"].pop("opened_or_used")
    source["ticket"] = {"status": "draft", "fields": {"opened_or_used": "已使用", "order_id": "SC-DEMO-4821"}}

    actions = select_handoff_actions(source)

    assert all(item["id"] != "return_refund.confirm_condition" for item in actions)


def test_direct_handoff_without_business_request_produces_one_greeting_action() -> None:
    source = _source()
    source["sub_requests"] = []
    source["intent_enrichment"] = {"intent": "human_handoff", "secondary_intents": []}

    actions = select_handoff_actions(source)

    assert len(actions) == 1
    assert actions[0]["id"] == "handoff.ask_request"
    assert actions[0]["text"] == "先问候并询问诉求"


def test_single_business_intent_without_subrequest_record_is_not_summarized_as_empty() -> None:
    source = _source()
    source["sub_requests"] = []
    source["intent_enrichment"] = {"intent": "order_status", "secondary_intents": []}
    store = _FakeStore(source)

    result = asyncio.run(generate_handoff_blocks(store, "case-1", "handoff-1", client=None))

    expected_name = BEAUTY_INTENT_BY_ID["order_status"]["name_zh"]
    assert expected_name in store.saved["summary"]["payload"]["text"]
    assert "暂无其他诉求" not in store.saved["summary"]["payload"]["text"]
    assert result["summary_status"] == "degraded"


class _FakeClient:
    def __init__(self) -> None:
        self.messages = self
        self.calls: list[dict[str, Any]] = []
        self._responses = [
            json.dumps({
                "statements": [{
                    "text": "顾客反馈使用后不适，同时申请退货并反映赠品未收到；AI 已核对订单，相关问题仍待处理，因此转人工。",
                    "sources": [{"type": "message", "seq": 7}, {"type": "message", "seq": 8}, {"type": "message", "seq": 9}, {"type": "action", "id": "action-1"}],
                }]
            }, ensure_ascii=False),
            "请问目前是否已经就医？我会一并核实退货退款条件和赠品发货记录，再向您说明处理结果。",
        ]

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._responses.pop(0))])


class _FakeStore:
    def __init__(self, source: dict[str, Any]) -> None:
        self.source = source
        self.saved: dict[str, dict[str, Any]] = {}

    def get_handoff_generation_input(self, handoff_id: str) -> dict[str, Any]:
        assert handoff_id == self.source["handoff_id"]
        return self.source

    def save_handoff_generated_block(self, handoff_id: str, **kwargs: Any) -> dict[str, Any]:
        self.saved[kwargs["block_type"]] = kwargs
        return {"status": kwargs["status"], "payload": kwargs["payload"]}


def test_generated_blocks_use_full_conversation_without_internal_logs_and_guard_reply() -> None:
    source = _source()
    store = _FakeStore(source)
    client = _FakeClient()

    result = asyncio.run(generate_handoff_blocks(store, "case-1", "handoff-1", client=client))

    assert result["summary_status"] == "ready"
    assert result["next_steps_status"] == "ready"
    assert "已为您转接人工" not in store.saved["next_steps"]["payload"]["suggested_reply"]
    assert store.saved["summary"]["payload"]["statements"][0]["sources"]
    prompts = "\n".join(call["messages"][0]["content"] for call in client.calls)
    assert "RAW_CHAT_MUST_NOT_REACH_MODEL_92a1" not in prompts
    assert "脸红肿" in prompts
    assert "已为您转接人工，请您稍等" in prompts
    assert "转人工后台已更新情绪与最终紧急度" not in prompts
    assert "action-internal" not in prompts
    assert "lookup_order" not in prompts and "adverse_reaction" not in prompts
    assert "工具 业务查询 返回成功" in prompts
    assert all("\"temperature\": 0" in str(call.get("extra_body")) or call.get("temperature") == 0 for call in client.calls)


def test_direct_handoff_summary_prompt_uses_consumer_dialogue_not_processing_logs() -> None:
    source = _source()
    source["sub_requests"] = []
    source["intent_enrichment"] = {"intent": "human_handoff", "secondary_intents": []}
    source["slots"] = {}
    source["conversation_messages"] = [
        {"message_seq": 10, "role": "消费者", "content": "转人工"},
        {"message_seq": 11, "role": "AI客服", "content": "已为您转接人工，请您稍等。"},
    ]
    store = _FakeStore(source)

    class DirectHandoffClient:
        def __init__(self) -> None:
            self.messages = self
            self.calls: list[dict[str, Any]] = []
            self.responses = [
                json.dumps({"statements": [{
                    "text": "顾客直接要求人工客服接手；AI客服已回复正在转接，目前没有其他已识别的业务诉求。",
                    "sources": [{"type": "message", "seq": 10}, {"type": "message", "seq": 11}],
                }]}, ensure_ascii=False),
                "您好，请问您这边具体需要处理什么问题？",
            ]

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.responses.pop(0))])

    client = DirectHandoffClient()
    result = asyncio.run(generate_handoff_blocks(store, "case-1", "handoff-1", client=client))
    prompt = client.calls[0]["messages"][0]["content"]

    assert result["summary_status"] == "ready"
    assert "顾客直接要求人工客服接手" in store.saved["summary"]["payload"]["text"]
    assert "转人工" in prompt and "已为您转接人工" in prompt
    assert "转人工后台已更新情绪与最终紧急度" not in prompt


def test_summary_and_next_steps_degrade_independently_when_model_is_unavailable() -> None:
    store = _FakeStore(_source())

    result = asyncio.run(generate_handoff_blocks(store, "case-1", "handoff-1", client=None))

    assert result["summary_status"] == "degraded"
    assert result["next_steps_status"] == "degraded"
    assert store.saved["summary"]["payload"]["text"]
    assert store.saved["next_steps"]["payload"]["items"]
    assert store.saved["next_steps"]["payload"]["suggested_reply"]


def test_summary_generation_failure_is_recorded_without_blocking_next_steps(monkeypatch) -> None:
    import handoff.generation as generation

    async def fail_summary(*args, **kwargs):
        raise RuntimeError("summary_generation_failed")

    monkeypatch.setattr(generation, "_generate_summary", fail_summary)
    store = _FakeStore(_source())

    result = asyncio.run(generate_handoff_blocks(store, "case-1", "handoff-1", client=None))

    assert result["summary_status"] == "failed"
    assert result["next_steps_status"] == "degraded"
    assert store.saved["summary"]["status"] == "failed"
    assert store.saved["summary"]["error_code"] == "RuntimeError"
    assert store.saved["next_steps"]["payload"]["items"]


def test_next_steps_generation_failure_is_recorded_without_blocking_summary(monkeypatch) -> None:
    import handoff.generation as generation

    async def fail_next_steps(*args, **kwargs):
        raise RuntimeError("next_steps_generation_failed")

    monkeypatch.setattr(generation, "_generate_next_steps", fail_next_steps)
    store = _FakeStore(_source())

    result = asyncio.run(generate_handoff_blocks(store, "case-1", "handoff-1", client=None))

    assert result["summary_status"] == "degraded"
    assert result["next_steps_status"] == "failed"
    assert store.saved["next_steps"]["status"] == "failed"
    assert store.saved["next_steps"]["error_code"] == "RuntimeError"
    assert store.saved["summary"]["payload"]["text"]


def test_bad_summary_source_reference_uses_nonempty_template_fallback() -> None:
    source = _source()
    store = _FakeStore(source)

    class BadReferenceClient:
        def __init__(self) -> None:
            self.messages = self
            self.responses = [
                json.dumps({"statements": [{"text": "无来源的虚构事实", "sources": [{"type": "message", "seq": 999}]}]}),
                "请问您目前是否已经就医？我会核实退货条件和赠品发货记录。",
            ]

        async def create(self, **kwargs: Any) -> Any:
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.responses.pop(0))])

    result = asyncio.run(generate_handoff_blocks(store, "case-1", "handoff-1", client=BadReferenceClient()))

    assert result["summary_status"] == "degraded"
    assert result["next_steps_status"] == "ready"
    assert store.saved["summary"]["payload"]["text"]
    assert "无来源的虚构事实" not in store.saved["summary"]["payload"]["text"]
    assert "脸部发红" in store.saved["summary"]["payload"]["text"]


def test_structured_contradiction_allows_at_most_one_cited_model_supplement() -> None:
    source = _source()
    source["sub_requests"] = [source["sub_requests"][1]]
    source["contradictions"] = [{"field_id": "return_reason", "values": ["过敏", "不适"], "raw_message": "RAW_CONFLICT_MESSAGE"}]
    source["slots"] = {"order_id": source["slots"]["order_id"], "opened_or_used": source["slots"]["opened_or_used"], "return_reason": source["slots"]["return_reason"]}
    store = _FakeStore(source)

    class SupplementClient:
        def __init__(self) -> None:
            self.messages = self
            self.calls: list[dict[str, Any]] = []
            self.responses = [
                json.dumps({"statements": [{"text": "顾客申请退货，相关条件仍需核对。", "sources": [{"type": "message", "seq": 7}, {"type": "message", "seq": 8}]}]}, ensure_ascii=False),
                json.dumps({"action": {"text": "确认退货原因的具体描述", "reason": "退货原因存在两种结构化记录", "type": "missing_field", "field_id": "return_reason", "source_refs": [{"type": "message", "seq": 8}]}}, ensure_ascii=False),
                "我会先核对退货条件，并确认退货原因记录，再向您说明。",
            ]

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.responses.pop(0))])

    client = SupplementClient()
    result = asyncio.run(generate_handoff_blocks(store, "case-1", "handoff-1", client=client))

    actions = store.saved["next_steps"]["payload"]["items"]
    assert result["next_steps_status"] == "ready"
    assert sum(item["id"] == "model.supplement" for item in actions) == 1
    prompts = "\n".join(call["messages"][0]["content"] for call in client.calls)
    assert "RAW_CONFLICT_MESSAGE" not in prompts
    assert len(client.calls) == 3


def test_registered_intent_without_matching_action_condition_allows_one_model_supplement() -> None:
    source = _source()
    source["sub_requests"] = []
    source["intent_enrichment"] = {"intent": "ingredient_fit", "secondary_intents": []}
    source["slots"] = {
        "product_name": {"value": "澄润屏障修护精华水", "source_message_seq": 10},
        "skin_type": {"value": "敏感肌", "source_message_seq": 10},
    }
    store = _FakeStore(source)

    class CoverageGapClient:
        def __init__(self) -> None:
            self.messages = self
            self.calls: list[dict[str, Any]] = []
            self.responses = [
                json.dumps({"statements": [{"text": "顾客咨询澄润屏障修护精华水的成分适配。", "sources": [{"type": "message", "seq": 10}, {"type": "message", "seq": 11}]}]}, ensure_ascii=False),
                json.dumps({"action": {"text": "核对敏感肌适配资料是否覆盖该商品", "reason": "现有动作条件未覆盖已明确的商品和肤质组合", "type": "ticket", "source_refs": [{"type": "message", "seq": 10}]}}, ensure_ascii=False),
                "我会核对这款精华水针对敏感肌的适配资料，再向您说明。",
            ]

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.responses.pop(0))])

    client = CoverageGapClient()
    result = asyncio.run(generate_handoff_blocks(store, "case-1", "handoff-1", client=client))

    actions = store.saved["next_steps"]["payload"]["items"]
    assert result["next_steps_status"] == "ready"
    assert [item["id"] for item in actions] == ["model.supplement"]
    assert len(client.calls) == 3


def test_identical_structured_state_reuses_ready_blocks_without_model_calls() -> None:
    source = _source()
    store = _FakeStore(source)
    first_client = _FakeClient()
    asyncio.run(generate_handoff_blocks(store, "case-1", "handoff-1", client=first_client))
    source["content_blocks"] = {
        block_type: {"status": value["status"], "state_hash": value["state_hash"], "payload": value["payload"]}
        for block_type, value in store.saved.items()
    }
    replay_store = _FakeStore(source)
    replay_client = _FakeClient()

    result = asyncio.run(generate_handoff_blocks(replay_store, "case-1", "handoff-1", client=replay_client))

    assert result["status"] == "cached"
    assert result["summary_status"] == "cached"
    assert result["next_steps_status"] == "cached"
    assert replay_client.calls == []


def test_generation_hash_invalidates_when_emotion_urgency_or_rule_version_changes(monkeypatch) -> None:
    import handoff.generation as generation

    source = _source()
    source["emotion_urgency"] = {"emotion": {"value": "不满"}, "final": {"value": "P2"}}
    source["urgency"] = {"temporary": {"value": "P3"}, "final": {"value": "P2"}}
    monkeypatch.setattr(generation, "load_handoff_emotion_config", lambda: {"version": "emotion-v1"})
    monkeypatch.setattr(generation, "load_handoff_urgency_config", lambda: {"version": "urgency-v1"})
    initial_hash = _generation_hash(source, generation.load_handoff_generation_config())

    source["emotion_urgency"]["emotion"]["value"] = "威胁升级"
    emotion_hash = _generation_hash(source, generation.load_handoff_generation_config())
    assert emotion_hash != initial_hash

    source["emotion_urgency"]["emotion"]["value"] = "不满"
    source["urgency"]["final"]["value"] = "P1"
    urgency_hash = _generation_hash(source, generation.load_handoff_generation_config())
    assert urgency_hash != initial_hash

    source["urgency"]["final"]["value"] = "P2"
    monkeypatch.setattr(generation, "load_handoff_urgency_config", lambda: {"version": "urgency-v2"})
    rule_hash = _generation_hash(source, generation.load_handoff_generation_config())
    assert rule_hash != initial_hash


def test_generation_hash_ignores_elapsed_time_without_source_state_change() -> None:
    import handoff.generation as generation

    source = _source()
    config = generation.load_handoff_generation_config()
    first_hash = _generation_hash(source, config)
    source["observed_at"] = "2099-01-01T00:00:00Z"

    assert _generation_hash(source, config) == first_hash


def test_generation_hash_is_scoped_to_conversation_id() -> None:
    import handoff.generation as generation

    source = _source()
    config = generation.load_handoff_generation_config()
    first_hash = _generation_hash(source, config)
    other_conversation = {**source, "conv_id": "case-2"}

    assert _generation_hash(other_conversation, config) != first_hash


def test_generation_hash_invalidates_for_empty_slot_state_and_other_case_changes() -> None:
    import handoff.generation as generation

    source = _source()
    config = generation.load_handoff_generation_config()
    initial_hash = _generation_hash(source, config)

    source["slots"]["medical_visit"]["confirmation_status"] = "declined"
    assert _generation_hash(source, config) != initial_hash

    source = _source()
    source["slots"]["order_id"]["value"] = "SC-DEMO-9200"
    assert _generation_hash(source, config) != initial_hash

    source = _source()
    source["sub_requests"][0]["status"] = "in_progress"
    assert _generation_hash(source, config) != initial_hash

    source = _source()
    source["ticket"]["version"] = 2
    assert _generation_hash(source, config) != initial_hash

    source = _source()
    source["action_log_version"] = 2
    assert _generation_hash(source, config) != initial_hash

    source = _source()
    source["source_message_seq"] = 11
    assert _generation_hash(source, config) != initial_hash

    changed_config = {**config, "template_version": "handoff-summary-template-v3"}
    assert _generation_hash(_source(), changed_config) != initial_hash


def test_sqlite_generation_input_reads_complete_chat_and_block_save_is_idempotent(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "generation.sqlite3")
    store.ensure_session("consumer-demo", "generation-case")
    for index in range(8):
        store.append_message(
            "generation-case",
            "consumer" if index % 2 == 0 else "ai",
            "consumer-demo" if index % 2 == 0 else "assistant-demo",
            f"早期对话第 {index + 1} 条",
        )
    handoff = store.start_handoff(
        "generation-case",
        user_id="consumer-demo",
        message="RAW_CONSUMER_CHAT_BODY_FORBIDDEN",
        idempotency_key="handoff:generation-case:one",
        reason="consumer_requested_human",
        temporary_summary={"text": "顾客直接要求转人工", "sources": []},
        temporary_priority="P3",
        temporary_priority_score=0,
        temporary_priority_reasons=["无额外风险规则命中"],
        temporary_priority_rule_version="test-v1",
        online_ack="已为您转接人工，请您稍等。",
        unknown_presence_ack="已登记，专员上线后会联系您。",
        callback_due_at=None,
    )
    store.sync_case_sub_requests(
        "generation-case",
        [{"id": "return:1", "intent": "return_refund"}],
        source_ref=handoff["consumer_message_id"],
        source_message_seq=handoff["source_message_seq"],
    )

    source = store.get_handoff_generation_input(handoff["handoff_id"])
    assert source is not None
    assert len(source["conversation_messages"]) == 10
    assert source["conversation_messages"][0]["content"] == "早期对话第 1 条"
    assert source["conversation_messages"][-2]["content"] == "RAW_CONSUMER_CHAT_BODY_FORBIDDEN"
    assert source["conversation_messages"][-1]["role"] == "AI客服"
    first = store.save_handoff_generated_block(
        handoff["handoff_id"], block_type="summary", status="ready", state_hash="hash-1",
        payload={"text": "交接摘要", "statements": [{"text": "交接摘要", "sources": [{"type": "message", "seq": 1}]}]},
        latency_ms=4.2, source_message_seq=handoff["source_message_seq"],
    )
    retry = store.save_handoff_generated_block(
        handoff["handoff_id"], block_type="summary", status="ready", state_hash="hash-1",
        payload={"text": "不应覆盖首次结果"}, latency_ms=8.0,
        source_message_seq=handoff["source_message_seq"],
    )

    assert first["status"] == "ready"
    assert retry["cached"] is True
    with sqlite3.connect(store.path) as db:
        count = db.execute(
            "SELECT COUNT(*) FROM handoff_events WHERE handoff_id=? AND event_type='handoff.block.updated'",
            (handoff["handoff_id"],),
        ).fetchone()[0]
    assert count == 1


def test_response_guard_replaces_transfer_copy_and_unverified_completed_claims() -> None:
    safe, rejected = guard_handoff_suggested_reply(
        "已为您转接人工，赠品也已经补发。",
        "我会先核实赠品的发货记录，再向您说明。",
    )

    assert rejected is True
    assert safe == "我会先核实赠品的发货记录，再向您说明。"
