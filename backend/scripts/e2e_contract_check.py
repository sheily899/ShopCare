"""不依赖外部模型/网络的 ShopCare 端到端契约验收。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intent.intent_recognizer import (
    CLASSIFY_INTENT_TOOL,
    LLM_INTENT_SCHEMA,
    IntentCategory,
    extract_entities,
    match_patterns,
)
from intent.beauty_intent_ticket_registry import BEAUTY_INTENT_TICKET_REGISTRY
from intent.beauty_intent_ticket_registry import BEAUTY_INTENT_TICKET_BY_ID
from agents.response_guard import guard_response
from storage.sqlite_case_store import SQLiteCaseStore

def main() -> int:
    assert len(BEAUTY_INTENT_TICKET_REGISTRY) >= 19
    allowed_intents = set(LLM_INTENT_SCHEMA["enum"]) | {
        IntentCategory.HUMAN_HANDOFF.value,
        IntentCategory.OTHER.value,
    }
    assert set(CLASSIFY_INTENT_TOOL["input_schema"]["properties"]["intent"]["enum"]) == allowed_intents
    assert extract_entities("\u8ba2\u5355\u53f7 SC-DEMO-4821 \u8fc7\u654f\u4e86，\u4e0d\u8981\u9000\u6b3e\u53ea\u6362\u8d27")["order_id"] == ["SC-DEMO-4821"]
    exchange_preference = match_patterns("\u6211\u4e0d\u8981\u9000\u6b3e\uff0c\u53ea\u6362\u8d27")
    assert exchange_preference["intent"].value == "other"
    assert "return_refund" not in exchange_preference["scores"]
    assert "wrong_item" not in exchange_preference["scores"]
    assert match_patterns("\u6211\u7684\u8ba2\u5355\u53f7 SC-DEMO-4821，\u7269\u6d41\u5230\u54ea\u91cc\u4e86")['intent'] is IntentCategory.ORDER_STATUS_BEAUTY
    assert match_patterns("\u7269\u6d41\u4e09\u5929\u6ca1\u66f4\u65b0\u4e86\uff0c\u5e2e\u6211\u50ac\u67e5")['intent'] is IntentCategory.LOGISTICS_STALLED
    assert "lookup_logistics" in BEAUTY_INTENT_TICKET_BY_ID["order_status"]["allowed_tools"]
    docs = list((ROOT / "data" / "knowledge_drafts").glob("*.md"))
    assert len(docs) >= 50, f"知识库源文档不足: {len(docs)}"
    with tempfile.TemporaryDirectory(prefix=".shopcare-e2e-", dir=ROOT) as temporary_directory:
        store = SQLiteCaseStore(Path(temporary_directory) / "contract.sqlite3")
        store.ensure_session("consumer-demo", "e2e")
        store.update_case_state("e2e", {"order_id": "SC-DEMO-4821"}, expected_version=0)
        try:
            store.update_case_state("e2e", {"x": 1}, expected_version=0)
        except ValueError:
            pass
        else:
            raise AssertionError("乐观并发版本校验未生效")
        store.record_slot_value("e2e", "order_id", "SC-DEMO-4821", "SC-DEMO-4821", source_type="text", source_ref="m1")
        store.record_risk_event("consumer-demo", "e2e", "P0", "adverse_reaction", "\u8fc7\u654f")
        event_id = store.append_stream_event("e2e", "ack", {"text": "ok"})
        assert store.list_stream_events("e2e", event_id - 1)[0]["event"] == "ack"
    assert len(guard_response("\u5df2\u8bc6\u522b\u610f\u56fe，\u8bf7\u7ee7\u7eed\u5904\u7406")) > 0
    print(json.dumps({"ok": True, "intent_count": len(BEAUTY_INTENT_TICKET_REGISTRY), "knowledge_documents": len(docs)}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
