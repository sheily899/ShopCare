"""槽位注册表读取与中文展示映射。"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

_PATH = Path(__file__).with_name("slot_registry.json")


def _load() -> dict[str, Any]:
    return json.loads(_PATH.read_text(encoding="utf-8"))


def get_slot(slot_id: str) -> Mapping[str, Any] | None:
    for item in _load().get("slots", []):
        if item.get("slot_id") == slot_id:
            return item
    return None


def normalize_value(slot_id: str, value: str) -> tuple[str, str]:
    slot = get_slot(slot_id)
    if not slot or not isinstance(value, str) or not value.strip():
        raise ValueError("槽位或槽位值无效")
    raw = value.strip()
    for option in slot.get("options", []):
        if raw in {option.get("value"), option.get("label")}:
            return str(option["value"]), str(option["label"])
    if slot.get("type") == "text":
        return raw, raw
    raise ValueError("不在槽位注册表选项内")


_NEW_REQUEST_MARKERS = re.compile(
    r"(?:\?|？|请问|我想问|我还想问|我想咨询|我想了解|另外|还有一个问题|另一个问题|"
    r"帮我查|查一下|查订单|查物流|(?:查询|查|问).{0,4}物流|物流进度|订单状态|成分|色号|怎么用|转人工|投诉|"
    r"我不想退款|我不要退款|我决定.{0,4}不(?:想|要|再)?退(?:款|货)|"
    r"先不(?:退款|退货)|暂时不(?:退款|退货)|不退了|取消(?:退款|退货))"
)
_PENDING_ANSWER_COMPATIBLE_INTENTS = {
    # 退款原因可能由消费者用不良反应描述；这仍是对退款追问的回答，不是切换诉求。
    "return_reason": {"adverse_reaction"},
}


def pending_text_answer_field_id(
    message: str,
    pending_fields: list[Any] | tuple[Any, ...] | None,
    *,
    current_intent_id: str = "",
) -> str | None:
    """返回本条消息可回答的、上轮实际询问的首个自由文本字段。

    拒绝空/过长文本、明显的新问题/新任务，以及和待答业务无关的分类结果。
    """

    text = str(message or "").strip()
    if not text or len(text) > 160 or _NEW_REQUEST_MARKERS.search(text):
        return None

    for pending in pending_fields or []:
        field_id = pending.get("field_id") if isinstance(pending, Mapping) else pending
        field_id = str(field_id or "").strip()
        context_intents = {
            str(value) for value in pending.get("context_intent_ids", [])
        } if isinstance(pending, Mapping) else set()
        normalized_intent = str(current_intent_id or "").strip()
        if (
            normalized_intent
            and normalized_intent != "other"
            and context_intents
            and normalized_intent not in context_intents
            and normalized_intent not in _PENDING_ANSWER_COMPATIBLE_INTENTS.get(field_id, set())
        ):
            continue
        slot = get_slot(field_id)
        if not field_id or not slot or slot.get("type") != "text":
            continue
        return field_id
    return None


def apply_pending_text_answer(
    message: str,
    entities: Mapping[str, Any] | None,
    pending_fields: list[Any] | tuple[Any, ...] | None,
    *,
    current_intent_id: str = "",
) -> dict[str, Any]:
    """将自然语言回答归入上轮等待的首个文本槽位，不覆盖本轮已抽取结果。"""

    result = dict(entities or {})
    field_id = pending_text_answer_field_id(
        message,
        pending_fields,
        current_intent_id=current_intent_id,
    )
    if field_id:
        values = result.get(field_id)
        values = values if isinstance(values, (list, tuple, set)) else [values]
        if not any(value is not None and str(value).strip() for value in values):
            result[field_id] = [str(message).strip()]
    return result
