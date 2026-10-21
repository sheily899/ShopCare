"""结构化事实驱动的客服交接摘要、动作和建议回复。"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import re
import time
from functools import lru_cache
from typing import Any, Callable, Mapping

import yaml

from agents.response_guard import guard_handoff_suggested_reply
from handoff.config import (
    handoff_config_path,
    load_handoff_copy_config,
    load_handoff_emotion_config,
    load_handoff_urgency_config,
)
from intent.beauty_intent_registry import BEAUTY_INTENT_BY_ID


logger = logging.getLogger(__name__)
_ACTION_TYPES = {"safety", "commitment", "missing_field", "ticket"}
_CONDITION_KINDS = {"missing_slot", "slot_value_in", "subrequest_open", "ticket_open", "callback_pending_due"}
_DONE_KINDS = {"slot_filled", "subrequest_resolved", "ticket_submitted", "callback_contacted"}
_PRIORITY = {"safety": 0, "commitment": 1, "missing_field": 2, "ticket": 3}
_INTENT_NAMES = {key: str(value.get("name_zh") or key) for key, value in BEAUTY_INTENT_BY_ID.items()}
_SENSITIVE_SLOT_NAMES = {"new_address", "address", "phone", "mobile", "contact", "email", "buyer_nickname"}
_ORDER_ID_PATTERN = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]{2,}){1,}\b", re.IGNORECASE)
_BUSINESS_ACTION_LABELS = {
    "tool": "业务工具核对", "rule": "业务规则核对", "followup": "向消费者追问",
    "ticket": "工单记录",
}
_REQUEST_STATUS_LABELS = {"open": "待处理", "in_progress": "处理中", "resolved": "已解决", "blocked": "暂时受阻"}
_ACTION_STATUS_LABELS = {"pending": "进行中", "success": "已完成", "failed": "未完成"}
_HANDOFF_REASON_LABELS = {"consumer_requested_human": "顾客直接要求人工客服接手"}
@lru_cache(maxsize=1)
def load_handoff_action_library() -> dict[str, list[dict[str, Any]]]:
    """读取并严格校验不含可执行表达式的声明式动作 YAML。"""
    with handoff_config_path("handoff_actions.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not config.get("version") or not isinstance(config.get("actions"), list):
        raise ValueError("handoff_actions.yaml 缺少版本或 actions 列表")
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    required = {"id", "intent", "when", "type", "text", "reason", "ui_anchor", "done_when", "reply"}
    for raw in config["actions"]:
        if not isinstance(raw, dict) or not required <= set(raw):
            raise ValueError("交接动作缺少必填字段")
        action = dict(raw)
        action_id = str(action["id"])
        intent_id = str(action["intent"])
        condition = action["when"]
        done_when = action["done_when"]
        if action_id in seen_ids:
            raise ValueError(f"交接动作 ID 重复: {action_id}")
        if intent_id not in BEAUTY_INTENT_BY_ID:
            raise ValueError(f"交接动作包含未注册意图: {intent_id}")
        if action["type"] not in _ACTION_TYPES:
            raise ValueError(f"交接动作类型无效: {action_id}")
        if not isinstance(condition, dict) or condition.get("kind") not in _CONDITION_KINDS:
            raise ValueError(f"交接动作条件无效: {action_id}")
        if not isinstance(done_when, dict) or done_when.get("kind") not in _DONE_KINDS:
            raise ValueError(f"交接动作完成条件无效: {action_id}")
        if action["type"] == "missing_field" and not action.get("field_id"):
            raise ValueError(f"缺字段动作没有 field_id: {action_id}")
        for key in ("text", "reason", "ui_anchor", "reply"):
            if not isinstance(action.get(key), str) or not action[key].strip():
                raise ValueError(f"交接动作 {action_id} 的 {key} 不能为空")
        seen_ids.add(action_id)
        grouped.setdefault(intent_id, []).append(action)
    if set(grouped) != set(BEAUTY_INTENT_BY_ID):
        missing = sorted(set(BEAUTY_INTENT_BY_ID) - set(grouped))
        extra = sorted(set(grouped) - set(BEAUTY_INTENT_BY_ID))
        raise ValueError(f"交接动作覆盖意图不一致: missing={missing}, extra={extra}")
    if any(not 3 <= len(items) <= 5 for items in grouped.values()):
        raise ValueError("每个已注册意图必须配置 3 到 5 条交接动作")
    return grouped


@lru_cache(maxsize=1)
def load_handoff_action_version() -> str:
    with handoff_config_path("handoff_actions.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    version = str((config or {}).get("version") or "")
    if not version:
        raise ValueError("handoff_actions.yaml 缺少版本")
    return version


@lru_cache(maxsize=1)
def load_handoff_generation_config() -> dict[str, Any]:
    with handoff_config_path("handoff_generation.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    required = (
        "version", "temperature", "llm_timeout_seconds", "summary_max_tokens", "reply_max_tokens",
        "supplement_max_tokens", "summary_system_prompt", "reply_system_prompt",
        "supplement_system_prompt", "direct_handoff_summary", "empty_actions_reply",
        "reply_guard_fallback", "template_version",
        "summary_requests_template", "summary_actions_template", "summary_no_completed_actions",
        "summary_slots_template", "slot_labels", "summary_unresolved_template",
        "summary_reason_template", "direct_handoff_action",
    )
    if any(key not in config for key in required):
        raise ValueError("handoff_generation.json 缺少必填配置")
    if float(config["temperature"]) != 0:
        raise ValueError("交接内容生成 temperature 必须为 0")
    for key in ("summary_max_tokens", "reply_max_tokens", "supplement_max_tokens"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} 必须为正数")
    if float(config["llm_timeout_seconds"]) <= 0:
        raise ValueError("llm_timeout_seconds 必须为正数")
    return config


def validate_handoff_configuration() -> dict[str, str]:
    """Load and validate every configuration required by the handoff flow.

    Returns:
        A map of configuration roles to their version identifiers.

    Raises:
        OSError: A required configuration file is missing or unreadable.
        ValueError: A configuration file does not satisfy its schema.
    """
    copy = load_handoff_copy_config()
    urgency = load_handoff_urgency_config()
    emotion = load_handoff_emotion_config()
    actions = load_handoff_action_library()
    action_version = load_handoff_action_version()
    generation = load_handoff_generation_config()
    if not actions:
        raise ValueError("handoff_actions.yaml 未配置有效动作")
    return {
        "copy": str(copy["version"]),
        "urgency": str(urgency.get("version") or ""),
        "emotion": str(emotion["version"]),
        "actions": action_version,
        "generation": str(generation["version"]),
    }


def _slot_value(source: Mapping[str, Any], field_id: str) -> Any:
    slots = source.get("slots") if isinstance(source.get("slots"), Mapping) else {}
    value = slots.get(field_id)
    value = value.get("value") if isinstance(value, Mapping) else value
    if not _slot_is_missing(value):
        return value
    ticket = source.get("ticket") if isinstance(source.get("ticket"), Mapping) else {}
    fields = ticket.get("fields") if isinstance(ticket.get("fields"), Mapping) else {}
    return fields.get(field_id)


def _slot_is_missing(value: Any) -> bool:
    if value is None:
        return True
    normalized = str(value).strip().casefold()
    return normalized in {"", "empty", "unknown", "未知", "不清楚", "未提供"}


def _requests(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item) for item in source.get("sub_requests", [])
        if isinstance(item, Mapping) and str(item.get("status") or "open") != "resolved"
    ]


def _is_done(action: Mapping[str, Any], source: Mapping[str, Any], request: Mapping[str, Any] | None) -> bool:
    done = action["done_when"]
    kind = done.get("kind")
    if kind == "slot_filled":
        return not _slot_is_missing(_slot_value(source, str(done.get("field_id") or "")))
    if kind == "subrequest_resolved":
        return not any(
            item.get("intent_id") == done.get("intent")
            and str(item.get("status") or "open") != "resolved"
            for item in source.get("sub_requests", []) if isinstance(item, Mapping)
        )
    if kind == "ticket_submitted":
        ticket = source.get("ticket") if isinstance(source.get("ticket"), Mapping) else {}
        return str(ticket.get("status") or "").lower() in {"submitted", "done", "closed", "resolved"}
    if kind == "callback_contacted":
        callback = source.get("callback_task") if isinstance(source.get("callback_task"), Mapping) else {}
        return str(callback.get("status") or "") == "contacted"
    return False


def _matches(action: Mapping[str, Any], source: Mapping[str, Any], active_intents: set[str]) -> bool:
    condition = action["when"]
    kind = condition.get("kind")
    if kind == "missing_slot":
        return action["intent"] in active_intents and _slot_is_missing(_slot_value(source, str(condition.get("field_id") or "")))
    if kind == "slot_value_in":
        value = _slot_value(source, str(condition.get("field_id") or ""))
        choices = {str(item).casefold() for item in condition.get("values", [])}
        if _slot_is_missing(value):
            return "empty" in choices and action["intent"] in active_intents
        return str(value).casefold() in choices and action["intent"] in active_intents
    if kind == "subrequest_open":
        return any(item.get("intent_id") == condition.get("intent") for item in _requests(source))
    if kind == "ticket_open":
        ticket = source.get("ticket") if isinstance(source.get("ticket"), Mapping) else {}
        return action["intent"] in active_intents and str(ticket.get("status") or "").lower() in {"draft", "pending", "needs_review"}
    if kind == "callback_pending_due":
        callback = source.get("callback_task") if isinstance(source.get("callback_task"), Mapping) else {}
        return str(callback.get("status") or "") == "pending" and bool(callback.get("due_at"))
    return False


def _refs_for_action(action: Mapping[str, Any], source: Mapping[str, Any]) -> list[dict[str, Any]]:
    condition = action["when"]
    refs: list[dict[str, Any]] = []
    if condition.get("kind") in {"subrequest_open", "ticket_open"}:
        for item in _requests(source):
            if item.get("intent_id") == (condition.get("intent") or action["intent"]):
                if item.get("source_message_seq") is not None:
                    refs.append({"type": "message", "seq": int(item["source_message_seq"])})
    field_id = str(action.get("field_id") or condition.get("field_id") or "")
    slot = (source.get("slots") or {}).get(field_id) if field_id else None
    if isinstance(slot, Mapping) and slot.get("source_message_seq") is not None:
        refs.append({"type": "message", "seq": int(slot["source_message_seq"])})
    if not refs and source.get("source_message_seq") is not None:
        refs.append({"type": "message", "seq": int(source["source_message_seq"])})
    return list({(item["type"], item.get("seq"), item.get("id")): item for item in refs}.values())


def _direct_handoff_action(source: Mapping[str, Any]) -> dict[str, Any]:
    action = dict(load_handoff_generation_config()["direct_handoff_action"])
    action["source_refs"] = ([{"type": "message", "seq": int(source["source_message_seq"])}]
                              if source.get("source_message_seq") is not None else [])
    return action


def select_handoff_actions(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """根据动作条件、已完成状态和优先级确定人工待办，最多三条。"""
    library = load_handoff_action_library()
    requests = _requests(source)
    active_intents = {str(item.get("intent_id") or "") for item in requests}
    enrichment = source.get("intent_enrichment") if isinstance(source.get("intent_enrichment"), Mapping) else {}
    primary = str(enrichment.get("intent") or "")
    if not active_intents and primary in BEAUTY_INTENT_BY_ID:
        active_intents.add(primary)
    if not active_intents:
        return [_direct_handoff_action(source)]
    candidates: list[dict[str, Any]] = []
    for intent_id in active_intents:
        for definition in library.get(intent_id, []):
            if not _matches(definition, source, active_intents):
                continue
            request = next((item for item in requests if item.get("intent_id") == intent_id), None)
            if _is_done(definition, source, request):
                continue
            action = {key: value for key, value in definition.items() if key not in {"when", "done_when"}}
            action["source_refs"] = _refs_for_action(definition, source)
            candidates.append(action)
    candidates.sort(key=lambda item: (_PRIORITY[str(item["type"])], next(
        (index for index, request in enumerate(requests) if request.get("intent_id") == item.get("intent")), 999
    )))
    return candidates[:3]


def _safe_slot_items(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    slots = source.get("slots") if isinstance(source.get("slots"), Mapping) else {}
    result = []
    for slot_id, raw in slots.items():
        if str(slot_id).casefold() in _SENSITIVE_SLOT_NAMES:
            continue
        item = dict(raw) if isinstance(raw, Mapping) else {"value": raw}
        value = item.get("value")
        if value is None or isinstance(value, (dict, list, tuple, set)):
            continue
        text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号已隐藏]", str(value))[:200]
        if str(slot_id).casefold() == "order_id":
            text = "[已提供订单号]"
        else:
            text = _ORDER_ID_PATTERN.sub("[订单号已隐藏]", text)
        record = {"field": str(slot_id), "value": text}
        if item.get("confirmation_status"):
            record["confirmation_status"] = str(item["confirmation_status"])
        if item.get("confidence") is not None:
            try:
                record["confidence"] = max(0.0, min(1.0, float(item["confidence"])))
            except (TypeError, ValueError):
                pass
        if item.get("source_message_seq") is not None:
            record["source"] = {"type": "message", "seq": int(item["source_message_seq"])}
        result.append(record)
    return result


def _safe_conflicts(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_conflicts = source.get("contradictions") if isinstance(source.get("contradictions"), list) else []
    safe_conflicts = []
    for item in raw_conflicts[:5]:
        if not isinstance(item, Mapping):
            continue
        safe = {
            key: str(item[key])[:60]
            for key in ("field", "field_id") if isinstance(item.get(key), str)
        }
        raw_values = item.get("values")
        if isinstance(raw_values, (list, tuple)):
            safe["values"] = [
                _ORDER_ID_PATTERN.sub("[订单号已隐藏]", re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号已隐藏]", str(value))[:120])
                for value in raw_values[:4] if isinstance(value, (str, int, float))
            ]
        raw_sequences = item.get("source_message_seqs")
        if isinstance(raw_sequences, (list, tuple)):
            safe["source_message_seqs"] = [int(value) for value in raw_sequences[:4] if isinstance(value, int) and not isinstance(value, bool)]
        if safe:
            safe_conflicts.append(safe)
    return safe_conflicts


def _safe_transcript_text(value: Any) -> str:
    """保留对话语义，同时遮蔽不需要进入摘要模型的常见个人标识。"""
    text = str(value or "")
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号已隐藏]", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[邮箱已隐藏]", text)
    text = re.sub(r"(?<!\d)\d{17}[\dXx](?!\w)", "[证件号已隐藏]", text)
    return _ORDER_ID_PATTERN.sub("[订单号已隐藏]", text)


def _conversation_messages(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    speakers = {
        "consumer": "消费者", "user": "消费者", "消费者": "消费者",
        "ai": "AI客服", "assistant": "AI客服", "AI客服": "AI客服",
        "customer_service": "人工客服", "agent": "人工客服", "人工客服": "人工客服",
    }
    result = []
    raw_messages = source.get("conversation_messages")
    for item in raw_messages if isinstance(raw_messages, list) else []:
        if not isinstance(item, Mapping):
            continue
        try:
            seq = int(item.get("message_seq", item.get("seq")))
        except (TypeError, ValueError):
            continue
        speaker = speakers.get(str(item.get("role") or item.get("sender_type") or ""))
        content = _safe_transcript_text(item.get("content", item.get("text", "")))
        if speaker and content.strip():
            result.append({"seq": seq, "speaker": speaker, "text": content})
    return result


def _human_handoff_reason(value: Any) -> str:
    reason = str(value or "").strip()
    if reason in _HANDOFF_REASON_LABELS:
        return _HANDOFF_REASON_LABELS[reason]
    if reason and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", reason):
        return reason
    return "需要人工客服继续处理"


def _structured_summary_input(source: Mapping[str, Any]) -> dict[str, Any]:
    requests = []
    for item in _requests(source):
        status = str(item.get("status") or "open")
        record = {
            "name": _INTENT_NAMES.get(str(item.get("intent_id") or ""), str(item.get("intent_id") or "其他诉求")),
            "status": _REQUEST_STATUS_LABELS.get(status, "待处理"),
            "resolved": status == "resolved",
        }
        if item.get("source_message_seq") is not None:
            record["source"] = {"type": "message", "seq": int(item["source_message_seq"])}
        requests.append(record)
    if not requests:
        enrichment = source.get("intent_enrichment") if isinstance(source.get("intent_enrichment"), Mapping) else {}
        primary = str(enrichment.get("intent") or "")
        if primary in BEAUTY_INTENT_BY_ID:
            record = {
                "name": _INTENT_NAMES.get(primary, primary),
                "status": "待处理",
                "resolved": False,
            }
            source_seq = source.get("source_message_seq")
            if source_seq is not None:
                record["source"] = {"type": "message", "seq": int(source_seq)}
            requests.append(record)
    actions = []
    for item in source.get("action_logs", []):
        if not isinstance(item, Mapping):
            continue
        action_type = str(item.get("action_type") or "")
        if action_type not in _BUSINESS_ACTION_LABELS:
            continue
        description = _safe_transcript_text(str(item.get("description") or "")[:240])
        if action_type == "rule" and (
            description.startswith("意图识别完成：") or description == "识别到转人工控制信号"
        ):
            continue
        # Tool names and snake_case slot/intent identifiers are backend vocabulary,
        # not customer-facing evidence. The visible dialogue carries the actual wording.
        description = re.sub(r"(?<=工具 )[A-Za-z][A-Za-z0-9_.-]*", "业务查询", description)
        description = re.sub(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b", "业务信息", description)
        action_status = str(item.get("status") or "pending")
        record = {
            "action_id": str(item.get("action_id") or ""),
            "kind": _BUSINESS_ACTION_LABELS[action_type],
            "description": description,
            "status": _ACTION_STATUS_LABELS.get(action_status, "未完成"),
            "completed": action_status == "success",
        }
        if item.get("source_message_seq") is not None:
            record["source"] = {"type": "message", "seq": int(item["source_message_seq"])}
        else:
            record["source"] = {"type": "action", "id": record["action_id"]}
        actions.append(record)
    return {
        "requests": requests,
        "actions": actions,
        "conversation": _conversation_messages(source),
        "slots": _safe_slot_items(source),
        "handoff_reason": _human_handoff_reason(source.get("handoff_reason")),
        "source_message_seq": source.get("source_message_seq"),
    }


def _allowed_source_keys(facts: Mapping[str, Any]) -> set[tuple[str, Any]]:
    allowed: set[tuple[str, Any]] = set()
    for item in facts.get("requests", []):
        source = item.get("source")
        if source:
            allowed.add((str(source.get("type")), source.get("seq", source.get("id"))))
    for item in facts.get("actions", []):
        source = item.get("source") or {}
        allowed.add((str(source.get("type")), source.get("seq", source.get("id"))))
    for item in facts.get("slots", []):
        source = item.get("source")
        if source:
            allowed.add((str(source.get("type")), source.get("seq", source.get("id"))))
    for item in facts.get("conversation", []):
        if item.get("seq") is not None:
            allowed.add(("message", int(item["seq"])))
    seq = facts.get("source_message_seq")
    if seq is not None:
        allowed.add(("message", int(seq)))
    return allowed


def _normalise_sources(value: Any, allowed: set[tuple[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("type") or "")
        ref = item.get("seq") if kind == "message" else item.get("id")
        if (kind, ref) not in allowed:
            continue
        result.append({"type": kind, "seq": int(ref)} if kind == "message" else {"type": kind, "id": str(ref)})
    return list({json.dumps(item, sort_keys=True): item for item in result}.values())


def _template_summary(facts: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    requests = list(facts.get("requests") or [])
    actions = list(facts.get("actions") or [])
    conversation = list(facts.get("conversation") or [])
    seq = facts.get("source_message_seq")
    request_text = "、".join(dict.fromkeys(str(item.get("name") or "其他诉求") for item in requests))
    if not request_text:
        consumer_messages = [item for item in conversation if item.get("speaker") == "消费者"]
        assistant_messages = [item for item in conversation if item.get("speaker") in {"AI客服", "人工客服"}]
        sources = [{"type": "message", "seq": int(item["seq"])} for item in consumer_messages[-1:]]
        sources.extend({"type": "message", "seq": int(item["seq"])} for item in assistant_messages[-1:])
        if not sources and seq is not None:
            sources = [{"type": "message", "seq": int(seq)}]
        text = str(facts.get("handoff_reason") or "需要人工客服继续处理")
        if assistant_messages:
            text += f"；客服已回复“{str(assistant_messages[-1].get('text') or '')[:160]}”"
        else:
            text += f"；{str(config['direct_handoff_summary'])}"
        if "没有其他" not in text and "暂无其他" not in text:
            text += "；目前没有其他已识别的业务诉求"
        statement = {
            "text": text + "。",
            "sources": sources,
        }
        return {"text": statement["text"], "statements": [statement], "template_version": config["template_version"]}
    statements: list[dict[str, Any]] = []
    request_sources = [item["source"] for item in requests if item.get("source")]
    if not request_sources and seq is not None:
        request_sources = [{"type": "message", "seq": int(seq)}]
    statements.append({"text": str(config["summary_requests_template"]).format(requests=request_text), "sources": request_sources})
    allowed_slot_ids = set(config.get("slot_labels") or {})
    summary_slots = [item for item in facts.get("slots", []) if item.get("field") in allowed_slot_ids][:5]
    if summary_slots:
        details = "；".join(
            f"{config['slot_labels'][item['field']]}：{item['value']}" for item in summary_slots
        )
        slot_sources = [item["source"] for item in summary_slots if item.get("source")]
        if slot_sources:
            statements.append({
                "text": str(config["summary_slots_template"]).format(slots=details),
                "sources": slot_sources,
            })
    assistant_messages = [item for item in conversation if item.get("speaker") in {"AI客服", "人工客服"}]
    if assistant_messages:
        replies = list(dict.fromkeys(str(item.get("text") or "") for item in assistant_messages[-2:] if item.get("text")))
        if replies:
            statements.append({
                "text": "客服实际回复：" + "；".join(replies),
                "sources": [{"type": "message", "seq": int(item["seq"])} for item in assistant_messages[-2:] if item.get("text")],
            })
    completed = [item for item in actions if item.get("completed")]
    if completed:
        descriptions = "；".join(dict.fromkeys(str(item.get("description") or "") for item in completed if item.get("description")))
        statements.append({
            "text": str(config["summary_actions_template"]).format(actions=descriptions),
            "sources": [{"type": "action", "id": str(item["action_id"])} for item in completed if item.get("action_id")],
        })
    else:
        statements.append({
            "text": str(config["summary_no_completed_actions"]),
            "sources": request_sources or ([{"type": "message", "seq": int(seq)}] if seq is not None else []),
        })
    unresolved = [item for item in requests if not item.get("resolved")]
    unresolved_names = "、".join(dict.fromkeys(str(item.get("name") or "其他诉求") for item in unresolved))
    source_refs = [item["source"] for item in unresolved if item.get("source")]
    if not source_refs and seq is not None:
        source_refs = [{"type": "message", "seq": int(seq)}]
    reason = str(facts.get("handoff_reason") or "需要人工继续核对")
    suffix = (
        str(config["summary_unresolved_template"]).format(requests=unresolved_names, reason=reason)
        if unresolved_names else str(config["summary_reason_template"]).format(reason=reason)
    )
    statements.append({"text": suffix, "sources": source_refs})
    statements = [item for item in statements if item["sources"]]
    return {
        "text": "".join(str(item["text"]) for item in statements),
        "statements": statements,
        "template_version": config["template_version"],
    }


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(str(getattr(item, "text", "")) for item in content if getattr(item, "type", "text") == "text").strip()
    return ""


async def _call_model(client: Any, *, model: str, max_tokens: int, temperature: float, timeout_seconds: float, system: str, prompt: str) -> str:
    result = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"thinking": {"type": "disabled"}},
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    response = await asyncio.wait_for(result, timeout=timeout_seconds) if inspect.isawaitable(result) else result
    return _extract_text(response)


def _parse_json(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```json"):
        value = value[7:]
    elif value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    parsed = json.loads(value.strip())
    return parsed if isinstance(parsed, dict) else {}


def _summary_from_model(text: str, facts: Mapping[str, Any]) -> dict[str, Any]:
    parsed = _parse_json(text)
    allowed = _allowed_source_keys(facts)
    statements = []
    for item in parsed.get("statements", [])[:3]:
        if not isinstance(item, Mapping):
            continue
        statement = str(item.get("text") or "").strip()[:500]
        sources = _normalise_sources(item.get("sources"), allowed)
        if statement and sources:
            statements.append({"text": statement, "sources": sources})
    if not statements:
        raise ValueError("叙述摘要缺少有效来源引用")
    cited = {
        (str(source.get("type")), source.get("seq", source.get("id")))
        for statement in statements for source in statement["sources"]
    }
    consumer_seqs = {int(item["seq"]) for item in facts.get("conversation", []) if item.get("speaker") == "消费者"}
    assistant_seqs = {int(item["seq"]) for item in facts.get("conversation", []) if item.get("speaker") in {"AI客服", "人工客服"}}
    if consumer_seqs and not any(("message", seq) in cited for seq in consumer_seqs):
        raise ValueError("叙述摘要没有引用消费者原始诉求")
    completed_action_ids = {str(item.get("action_id")) for item in facts.get("actions", []) if item.get("completed") and item.get("action_id")}
    if assistant_seqs and not (
        any(("message", seq) in cited for seq in assistant_seqs)
        or any(("action", action_id) in cited for action_id in completed_action_ids)
    ):
        raise ValueError("叙述摘要没有引用客服实际回复或已完成业务动作")
    return {"text": "".join(item["text"] for item in statements), "statements": statements}


def _generation_hash(source: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    facts = _structured_summary_input(source)
    actions = select_handoff_actions(source)
    raw_slots = source.get("slots") if isinstance(source.get("slots"), Mapping) else {}
    slot_state = []
    for field_id, raw in raw_slots.items():
        item = dict(raw) if isinstance(raw, Mapping) else {"value": raw}
        slot_state.append({
            "field_id": str(field_id),
            "value": item.get("value"),
            "status": str(item.get("confirmation_status") or item.get("status") or "unknown"),
            "source_type": str(item.get("source_type") or ""),
            "source_ref": str(item.get("source_ref") or item.get("source_message_id") or ""),
            "source_message_seq": item.get("source_message_seq"),
            "confidence": item.get("confidence"),
        })
    slot_state.sort(key=lambda item: item["field_id"])
    material = {
        "facts": facts,
        "conversation_id": str(source.get("conv_id") or ""),
        "sub_request_state": [
            {
                "intent_id": str(item.get("intent_id") or ""),
                "status": str(item.get("status") or "open"),
                "source_message_seq": item.get("source_message_seq"),
            }
            for item in source.get("sub_requests", []) if isinstance(item, Mapping)
        ],
        "slot_state": slot_state,
        "selected_actions": actions,
        "contradictions": _safe_conflicts(source),
        "ticket": source.get("ticket") or {},
        "ticket_version": (source.get("ticket") or {}).get("version", 0),
        "callback_task": source.get("callback_task") or {},
        "emotion_urgency": source.get("emotion_urgency") or {},
        "urgency": source.get("urgency") or {},
        "action_log_version": int(source.get("action_log_version", 0) or 0),
        "action_config_version": load_handoff_action_version(),
        "emotion_rule_version": load_handoff_emotion_config().get("version"),
        "urgency_rule_version": load_handoff_urgency_config().get("version"),
        "prompt_version": config.get("version"),
        "template_version": config.get("template_version"),
        "source_message_seq": source.get("source_message_seq"),
    }
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def _generate_summary(store: Any, source: Mapping[str, Any], client: Any, config: Mapping[str, Any], state_hash: str) -> dict[str, Any]:
    existing = (source.get("content_blocks") or {}).get("summary")
    if (
        isinstance(existing, Mapping)
        and existing.get("status") in {"ready", "degraded"}
        and str(existing.get("state_hash") or "") == state_hash
    ):
        return {"status": "cached", "latency_ms": float(existing.get("latency_ms") or 0.0)}
    started = time.perf_counter()
    facts = _structured_summary_input(source)
    status = "ready"
    error_code = ""
    payload: dict[str, Any]
    try:
        if client is None:
            raise RuntimeError("llm_unavailable")
        raw = await _call_model(
            client,
            model=str(config.get("model") or "claude-3-5-haiku-latest"),
            max_tokens=int(config["summary_max_tokens"]),
            temperature=float(config["temperature"]),
            timeout_seconds=float(config["llm_timeout_seconds"]),
            system=str(config["summary_system_prompt"]),
            prompt=json.dumps(facts, ensure_ascii=False, separators=(",", ":")),
        )
        payload = _summary_from_model(raw, facts)
    except Exception as exc:
        status = "degraded"
        error_code = type(exc).__name__
        payload = _template_summary(facts, config)
    payload.update({"degraded": status == "degraded", "source_message_seq": source.get("source_message_seq")})
    latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
    saved = await asyncio.to_thread(
        store.save_handoff_generated_block,
        str(source["handoff_id"]),
        block_type="summary", status=status, state_hash=state_hash, payload=payload,
        error_code=error_code, latency_ms=latency_ms,
        source_message_seq=source.get("source_message_seq"),
    )
    return {"status": str(saved.get("status") or status), "latency_ms": latency_ms}


def _fallback_reply(actions: list[Mapping[str, Any]], config: Mapping[str, Any]) -> str:
    candidates = list(dict.fromkeys(str(item.get("reply") or "").strip() for item in actions if item.get("reply")))
    if not candidates:
        return str(config["empty_actions_reply"])
    return "".join(candidates[:3])


async def _optional_supplement(client: Any, source: Mapping[str, Any], actions: list[dict[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    contradictions = _safe_conflicts(source)
    requests = _requests(source)
    enrichment = source.get("intent_enrichment") if isinstance(source.get("intent_enrichment"), Mapping) else {}
    primary = str(enrichment.get("intent") or "")
    active_intents = {str(item.get("intent_id") or "") for item in requests}
    if not active_intents and primary:
        active_intents.add(primary)
    action_library = load_handoff_action_library()
    uncovered = []
    for intent_id in active_intents:
        definitions = action_library.get(intent_id, [])
        if intent_id not in BEAUTY_INTENT_BY_ID or not any(
            _matches(definition, source, active_intents) for definition in definitions
        ):
            uncovered.append(intent_id)
    if client is None or (not contradictions and not uncovered):
        return actions
    facts = _structured_summary_input(source)
    allowed = _allowed_source_keys(facts)
    slot_labels = config.get("slot_labels") if isinstance(config.get("slot_labels"), Mapping) else {}
    readable_conflicts = [
        {
            "field": str(slot_labels.get(str(item.get("field_id") or "")) or "待确认信息"),
            "values": item.get("values") or [],
            "source_message_seqs": item.get("source_message_seqs") or [],
        }
        for item in contradictions
    ]
    readable_intents = [
        str(BEAUTY_INTENT_BY_ID.get(intent_id, {}).get("name_zh") or "其他业务诉求")
        for intent_id in uncovered
    ]
    prompt = json.dumps({
        "facts": facts,
        "conflicts": readable_conflicts,
        "uncovered_intents": readable_intents,
        "current_actions": [
            {key: item.get(key) for key in ("text", "reason", "source_refs")}
            for item in actions
        ],
    }, ensure_ascii=False)
    try:
        raw = await _call_model(
            client, model=str(config.get("model") or "claude-3-5-haiku-latest"),
            max_tokens=int(config["supplement_max_tokens"]), temperature=float(config["temperature"]),
            timeout_seconds=float(config["llm_timeout_seconds"]),
            system=str(config["supplement_system_prompt"]), prompt=prompt,
        )
        action = _parse_json(raw).get("action")
        if not isinstance(action, Mapping):
            return actions
        refs = _normalise_sources(action.get("source_refs"), allowed)
        text = str(action.get("text") or "").strip()[:200]
        reason = str(action.get("reason") or "").strip()[:240]
        action_type = str(action.get("type") or "ticket")
        if not text or not reason or not refs or action_type not in _ACTION_TYPES:
            return actions
        candidate = {
            "id": "model.supplement", "intent": "supplement", "type": action_type,
            "field_id": str(action.get("field_id") or ""), "text": text, "reason": reason,
            "ui_anchor": "case.sub_requests", "reply": "", "source_refs": refs,
        }
        return sorted([*actions, candidate], key=lambda item: _PRIORITY[str(item["type"])])[:3]
    except Exception as exc:
        logger.warning("交接动作补充失败，保留规则动作 error_type=%s", type(exc).__name__)
        return actions


async def _generate_next_steps(store: Any, source: Mapping[str, Any], client: Any, config: Mapping[str, Any], state_hash: str) -> dict[str, Any]:
    existing = (source.get("content_blocks") or {}).get("next_steps")
    if (
        isinstance(existing, Mapping)
        and existing.get("status") in {"ready", "degraded"}
        and str(existing.get("state_hash") or "") == state_hash
    ):
        return {"status": "cached", "latency_ms": float(existing.get("latency_ms") or 0.0)}
    started = time.perf_counter()
    actions = select_handoff_actions(source)
    status = "ready"
    error_code = ""
    try:
        actions = await _optional_supplement(client, source, actions, config)
    except Exception as exc:  # pragma: no cover - supplement is independently optional
        logger.warning("交接动作补充不可用 error_type=%s", type(exc).__name__)
    fallback = _fallback_reply(actions, config)
    reply = fallback
    if client is None:
        status = "degraded"
        error_code = "llm_unavailable"
    else:
        try:
            allowed_facts = {
                "verified_slots": _safe_slot_items(source),
                "actions": [
                    {key: item.get(key) for key in ("text", "reason", "source_refs")}
                    for item in actions
                ],
            }
            raw = await _call_model(
                client,
                model=str(config.get("model") or "claude-3-5-haiku-latest"),
                max_tokens=int(config["reply_max_tokens"]),
                temperature=float(config["temperature"]),
                timeout_seconds=float(config["llm_timeout_seconds"]),
                system=str(config["reply_system_prompt"]),
                prompt=json.dumps(allowed_facts, ensure_ascii=False, separators=(",", ":")),
            )
            guarded, rejected = guard_handoff_suggested_reply(raw, fallback)
            if rejected:
                raise ValueError("response_guard_rejected_suggested_reply")
            reply = guarded
        except Exception as exc:
            status = "degraded"
            error_code = type(exc).__name__
            reply = fallback
    reply, rejected = guard_handoff_suggested_reply(reply, str(config["reply_guard_fallback"]))
    if rejected:
        status = "degraded"
        error_code = error_code or "response_guard_rejected_fallback"
    payload = {
        "items": actions,
        "suggested_reply": reply,
        "degraded": status == "degraded",
        "source_message_seq": source.get("source_message_seq"),
        "action_library_version": load_handoff_action_version(),
        "prompt_version": config.get("version"),
    }
    latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
    saved = await asyncio.to_thread(
        store.save_handoff_generated_block,
        str(source["handoff_id"]),
        block_type="next_steps", status=status, state_hash=state_hash, payload=payload,
        error_code=error_code, latency_ms=latency_ms,
        source_message_seq=source.get("source_message_seq"),
    )
    return {"status": str(saved.get("status") or status), "latency_ms": latency_ms}


async def generate_handoff_blocks(
    store: Any,
    conv_id: str,
    handoff_id: str,
    *,
    client: Any = None,
) -> dict[str, Any]:
    """独立生成摘要与人工下一步，任一块降级不阻塞另一块。"""
    source = await asyncio.to_thread(store.get_handoff_generation_input, handoff_id)
    if not source or str(source.get("conv_id")) != str(conv_id):
        return {"status": "stale", "handoff_id": handoff_id}
    config = load_handoff_generation_config()
    state_hash = _generation_hash(source, config)

    async def generate_independently(block_type: str, operation: Any) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return await operation()
        except Exception as exc:
            latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
            logger.warning(
                "交接内容块生成失败 handoff_id=%s block_type=%s error_type=%s",
                handoff_id, block_type, type(exc).__name__,
            )
            try:
                await asyncio.to_thread(
                    store.save_handoff_generated_block,
                    handoff_id,
                    block_type=block_type,
                    status="failed",
                    state_hash=state_hash,
                    payload={},
                    error_code=type(exc).__name__,
                    latency_ms=latency_ms,
                    source_message_seq=source.get("source_message_seq"),
                )
            except Exception as persist_exc:
                logger.warning(
                    "交接内容块失败状态无法落库 handoff_id=%s block_type=%s error_type=%s",
                    handoff_id, block_type, type(persist_exc).__name__,
                )
            return {"status": "failed", "latency_ms": latency_ms}

    summary, next_steps = await asyncio.gather(
        generate_independently(
            "summary", lambda: _generate_summary(store, source, client, config, state_hash)
        ),
        generate_independently(
            "next_steps", lambda: _generate_next_steps(store, source, client, config, state_hash)
        ),
    )
    statuses = {summary["status"], next_steps["status"]}
    return {
        "status": "cached" if statuses == {"cached"} else "completed" if statuses <= {"ready", "degraded", "cached"} else "partial",
        "handoff_id": handoff_id,
        "summary_status": summary["status"],
        "summary_latency_ms": summary.get("latency_ms"),
        "next_steps_status": next_steps["status"],
        "next_steps_latency_ms": next_steps.get("latency_ms"),
    }
