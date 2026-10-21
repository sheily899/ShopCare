"""转人工时生成客服可审核的摘要、工单草稿与话术参考。"""

from __future__ import annotations

from datetime import datetime, timezone
import re
import uuid
from typing import Any, Dict, Iterable, Mapping

from agents.case_facts import CaseFacts
from intent.beauty_intent_ticket_registry import BEAUTY_INTENT_TICKET_BY_ID
from intent.next_step_policy import build_next_steps, explicitly_missing_gift


def _priority(urgency: Any, intent_id: str) -> str:
    raw = getattr(urgency, "name", str(urgency or "LOW")).upper()
    if intent_id == "adverse_reaction" and raw in {"CRITICAL", "HIGH"}:
        return "P0" if raw == "CRITICAL" else "P1"
    return {"CRITICAL": "P0", "HIGH": "P1", "MEDIUM": "P2"}.get(raw, "P3")


def _compact_request(message: str, intent_ids: Iterable[str]) -> list[str]:
    text = message or ""
    labels = {
        "ingredient_fit": "成分与肤质适配咨询", "shade_match": "色号适配咨询",
        "usage_howto": "商品使用方法咨询", "product_info": "商品信息咨询",
        "promo_rules": "活动规则咨询", "pregnancy_safety": "特殊人群使用安全咨询",
        "order_status": "查询订单状态", "logistics_stalled": "查询物流进度",
        "signed_not_received": "签收未收到", "intercept_change_address": "拦截或修改地址",
        "missing_item": "商品或赠品漏发", "wrong_item": "商品错发",
        "damaged_on_arrival": "商品到货破损", "return_refund": "退货退款",
        "adverse_reaction": "商品使用不适", "suspected_abnormal_refund": "异常退款核查",
        "price_protection": "价保补差", "refund_not_received": "退款未到账",
        "shipping_fee_reimbursement": "运费补偿",
    }
    items = [labels[item] for item in intent_ids if item in labels]
    if explicitly_missing_gift(text):
        items.append("赠品/小样未收到")
    return items or ["人工复核"]


def _key_facts(message: str, values: Mapping[str, list[str]]) -> list[str]:
    text = message or ""
    facts = []
    if any(word in text for word in ("整脸发红", "全脸发红")): facts.append("整脸发红")
    elif "发红" in text or "红肿" in text: facts.append("发红/红肿")
    if "发痒" in text or "瘙痒" in text: facts.append("发痒")
    if "越来越严重" in text or "加重" in text: facts.append("症状加重")
    for value in values.get("symptom", []):
        if value and str(value) not in facts:
            facts.append(str(value))
    if values.get("stopped_use") and str(values["stopped_use"][-1]).lower() in {"true", "1", "yes", "已停用", "stopped"}:
        facts.append("已停用")
    if values.get("medical_visit") and str(values["medical_visit"][-1]).lower() in {"true", "1", "yes", "已就医", "visited"}:
        facts.append("已就医")
    return facts


def _display_facts(values: Mapping[str, list[str]]) -> list[str]:
    labels = {
        "symptom": "症状", "symptom_area": "不适部位", "onset_time": "出现时间",
        "stopped_use": "使用状态", "medical_visit": "就医情况", "order_id": "订单号",
        "product_name": "商品", "tracking_no": "物流单号", "return_reason": "退货原因",
    }
    result = []
    for key, items in values.items():
        if items and key in labels:
            value = items[-1]
            value = "已停用" if key == "stopped_use" and value in {"true", "1", "yes"} else value
            result.append(f"{labels[key]}：{value}")
    return result


def _commitments_from_response(response: str) -> list[str]:
    """只提取明确的客服承诺，不替消费者侧答复补写承诺。"""
    candidates = re.split(r"(?<=[。！？])", str(response or ""))
    markers = ("我会帮您", "我会为您", "我们会帮您", "我们会为您", "将为您", "会向您反馈")
    return list(dict.fromkeys(
        sentence.strip() for sentence in candidates
        if any(marker in sentence for marker in markers) and sentence.strip()
    ))


def emotion_band(label: str, intensity: float) -> str:
    """将情绪归一为五档客服标签；未知值才按强度降级。"""
    aliases = {
        "neutral": "calm", "平静": "calm", "calm": "calm",
        "worried": "anxious", "焦虑": "anxious", "担心": "anxious", "anxious": "anxious",
        "disappointed": "upset", "dissatisfied": "upset", "frustrated": "upset",
        "不满": "upset", "不耐烦": "upset", "upset": "upset",
        "愤怒": "angry", "anger": "angry", "angry": "angry",
        "威胁升级": "threatening", "threat": "threatening", "threatening": "threatening",
    }
    normalized = aliases.get(str(label or "").strip().lower())
    if normalized:
        return normalized
    if label in {"calm", "anxious", "upset", "angry", "threatening"}:
        return label
    if intensity >= 0.85:
        return "angry"
    if intensity >= 0.65:
        return "upset"
    if intensity >= 0.35:
        return "anxious"
    return "calm"


def build_handoff_payload(
    *,
    message: str,
    intent: Any,
    entities: Mapping[str, Iterable[str]] | None,
    emotion: str = "neutral",
    emotion_intensity: float = 0.0,
    urgency: Any = None,
    should_handoff: bool = False,
    conv_id: str | None = None,
    buyer_nickname: str | None = None,
    case_facts: CaseFacts | Mapping[str, Any] | None = None,
    secondary_intents: Iterable[Any] | None = None,
    sub_requests: Iterable[Mapping[str, Any]] | None = None,
    ai_response: str = "",
) -> Dict[str, Any]:
    """按业务意图生成待审核工单；交接摘要与客服话术仅在转人工时生成。"""

    intent_id = str(getattr(intent, "value", intent) or "other")
    intent_ids = [intent_id]
    for candidate in secondary_intents or []:
        value = str(getattr(candidate, "value", candidate) or "")
        if value:
            intent_ids.append(value)
    for item in sub_requests or []:
        value = item.get("intent") or item.get("intent_id")
        value = str(getattr(value, "value", value) or "")
        if value:
            intent_ids.append(value)
    intent_ids = list(dict.fromkeys(value for value in intent_ids if value in BEAUTY_INTENT_TICKET_BY_ID))
    if not intent_ids:
        intent_ids = [intent_id]
    ticket_contracts = [
        BEAUTY_INTENT_TICKET_BY_ID[value]
        for value in intent_ids
        if value in BEAUTY_INTENT_TICKET_BY_ID
        and BEAUTY_INTENT_TICKET_BY_ID[value]["needs_ticket"]
    ]
    if not should_handoff and not ticket_contracts:
        return {"handoff_required": False, "summary": None, "ticket_draft": None, "suggested_replies": []}
    if isinstance(case_facts, CaseFacts):
        facts = CaseFacts.from_dict(case_facts.to_dict())
    else:
        facts = CaseFacts.from_dict(case_facts if isinstance(case_facts, Mapping) else None)
    if not facts.facts:
        # 兼容直接调用 builder 的旧入口；主运行链路由 API 提供含来源的 CaseFacts。
        for field_name, raw_values in (entities or {}).items():
            for value in raw_values or []:
                facts.record(
                    str(field_name), value, source_type="message",
                    source_ref="handoff:legacy-entity", confidence=0.8,
                )
    values = facts.to_prompt_dict()
    fact_emotion = facts.values("emotion")
    fact_intensity = facts.values("emotion_intensity")
    effective_emotion = str(fact_emotion[-1]) if fact_emotion else emotion
    try:
        effective_intensity = float(fact_intensity[-1]) if fact_intensity else float(emotion_intensity or 0.0)
    except (TypeError, ValueError):
        effective_intensity = float(emotion_intensity or 0.0)
    band = emotion_band(effective_emotion, effective_intensity)
    emotion_record = facts.latest("emotion") or {}
    emotion_evidence = [
        str(item) for item in facts.values("emotion_evidence")
        if isinstance(item, str) and item.strip()
    ]
    known = _display_facts(values)
    key_facts = _key_facts(message, values)
    request_items = _compact_request(message, intent_ids)
    field_names = {
        "order_id": "订单号", "product_name": "商品", "symptom": "症状",
        "symptom_area": "不适部位", "onset_time": "症状出现时间",
        "stopped_use": "是否停用", "medical_visit": "是否就医",
        "return_reason": "退货原因", "opened_or_used": "拆封/使用状态",
    }
    blocking_fields = list(dict.fromkeys(
        field for item in ticket_contracts for field in item["blocking_fields"]
    ))
    pending = [field_names.get(field, field) for field in blocking_fields if not values.get(field)]
    priority = _priority(urgency, str(intent_id))
    next_step_results = [
        build_next_steps(
            current_intent,
            message=message,
            case_state={"slots": {key: (items[-1] if items else None) for key, items in values.items()}},
            order_result={"available": bool(values.get("order_id"))},
        )
        for current_intent in intent_ids
    ]
    next_steps = list(dict.fromkeys(
        step for result in next_step_results for step in result["steps"]
    ))
    next_step_sources = list(dict.fromkeys(
        source for result in next_step_results for source in result["sources"]
    ))
    summary = {
        "intent": str(intent_id),
        "message": message,
        "request": " · ".join(request_items),
        "request_items": request_items,
        "commitments": _commitments_from_response(ai_response),
        "emotion": band,
        "emotion_intensity": round(effective_intensity, 2),
        "emotion_source": emotion_record.get("source_type", "emotion_analysis"),
        "emotion_evidence": emotion_evidence,
        "urgency": priority,
        "known_facts": known,
        "key_facts": key_facts,
        "pending_items": pending,
        "next_steps": next_steps,
        "next_steps_sources": next_step_sources,
        "sub_intents": intent_ids,
        "handoff_reason": "风险或处理权限需要人工接管",
    } if should_handoff else None
    ticket = None
    # 建议回复面向已经接管的人工客服，必须围绕 next_steps 组织，不能再说“已为您转接人工”。
    replies = []
    for step in next_steps[:4] if should_handoff else []:
        if "呼吸不适" in step or "症状持续加重" in step:
            replies.append("想再确认一下：目前有没有呼吸不适，或症状还在持续加重？如果有，请尽快就医。")
        elif "退货" in step or "退款" in step:
            replies.append("我来帮您核对订单和退货退款条件，符合条件后再继续处理，结果会向您说明。")
        elif "赠品" in step or "小样" in step:
            replies.append("我再帮您核对直播赠品的发货记录和处理结果，并把结果反馈给您。")
        elif "风险" in step:
            replies.append("我会把当前不适情况和风险记录完整，交由人工继续核对。")
    if should_handoff and not replies and next_steps:
        replies = [f"我先帮您{next_steps[0]}，核对后把结果告诉您。"]
    if ticket_contracts:
        ticket_fields = list(dict.fromkeys(field for item in ticket_contracts for field in item["ticket_fields"]))
        fields = {field: (values.get(field) or [None])[-1] for field in ticket_fields}
        sub_tasks_by_department: dict[str, dict[str, Any]] = {}
        for current_contract in ticket_contracts:
            current_intent = current_contract["intent_id"]
            for department in current_contract["departments"]:
                task = sub_tasks_by_department.setdefault(
                    department,
                    {"department": department, "intents": [], "fields": [], "status": "待处理"},
                )
                if current_intent not in task["intents"]:
                    task["intents"].append(current_intent)
                task["fields"] = list(dict.fromkeys([*task["fields"], *current_contract["ticket_fields"]]))
        main_type = ticket_contracts[0]["ticket_type"]
        missing_fields = [field for field in blocking_fields if not fields.get(field)]
        department_tasks = {
            "质量": "核对商品质量与风险信息",
            "仓储": "核对订单明细及漏发、错发或补发情况",
            "财务": "核对退款、价保及费用处理条件",
            "物流": "核对物流节点、签收或地址处理情况",
            "售后": "核对售后申请条件并反馈处理结果",
            "风控": "复核异常退款或高风险事项",
        }
        sub_tasks = list(sub_tasks_by_department.values())
        for task in sub_tasks:
            task["task"] = department_tasks.get(task["department"], "核对并处理相关诉求")
        ticket = {
            "status": "draft",
            "needs_review": True,
            "intent": str(intent_id),
            "sub_intents": intent_ids,
            "ticket_type": main_type,
            "risk_level": "P0" if priority == "P0" else min(
                (item["risk_level"] for item in ticket_contracts),
                key=lambda value: {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(value, 4),
            ),
            "fields": fields,
            "blocking_fields": blocking_fields,
            "missing_blocking_fields": missing_fields,
            "sub_tasks": sub_tasks,
            "system_fields": {
                "ticket_id": f"T-{uuid.uuid4().hex[:8]}",
                "conv_id": conv_id or "",
                "buyer_nickname": buyer_nickname or "",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }
    return {
        "handoff_required": bool(should_handoff),
        "summary": summary,
        "ticket_draft": ticket,
        "suggested_replies": replies,
    }
