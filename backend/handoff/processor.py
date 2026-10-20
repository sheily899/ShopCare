"""转人工后的异步事实补齐；不运行领域 Agent，也不执行业务写工具。"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import time
from typing import Any, Callable, Mapping

from agents.case_facts import CaseFacts
from agents.handoff import build_handoff_payload
from handoff.config import load_handoff_emotion_config, load_handoff_urgency_config
from handoff.emotion import analyze_handoff_emotion
from handoff.generation import generate_handoff_blocks
from handoff.urgency import score_final_urgency
from intent.beauty_intent_ticket_registry import BEAUTY_INTENT_TICKET_BY_ID
from intent.intent_recognizer import recognize_intent


logger = logging.getLogger(__name__)


def _detect_exclusive_slot_conflicts(entities: Mapping[str, Any], source_message_seq: int) -> list[dict[str, Any]]:
    """Record only explicit same-message opposing values for mutually exclusive fields."""
    polarities = {
        "medical_visit": {
            "positive": {"已就医", "看过医生"},
            "negative": {"没就医", "未就医"},
        },
        "stopped_use": {
            "positive": {"已停用", "停止使用"},
            "negative": {"还在用", "继续使用"},
        },
        "opened_or_used": {
            "positive": {"已拆封未使用", "已使用", "拆封", "用过"},
            "negative": {"未拆封"},
        },
    }
    conflicts = []
    for field_id, groups in polarities.items():
        raw = entities.get(field_id, [])
        values = raw if isinstance(raw, (list, tuple, set)) else [raw]
        normalized = list(dict.fromkeys(str(value).strip() for value in values if value is not None and str(value).strip()))
        positive = any(value in groups["positive"] for value in normalized)
        negative = any(value in groups["negative"] for value in normalized)
        if positive and negative:
            conflicts.append({
                "field_id": field_id,
                "values": normalized[:4],
                "source_message_seqs": [int(source_message_seq)],
            })
    return conflicts


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _recent_context(history: list[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"{'消费者' if item.get('role') == 'user' else '客服'}：{str(item.get('content') or '')}"
        for item in history[-6:]
        if item.get("content")
    )


def _safe_intent_diagnostics(value: Any) -> dict[str, Any]:
    """Persist only bounded, non-conversational recognizer metadata, never queries or messages."""
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in ("fast_path", "fast_path_reason", "knowledge_required", "final_intent"):
        item = value.get(key)
        if isinstance(item, (bool, int, float, str)):
            result[key] = item
    for key in ("source_intents", "source_scores"):
        item = value.get(key)
        if isinstance(item, Mapping):
            safe_values = {
                str(source)[:40]: scalar
                for source, scalar in item.items()
                if isinstance(scalar, (str, int, float, bool))
            }
            if safe_values:
                result[key] = safe_values
    return result


def _emotion_state_hash(
    handoff_id: str,
    source_message_seq: int,
    message: str,
    emotion_config: Mapping[str, Any],
    urgency_config: Mapping[str, Any],
) -> str:
    state = {
        "handoff_id": handoff_id,
        "source_message_seq": int(source_message_seq),
        "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        "emotion_rules_version": str(emotion_config.get("version") or ""),
        "emotion_model": str(emotion_config.get("model") or ""),
        "urgency_rules_version": str(urgency_config.get("version") or ""),
    }
    canonical = json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def _call_analyzer(analyzer: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    result = analyzer(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


async def enrich_handoff(
    store: Any,
    conv_id: str,
    handoff_id: str,
    *,
    emotion_client: Any = None,
    intent_analyzer: Callable[..., Any] = recognize_intent,
    emotion_analyzer: Callable[..., Any] = analyze_handoff_emotion,
) -> dict[str, Any]:
    """并行补齐最后一条消息的意图/槽位与情绪；两个结果独立持久化。"""
    source = await asyncio.to_thread(store.get_handoff_enrichment_input, handoff_id)
    if not source or str(source.get("conv_id")) != str(conv_id):
        return {"status": "stale", "handoff_id": handoff_id}
    source_seq = int(source["source_message_seq"])
    message = str(source.get("message") or "")
    history = list(source.get("history") or [])
    recent = _recent_context(history)
    emotion_config = load_handoff_emotion_config()
    urgency_config = load_handoff_urgency_config()
    saved_intent_status = str((source.get("intent_enrichment") or {}).get("status") or "pending")
    need_intent = saved_intent_status == "pending"
    need_emotion = str(source.get("emotion_block_status") or "pending") == "pending"
    need_content = (
        str(source.get("summary_block_status") or "pending") == "pending"
        or str(source.get("next_steps_block_status") or "pending") == "pending"
    )
    if not need_intent and not need_emotion and not need_content:
        return {
            "status": "cached",
            "handoff_id": handoff_id,
            "intent_status": saved_intent_status,
            "emotion_status": str(source.get("emotion_block_status") or "pending"),
        }

    async def enrich_intent() -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = await _call_analyzer(intent_analyzer, message, history=history)
            intent_id = str(_value(getattr(result, "intent", "other")) or "other")
            secondary = [str(_value(item)) for item in (getattr(result, "secondary_intents", []) or [])]
            entities = getattr(result, "entities", {}) or {}
            sub_requests = getattr(result, "sub_requests", []) or []
            diagnostics = _safe_intent_diagnostics(getattr(result, "intent_diagnostics", {}) or {})
            persisted = await asyncio.to_thread(
                store.save_handoff_intent_enrichment,
                handoff_id,
                source_message_seq=source_seq,
                source_message_id=str(source["source_message_id"]),
                intent_id=intent_id,
                confidence=float(getattr(result, "confidence", 0.0) or 0.0),
                intent_group=str(getattr(result, "intent_group", "other") or "other"),
                secondary_intents=secondary,
                entities=dict(entities) if isinstance(entities, Mapping) else {},
                sub_requests=[dict(item) for item in sub_requests if isinstance(item, Mapping)],
                contradictions=_detect_exclusive_slot_conflicts(
                    dict(entities) if isinstance(entities, Mapping) else {}, source_seq
                ),
                latency_ms=max(0.0, (time.perf_counter() - started) * 1000),
                diagnostics=diagnostics,
            )
            return {"status": str(persisted.get("status") or "ready"), "latency_ms": (time.perf_counter() - started) * 1000}
        except Exception as exc:
            logger.warning("转人工意图字段补齐失败 handoff_id=%s error_type=%s", handoff_id, type(exc).__name__)
            persisted = await asyncio.to_thread(
                store.save_handoff_intent_failure,
                handoff_id,
                source_message_seq=source_seq,
                error_code=type(exc).__name__,
            )
            return {"status": str(persisted.get("status") or "failed"), "error_code": type(exc).__name__}

    async def enrich_emotion() -> dict[str, Any]:
        started = time.perf_counter()
        try:
            try:
                result = await _call_analyzer(
                    emotion_analyzer,
                    emotion_client,
                    message,
                    recent,
                    emotion_config,
                )
            except Exception as exc:
                logger.warning("转人工情绪判断异常，使用规则降级 handoff_id=%s error_type=%s", handoff_id, type(exc).__name__)
                result = await analyze_handoff_emotion(None, message, recent, emotion_config)
                result["degraded"] = True
                result["degradation_reason"] = type(exc).__name__
            latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
            urgency = score_final_urgency(
                message,
                urgency_config,
                result,
                intent=str(source.get("intent") or ""),
                intents=source.get("intents") or [],
            )
            state_hash = _emotion_state_hash(
                handoff_id, source_seq, message, emotion_config, urgency_config
            )
            persisted = await asyncio.to_thread(
                store.save_handoff_emotion_enrichment,
                handoff_id,
                source_message_seq=source_seq,
                emotion_result=dict(result),
                final_urgency=urgency,
                state_hash=state_hash,
                latency_ms=latency_ms,
            )
            return {"status": str(persisted.get("status") or "ready"), "latency_ms": latency_ms}
        except Exception as exc:
            latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
            state_hash = _emotion_state_hash(
                handoff_id, source_seq, message, emotion_config, urgency_config
            )
            logger.warning("转人工情绪块失败 handoff_id=%s error_type=%s", handoff_id, type(exc).__name__)
            try:
                await asyncio.to_thread(
                    store.save_handoff_block_failure,
                    handoff_id,
                    block_type="emotion_urgency",
                    source_message_seq=source_seq,
                    state_hash=state_hash,
                    error_code=type(exc).__name__,
                    latency_ms=latency_ms,
                )
            except Exception as persist_exc:
                logger.warning("转人工情绪块失败状态无法落库 handoff_id=%s error_type=%s", handoff_id, type(persist_exc).__name__)
            return {"status": "failed", "error_code": type(exc).__name__, "latency_ms": latency_ms}

    intent_task = asyncio.create_task(enrich_intent()) if need_intent else None
    emotion_task = asyncio.create_task(enrich_emotion()) if need_emotion else None
    intent_result = await intent_task if intent_task else {"status": saved_intent_status}

    emotion_result = (
        await emotion_task if emotion_task
        else {"status": str(source.get("emotion_block_status") or "pending")}
    )
    # 内容在最后一轮意图/槽位与情绪结果落库后统一取快照，避免用临时情绪哈希生成后再重复调用模型。
    content_result = (
        await generate_handoff_blocks(store, conv_id, handoff_id, client=emotion_client)
        if need_content else {"status": "cached"}
    )
    return {
        "status": "completed" if intent_result["status"] in {"ready", "degraded"} and emotion_result["status"] in {"ready", "degraded"} and content_result.get("status") in {"completed", "cached"} else "partial",
        "handoff_id": handoff_id,
        "intent_status": intent_result["status"],
        "intent_latency_ms": intent_result.get("latency_ms"),
        "emotion_status": emotion_result["status"],
        "emotion_latency_ms": emotion_result.get("latency_ms"),
        "summary_status": content_result.get("summary_status"),
        "summary_latency_ms": content_result.get("summary_latency_ms"),
        "next_steps_status": content_result.get("next_steps_status"),
        "next_steps_latency_ms": content_result.get("next_steps_latency_ms"),
    }


def _refresh_ticket_draft(store: Any, conv_id: str, message: str, *, _attempt: int = 0) -> bool:
    """用已有结构化事实补齐本地待审核工单；不覆盖已有非空字段或已提交工单。"""
    conversation = store.get_conversation(conv_id, viewer_role="agent")
    state = dict(conversation.get("case_state") or {})
    existing = state.get("ticket_draft") if isinstance(state.get("ticket_draft"), Mapping) else {}
    if str(existing.get("status") or "draft") in {"submitted", "done", "closed"}:
        return False

    requests = [
        dict(item) for item in store.list_case_sub_requests(conv_id)
        if str(item.get("status") or "open") != "resolved"
    ]
    enrichment = state.get("handoff_intent_enrichment")
    latest_intent = str(enrichment.get("intent") or "") if isinstance(enrichment, Mapping) else ""
    intent_ids = list(dict.fromkeys(
        [str(item.get("intent_id") or "") for item in requests]
        + ([latest_intent] if latest_intent else [])
        + ([str(existing.get("intent") or "")] if existing.get("intent") else [])
    ))
    ticket_intents = [
        intent_id for intent_id in intent_ids
        if intent_id in BEAUTY_INTENT_TICKET_BY_ID
        and BEAUTY_INTENT_TICKET_BY_ID[intent_id]["needs_ticket"]
    ]
    if not ticket_intents:
        return False

    primary = str(existing.get("intent") or "")
    if primary not in ticket_intents:
        primary = ticket_intents[0]
    sub_requests = [
        {"id": str(item.get("sub_request_key") or item.get("sub_request_id") or ""),
         "intent_id": str(item.get("intent_id") or "")}
        for item in requests
    ]
    facts = CaseFacts.from_dict(state.get("case_facts"))
    built = build_handoff_payload(
        message=message,
        intent=primary,
        entities={},
        should_handoff=False,
        conv_id=conv_id,
        case_facts=facts,
        secondary_intents=[item for item in ticket_intents if item != primary],
        sub_requests=sub_requests,
    ).get("ticket_draft")
    if not isinstance(built, Mapping):
        return False
    ticket = dict(built)
    old_fields = existing.get("fields") if isinstance(existing.get("fields"), Mapping) else {}
    new_fields = dict(ticket.get("fields") or {})
    for field_id, value in old_fields.items():
        if value is not None and (not isinstance(value, str) or value.strip()):
            new_fields[field_id] = value
    ticket["fields"] = new_fields
    ticket["status"] = str(existing.get("status") or "draft")
    ticket["needs_review"] = True
    ticket["system_fields"] = dict(existing.get("system_fields") or ticket.get("system_fields") or {})
    if "version" in existing:
        ticket["version"] = existing["version"]
    contract = BEAUTY_INTENT_TICKET_BY_ID.get(str(ticket.get("intent") or ""), {})
    blocking_fields = list(ticket.get("blocking_fields") or contract.get("blocking_fields", []))
    ticket["missing_blocking_fields"] = [field for field in blocking_fields if not new_fields.get(field)]
    if ticket == existing:
        return False
    try:
        saved = store.update_case_state(conv_id, {"ticket_draft": ticket}, state.get("version"))
    except ValueError as exc:
        if "case_state version conflict" not in str(exc) or _attempt >= 2:
            raise
        # 客服可能在刷新期间编辑了工单；重读后再合并，避免旧快照覆盖人工输入。
        return _refresh_ticket_draft(store, conv_id, message, _attempt=_attempt + 1)
    source_id = str(next(
        (item.get("message_id") for item in reversed(conversation.get("messages") or [])
         if item.get("sender_type") == "consumer"),
        "",
    ))
    store.append_case_action(
        conv_id,
        action_type="ticket",
        description="根据消费者补充信息更新待审核工单",
        source_ref=source_id,
        idempotency_key=f"handoff-ticket-refresh:{conv_id}:{saved.get('version', 0)}",
        status="success",
    )
    return True


async def refresh_handoff_after_messages(
    store: Any,
    conv_id: str,
    handoff_id: str,
    *,
    emotion_client: Any = None,
    intent_analyzer: Callable[..., Any] = recognize_intent,
) -> dict[str, Any]:
    """只补处理转人工后未识别的消费者消息，再重算工单草稿和交接内容块。"""
    source = await asyncio.to_thread(store.get_handoff_refresh_input, handoff_id)
    if not source or str(source.get("conv_id")) != str(conv_id):
        return {"status": "stale", "handoff_id": handoff_id, "processed_messages": 0}

    processed = 0
    failed = 0
    last_message = ""
    for item in source.get("messages") or []:
        source_seq = int(item["message_seq"])
        source_message_id = str(item["message_id"])
        message = str(item.get("content") or "")
        last_message = message
        started = time.perf_counter()
        try:
            result = await _call_analyzer(intent_analyzer, message, history=list(item.get("history") or []))
            intent_id = str(_value(getattr(result, "intent", "other")) or "other")
            entities = getattr(result, "entities", {}) or {}
            persisted = await asyncio.to_thread(
                store.save_handoff_intent_enrichment,
                handoff_id,
                source_message_seq=source_seq,
                source_message_id=source_message_id,
                intent_id=intent_id,
                confidence=float(getattr(result, "confidence", 0.0) or 0.0),
                intent_group=str(getattr(result, "intent_group", "other") or "other"),
                secondary_intents=[str(_value(value)) for value in (getattr(result, "secondary_intents", []) or [])],
                entities=dict(entities) if isinstance(entities, Mapping) else {},
                sub_requests=[dict(value) for value in (getattr(result, "sub_requests", []) or []) if isinstance(value, Mapping)],
                contradictions=_detect_exclusive_slot_conflicts(
                    dict(entities) if isinstance(entities, Mapping) else {}, source_seq
                ),
                latency_ms=max(0.0, (time.perf_counter() - started) * 1000),
                diagnostics=_safe_intent_diagnostics(getattr(result, "intent_diagnostics", {}) or {}),
                advance_refresh_cursor=True,
            )
            if persisted.get("status") == "stale":
                return {"status": "stale", "handoff_id": handoff_id, "processed_messages": processed}
            processed += 1
        except Exception as exc:
            failed += 1
            logger.warning(
                "交接后新消息补齐失败 handoff_id=%s message_seq=%s error_type=%s",
                handoff_id, source_seq, type(exc).__name__,
            )
            await asyncio.to_thread(
                store.save_handoff_intent_failure,
                handoff_id,
                source_message_seq=source_seq,
                error_code=type(exc).__name__,
            )
            # 保留失败消息之后的游标，让下一次手动刷新可以重试。
            break

    generation_input = await asyncio.to_thread(store.get_handoff_generation_input, handoff_id)
    if not generation_input:
        return {"status": "stale", "handoff_id": handoff_id, "processed_messages": processed}
    if not last_message:
        last_message = next(
            (str(item.get("content") or "") for item in reversed(generation_input.get("conversation_messages") or [])
             if item.get("role") == "消费者"),
            "",
        )
    ticket_updated = await asyncio.to_thread(_refresh_ticket_draft, store, conv_id, last_message)
    content_result = await generate_handoff_blocks(store, conv_id, handoff_id, client=emotion_client)
    return {
        "status": "partial" if failed or content_result.get("status") == "partial" else "completed",
        "handoff_id": handoff_id,
        "processed_messages": processed,
        "failed_messages": failed,
        "ticket_updated": ticket_updated,
        "summary_status": content_result.get("summary_status"),
        "next_steps_status": content_result.get("next_steps_status"),
    }
