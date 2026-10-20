"""Deterministic urgency scoring used for the synchronous handoff snapshot."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


def is_explicit_handoff_requested(message: str, config: Mapping[str, Any]) -> bool:
    """Check explicit transfer requests against the shared handoff policy.

    Args:
        message: Latest consumer message.
        config: Validated ``handoff_urgency.json`` configuration.

    Returns:
        ``True`` only when a configured transfer phrase is not locally negated.
    """
    text = str(message or "").casefold()
    policy = config.get("handoff_policy") or {}
    phrases = policy.get("explicit_request_phrases") or (
        "转人工", "人工客服", "我要找人工", "找人工", "请人工处理", "联系人工",
        "人工处理", "人工介入", "升级人工", "请升级处理", "supervisor",
    )
    negated_prefixes = tuple(
        str(item).strip().casefold()
        for item in policy.get("negative_request_prefixes") or (
            "不想", "不要", "不需要", "无需", "暂时不", "先不", "不用",
            "不希望", "不考虑", "不是要", "别",
        )
        if str(item).strip()
    )
    for phrase_value in phrases:
        phrase = str(phrase_value or "").strip().casefold()
        if not phrase:
            continue
        for match in re.finditer(re.escape(phrase), text):
            prefix = text[max(0, match.start() - 12):match.start()].rstrip()
            negative_pattern = "(?:" + "|".join(map(re.escape, negated_prefixes)) + ")"
            if re.search(negative_pattern + r"(?:再|去|找|转|联系|要|想)*$", prefix):
                continue
            return True
    return False


def score_temporary_urgency(
    message: str,
    config: Mapping[str, Any],
    intent: str | None = None,
    intents: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Score only clear signals in the latest message; the transfer phrase adds zero."""
    text = str(message or "")
    lowered = text.lower()
    handoff_policy = config.get("handoff_policy") or {}
    ordered_intents = list(dict.fromkeys(
        value for value in [str(intent or ""), *(str(item) for item in intents or [])]
        if value
    ))
    active_intents = set(ordered_intents)
    automatic_intent_handoff = bool(
        active_intents.intersection(handoff_policy.get("automatic_handoff_intents") or [])
    )
    automatic_intent_action = str(
        handoff_policy.get("automatic_handoff_intent_action")
        or "该业务诉求需要人工核对"
    )
    resolved_history_end = max(
        (match.end() for pattern in config.get("historical_resolved_patterns", [])
         for match in re.finditer(pattern, lowered)),
        default=-1,
    )

    def is_negated(start: int) -> bool:
        prefix = lowered[max(0, start - 12):start]
        for negative in config.get("negative_prefixes", []):
            position = prefix.rfind(str(negative))
            if position < 0:
                continue
            bridge = prefix[position + len(str(negative)):]
            if len(bridge) <= 6 and not any(mark in bridge for mark in ("但", "不过", "然而", "可是", "；", ";", "。", "，", ",")):
                return True
        return False

    hard_hits: list[tuple[int, int, str, str, str]] = []
    for signal in config.get("hard_signals", []):
        for pattern in signal.get("patterns", []):
            for match in re.finditer(re.escape(str(pattern)), text, re.IGNORECASE):
                if match.start() <= resolved_history_end or is_negated(match.start()):
                    continue
                hard_hits.append((
                    match.start(), match.end(), str(signal.get("id") or "risk.unknown"),
                    str(signal.get("reason") or "命中安全规则"), str(match.group(0)),
                ))
    if hard_hits:
        # Prefer the most specific phrase when configured phrases overlap, so
        # “没有脸红肿” is not reclassified by its shorter “红肿” suffix.
        hard_hits.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        start, end, rule_id, reason, evidence = hard_hits[0]
        return {
            "priority": str(config.get("hard_priority", "P0")),
            "score": int(config.get("hard_score", 999)),
            "reasons": [{"rule_id": rule_id, "text": reason, "evidence": evidence, "start": start, "end": end}],
            "rule_version": str(config.get("version") or ""),
            "action": automatic_intent_action if automatic_intent_handoff and str(
                config.get("hard_priority", "P0")
            ) not in set(handoff_policy.get("automatic_handoff_priorities") or ["P0"]) else str(
                (handoff_policy.get("priority_actions") or {}).get(
                    str(config.get("hard_priority", "P0")), "先提供安全处理并升级人工处理"
                )
            ),
            "automatic_handoff": automatic_intent_handoff or str(
                config.get("hard_priority", "P0")
            ) in set(handoff_policy.get("automatic_handoff_priorities") or ["P0"]),
        }

    score = 0
    reasons: list[dict[str, str]] = []
    for signal in config.get("scored_signals", []):
        patterns = [str(item) for item in signal.get("patterns", [])]
        matches = [
            (match.start(), match.group(0))
            for pattern in patterns
            for match in re.finditer(re.escape(pattern), text, re.IGNORECASE)
            if match.start() > resolved_history_end and not is_negated(match.start())
        ]
        if not matches:
            continue
        _, matched = min(matches, key=lambda item: item[0])
        weight = int(signal.get("score", 0))
        score += weight
        reasons.append({
            "rule_id": str(signal.get("id") or "risk.unknown"),
            "text": str(signal.get("reason") or "命中紧急度规则"),
            "evidence": matched,
        })
    # Why：不良反应与消费者投诉即使表述简短，仍需至少进入相应风险优先级；
    # 转人工控制信号不在该表内，避免“要求人工”本身抬高风险分。
    floors = handoff_policy.get("intent_priority_floor") or {}
    floor_candidates = [
        (active_intent, str(floors.get(active_intent) or ""))
        for active_intent in ordered_intents
        if str(floors.get(active_intent) or "") in {"P0", "P1", "P2"}
    ]
    floor_rank = {"P0": 0, "P1": 1, "P2": 2}
    floor_intent, floor = min(
        floor_candidates, key=lambda item: (floor_rank[item[1]], ordered_intents.index(item[0]))
    ) if floor_candidates else ("", "")
    if floor in {"P0", "P1", "P2"}:
        floor_threshold = {
            "P0": int(config.get("hard_score", 999)),
            "P1": int((config.get("thresholds") or {}).get("p1_min_score", 3)),
            "P2": int((config.get("thresholds") or {}).get("p2_min_score", 1)),
        }[floor]
        if score < floor_threshold:
            score = floor_threshold
            reasons.append({
                "rule_id": f"intent.priority_floor.{floor_intent}",
                "text": f"{floor_intent} 业务风险最低按 {floor} 处理",
                "evidence": "",
            })
    default = config.get("default") or {}
    thresholds = config.get("thresholds") or {}
    if score >= int(thresholds.get("p1_min_score", 3)):
        priority = "P1"
    elif score >= int(thresholds.get("p2_min_score", 1)):
        priority = "P2"
    else:
        priority = str(default.get("priority", "P3"))
    if not reasons:
        reasons = [{
            "rule_id": "default.no_higher_signal",
            "text": str(default.get("reason") or "未命中更高紧急度规则"),
            "evidence": "",
        }]
    automatic_priority_handoff = priority in set(
        handoff_policy.get("automatic_handoff_priorities") or ["P0"]
    )
    return {
        "priority": priority,
        "score": score,
        "reasons": reasons,
        "rule_version": str(config.get("version") or ""),
        "action": automatic_intent_action if automatic_intent_handoff and not automatic_priority_handoff else str(
            (handoff_policy.get("priority_actions") or {}).get(priority, "继续普通业务流程")
        ),
        "automatic_handoff": automatic_intent_handoff or automatic_priority_handoff,
    }


def score_final_urgency(
    message: str,
    config: Mapping[str, Any],
    emotion_result: Mapping[str, Any],
    intent: str | None = None,
    intents: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Recompute P0-P3 from message rules and the async emotion band, without double counting."""
    emotion = str(emotion_result.get("emotion") or "平静")
    emotion_evidence = emotion_result.get("evidence") if isinstance(emotion_result.get("evidence"), Mapping) else {}
    if emotion == "威胁升级":
        quote = str(emotion_evidence.get("threat") or "")
        if quote and quote in str(message or ""):
            threat_rule = config.get("async_emotion_threat") or {}
            return {
                "priority": str(config.get("hard_priority", "P0")),
                "score": int(config.get("hard_score", 999)),
                "reasons": [{
                    "rule_id": str(threat_rule.get("id") or "emotion.threat_escalation"),
                    "text": str(threat_rule.get("reason") or "异步情绪命中威胁升级规则"),
                    "evidence": quote,
                }],
                "rule_version": str(config.get("version") or ""),
                "action": str(((config.get("handoff_policy") or {}).get("priority_actions") or {}).get(str(config.get("hard_priority", "P0")), "先提供安全处理并升级人工处理")),
                "automatic_handoff": str(config.get("hard_priority", "P0")) in set(
                    (config.get("handoff_policy") or {}).get("automatic_handoff_priorities") or ["P0"]
                ),
            }

    result = score_temporary_urgency(
        message, config, intent=intent, intents=intents
    )
    if result["priority"] == str(config.get("hard_priority", "P0")):
        return result
    score = int(result["score"])
    reasons = [dict(item) for item in result["reasons"]]
    for signal in config.get("scored_signals", []):
        band = str(signal.get("emotion") or "")
        question = str(signal.get("emotion_question") or "")
        if not band or emotion != band:
            continue
        if any(item.get("rule_id") == signal.get("id") for item in reasons):
            continue
        quote = str(emotion_evidence.get(question) or "")
        if not quote or quote not in str(message or ""):
            continue
        score += int(signal.get("score", 0))
        reasons.append({
            "rule_id": str(signal.get("id") or "emotion.unknown"),
            "text": str(signal.get("reason") or "异步情绪判断命中紧急度规则"),
            "evidence": quote,
        })
    thresholds = config.get("thresholds") or {}
    if score >= int(thresholds.get("p1_min_score", 3)):
        priority = "P1"
    elif score >= int(thresholds.get("p2_min_score", 1)):
        priority = "P2"
    else:
        priority = str((config.get("default") or {}).get("priority", "P3"))
    return {
        "priority": priority,
        "score": score,
        "reasons": reasons,
        "rule_version": str(config.get("version") or ""),
        "action": str(((config.get("handoff_policy") or {}).get("priority_actions") or {}).get(priority, "继续普通业务流程")),
        "automatic_handoff": bool(result["automatic_handoff"])
        or priority in set(
            (config.get("handoff_policy") or {}).get("automatic_handoff_priorities") or ["P0"]
        ),
    }
