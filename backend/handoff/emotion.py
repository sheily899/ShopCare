"""交接后台的 Q1-Q4 情绪判断；模型答边界问题，类别由代码推导。"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


QUESTION_ORDER = ("threat", "anger", "dissatisfaction", "anxiety")


def classify_handoff_answers(answers: Mapping[str, bool]) -> str:
    if answers.get("threat"):
        return "威胁升级"
    if answers.get("anger"):
        return "愤怒"
    if answers.get("dissatisfaction"):
        return "不满"
    if answers.get("anxiety"):
        return "焦虑"
    return "平静"


def _is_negated(text: str, start: int, prefixes: list[str]) -> bool:
    before = text[max(0, start - 14):start]
    for prefix in prefixes:
        index = before.rfind(str(prefix))
        if index < 0:
            continue
        bridge = before[index + len(str(prefix)):]
        if len(bridge) <= 7 and not any(mark in bridge for mark in ("但", "不过", "然而", "可是", "；", ";", "。", "，", ",")):
            return True
    return False


def _fallback_emotion(message: str, config: Mapping[str, Any]) -> dict[str, Any]:
    lowered = str(message or "").lower()
    historical_end = max(
        (match.end() for pattern in config.get("historical_resolved_patterns", [])
         for match in re.finditer(str(pattern), lowered)),
        default=-1,
    )
    answers: dict[str, bool] = {}
    evidence: dict[str, str] = {}
    for key in QUESTION_ORDER:
        candidate = None
        for phrase in config.get("fallback_rules", {}).get(key, []):
            match = re.search(re.escape(str(phrase)), lowered, re.IGNORECASE)
            if not match or match.start() <= historical_end:
                continue
            if _is_negated(lowered, match.start(), list(config.get("negative_prefixes", []))):
                continue
            candidate = match.group(0)
            break
        answers[key] = candidate is not None
        if candidate is not None:
            evidence[key] = candidate
    return {
        "emotion": classify_handoff_answers(answers),
        "answers": answers,
        "evidence": evidence,
        "source": "fallback_rules",
        "degraded": True,
        "degradation_reason": "handoff_emotion_model_unavailable_or_invalid",
        "rule_version": str(config.get("version") or ""),
    }


async def analyze_handoff_emotion(
    client: Any,
    message: str,
    recent: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify only the latest message; recent turns disambiguate context."""
    if client is None:
        return _fallback_emotion(message, config)
    question_lines = "\n".join(str(item["text"]) for item in config["questions"])
    prompt = (
        f"{config['user_prompt_prefix']}\n\n{question_lines}\n"
        f"\n最近三轮对话（仅作语境参考）：\n{str(recent or '')[-1800:]}"
        f"\n\n当前待判断消息（只判断这一条）：\n{str(message or '')}"
    )
    try:
        response = await client.messages.create(
            model=str(config["model"]),
            max_tokens=int(config["max_tokens"]),
            temperature=0,
            system=str(config["system_prompt"]),
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(
            str(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", ""))
            for block in getattr(response, "content", [])
        )
        payload = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        answers: dict[str, bool] = {}
        evidence: dict[str, str] = {}
        current = str(message or "")
        for key in QUESTION_ORDER:
            item = payload.get(key)
            if not isinstance(item, dict):
                raise ValueError("emotion_question_shape")
            answer = item.get("answer")
            quote = item.get("evidence", "")
            if not isinstance(answer, bool) or not isinstance(quote, str):
                raise ValueError("emotion_answer_type")
            if quote and quote not in current:
                raise ValueError("emotion_evidence_not_in_current_message")
            if answer and not quote:
                raise ValueError("emotion_positive_answer_without_evidence")
            if not answer and quote:
                raise ValueError("emotion_negative_answer_with_evidence")
            answers[key] = answer
            if quote:
                evidence[key] = quote
        return {
            "emotion": classify_handoff_answers(answers),
            "answers": answers,
            "evidence": evidence,
            "source": "llm",
            "degraded": False,
            "rule_version": str(config.get("version") or ""),
        }
    except Exception:
        fallback = _fallback_emotion(message, config)
        fallback["degradation_reason"] = "handoff_emotion_model_unavailable_or_invalid"
        return fallback
