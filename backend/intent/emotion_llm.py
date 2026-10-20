"""异步 Q1-Q4 情绪判断：模型只回答边界问题，类别由代码决定。"""
from __future__ import annotations

import json
from typing import Any, Dict

ORDER = ("threat", "anger", "dissatisfaction", "anxiety")


def _fallback_band(value: str) -> str:
    aliases = {
        "neutral": "平静", "calm": "平静", "平静": "平静",
        "anxious": "焦虑", "worried": "焦虑", "焦虑": "焦虑",
        "upset": "不满", "disappointed": "不满", "dissatisfied": "不满", "不满": "不满",
        "angry": "愤怒", "愤怒": "愤怒",
        "threatening": "威胁升级", "威胁升级": "威胁升级",
    }
    return aliases.get(str(value or "").strip().lower(), "平静")

def classify_answers(answers: Dict[str, bool]) -> str:
    if answers.get("threat"): return "威胁升级"
    if answers.get("anger"): return "愤怒"
    if answers.get("dissatisfaction"): return "不满"
    if answers.get("anxiety"): return "焦虑"
    return "平静"

async def analyze_emotion(
    client: Any,
    message: str,
    recent: str = "",
    fallback_emotion: str = "平静",
) -> Dict[str, Any]:
    if client is None:
        return {"emotion": _fallback_band(fallback_emotion), "source": "fallback", "degraded": True}
    prompt = (
        "对下面消费者消息只回答 JSON，不要解释。Q1 threat=是否明确投诉/12315/曝光/起诉或限期后果；"
        "Q2 anger=是否辱骂、讽刺、指控欺骗赖账或强烈语气；Q3 dissatisfaction=是否批评服务/处理进度；"
        "Q4 anxiety=是否担心安全、时间或寻求保证。每项 answer 必须是 JSON 布尔值，"
        "evidence 必须为空字符串或从下文逐字摘录的原句片段；禁止把字符串 true/false 当布尔值。"
        "代码按 Q1 到 Q4 的优先顺序决定类别，不要自行选择类别。"
        f"\n最近消息：{recent[-1200:]}\n当前消息：{message}"
    )
    try:
        response = await client.messages.create(
            model="claude-3-5-haiku-latest", max_tokens=220, temperature=0,
            system="你是客服质检分类器，只输出 JSON。",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            str(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", ""))
            for block in getattr(response, "content", [])
        )
        payload = json.loads(text[text.find("{"):text.rfind("}") + 1])
        observed = f"{recent}\n{message}"
        answers: Dict[str, bool] = {}
        evidence: Dict[str, str] = {}
        for key in ORDER:
            item = payload.get(key)
            answer = item.get("answer") if isinstance(item, dict) else item
            quote = item.get("evidence", "") if isinstance(item, dict) else ""
            if not isinstance(answer, bool) or not isinstance(quote, str):
                raise ValueError(f"情绪判断字段格式错误: {key}")
            if quote and quote not in observed:
                raise ValueError(f"情绪证据不在输入消息中: {key}")
            if answer and not quote:
                raise ValueError(f"肯定判断缺少原文证据: {key}")
            answers[key] = answer
            if quote:
                evidence[key] = quote
        return {
            "emotion": classify_answers(answers), "answers": answers,
            "evidence": evidence, "source": "llm", "degraded": False,
        }
    except Exception as exc:
        return {
            "emotion": _fallback_band(fallback_emotion), "source": "fallback",
            "degraded": True, "degradation_reason": str(exc)[:200],
        }
