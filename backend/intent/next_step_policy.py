"""人工下一步策略：规则注册表驱动，不调用大模型。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from intent.beauty_intent_registry import BEAUTY_INTENT_BY_ID

_PATH = Path(__file__).with_name("next_step_policy.json")
_POLICIES = json.loads(_PATH.read_text(encoding="utf-8")).get("policies", {})


def validate_policy_registry() -> None:
    missing = set(BEAUTY_INTENT_BY_ID) - set(_POLICIES)
    if missing:
        raise ValueError(f"人工下一步策略缺少意图: {sorted(missing)}")


def explicitly_missing_gift(message: str) -> bool:
    """只在消息明确表达赠品/小样未收到时标记漏发诉求。"""
    text = str(message or "")
    if not any(word in text for word in ("赠品", "小样")):
        return False
    if any(word in text for word in ("已经收到", "已收到", "收到了", "不需要赠品", "不用赠品")):
        return False
    missing_markers = (
        "没收到", "没有收到", "未收到", "漏发", "少发", "没发", "未发",
        "没发来", "没有发来", "没有一起发来", "未随单发出", "缺少",
    )
    return any(marker in text for marker in missing_markers)


def build_next_steps(
    intent_id: str,
    *,
    message: str = "",
    case_state: Mapping[str, Any] | None = None,
    order_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """根据注册表和当前事实返回稳定的中文待办及其来源。"""
    validate_policy_registry()
    policy = _POLICIES.get(intent_id) or _POLICIES["product_info"]
    state = case_state or {}
    slots = state.get("slots") if isinstance(state.get("slots"), Mapping) else {}
    text = str(message or "")
    steps = list(policy.get("steps", []))
    # 赠品是独立漏发诉求；仅在该子意图且消息明确说未收到时加入专项待办。
    if intent_id == "missing_item" and explicitly_missing_gift(text):
        steps.append("核对赠品发货情况")
    if intent_id == "adverse_reaction" and slots.get("medical_visit") in {"visited", "已就医"}:
        steps = [step for step in steps if "呼吸不适" not in step]
    # 完成项不再作为待办重复展示。
    if slots.get("order_id") or (order_result and order_result.get("available")):
        steps = [step for step in steps if "核对订单" not in step]
    return {"steps": list(dict.fromkeys(steps)), "sources": list(policy.get("sources", [])), "policy_id": intent_id}


validate_policy_registry()
