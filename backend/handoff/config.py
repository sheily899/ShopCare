"""Human-handoff policy loaded from the repository or an explicit runtime config directory."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


_REPOSITORY_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def handoff_config_path(filename: str) -> Path:
    """返回统一的交接配置路径，允许容器显式指定只读挂载目录。

    Args:
        filename: ``config/`` 中的配置文件名。

    Returns:
        配置文件的绝对路径。
    """
    configured_dir = os.getenv("SHOPCARE_CONFIG_DIR")
    base_dir = Path(configured_dir).expanduser() if configured_dir else _REPOSITORY_CONFIG_DIR
    return base_dir / filename


@lru_cache(maxsize=1)
def load_handoff_copy_config() -> dict[str, Any]:
    """Load and validate fixed transfer copy; missing config fails before any write."""
    with handoff_config_path("handoff_copy.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    required = (
        "version", "online_ack", "unknown_presence_ack", "callback_reason",
        "temporary_summary_direct", "temporary_summary_prior",
        "temporary_summary_open_requests",
    )
    if any(not isinstance(config.get(key), str) or not config[key].strip() for key in required):
        raise ValueError("handoff_copy.json 缺少必填文案或版本")
    if config.get("callback_sla_minutes") is not None:
        minutes = int(config["callback_sla_minutes"])
        if minutes <= 0:
            raise ValueError("callback_sla_minutes 必须为正数或 null")
        config["callback_sla_minutes"] = minutes
    return config


@lru_cache(maxsize=1)
def load_handoff_urgency_config() -> dict[str, Any]:
    """Load configured rule thresholds and verify their schema."""
    with handoff_config_path("handoff_urgency.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    # Why：兼容旧版只配置 P0-P3 分值的挂载配置，同时把转人工策略收敛到一个契约。
    config.setdefault("handoff_policy", {})
    handoff_policy = config["handoff_policy"]
    handoff_policy.setdefault(
        "explicit_request_phrases",
        ["转人工", "人工客服", "我要找人工", "找人工", "请人工处理", "联系人工", "人工处理", "人工介入", "升级人工", "请升级处理", "supervisor"],
    )
    handoff_policy.setdefault(
        "negative_request_prefixes",
        ["不想", "不要", "不需要", "无需", "暂时不", "先不", "不用", "不希望", "不考虑", "不是要", "别"],
    )
    handoff_policy.setdefault("automatic_handoff_priorities", ["P0"])
    handoff_policy.setdefault("automatic_handoff_intents", [])
    handoff_policy.setdefault(
        "automatic_handoff_intent_action", "该业务诉求需要人工核对"
    )
    handoff_policy.setdefault(
        "priority_actions",
        {
            "P0": "先给安全处理答复并进入人工高优先级处理",
            "P1": "进入高优先级待处理队列，不因紧急度单独打断当前业务答复",
            "P2": "继续普通业务流程",
            "P3": "继续普通业务流程",
        },
    )
    handoff_policy.setdefault("intent_priority_floor", {"adverse_reaction": "P1", "complaint": "P2"})
    handoff_policy.setdefault("money_alone_auto_handoff", False)
    if config.get("default", {}).get("priority") not in {"P0", "P1", "P2", "P3"}:
        raise ValueError("handoff_urgency.json default.priority 必须是 P0-P3")
    if config.get("hard_priority", "P0") not in {"P0", "P1", "P2", "P3"}:
        raise ValueError("handoff_urgency.json hard_priority 必须是 P0-P3")
    thresholds = config.get("thresholds") or {}
    if not isinstance(thresholds.get("p1_min_score"), int) or not isinstance(thresholds.get("p2_min_score"), int):
        raise ValueError("handoff_urgency.json 缺少整数阈值")
    if not isinstance(config.get("hard_score"), int) or config["hard_score"] <= 0:
        raise ValueError("handoff_urgency.json hard_score 必须为正整数")
    threat_rule = config.get("async_emotion_threat") or {}
    if not threat_rule.get("id") or not threat_rule.get("reason"):
        raise ValueError("handoff_urgency.json 缺少异步威胁情绪规则")
    phrases = handoff_policy.get("explicit_request_phrases")
    if not isinstance(phrases, list) or not phrases or any(not str(item).strip() for item in phrases):
        raise ValueError("handoff_urgency.json 必须配置明确转人工表达")
    negative_phrases = handoff_policy.get("negative_request_prefixes")
    if not isinstance(negative_phrases, list) or any(not str(item).strip() for item in negative_phrases):
        raise ValueError("handoff_urgency.json negative_request_prefixes 必须为非空文本列表")
    automatic_priorities = handoff_policy.get("automatic_handoff_priorities")
    if not isinstance(automatic_priorities, list) or any(
        item not in {"P0", "P1", "P2", "P3"} for item in automatic_priorities
    ):
        raise ValueError("handoff_urgency.json automatic_handoff_priorities 必须为 P0-P3 列表")
    if str(config.get("hard_priority", "P0")) not in automatic_priorities:
        raise ValueError("硬风险等级必须包含在 automatic_handoff_priorities 中")
    automatic_intents = handoff_policy.get("automatic_handoff_intents")
    if not isinstance(automatic_intents, list) or any(
        not isinstance(item, str) or not item.strip() for item in automatic_intents
    ):
        raise ValueError("handoff_urgency.json automatic_handoff_intents 必须为非空意图 ID 列表")
    if not isinstance(handoff_policy.get("money_alone_auto_handoff"), bool):
        raise ValueError("handoff_urgency.json money_alone_auto_handoff 必须为布尔值")
    if handoff_policy["money_alone_auto_handoff"]:
        raise ValueError("当前版本不支持仅因涉及金额自动转人工")
    return config


@lru_cache(maxsize=1)
def load_handoff_emotion_config() -> dict[str, Any]:
    """Load the Q1-Q4 prompt and deterministic fallback rules for handoff emotion."""
    with handoff_config_path("handoff_emotion.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    questions = config.get("questions")
    if not isinstance(questions, list) or [item.get("id") for item in questions] != [
        "threat", "anger", "dissatisfaction", "anxiety"
    ]:
        raise ValueError("handoff_emotion.json 的问题顺序必须为 Q1-Q4")
    if int(config.get("temperature", -1)) != 0 or int(config.get("max_tokens", 0)) <= 0:
        raise ValueError("handoff_emotion.json 模型参数无效")
    if any(not config.get(key) for key in ("version", "model", "system_prompt", "user_prompt_prefix")):
        raise ValueError("handoff_emotion.json 缺少版本、模型或提示词")
    if set(config.get("fallback_rules") or {}) != {"threat", "anger", "dissatisfaction", "anxiety"}:
        raise ValueError("handoff_emotion.json fallback_rules 必须覆盖 Q1-Q4")
    return config


def callback_due_at(config: dict[str, Any]) -> str | None:
    """Return a due time only when an explicit positive SLA is configured."""
    minutes = config.get("callback_sla_minutes")
    if minutes is None:
        return None
    return (datetime.now(timezone.utc) + timedelta(minutes=int(minutes))).isoformat()
