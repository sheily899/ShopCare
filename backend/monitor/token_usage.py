"""旁路 Token 用量台账。

只保存供应商真实返回的 usage，不把缺失字段伪造成 0；记录失败也不会影响主链路。
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Mapping, Optional


_TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def _normalize_usage(value: Any) -> Dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, int] = {}
    for key in _TOKEN_KEYS:
        raw = value.get(key)
        if isinstance(raw, bool):
            continue
        try:
            number = int(raw)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            result[key] = number
    if not result:
        return {}
    if "total_tokens" not in result and {"input_tokens", "output_tokens"} <= result.keys():
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def _add_usage(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for key in _TOKEN_KEYS:
        values = [item.get(key) for item in (left, right) if isinstance(item, Mapping)]
        numbers = [int(value) for value in values if isinstance(value, int) and not isinstance(value, bool)]
        if numbers:
            result[key] = sum(numbers)
    if "total_tokens" not in result and {"input_tokens", "output_tokens"} <= result.keys():
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


class TokenUsageLedger:
    """按请求保存最近的分阶段 Token 观测，默认最多 200 条。"""

    def __init__(self, max_items: int = 200) -> None:
        self._items: deque[Dict[str, Any]] = deque(maxlen=max(1, int(max_items)))

    def record_request(
        self,
        request_id: str,
        usage_by_stage: Mapping[str, Mapping[str, Any]] | None,
        *,
        latency_ms: Optional[float] = None,
    ) -> None:
        """记录一次请求；缺少供应商 usage 时保留 unavailable 状态。"""

        try:
            stages = {
                str(stage): normalized
                for stage, usage in dict(usage_by_stage or {}).items()
                if (normalized := _normalize_usage(usage))
            }
            total: Dict[str, int] = {}
            for usage in stages.values():
                total = _add_usage(total, usage)
            item: Dict[str, Any] = {
                "request_id": str(request_id),
                "status": "available" if stages else "unavailable",
                "stages": stages,
                "total": total,
            }
            if latency_ms is not None:
                item["latency_ms"] = round(max(0.0, float(latency_ms)), 3)
            self._items.append(item)
        except Exception:
            # 观测是旁路能力，任何异常都不能影响请求返回。
            return

    def get(self, request_id: str) -> Optional[Dict[str, Any]]:
        for item in reversed(self._items):
            if item.get("request_id") == str(request_id):
                return _copy_item(item)
        return None

    def snapshot(self, limit: int = 200) -> list[Dict[str, Any]]:
        bounded = max(1, min(int(limit or 200), len(self._items)))
        return [_copy_item(item) for item in list(self._items)[-bounded:][::-1]]

    def summary(self, limit: int = 200) -> Dict[str, Any]:
        items = self.snapshot(limit)
        by_stage: Dict[str, Dict[str, int]] = {}
        for item in items:
            for stage, usage in item.get("stages", {}).items():
                by_stage[stage] = _add_usage(by_stage.get(stage, {}), usage)
        return {
            "requests": len(items),
            "available_requests": sum(item.get("status") == "available" for item in items),
            "missing_usage_requests": sum(item.get("status") != "available" for item in items),
            "by_stage": by_stage,
            "records": items,
            "storage": "memory",
            "restart_loses_data": True,
        }


def _copy_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        **item,
        "stages": {stage: dict(usage) for stage, usage in item.get("stages", {}).items()},
        "total": dict(item.get("total", {})),
    }

