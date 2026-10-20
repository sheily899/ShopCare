"""会话处理链路共用的可追溯事实对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass
class CaseFacts:
    """保存字段值及其来源，供回答、摘要、工单和工作台共用。

    `facts` 使用字段到记录列表的映射，以兼容同一事实由消费者、客服补录和
    工具分别确认的情况；每条记录都保留独立来源，不把推断冒充为工具结果。
    """

    facts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    schema_version: int = 1
    MULTI_VALUE_FIELDS = frozenset({"symptom", "emotion_evidence", "knowledge_evidence", "risk_tags"})

    @classmethod
    def from_entities(
        cls,
        entities: Mapping[str, Iterable[Any]] | None,
        *,
        source_ref: str,
        confidence: float = 0.9,
        confirmation_status: str = "unconfirmed",
    ) -> "CaseFacts":
        facts = cls()
        for field_name, values in (entities or {}).items():
            items = values if isinstance(values, (list, tuple, set)) else [values]
            for value in items:
                if value is None or str(value).strip() == "":
                    continue
                facts.record(
                    str(field_name),
                    value,
                    source_type="message",
                    source_ref=source_ref,
                    confidence=confidence,
                    confirmation_status=confirmation_status,
                )
        return facts

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "CaseFacts":
        raw = payload if isinstance(payload, Mapping) else {}
        result = cls(schema_version=int(raw.get("schema_version", 1) or 1))
        items = raw.get("facts") if isinstance(raw.get("facts"), Mapping) else {}
        for field_name, entries in items.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping) or "value" not in entry:
                    continue
                result.record(
                    str(field_name),
                    entry["value"],
                    source_type=str(entry.get("source_type") or "legacy"),
                    source_ref=str(entry.get("source_ref") or ""),
                    confidence=entry.get("confidence", 0.5),
                    confirmation_status=str(entry.get("confirmation_status") or "unconfirmed"),
                )
        return result

    def record(
        self,
        field_name: str,
        value: Any,
        *,
        source_type: str,
        source_ref: str = "",
        confidence: float = 1.0,
        confirmation_status: str = "confirmed",
    ) -> None:
        """追加一条来源明确的事实，不覆盖其他来源记录。"""
        cleared = confirmation_status == "cleared"
        if not field_name or (value is None and not cleared) or (isinstance(value, str) and not value.strip() and not cleared):
            return
        try:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_value = 0.5
        item = {
            "value": value,
            "source_type": str(source_type or "unknown"),
            "source_ref": str(source_ref or ""),
            "confidence": confidence_value,
            "confirmation_status": str(confirmation_status or "unconfirmed"),
        }
        records = self.facts.setdefault(str(field_name), [])
        if item not in records:
            records.append(item)

    def clear(
        self,
        field_name: str,
        *,
        source_type: str,
        source_ref: str = "",
    ) -> None:
        """记录客服明确清空字段，避免旧值继续被摘要或模型当作有效事实。"""
        self.record(
            field_name,
            None,
            source_type=source_type,
            source_ref=source_ref,
            confidence=1.0,
            confirmation_status="cleared",
        )

    def record_tool_result(
        self,
        tool_name: str,
        result: Mapping[str, Any],
        *,
        source_ref: str,
    ) -> None:
        """把成功工具返回的业务字段写入事实对象；失败结果不产生事实。"""
        if not isinstance(result, Mapping) or not result.get("success"):
            return
        source_type = f"tool:{tool_name}"
        if tool_name in {"lookup_order", "lookup_logistics", "check_refund_status"} and not result.get("available"):
            self.record(
                f"{tool_name}_available",
                False,
                source_type=source_type,
                source_ref=source_ref,
            )
            return

        def normalized_value(name: str, value: Any) -> Any:
            if name == "gift_items" and isinstance(value, (list, tuple, set)):
                return "、".join(str(item) for item in value if item)
            return value

        direct_fields = {
            "order_id", "tracking_no", "carrier", "logistics_status", "refund_status",
            "refund_amount", "price_difference", "quantity", "status",
            "order_status", "signed_info", "received_by", "gift_items",
        }
        for name in direct_fields:
            value = result.get(name)
            if value is not None and value != "":
                target_name = "order_status" if name == "status" and tool_name == "lookup_order" else name
                self.record(target_name, normalized_value(name, value), source_type=source_type, source_ref=source_ref)

        order = result.get("order")
        if isinstance(order, Mapping):
            for name in (
                "order_id", "product_name", "sku", "quantity", "status", "order_status",
                "tracking_no", "carrier", "logistics_status", "signed_info", "received_by",
                "gift_items", "refund_status",
            ):
                value = order.get(name)
                if value is not None and value != "":
                    target_name = "order_status" if name == "status" else name
                    self.record(target_name, normalized_value(name, value), source_type=source_type, source_ref=source_ref)

        if tool_name == "search_knowledge_base":
            chunks = result.get("chunks") or result.get("results") or []
            if isinstance(chunks, list):
                for index, chunk in enumerate(chunks):
                    if not isinstance(chunk, Mapping):
                        continue
                    evidence = {
                        "title": str(chunk.get("title") or ""),
                        "content": str(chunk.get("content") or ""),
                        "source_chunk_id": str(chunk.get("source_chunk_id") or chunk.get("id") or index),
                    }
                    if evidence["content"]:
                        self.record(
                            "knowledge_evidence",
                            evidence,
                            source_type=source_type,
                            source_ref=evidence["source_chunk_id"],
                        )

    def merge(self, other: "CaseFacts | Mapping[str, Any] | None") -> None:
        source = other if isinstance(other, CaseFacts) else self.from_dict(other)
        for field_name, records in source.facts.items():
            for record in records:
                self.record(
                    field_name,
                    record.get("value"),
                    source_type=record.get("source_type", "unknown"),
                    source_ref=record.get("source_ref", ""),
                    confidence=record.get("confidence", 0.5),
                    confirmation_status=record.get("confirmation_status", "unconfirmed"),
                )

    def latest(self, field_name: str) -> dict[str, Any] | None:
        records = self.facts.get(field_name) or []
        return dict(records[-1]) if records else None

    def values(self, field_name: str) -> list[Any]:
        values: list[Any] = []
        records = self.facts.get(field_name) or []
        if field_name not in self.MULTI_VALUE_FIELDS and records:
            latest = records[-1]
            if latest.get("confirmation_status") == "cleared" or latest.get("value") is None:
                return []
            records = [latest]
        elif field_name in self.MULTI_VALUE_FIELDS and records:
            # 多值字段也支持客服清空：清空标记之前的旧值不能继续当成当前事实。
            last_clear = max(
                (
                    index
                    for index, item in enumerate(records)
                    if item.get("confirmation_status") == "cleared"
                    or item.get("value") is None
                ),
                default=-1,
            )
            records = records[last_clear + 1 :]
        for item in records:
            if item.get("confirmation_status") == "cleared" or item.get("value") is None:
                continue
            value = item.get("value")
            if value not in values:
                values.append(value)
        return values

    def effective_values(
        self,
        current_entities: Mapping[str, Any] | None = None,
    ) -> dict[str, list[Any]]:
        """返回回答当前轮时的事实视图，让当前消息覆盖较旧的案件事实。

        当前消息的实体抽取若为空，表示本轮没有更新该字段，不会抹掉历史值；
        显式清空仍由 ``clear()`` 的 ``cleared`` 状态表达。
        """

        effective = self.to_prompt_dict()
        entities = current_entities if isinstance(current_entities, Mapping) else {}
        for field_name, raw_values in entities.items():
            values = raw_values if isinstance(raw_values, (list, tuple, set)) else [raw_values]
            current_values = [
                value
                for value in values
                if value is not None and str(value).strip()
            ]
            if current_values:
                unique_values: list[Any] = []
                for value in current_values:
                    if value not in unique_values:
                        unique_values.append(value)
                effective[str(field_name)] = unique_values
            else:
                effective.setdefault(str(field_name), [])
        return effective

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "facts": {
                field_name: [dict(item) for item in records]
                for field_name, records in self.facts.items()
            },
        }

    def to_prompt_dict(self) -> dict[str, list[Any]]:
        """返回去除存储元数据的值视图，供最终答复模型读取。"""
        return {name: self.values(name) for name in self.facts}
