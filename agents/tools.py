"""ShopCare Agent 工具契约与通用只读工具。

本模块属于 Agent 能力层，负责把内存中的确定性计算包装成符合 Anthropic
Tool Use 协议的声明。它不连接订单、支付或人工坐席系统，也不执行任何外部
业务写操作；领域专属工具将在后续施工步骤中独立实现。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
from typing import Any, Callable, Dict, Mapping, Optional


Handler = Callable[..., Any]


@dataclass(frozen=True)
class AgentToolSpec:
    """封装一个 Agent 工具的元信息、JSON Schema 与执行函数。

    Args:
        name: 工具在 Anthropic Tool Use 中使用的唯一名称。
        description: 面向模型的工具用途说明。
        input_schema: JSON Schema 对象；必须显式禁止额外属性。
        handler: 接收参数字典的同步或异步执行函数。

    Raises:
        TypeError: 工具名称、描述、schema 或 handler 类型不符合契约时抛出。
        ValueError: schema 不是 object，或未显式设置
            ``additionalProperties=False`` 时抛出。
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Handler

    def __post_init__(self) -> None:
        """在注册前锁定 Anthropic 工具声明的安全边界。"""

        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("工具 name 必须是非空字符串")
        if not isinstance(self.description, str) or not self.description.strip():
            raise TypeError("工具 description 必须是非空字符串")
        if not isinstance(self.input_schema, dict):
            raise TypeError("工具 input_schema 必须是字典")
        if self.input_schema.get("type") != "object":
            raise ValueError('工具 input_schema.type 必须为 "object"')
        # 显式关闭额外字段，避免模型通过未声明参数越过工具契约边界。
        if self.input_schema.get("additionalProperties") is not False:
            raise ValueError("工具 input_schema 必须显式设置 additionalProperties=False")
        if not callable(self.handler):
            raise TypeError("工具 handler 必须可调用")

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """导出 Anthropic SDK 使用的工具声明。

        Returns:
            包含 ``name``、``description`` 和独立 schema 副本的字典。
        """

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": deepcopy(self.input_schema),
        }

    async def execute(
        self,
        params: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """执行工具处理器，并将未预期异常转换为安全错误结果。

        Args:
            params: 模型传入的工具参数。
            context: 可选的当前请求内存快照，仅传给声明支持该参数的 handler。

        Returns:
            handler 的结果；异常时返回 ``success=False`` 的错误字典。
        """

        try:
            signature = inspect.signature(self.handler)
            supports_context = "context" in signature.parameters or any(
                parameter.kind is parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if supports_context:
                result = self.handler(params, context=context)
            else:
                result = self.handler(params)
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as exc:  # 工具边界必须隔离异常，避免模型调用链崩溃。
            return {"success": False, "error": f"工具执行失败: {exc}"}


def make_tool(
    name: str,
    description: str,
    input_schema: Dict[str, Any],
    handler: Handler,
) -> AgentToolSpec:
    """创建并立即校验一个 ``AgentToolSpec``。

    Args:
        name: 工具名称。
        description: 工具描述。
        input_schema: 符合 object 与 additionalProperties=false 约束的 schema。
        handler: 同步或异步执行函数。

    Returns:
        校验完成的工具规格实例。
    """

    return AgentToolSpec(name, description, input_schema, handler)


def _error(message: str) -> Dict[str, Any]:
    """构造通用工具的确定性错误结果。"""

    return {"success": False, "error": message}


def _mask_user_id(user_id: str) -> str:
    """对用户标识做不可逆的展示级脱敏。"""

    if not user_id or user_id == "anonymous":
        return "anonymous"
    if len(user_id) <= 2:
        return "*" * len(user_id)
    if len(user_id) <= 4:
        return f"{user_id[0]}***"
    return f"{user_id[:2]}***{user_id[-2:]}"


async def inspect_request_context(
    params: Any,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """返回当前请求的只读、脱敏上下文快照。

    Args:
        params: 工具入参，契约要求为空对象。
        context: 当前请求的内存上下文，不能包含外部业务写操作句柄。

    Returns:
        包含 request_id、intent、脱敏 user_id 与系统时间戳的字典；入参异常时
        返回 ``success=False`` 的安全错误。
    """

    if not isinstance(params, dict):
        return _error("工具参数必须是对象")
    if params:
        return _error("inspect_request_context 不接受额外参数")
    source = context if isinstance(context, Mapping) else {}
    request_id = str(source.get("request_id", ""))
    intent = str(source.get("intent", "other"))
    user_id = str(source.get("user_id", "anonymous"))
    # 测试或上游已有可信时间戳时复用它，否则使用当前 UTC 时间完成快照。
    timestamp = str(source.get("timestamp") or datetime.now(timezone.utc).isoformat())
    return {
        "success": True,
        "request_id": request_id,
        "intent": intent,
        "user_id": _mask_user_id(user_id),
        "timestamp": timestamp,
    }


_REQUIRED_FIELDS_BY_INTENT: Dict[str, tuple[str, ...]] = {
    "refund": ("order_id", "amount"),
    "billing": ("order_id", "amount"),
    "payment_issue": ("order_id", "amount"),
    "invoice": ("order_id", "amount"),
    "order_status": ("order_id",),
    "logistics": ("order_id",),
    "technical": ("error_code", "product"),
    "technical_login": ("error_code",),
    "technical_crash": ("error_code", "product"),
    "account_security": ("date",),
}


async def suggest_required_fields(
    params: Any,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """根据意图给出下一轮追问的字段建议。

    Args:
        params: 必须包含非空字符串字段 ``intent`` 的对象。
        context: 保留参数，当前实现不读取用户画像或外部系统。

    Returns:
        包含 ``required_fields`` 与中文 ``suggestion`` 的字典；参数非法时返回
        ``success=False`` 的安全错误。
    """

    del context  # 明确该工具不读取会话私有上下文，避免信息越界。
    if not isinstance(params, dict):
        return _error("工具参数必须是对象")
    intent = params.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        return _error("intent 必须是非空字符串")

    normalized_intent = intent.strip().lower()
    fields = list(_REQUIRED_FIELDS_BY_INTENT.get(normalized_intent, ()))
    if fields:
        field_text = "、".join(fields)
        suggestion = f"建议用户补充订单或问题字段：{field_text}，以便继续只读核验。"
    else:
        suggestion = "当前意图暂无固定必填字段，建议用户补充问题细节。"
    return {
        "success": True,
        "intent": normalized_intent,
        "required_fields": fields,
        "suggestion": suggestion,
    }


INSPECT_REQUEST_CONTEXT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}

SUGGEST_REQUIRED_FIELDS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "description": "当前会话主意图。"},
    },
    "required": ["intent"],
    "additionalProperties": False,
}

INSPECT_REQUEST_CONTEXT_TOOL = make_tool(
    "inspect_request_context",
    "返回当前请求的只读脱敏上下文快照，不调用外部业务系统。",
    INSPECT_REQUEST_CONTEXT_SCHEMA,
    inspect_request_context,
)

SUGGEST_REQUIRED_FIELDS_TOOL = make_tool(
    "suggest_required_fields",
    "根据当前意图建议下一轮追问需要补充的字段，不执行任何外部写操作。",
    SUGGEST_REQUIRED_FIELDS_SCHEMA,
    suggest_required_fields,
)

GENERAL_TOOLS: Dict[str, AgentToolSpec] = {
    INSPECT_REQUEST_CONTEXT_TOOL.name: INSPECT_REQUEST_CONTEXT_TOOL,
    SUGGEST_REQUIRED_FIELDS_TOOL.name: SUGGEST_REQUIRED_FIELDS_TOOL,
}


def general_tools() -> Dict[str, AgentToolSpec]:
    """返回通用只读工具注册表的浅拷贝。

    Returns:
        以工具名为 key 的 ``AgentToolSpec`` 字典；调用方修改注册表不会影响模块
        内置定义。
    """

    return dict(GENERAL_TOOLS)

