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
import math
import re
import uuid
from numbers import Real
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


_ERROR_CODE_CATALOG: Dict[str, Dict[str, Any]] = {
    "401": {
        "meaning": "认证失败：登录凭证缺失、过期或无效。",
        "suggestions": ["确认账号仍处于登录状态", "重新登录后重试"],
    },
    "403": {
        "meaning": "权限不足：当前账号无权访问该资源。",
        "suggestions": ["确认账号与目标资源匹配", "联系管理员核对访问权限"],
    },
    "404": {
        "meaning": "资源不存在：请求的页面或接口路径未找到。",
        "suggestions": ["检查访问链接是否完整", "返回上一页后重新打开"],
    },
    "500": {
        "meaning": "服务器内部错误：服务处理请求时发生异常。",
        "suggestions": ["记录发生时间与操作步骤", "稍后重试并保留错误截图"],
    },
    "400": {
        "meaning": "请求参数无效：提交的数据未通过格式校验。",
        "suggestions": ["检查必填字段和输入格式", "移除无效字符后重试"],
    },
    "429": {
        "meaning": "请求过于频繁：当前访问触发了频率限制。",
        "suggestions": ["等待片刻后重试", "避免短时间重复提交"],
    },
}


async def lookup_error_code(params: Any) -> Dict[str, Any]:
    """根据静态状态码目录返回低风险排查建议。

    Args:
        params: 必须包含 ``error_code: str`` 的工具参数对象。

    Returns:
        包含状态码含义、静态建议和 ``server_log_checked=False`` 的字典；参数
        错误时返回安全错误结果。该方法不读取服务端日志。
    """

    base = {"server_log_checked": False, "source": "static_catalog"}
    if not isinstance(params, dict):
        return {**base, **_error("工具参数必须是对象")}
    raw_code = params.get("error_code")
    if not isinstance(raw_code, str) or not raw_code.strip():
        return {**base, **_error("error_code 必须是非空字符串")}

    code = raw_code.strip().upper().removeprefix("HTTP ").strip()
    entry = _ERROR_CODE_CATALOG.get(code)
    if entry is None:
        return {
            **base,
            "success": True,
            "error_code": code,
            "meaning": "未知状态码：当前静态目录没有该状态码的解释。",
            "suggestions": ["提供完整错误信息和发生步骤", "不要将本结果视为后台日志结论"],
        }
    return {
        **base,
        "success": True,
        "error_code": code,
        "meaning": entry["meaning"],
        "suggestions": list(entry["suggestions"]),
    }


async def build_diagnostic_plan(params: Any) -> Dict[str, Any]:
    """根据症状生成不改变系统状态的低风险排查计划。

    Args:
        params: 包含 ``symptom: str`` 和可选 ``device_type: str`` 的参数对象。

    Returns:
        包含结构化 ``steps`` 列表的排查计划。所有步骤均标记为 low risk，且不
        执行系统重置、删除数据或其他高危操作。
    """

    if not isinstance(params, dict):
        return _error("工具参数必须是对象")
    symptom = params.get("symptom")
    device_type = params.get("device_type")
    if not isinstance(symptom, str) or not symptom.strip():
        return _error("symptom 必须是非空字符串")
    if device_type is not None and not isinstance(device_type, str):
        return _error("device_type 必须是字符串或省略")

    steps = [
        {
            "step": 1,
            "action": "记录完整错误提示、发生时间和复现步骤",
            "reason": "为后续人工或技术排查保留事实信息",
            "risk": "low",
        },
        {
            "step": 2,
            "action": "检查网络连接并在稳定网络下重试",
            "reason": "排除临时网络波动影响",
            "risk": "low",
        },
        {
            "step": 3,
            "action": "清理应用缓存后重新打开当前页面",
            "reason": "排除本地缓存导致的加载异常",
            "risk": "low",
        },
        {
            "step": 4,
            "action": "完全退出应用并重新启动",
            "reason": "刷新当前会话与临时资源",
            "risk": "low",
        },
    ]
    return {
        "success": True,
        "symptom": symptom.strip(),
        "device_type": device_type.strip() if isinstance(device_type, str) else None,
        "steps": steps,
        "high_risk_action": False,
    }


_BILLING_FIELDS = ("order_id", "amount", "date", "payment_channel")


async def check_billing_fields(params: Any) -> Dict[str, Any]:
    """检查账单核验所需字段是否齐全，不执行退款或支付操作。

    Args:
        params: 可选的 ``order_id``、``amount``、``date`` 和
            ``payment_channel`` 字符串字段。

    Returns:
        包含字段状态、缺失字段和固定 ``can_confirm_refund=False`` 的字典。
        该结果只表示信息完整度，不代表退款成功或后台已核验。
    """

    if not isinstance(params, dict):
        return {"can_confirm_refund": False, **_error("工具参数必须是对象")}
    values: Dict[str, Optional[str]] = {}
    for field_name in _BILLING_FIELDS:
        value = params.get(field_name)
        if value is not None and not isinstance(value, str):
            return {
                "can_confirm_refund": False,
                **_error(f"{field_name} 必须是字符串或省略"),
            }
        values[field_name] = value.strip() if isinstance(value, str) else None

    missing_fields = [name for name, value in values.items() if not value]
    field_status = {name: bool(value) for name, value in values.items()}
    return {
        "success": True,
        "fields": values,
        "field_status": field_status,
        "missing_fields": missing_fields,
        "complete": not missing_fields,
        # 账单工具只能检查输入完整度，绝不越权确认退款结果。
        "can_confirm_refund": False,
        "read_only": True,
    }


async def compare_amounts(params: Any) -> Dict[str, Any]:
    """计算预期金额与实际金额的差值和大小关系。

    Args:
        params: 包含 ``expected_amount`` 与 ``actual_amount`` 数值字段的对象。

    Returns:
        包含原值、``difference``（实际值减预期值）和 comparison 的字典；非法
        数值返回安全错误。该方法不推断重复扣款或其他法律结论。
    """

    if not isinstance(params, dict):
        return _error("工具参数必须是对象")
    expected = params.get("expected_amount")
    actual = params.get("actual_amount")
    if (
        isinstance(expected, bool)
        or isinstance(actual, bool)
        or not isinstance(expected, Real)
        or not isinstance(actual, Real)
    ):
        return _error("expected_amount 和 actual_amount 必须是数值")

    expected_value = float(expected)
    actual_value = float(actual)
    if not math.isfinite(expected_value) or not math.isfinite(actual_value):
        return _error("金额必须是有限数值")
    difference = actual_value - expected_value
    comparison = (
        "actual_higher"
        if difference > 0
        else "actual_lower"
        if difference < 0
        else "equal"
    )
    return {
        "success": True,
        "expected_amount": expected_value,
        "actual_amount": actual_value,
        "difference": difference,
        "comparison": comparison,
        "read_only": True,
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


LOOKUP_ERROR_CODE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"error_code": {"type": "string"}},
    "required": ["error_code"],
    "additionalProperties": False,
}

BUILD_DIAGNOSTIC_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "symptom": {"type": "string"},
        "device_type": {"type": "string"},
    },
    "required": ["symptom"],
    "additionalProperties": False,
}

CHECK_BILLING_FIELDS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "amount": {"type": "string"},
        "date": {"type": "string"},
        "payment_channel": {"type": "string"},
    },
    "required": [],
    "additionalProperties": False,
}

COMPARE_AMOUNTS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "expected_amount": {"type": "number"},
        "actual_amount": {"type": "number"},
    },
    "required": ["expected_amount", "actual_amount"],
    "additionalProperties": False,
}

LOOKUP_ERROR_CODE_TOOL = make_tool(
    "lookup_error_code",
    "根据静态目录解释常见错误码并给出低风险建议，不读取服务端日志。",
    LOOKUP_ERROR_CODE_SCHEMA,
    lookup_error_code,
)

BUILD_DIAGNOSTIC_PLAN_TOOL = make_tool(
    "build_diagnostic_plan",
    "根据技术症状生成只读、低风险的分步排查计划。",
    BUILD_DIAGNOSTIC_PLAN_SCHEMA,
    build_diagnostic_plan,
)

CHECK_BILLING_FIELDS_TOOL = make_tool(
    "check_billing_fields",
    "检查账单核验字段是否齐全，不确认退款、不调用支付系统。",
    CHECK_BILLING_FIELDS_SCHEMA,
    check_billing_fields,
)

COMPARE_AMOUNTS_TOOL = make_tool(
    "compare_amounts",
    "只计算预期金额与实际金额的数学差值，不做业务或法律定性。",
    COMPARE_AMOUNTS_SCHEMA,
    compare_amounts,
)

TECHNICAL_TOOLS: Dict[str, AgentToolSpec] = {
    LOOKUP_ERROR_CODE_TOOL.name: LOOKUP_ERROR_CODE_TOOL,
    BUILD_DIAGNOSTIC_PLAN_TOOL.name: BUILD_DIAGNOSTIC_PLAN_TOOL,
}

BILLING_TOOLS: Dict[str, AgentToolSpec] = {
    CHECK_BILLING_FIELDS_TOOL.name: CHECK_BILLING_FIELDS_TOOL,
    COMPARE_AMOUNTS_TOOL.name: COMPARE_AMOUNTS_TOOL,
}


def technical_tools() -> Dict[str, AgentToolSpec]:
    """返回 Technical 专属工具注册表的浅拷贝。

    Returns:
        以工具名为 key 的 Technical 工具字典。
    """

    return dict(TECHNICAL_TOOLS)


def billing_tools() -> Dict[str, AgentToolSpec]:
    """返回 Billing 专属工具注册表的浅拷贝。

    Returns:
        以工具名为 key 的 Billing 工具字典。
    """

    return dict(BILLING_TOOLS)


# 这些字段属于交接摘要的禁止采集范围；过滤发生在输出边界，避免敏感值进入摘要。
_FORBIDDEN_SENSITIVE_FIELDS = frozenset(
    {"password", "verification_code", "full_card_number"}
)
_SENSITIVE_TEXT_PATTERN = re.compile(
    r"password|verification_code|full_card_number|密码|短信验证码|验证码|银行卡号",
    flags=re.IGNORECASE,
)


def _sanitize_handoff_text(value: str) -> str:
    """移除交接文本中的敏感凭证提示词。"""

    return _SENSITIVE_TEXT_PATTERN.sub("[已脱敏字段]", value)


def _sanitize_handoff_value(value: Any) -> Any:
    """递归清理交接实体中的敏感字段与文本。"""

    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_handoff_value(item)
            for key, item in value.items()
            if str(key).strip().lower() not in _FORBIDDEN_SENSITIVE_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_handoff_value(item) for item in value]
    if isinstance(value, str):
        return _sanitize_handoff_text(value)
    return value


def _handoff_error(message: str) -> Dict[str, Any]:
    """构造不创建工单且不要求敏感数据的安全错误结果。"""

    return {
        "success": False,
        "error": message,
        "sensitive_data_required": False,
        "external_ticket_created": False,
    }


async def create_handoff_summary(params: Any) -> Dict[str, Any]:
    """生成仅供当前会话使用的人工交接摘要与升级建议。

    Args:
        params: 工具参数对象，包含 ``reason``、``urgency``，以及可选的
            ``entities`` 和 ``context_summary``。

    Returns:
        标准化的交接摘要字典。该结果只表示升级建议，不代表人工已接入，
        且不会创建外部工单或收集密码、验证码、完整银行卡号。
    """

    if not isinstance(params, Mapping):
        return _handoff_error("工具参数必须是对象")

    reason = params.get("reason")
    urgency = params.get("urgency")
    entities = params.get("entities")
    context_summary = params.get("context_summary")

    if not isinstance(reason, str) or not reason.strip():
        return _handoff_error("reason 必须是非空字符串")
    # 紧急度固定在 1-4，便于下游按既定等级处理而不引入隐式转换。
    if isinstance(urgency, bool) or not isinstance(urgency, int) or not 1 <= urgency <= 4:
        return _handoff_error("urgency 必须是 1 到 4 的整数")
    if entities is not None and not isinstance(entities, Mapping):
        return _handoff_error("entities 必须是对象")
    if context_summary is not None and not isinstance(context_summary, str):
        return _handoff_error("context_summary 必须是字符串")

    return {
        "success": True,
        "request_id": uuid.uuid4().hex[:8],
        "reason": _sanitize_handoff_text(reason.strip()),
        "urgency": urgency,
        "entities": _sanitize_handoff_value(dict(entities or {})),
        "context_summary": _sanitize_handoff_text(context_summary.strip())
        if isinstance(context_summary, str)
        else "",
        "sensitive_data_required": False,
        "external_ticket_created": False,
        "handoff_status": "recommended",
        "handoff_message": "已生成交接摘要与升级建议，等待后续人工流程处理。",
    }


CREATE_HANDOFF_SUMMARY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "urgency": {"type": "integer", "minimum": 1, "maximum": 4},
        "entities": {"type": "object"},
        "context_summary": {"type": "string"},
    },
    "required": ["reason", "urgency"],
    "additionalProperties": False,
}

CREATE_HANDOFF_SUMMARY_TOOL = make_tool(
    "create_handoff_summary",
    "生成不创建外部工单的人工交接摘要与升级建议。",
    CREATE_HANDOFF_SUMMARY_SCHEMA,
    create_handoff_summary,
)

ESCALATION_TOOLS: Dict[str, AgentToolSpec] = {
    CREATE_HANDOFF_SUMMARY_TOOL.name: CREATE_HANDOFF_SUMMARY_TOOL,
}


def escalation_tools() -> Dict[str, AgentToolSpec]:
    """返回 Escalation 人工交接工具注册表的浅拷贝。

    Returns:
        以工具名为 key 的 Escalation 工具字典。
    """

    return dict(ESCALATION_TOOLS)

