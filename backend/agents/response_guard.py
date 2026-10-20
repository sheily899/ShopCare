"""消费者回复的确定性安全守卫。"""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Any, Mapping

_INTERNAL_MARKERS = ("symptom:", "medical_visit", "order_id", "转交给专员", "建议转人工")
_UNSAFE_CLAIMS = ("一定退款", "一定会退款", "肯定会退款", "保证退款", "绝对会好", "一定治好", "肯定治好", "绝对治好")
_SAFE_REPLACEMENT = "这项信息目前还无法确认，我会先按已核实的信息继续处理。"
_NEGATED_OR_QUESTIONED = ("不能保证", "不保证", "无法保证", "不承诺", "不一定", "不能确认", "无法确认", "是否", "会不会")
_UNSAFE_HANDOFF_SUGGESTIONS = (
    "已为您转接人工", "为您转接人工", "已转人工", "已补发", "已经补发", "已退款", "已经退款",
    "已完成", "已经完成", "已处理", "已经处理", "已核实", "已经核实", "已确认", "已经确认",
    "已查到", "已经查到", "已安排", "已经安排", "已提交", "已经提交", "工单已",
)

_FIELD_QUESTION_ALIASES = {
    "order_id": ("订单号", "订单编号"),
    "product_name": ("商品名称", "产品名称", "商品名", "产品名"),
    "sku": ("商品规格", "产品规格", "规格"),
    "quantity": ("数量", "几件", "几个", "多少件", "多少个", "多少支"),
    "shade": ("色号",),
    "shade_received": ("收到的色号", "收到什么色号", "收到色号"),
    "opened_or_used": ("拆封", "开封", "使用情况", "是否使用", "用过", "使用过"),
    "gift_opened_or_used": ("小样拆封", "小样是否拆封", "赠品拆封", "赠品是否拆封"),
    "return_reason": ("退货原因", "退款原因", "退货理由", "退款理由", "为什么退货", "为什么退款"),
    "return_scope": ("整单退款", "整单退", "全单退款", "正装和小样", "赠品是否一起退", "小样是否一起退"),
    "missing_item": ("漏发", "少发", "缺少的商品", "缺少什么", "漏了什么"),
    "wrong_item": ("错发", "发错", "收到的商品", "收到什么商品"),
    "damaged_area": ("破损部位", "哪里破损", "损坏部位"),
    "tracking_no": ("物流单号", "运单号", "快递单号"),
    "signed_info": ("签收情况", "签收信息", "是否签收"),
    "received_by": ("签收人", "谁签收", "谁收的"),
    "intercept_or_change": ("拦截需求", "改地址需求", "修改地址需求"),
    "new_address": ("新地址", "收货地址"),
    "symptom": ("不适症状", "具体症状", "出现什么症状", "哪里不舒服"),
    "symptom_area": ("不适部位", "发红部位", "肿胀部位", "具体部位", "具体位置", "哪里红肿", "红肿位置"),
    "onset_time": ("出现时间", "什么时候开始", "用了多久出现"),
    "stopped_use": ("是否停用", "停用情况", "还在使用吗", "是否还在用"),
    "medical_visit": ("就医情况", "是否就医", "有没有就医", "看过医生吗"),
    "age": ("年龄", "年龄段"),
    "refund_amount": ("退款金额", "申请金额"),
    "refund_status": ("退款进度", "退款状态"),
    "price_difference": ("差价", "价差", "差额"),
    "gift_items": ("赠品", "小样"),
    "carrier": ("快递公司", "承运商"),
    "logistics_status": ("物流进度", "物流状态"),
}
_FIELD_QUESTION_MARKERS = (
    "?", "？", "吗", "呢", "是否", "能否", "可否", "是不是", "有没有", "有无", "还是", "请问",
    "什么", "哪", "多少", "几", "哪里", "何时", "什么时候", "如何", "怎样",
    "麻烦您确认", "方便确认", "请您确认", "请确认", "请提供", "请补充",
    "请告知", "麻烦提供", "麻烦补充", "还需要提供",
)
_FIELD_QUESTION_PATTERNS = {
    "order_id": (
        r"(?:请|麻烦|还需要|需要您).{0,10}(?:提供|告知|补充|发送|发来|发一下|告诉).{0,8}(?:订单号|订单编号)",
        r"(?:订单号|订单编号)(?:(?!对应).){0,6}(?:是多少|是什么|几号|发一下|补充一下|提供一下)",
        r"(?:再|请|麻烦).{0,8}(?:发|补|提供|告知).{0,8}(?:订单号|订单编号)",
    ),
    "opened_or_used": (
        r"(?:是否|有没有|是不是|是否已|是否已经).{0,6}(?:拆封|开封|拆开|使用|用过)",
        r"(?:拆封|开封|拆开|使用|用过)(?:了|过)?(?:吗|没有|没|呢)",
        r"(?:商品|产品|包装).{0,8}(?:拆封|开封|拆开|使用)(?:情况|状态)(?:是什么|如何|怎样|吗|呢)",
        r"(?:商品|产品|包装).{0,8}(?:拆封|开封|拆开|使用)(?:了|过)?(?:吗|没有|没|呢)",
        r"(?:商品|产品|包装).{0,8}(?:未拆封|已拆封|已使用|拆封未使用).{0,4}(?:吗|呢|对吗)",
    ),
    "return_reason": (
        r"(?:退货|退款).{0,12}(?:原因|理由).{0,10}(?:是什么|是哪|是啥|吗|呢)",
        r"(?:请|麻烦|还需要|需要您).{0,10}(?:提供|告知|补充|说明).{0,8}(?:退货原因|退款原因|退货理由|退款理由)",
        r"(?:为什么|为何).{0,4}(?:退货|退款)",
        r"(?:退货|退款).{0,10}(?:因为什么|为何|为啥)",
    ),
    "gift_items": (
        r"(?:赠品|小样).{0,6}(?:收到|有没有|是否|还在吗|齐吗)",
        r"(?:是否|有没有|有无).{0,6}(?:收到|拿到).{0,6}(?:赠品|小样)",
    ),
    "symptom": (
        r"(?:请问|请|麻烦|还需要|需要您).{0,8}(?:具体症状|出现什么症状|哪里不舒服|不适症状)",
        r"(?:具体症状|出现什么症状|哪里不舒服|不适症状).{0,4}(?:是什么|吗|呢)",
    ),
    "refund_status": (
        r"(?:是否|有没有|是不是|是否已|是否已经).{0,8}(?:提交|申请).{0,6}退款",
        r"(?:退款申请|退款).{0,8}(?:是否|有没有|进度|状态)",
    ),
}

_ASSISTANT_ROLES = frozenset({"assistant", "ai", "shopcare-ai", "customer_service", "客服", "人工客服"})
_REPEAT_REQUEST = re.compile(r"再说一遍|再说一次|重复(?:一下|一遍)?|再解释一下|重新说明|详细说(?:一下)?|展开说(?:一下)?")
_SAFETY_QUESTION = re.compile(
    r"怎么办|怎么处理|如何处理|怎么做|还能用吗|要不要就医|需要就医|安全吗|"
    r"(?:还能|是否可以|可不可以|能不能|可以|能)继续(?:使用|用)|要不要停用"
)
_SAFETY_SPECIFIC_QUESTION = re.compile(
    r"还能用|继续(?:使用|用)|要不要就医|需要就医|安全吗|是否安全|要不要停用|是否停用"
)
_WORSENING_SIGNAL = re.compile(r"加重|加剧|越来越|更红|更痒|更肿|肿得更|呼吸困难|水疱|水泡|渗液|急诊|发烧")
_SAFETY_SYMPTOMS = {
    "redness": re.compile(r"泛红|发红|红肿|脸(?:部)?(?:还|仍)?红"),
    "itch": re.compile(r"发痒|瘙痒|痒"),
    "sting": re.compile(r"刺痛|刺痒"),
    "rash": re.compile(r"起疹|疹子"),
    "swelling": re.compile(r"肿胀|脸肿|肿"),
    "peeling": re.compile(r"脱皮"),
    "allergy": re.compile(r"过敏"),
    "breathing": re.compile(r"呼吸困难|喘不上气"),
}
_SAFETY_ADVICE = {
    "stop_use": re.compile(r"停用|停止使用|暂停使用|先别用"),
    "cleanse": re.compile(r"清水|温水|清洁|冲洗"),
    "avoid_irritation": re.compile(r"不要抓|别抓|抓挠|不要热敷|避免热敷|不要叠加|避免叠加"),
    "seek_care": re.compile(r"就医|急诊|皮肤科|看医生"),
    "preserve_batch": re.compile(r"保留.{0,8}(?:包装|批次|外盒)|(?:包装|批次).{0,8}保留"),
}
_ADVICE_MARKER = re.compile(r"请|建议|先|尽快|避免|不要|别|暂停|停用|保留|应当|需要")
_CLAUSE_SPLIT = re.compile(r"([。！？!?；;\n]+)")
_CLAUSE_NORMALIZE = re.compile(r"[\s\W_]+", re.UNICODE)
_MATERIAL_STATUS_MARKERS = (
    "尚未", "未", "没有", "不", "已经", "已", "正在", "待", "审核中", "签收",
    "未签收", "成功", "失败", "不能", "可以", "不支持", "支持",
)
_EXTERNAL_REFUND_ACTION = re.compile(
    r"(?:订单页|订单页面|订单详情页|平台订单页).{0,50}(?:发起|提交|选择|点击|申请)"
    r"|(?:打开|进入|点击|前往).{0,16}(?:订单页|订单页面|订单详情页|平台页面)"
    r"|(?:自行|自己).{0,10}(?:发起|提交|申请).{0,12}(?:退款|退货)"
)


def _history_role_content(item: Any) -> tuple[str, str]:
    """读取历史消息的角色与文本，兼容 Message、字典及 Enum 角色。"""
    if isinstance(item, Mapping):
        role = item.get("role", "")
        content = item.get("content", "")
    else:
        role = getattr(item, "role", "")
        content = getattr(item, "content", "")
    role = getattr(role, "value", role)
    return str(role or "").strip().casefold(), str(content or "").strip()


def _split_clauses(text: str) -> list[str]:
    """按句末标点和换行拆分答复，保留可独立去重的完整句子。"""
    return [part.strip() for part in _CLAUSE_SPLIT.split(str(text or "")) if part.strip()]


def _is_negated(text: str, start: int) -> bool:
    """判断一个风险词是否紧跟在常见中文否定表达之后。"""
    prefix = text[max(0, start - 8):start]
    return bool(re.search(r"(?:没有|没|并未|未|无|不|并不)(?:出现|有|感到|觉得)?\s*$", prefix))


def _safety_topic_reopened(current_message: str, prior_user_messages: list[str]) -> bool:
    """判断本轮是否询问安全处理或带来新/加重症状。"""
    if _SAFETY_SPECIFIC_QUESTION.search(current_message):
        return True
    if _SAFETY_QUESTION.search(current_message) and any(
        pattern.search(current_message) for pattern in _SAFETY_SYMPTOMS.values()
    ):
        return True
    if any(
        match and not _is_negated(current_message, match.start())
        for match in _WORSENING_SIGNAL.finditer(current_message)
    ):
        return True

    previous_symptoms = {
        name
        for message in prior_user_messages
        for name, pattern in _SAFETY_SYMPTOMS.items()
        if any(not _is_negated(message, match.start()) for match in pattern.finditer(message))
    }
    current_symptoms = {
        name
        for name, pattern in _SAFETY_SYMPTOMS.items()
        if any(not _is_negated(current_message, match.start()) for match in pattern.finditer(current_message))
    }
    return bool(current_symptoms - previous_symptoms)


def _safety_advice_categories(text: str) -> set[str]:
    """提取回复中明确给出的安全动作类别。"""
    return {
        name
        for name, pattern in _SAFETY_ADVICE.items()
        if pattern.search(text) and _ADVICE_MARKER.search(text)
    }


def _is_standalone_safety_instruction(clause: str) -> bool:
    """只将独立的安全建议视作可按类别去重，避免吞掉新业务问题。"""
    categories = _safety_advice_categories(clause)
    if not categories or not _ADVICE_MARKER.search(clause):
        return False
    # 含有业务问题、状态说明或对历史事实的描述时，不能整句删除。
    if re.search(
        r"请问|请确认|请告知|请提供|请补充|退款|退货|订单|申请|进度|原因|状态|"
        r"商品|产品是否|已记录|已说明|此前|之前|刚才|您说|您已|我已|吗|呢|[？?]",
        clause,
    ):
        return False
    return True


def _has_material_difference(candidate: str, prior: str) -> bool:
    """避免只因措辞相似，就吞掉状态、否定或数字已经变化的新事实。"""
    candidate_numbers = re.findall(r"\d+(?:\.\d+)?|[A-Z]{2,}-[A-Z0-9-]+", candidate, re.IGNORECASE)
    prior_numbers = re.findall(r"\d+(?:\.\d+)?|[A-Z]{2,}-[A-Z0-9-]+", prior, re.IGNORECASE)
    if candidate_numbers != prior_numbers:
        return True
    candidate_markers = {marker for marker in _MATERIAL_STATUS_MARKERS if marker in candidate}
    prior_markers = {marker for marker in _MATERIAL_STATUS_MARKERS if marker in prior}
    return candidate_markers != prior_markers


def _known_case_field_values(facts: Any) -> dict[str, list[Any]]:
    """仅从当前有效案件事实提取可用于阻止重复追问的字段。"""
    raw_facts = facts.get("facts", {}) if isinstance(facts, Mapping) else {}
    if not isinstance(raw_facts, Mapping):
        return {}
    unknown_values = {"", "unknown", "null", "none", "未知", "不清楚", "不确定", "不知道", "未提供", "待确认"}
    known: dict[str, list[Any]] = {}
    for field in _FIELD_QUESTION_ALIASES:
        records = raw_facts.get(field)
        if not isinstance(records, list) or not records:
            continue
        if field in {"symptom"}:
            last_clear = max(
                (
                    index
                    for index, record in enumerate(records)
                    if isinstance(record, Mapping)
                    and (record.get("confirmation_status") == "cleared" or record.get("value") is None)
                ),
                default=-1,
            )
            effective_records = records[last_clear + 1:]
        else:
            effective_records = records[-1:]
        values = [
            record.get("value")
            for record in effective_records
            if isinstance(record, Mapping)
            and record.get("confirmation_status") != "cleared"
            and record.get("value") is not None
            and str(record.get("value")).strip().casefold() not in unknown_values
        ]
        if values:
            known[field] = values
    return known


def filter_repeated_answer_content(
    text: str,
    history: Any,
    *,
    current_message: str = "",
    current_message_ref: str = "",
    case_facts: Any = None,
) -> tuple[str, int]:
    """删除本轮对历史答复的重复句，保留新问题内容与风险升级提醒。

    Args:
        text: Agent 或 ResponseComposer 生成的本轮客户答复。
        history: 近期消息对象或包含 role/content 的映射序列。
        current_message: 消费者本轮原文，用于识别明确复述请求及新增风险。
        case_facts: 会话事实字典，可提供超出近期窗口的历史 AI 答复。

    Returns:
        去重后的文本和被删除的重复句数量。若消费者要求复述、询问安全处理，
        或报告新/加重症状，则原样保留相关内容。
    """
    value = str(text or "").strip()
    if not value or (not history and not case_facts):
        return value, 0

    previous_assistant: list[str] = []
    prior_user_messages: list[str] = []
    for item in history:
        role, content = _history_role_content(item)
        if not content:
            continue
        if role in _ASSISTANT_ROLES:
            previous_assistant.append(content)
        elif role in {"user", "consumer", "客户", "消费者"}:
            prior_user_messages.append(content)
    facts = case_facts.get("facts", {}) if isinstance(case_facts, Mapping) else {}
    ai_responses = facts.get("ai_response", []) if isinstance(facts, Mapping) else []
    if isinstance(ai_responses, list):
        for record in ai_responses:
            if not isinstance(record, Mapping) or record.get("confirmation_status") == "cleared":
                continue
            content = record.get("value")
            if isinstance(content, str) and content.strip():
                previous_assistant.append(content.strip())
    symptom_records = facts.get("symptom", []) if isinstance(facts, Mapping) else []
    if isinstance(symptom_records, list):
        for record in symptom_records:
            if not isinstance(record, Mapping) or record.get("confirmation_status") == "cleared":
                continue
            if current_message_ref and str(record.get("source_ref") or "") == current_message_ref:
                continue
            symptom = record.get("value")
            if isinstance(symptom, str) and symptom.strip():
                prior_user_messages.append(symptom.strip())
    known_fields = _known_case_field_values(case_facts)
    if not previous_assistant:
        filtered, repeated_fields = filter_redundant_field_questions(value, known_fields)
        return filtered, len(repeated_fields)
    current_message = str(current_message or "")
    if _REPEAT_REQUEST.search(current_message):
        return value, 0
    safety_topic_reopened = _safety_topic_reopened(current_message, prior_user_messages)

    prior_clauses = [clause for answer in previous_assistant for clause in _split_clauses(answer)]
    normalized_prior = [
        _CLAUSE_NORMALIZE.sub("", clause).casefold()
        for clause in prior_clauses
    ]
    covered_safety_actions = set().union(*(_safety_advice_categories(answer) for answer in previous_assistant))
    parts = _CLAUSE_SPLIT.split(value)
    kept: list[str] = []
    removed = 0
    for index in range(0, len(parts), 2):
        clause = parts[index]
        delimiter = parts[index + 1] if index + 1 < len(parts) else ""
        normalized = _CLAUSE_NORMALIZE.sub("", clause).casefold()
        if not normalized:
            kept.append(clause + delimiter)
            continue

        duplicate = any(
            normalized == prior
            or (
                len(normalized) >= 24
                and len(prior) >= 24
                and not _has_material_difference(clause, prior_clause)
                and SequenceMatcher(None, normalized, prior, autojunk=False).ratio() >= 0.82
            )
            for prior_clause, prior in zip(prior_clauses, normalized_prior)
            if prior
        )
        repeated_safety_action = bool(
            _is_standalone_safety_instruction(clause)
            and covered_safety_actions.intersection(_safety_advice_categories(clause))
        )
        if safety_topic_reopened and _safety_advice_categories(clause):
            # 新风险/安全追问允许重述安全建议，但不豁免退款等其它内容的去重。
            duplicate = False
            repeated_safety_action = False
        if duplicate or repeated_safety_action:
            removed += 1
            continue
        kept.append(clause + delimiter)

    filtered = "".join(kept).strip()
    filtered, repeated_fields = filter_redundant_field_questions(filtered, known_fields)
    total_removed = removed + len(repeated_fields)
    if total_removed and filtered:
        filtered = filtered.rstrip("，,；;、")
        if filtered and filtered[-1] not in "。！？!?":
            filtered += "。"
    return filtered, total_removed


def filter_redundant_field_questions(
    text: str,
    known_fields: Mapping[str, Any],
    *,
    allowed_fields: set[str] | None = None,
) -> tuple[str, list[str]]:
    """移除已回答字段及业务契约未允许的字段追问。

    ``allowed_fields`` 为当前意图注册表明确要求且尚未补齐的字段；传入时，
    其它可识别字段的问句不允许由模型自行扩大。没有业务字段契约的咨询场景
    仍只过滤已知字段，保留模型必要的澄清问题。
    """

    value = str(text or "").strip()
    if not value:
        return value, []

    known = known_fields if isinstance(known_fields, Mapping) else {}
    if not known and allowed_fields is None:
        return value, []

    known_scope_question = bool(known.get("return_scope")) and bool(re.search(
        r"(?:整单|全单|只退|正装|小样|赠品).{0,16}(?:还是|或).{0,16}(?:整单|全单|只退|正装|小样|赠品)",
        value,
    ))
    parts = re.split(r"([，,；;。！？?!\n]+)", value)
    kept: list[str] = []
    removed_fields: list[str] = []
    for index in range(0, len(parts), 2):
        clause = parts[index]
        delimiter = parts[index + 1] if index + 1 < len(parts) else ""
        normalized = clause.strip()
        has_question_marker = (
            any(marker in normalized for marker in _FIELD_QUESTION_MARKERS)
            or delimiter in {"?", "？"}
        )
        matched_fields = []
        for field, aliases in _FIELD_QUESTION_ALIASES.items():
            is_known = bool(known.get(field))
            is_allowed = allowed_fields is None or field in allowed_fields
            if not is_known and is_allowed:
                continue
            if (
                field == "return_scope"
                and known_scope_question
                and any(term in normalized for term in ("整单", "全单", "只退", "正装", "小样", "赠品"))
            ):
                matched_fields.append(field)
                continue
            patterns = _FIELD_QUESTION_PATTERNS.get(field)
            if patterns:
                if any(re.search(pattern, normalized) for pattern in patterns) and (
                    is_known or has_question_marker
                ):
                    matched_fields.append(field)
            elif has_question_marker and any(alias in normalized for alias in aliases):
                matched_fields.append(field)
        if matched_fields:
            removed_fields.extend(matched_fields)
            continue
        if clause or delimiter:
            kept.append(clause + delimiter)

    filtered = "".join(kept).strip(" ，,；;。！？?!\n")
    filtered = re.sub(
        r"(^|[，,；;。！？?!\n])\s*(?:另外|此外|还有|同时|并且|以及|然后)\s*(?=[，,；;。！？?!\n]|$)",
        r"\1",
        filtered,
    ).strip(" ，,；;。！？?!\n")
    if filtered and value.endswith(tuple("。！？?!")) and filtered[-1] not in "。！？?!":
        filtered += value[-1]
    return filtered, list(dict.fromkeys(removed_fields))


def asked_fields_in_text(text: str, candidate_fields: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    """识别回复中实际提出的字段问题，用于记录下一轮等待的自由文本槽位。"""

    clauses = re.split(r"[。！？!?；;\n]+", str(text or ""))
    asked: list[str] = []
    for field in candidate_fields:
        aliases = _FIELD_QUESTION_ALIASES.get(str(field), ())
        patterns = _FIELD_QUESTION_PATTERNS.get(str(field), ())
        for clause in clauses:
            normalized = clause.strip()
            if not normalized:
                continue
            if any(re.search(pattern, normalized) for pattern in patterns):
                asked.append(str(field))
                break
            has_question_marker = any(marker in normalized for marker in _FIELD_QUESTION_MARKERS)
            if has_question_marker and any(alias in normalized for alias in aliases):
                asked.append(str(field))
                break
    return asked


def build_return_refund_request_reply(
    ticket_draft: Mapping[str, Any] | None,
    previous_ticket: Mapping[str, Any] | None = None,
) -> str:
    """基于本地待审核草稿回应退款申请，不把消费者导向聊天框之外。

    Args:
        ticket_draft: 本轮按退款意图生成的本地工单草稿。
        previous_ticket: 同一会话仍未审核的草稿，用于复用其已知字段。

    Returns:
        只确认申请草稿或询问首个缺失退款字段的消费者话术；不表示退款已执行。
    """

    ticket = ticket_draft if isinstance(ticket_draft, Mapping) else {}
    fields = dict(ticket.get("fields") or {})
    previous = previous_ticket if isinstance(previous_ticket, Mapping) else {}
    if previous.get("status") == "draft":
        prior_fields = previous.get("fields")
        if isinstance(prior_fields, Mapping):
            for field, value in prior_fields.items():
                if fields.get(field) in (None, "") and value not in (None, ""):
                    fields[field] = value

    blocking_fields = ticket.get("blocking_fields")
    if not isinstance(blocking_fields, (list, tuple)) or not blocking_fields:
        blocking_fields = ("order_id", "return_reason", "opened_or_used")
    missing = [field for field in blocking_fields if fields.get(field) in (None, "")]
    if missing:
        prompts = {
            "order_id": "请把订单号直接发在这里。",
            "return_reason": "这次退款的原因是什么？",
            "opened_or_used": "商品目前是未拆封、已拆封，还是已经使用？",
        }
        prompt = prompts.get(str(missing[0]), "请补充办理退款所必需的信息。")
        if ticket.get("status") == "draft":
            return f"我先把退款申请登记为待客服审核的草稿。目前还缺少信息：{prompt}"
        return f"我可以在这里帮你登记退款申请。{prompt}"

    recorded = []
    order_id = str(fields.get("order_id") or "").strip()
    reason = str(fields.get("return_reason") or "").strip()
    product_status = str(fields.get("opened_or_used") or "").strip()
    if order_id:
        recorded.append(f"订单 {order_id}")
    if reason:
        recorded.append(f"退款原因“{reason[:80]}”")
    if product_status:
        recorded.append(f"商品状态“{product_status[:80]}”")
    details = f"（已记录：{'；'.join(recorded)}）" if recorded else ""
    return (
        f"这笔退款申请已登记为待客服审核的草稿{details}。"
        "审核结果会在这里回复；这还不代表退款已经完成。"
    )


def filter_external_refund_instructions(text: str) -> tuple[str, bool]:
    """移除要求消费者离开聊天框到订单/平台页面办理退款的句子。

    Args:
        text: Agent 或合成器生成的答复。

    Returns:
        过滤后的正文及是否移除了外部页面操作指引。
    """

    parts = re.split(r"([。！？!?；;\n]+)", str(text or ""))
    kept: list[str] = []
    removed = False
    for index in range(0, len(parts), 2):
        clause = parts[index]
        delimiter = parts[index + 1] if index + 1 < len(parts) else ""
        if _EXTERNAL_REFUND_ACTION.search(clause):
            removed = True
            continue
        if clause or delimiter:
            kept.append(clause + delimiter)
    return "".join(kept).strip(), removed


def guard_response(text: str) -> str:
    """保留模型原话；发现内部字段或不安全承诺时整句替换，不做词语级删改。"""
    value = str(text or "").strip()
    normalized = value.casefold()
    if any(marker.casefold() in normalized for marker in _INTERNAL_MARKERS):
        return _SAFE_REPLACEMENT
    for marker in _UNSAFE_CLAIMS:
        start = 0
        while (index := normalized.find(marker.casefold(), start)) >= 0:
            prefix = normalized[max(0, index - 12):index]
            suffix = normalized[index + len(marker):index + len(marker) + 3]
            if not any(phrase in prefix for phrase in _NEGATED_OR_QUESTIONED) and not suffix.startswith(("吗", "呢", "？", "?")):
                return _SAFE_REPLACEMENT
            start = index + len(marker)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return value[:1200]


def guard_handoff_suggested_reply(text: str, fallback: str) -> tuple[str, bool]:
    """Reject transfer copy or unverified completion claims in an agent-side suggestion."""
    value = guard_response(text)
    normalized = value.casefold()
    if not value or value == _SAFE_REPLACEMENT or any(marker.casefold() in normalized for marker in _UNSAFE_HANDOFF_SUGGESTIONS):
        safe_fallback = guard_response(fallback)
        return (safe_fallback or _SAFE_REPLACEMENT), True
    return value, False
